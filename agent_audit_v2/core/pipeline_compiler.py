"""
agent_audit_v2/core/pipeline_compiler.py

Minimal LangGraph pipeline compiler.

Purpose
-------
Compile a small declarative pipeline of Stage objects into a LangGraph execution
graph with support for:

- sequential stages
- optional audit nodes after a stage
- optional retry loops via conditional routing

This is intentionally minimal. It does NOT yet support:

- parallel branches
- YAML config loading
- logging / S3 integration
- advanced reducers
- dynamic loop bounds beyond per-stage retry logic

Design notes
------------
- State is assumed to be a plain dict.
- Dynamic execution metadata lives under:
      state["round_context"]["audit"]
      state["round_context"]["retry"]
- Audit functions are assumed to be pure callables:
      audit_fn(state: dict) -> dict | Any
- Retry condition is assumed to be:
      retry_condition(state: dict) -> bool
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from langgraph.graph import END, START, StateGraph


State = dict[str, Any]
NodeFn = Callable[[State], State]
AuditFn = Callable[[State], Any]
RetryConditionFn = Callable[[State], bool]


@dataclass
class Stage:
    """
    Minimal stage abstraction.

    Attributes
    ----------
    name:
        Unique stage name within the compiled pipeline.
    fn:
        LangGraph-compatible node function: fn(state) -> state_update_dict
        or fn(state) -> full_state_dict.
    audits:
        Optional list of audit callables to run after the stage.
    retry_condition:
        Optional predicate. If True, retry the same stage, bounded by max_retries.
    max_retries:
        Maximum number of retries for this stage.
    """

    name: str
    fn: NodeFn
    audits: list[AuditFn] = field(default_factory=list)
    retry_condition: Optional[RetryConditionFn] = None
    max_retries: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Stage.name must be non-empty")
        if not callable(self.fn):
            raise TypeError(f"Stage.fn for {self.name!r} must be callable")
        if self.max_retries < 0:
            raise ValueError(f"Stage.max_retries for {self.name!r} must be >= 0")


class PipelineCompiler:
    """
    Compile a list of Stage objects into a LangGraph execution graph.
    """

    def __init__(self, state_type: type = dict) -> None:
        self.state_type = state_type

    def compile(self, stages: Sequence[Stage]):
        """
        Build and compile a LangGraph graph.

        Parameters
        ----------
        stages:
            Ordered sequence of Stage objects.

        Returns
        -------
        Compiled LangGraph graph.

        Graph shape
        -----------
        For a normal stage:
            stage_i -> stage_{i+1}

        For a stage with audits:
            stage_i -> stage_i__audit -> stage_{i+1}

        For a stage with retry condition:
            stage_i -> [audit?] -> stage_i__retry_gate
                                      ├── retry -> stage_i
                                      └── continue -> next
        """
        if not stages:
            raise ValueError("PipelineCompiler.compile() requires at least one stage")

        self._validate_unique_names(stages)

        builder = StateGraph(self.state_type)

        # 1. Add all stage nodes + helper nodes
        for stage in stages:
            builder.add_node(stage.name, self._make_stage_node(stage))

            if stage.audits:
                builder.add_node(self._audit_node_name(stage), self._make_audit_node(stage))

            if stage.retry_condition is not None:
                builder.add_node(self._retry_gate_node_name(stage), self._make_retry_gate_node(stage))

        # 2. Wire start -> first stage
        builder.add_edge(START, stages[0].name)

        # 3. Wire each stage chain
        for i, stage in enumerate(stages):
            next_target = stages[i + 1].name if i + 1 < len(stages) else END

            terminal_after_stage = stage.name

            if stage.audits:
                audit_name = self._audit_node_name(stage)
                builder.add_edge(stage.name, audit_name)
                terminal_after_stage = audit_name

            if stage.retry_condition is not None:
                gate_name = self._retry_gate_node_name(stage)
                builder.add_edge(terminal_after_stage, gate_name)

                builder.add_conditional_edges(
                    gate_name,
                    self._make_retry_router(stage, next_target),
                    {
                        "retry": stage.name,
                        "continue": next_target,
                    },
                )
            else:
                builder.add_edge(terminal_after_stage, next_target)

        return builder.compile()

    @staticmethod
    def _validate_unique_names(stages: Sequence[Stage]) -> None:
        seen: set[str] = set()
        for stage in stages:
            if stage.name in seen:
                raise ValueError(f"Duplicate stage name: {stage.name}")
            seen.add(stage.name)

    @staticmethod
    def _ensure_round_context(state: State) -> None:
        rc = state.setdefault("round_context", {})
        if not isinstance(rc, dict):
            raise TypeError("state['round_context'] must be a dict")

        rc.setdefault("control", {})
        rc.setdefault("audit", {})
        rc.setdefault("retry", {})

    @staticmethod
    def _merge_state(original: State, update: State | None) -> State:
        """
        Merge a stage/audit/gate update into a copied state dict.

        Convention:
        - If node returns None, treat as no-op.
        - If node returns a dict, shallow-merge into copied state.
        """
        merged = dict(original)
        if update:
            merged.update(update)
        return merged

    @staticmethod
    def _audit_node_name(stage: Stage) -> str:
        return f"{stage.name}__audit"

    @staticmethod
    def _retry_gate_node_name(stage: Stage) -> str:
        return f"{stage.name}__retry_gate"

    def _make_stage_node(self, stage: Stage) -> NodeFn:
        def node(state: State) -> State:
            self._ensure_round_context(state)
            update = stage.fn(dict(state))
            return self._merge_state(state, update)
        return node

    def _make_audit_node(self, stage: Stage) -> NodeFn:
        def node(state: State) -> State:
            self._ensure_round_context(state)

            rc = state["round_context"]
            audit_bucket = rc["audit"]
            stage_bucket = audit_bucket.setdefault(stage.name, {})

            # Run all audits and store results keyed by audit function name.
            for audit_fn in stage.audits:
                audit_name = getattr(audit_fn, "__name__", audit_fn.__class__.__name__)
                result = audit_fn(dict(state))
                stage_bucket[audit_name] = result

            return dict(state)
        return node

    def _make_retry_gate_node(self, stage: Stage) -> NodeFn:
        """
        A no-op node whose purpose is to provide a routing point after audits/stage.
        """
        def node(state: State) -> State:
            self._ensure_round_context(state)
            return dict(state)
        return node

    def _make_retry_router(self, stage: Stage, next_target: str):
        """
        Returns a conditional router function for LangGraph.

        The router:
        - checks the stage retry condition
        - checks current retry count
        - increments retry count if retrying
        - resets/initializes retry count when continuing
        """
        def router(state: State) -> str:
            self._ensure_round_context(state)

            rc = state["round_context"]
            retry_bucket = rc["retry"]
            current_count = int(retry_bucket.get(stage.name, 0))

            should_retry = False
            if stage.retry_condition is not None:
                should_retry = bool(stage.retry_condition(dict(state)))

            if should_retry and current_count < stage.max_retries:
                retry_bucket[stage.name] = current_count + 1
                return "retry"

            # Keep count in state for observability; do not delete automatically.
            retry_bucket.setdefault(stage.name, current_count)
            return "continue"

        return router


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    def propose_node(state: State) -> State:
        x = int(state.get("x", 0))
        messages = list(state.get("messages", []))
        messages.append(f"propose(x={x})")
        return {
            "x": x + 1,
            "messages": messages,
        }

    def critique_node(state: State) -> State:
        x = int(state.get("x", 0))
        messages = list(state.get("messages", []))
        messages.append(f"critique(x={x})")
        return {
            "messages": messages,
            "score": 0.4 if x < 3 else 0.9,  # force retries early
        }

    def judge_node(state: State) -> State:
        x = int(state.get("x", 0))
        messages = list(state.get("messages", []))
        messages.append(f"judge(x={x})")
        return {
            "final": f"done:{x}",
            "messages": messages,
        }

    def crit_audit(state: State) -> dict[str, float]:
        return {"crit_score": float(state.get("score", 1.0))}

    def retry_if_low_score(state: State) -> bool:
        rc = state.get("round_context", {})
        audit = rc.get("audit", {})
        critique_audits = audit.get("critique", {})
        audit_result = critique_audits.get("crit_audit", {})
        crit_score = audit_result.get("crit_score", 1.0) if isinstance(audit_result, dict) else 1.0
        return crit_score < 0.7

    pipeline = [
        Stage(name="propose", fn=propose_node),
        Stage(
            name="critique",
            fn=critique_node,
            audits=[crit_audit],
            retry_condition=retry_if_low_score,
            max_retries=2,
        ),
        Stage(name="judge", fn=judge_node),
    ]

    compiler = PipelineCompiler(dict)
    graph = compiler.compile(pipeline)

    initial_state: State = {
        "x": 0,
        "messages": [],
        "round_context": {
            "control": {},
            "audit": {},
            "retry": {},
        },
    }

    result = graph.invoke(initial_state)

    print("\n=== FINAL STATE ===")
    for key, value in result.items():
        print(f"{key}: {value}")