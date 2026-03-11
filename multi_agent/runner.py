"""
High-level runner for the multi-agent debate system.

WHAT CHANGED AND WHY
---------------------
Previously, the runner compiled a single monolithic LangGraph and called
graph.invoke() once.  The debate loop (propose → critique → revise →
repeat → judge) was managed entirely inside the graph via a conditional
edge (should_continue).  This made it impossible for external code to
inspect or modify state between rounds — there was no seam to hook into.

This module now decomposes execution into three phases, each backed by
its own LangGraph sub-graph:

    Phase 1 — Pipeline:      news → data → build_context
    Phase 2 — Debate rounds:  for t in range(max_rounds):
                                  propose → critique → revise → CRIT → PID
    Phase 3 — Finalize:       judge → build_trace

The runner owns the iteration loop (Phase 2), calling per-phase sub-graphs
once per round.  CRIT runs once per round after revise, scoring each agent
independently via parallel LLM calls (no cross-agent contamination).
PID updates beta once per round from the aggregate CRIT score.

    for t in range(max_rounds):
        state = self._run_round_with_pid(state, t + 1)
        # Inside _run_round_with_pid:
        #   propose(β) → critique(β) → revise(β) → _crit_and_pid_step()
        if self.should_terminate(state):
            break

HOW INVARIANTS ARE PRESERVED
-----------------------------
1. All three phases use graph.invoke() — never manual state.update().
   This ensures LangGraph's reducers (Annotated[list, operator.add]
   for debate_turns) are applied correctly at every phase boundary.

2. Between sub-graph invocations the runner passes the full state dict.
   LangGraph treats the input dict as starting state and applies each
   node's returned dict through the reducer.  So debate_turns accumulates
   across pipeline → rounds → finalize seamlessly.

3. propose_node has an idempotency guard: if proposals already exist,
   it returns {} (no-op).  This means the single_round_graph can include
   propose in every round, but proposals are only generated once (round 1).

4. current_round is set by the runner to t+1 before each round AND is
   returned by the node functions (for monolithic graph compatibility).
   The two sources agree: runner sets t+1, propose returns 1 (round 1 only),
   revise returns current_round+1 (same as the next iteration's t+1).

5. proposals (plain list, no reducer) — set once, preserved by guard.
   critiques/revisions (plain list, no reducer) — replaced each round.
   debate_turns (Annotated[list, operator.add]) — appended each round.
   These behaviors match the monolithic graph exactly.

6. run_returning_state() duplicates run()'s logic but returns raw state
   instead of (Action, AgentTrace), without disk I/O.  This enables
   equivalence testing against the old monolithic path.

Usage:
    from multi_agent import MultiAgentRunner, DebateConfig, Observation

    config = DebateConfig(
        roles=["macro", "value", "risk", "technical"],
        max_rounds=2,
        agreeableness=0.3,
        enable_adversarial=True,
        mock=True,
    )
    runner = MultiAgentRunner(config)
    action, trace = runner.run(observation)
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Dedicated PID loggers.  pid.metrics logs at INFO so PID output is
# visible by default (pid_log_metrics defaults to True).  Instance flags
# gate whether info() calls are made, so multiple runners with different
# pid_log_* settings don't conflict on the singleton.
pid_metrics_logger = logging.getLogger("pid.metrics")
pid_metrics_logger.setLevel(logging.INFO)
pid_llm_logger = logging.getLogger("pid.llm")


def _round_floats(obj, ndigits=4):
    """Recursively round floats in nested dicts/lists for clean JSON output."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    return obj


from .config import DebateConfig
from .prompts.registry import beta_to_bucket, build_prompt_manifest, reset_registry_cache
from .terminal_display import (
    render_round_header,
    render_phase_label,
    render_portfolio_table,
    render_phase_metrics,
    render_judge_result,
    render_debate_end,
)
from .graph import (
    compile_finalize_graph,
    compile_parallel_single_round_graph,
    compile_pipeline_graph,
    compile_single_round_graph,
    # Per-phase graphs (for PID intervention)
    compile_propose_graph,
    compile_critique_graph,
    compile_revise_graph,
    compile_parallel_propose_graph,
    compile_parallel_critique_graph,
    compile_parallel_revise_graph,
    _call_llm,
    _print_comparison_table,
)
from .graph.llm import _extract_snapshot_id, prompt_logger
from .models import (
    Action,
    AgentTrace,
    Claim,
    ControllerOutput,
    DebateTurn,
    Observation,
    Order,
    PIDEvent,
    RoundMetrics,
)


def _extract_thesis(action_dict: dict, role: str = "", phase: str = "") -> str:
    """Extract thesis from action_dict, supporting canonical and legacy formats.

    Reading priority: thesis → portfolio_rationale → justification.
    Falls back to empty string.
    """
    tag = f"[{role}/{phase}] " if role else ""
    thesis = action_dict.get("thesis")
    if thesis:
        return thesis
    thesis = action_dict.get("portfolio_rationale")
    if thesis:
        logger.debug("%sUsing 'portfolio_rationale' for thesis (no 'thesis' field).", tag)
        return thesis
    thesis = action_dict.get("justification")
    if thesis:
        logger.debug("%sUsing 'justification' for thesis (no 'thesis' or 'portfolio_rationale' field).", tag)
        return thesis
    logger.warning(
        "%sCRIT bundle thesis is EMPTY — agent output has no "
        "'thesis', 'portfolio_rationale', or 'justification'.  "
        "action_dict keys: %s  "
        "This will degrade CRIT evidential_support scoring.",
        tag, sorted(action_dict.keys()),
    )
    return ""


def _normalize_claims(claims: list[dict], normalize_evidence_id) -> list[dict]:
    """Normalize claims for CRIT bundle. Ensures all canonical fields exist."""
    normalized = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        if not claim.get("claim_text"):
            continue
        raw_evidence = claim.get("evidence", [])
        normalized.append({
            "claim_id": claim.get("claim_id", ""),
            "claim_text": claim.get("claim_text", ""),
            "claim_type": claim.get("claim_type", "unknown"),
            "reasoning_type": claim.get("reasoning_type", ""),
            "evidence": raw_evidence,
            "evidence_ids": sorted(set(
                normalize_evidence_id(e) for e in raw_evidence if isinstance(e, str)
            )),
            "assumptions": claim.get("assumptions", []),
            "falsifiers": claim.get("falsifiers", []),
            "impacts_positions": claim.get("impacts_positions", []),
            "confidence": claim.get("confidence", 0.5),
        })
    return normalized


def _normalize_position_rationale(entries: list[dict]) -> list[dict]:
    """Normalize position rationale for CRIT bundle."""
    normalized = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        normalized.append({
            "ticker": entry.get("ticker", ""),
            "weight": entry.get("weight", 0.0),
            "supporting_claims": entry.get("supporting_claims") or entry.get("supported_by_claims", []),
            "explanation": entry.get("explanation") or entry.get("rationale") or "",
        })
    return normalized


def _extract_reasoning(
    action_dict: dict, normalize_evidence_id, *, revision_notes: str = "",
) -> dict:
    """Extract structured reasoning from an action_dict for CRIT bundle.

    This function defines the canonical reasoning bundle schema used by CRIT.
    All downstream evaluators must rely on this structure rather than parsing
    raw LLM text. If fields change here, CRIT prompts and evaluation schemas
    must be updated together.
    """
    thesis = (action_dict.get("thesis")
               or action_dict.get("portfolio_rationale")
               or action_dict.get("justification"))
    if not thesis:
        raise RuntimeError(
            f"action_dict has no thesis/portfolio_rationale/justification. "
            f"Keys: {list(action_dict.keys())}"
        )

    result = {
        "claims": _normalize_claims(action_dict["claims"], normalize_evidence_id),
        "position_rationale": _normalize_position_rationale(
            action_dict["position_rationale"]
        ),
        "thesis": thesis,
        "confidence": action_dict["confidence"],
        "risks_or_falsifiers": action_dict["risks_or_falsifiers"],
    }
    if revision_notes:
        result["revision_notes"] = revision_notes
        result["critique_responses"] = action_dict["critique_responses"]
    return result


def _extract_citations(
    action_dict: dict, raw_text: str, extract_evidence_ids,
    *, role: str = "", phase: str = "",
) -> list[dict]:
    """Extract evidence citations from action_dict or raw text.

    Priority order:
    1. Top-level ``evidence_citations`` (base format)
    2. ``claims[].evidence`` arrays (enriched format)
    3. Regex extraction from raw LLM response text (fallback)
    """
    import copy
    from eval.evidence import normalize_evidence_id

    tag = f"[{role}/{phase}] " if role else ""

    # 1. Top-level structured citations (base format)
    top_level = action_dict.get("evidence_citations", [])
    if top_level:
        return copy.deepcopy(top_level)

    # 2. Extract from claims[].evidence (enriched format)
    claims = action_dict.get("claims", [])
    if claims:
        seen = set()
        citations = []
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            for ev in claim.get("evidence", []):
                # Evidence entries may be "[AAPL-RET60]" or "AAPL-RET60"
                eid = normalize_evidence_id(ev)
                if eid and eid not in seen:
                    seen.add(eid)
                    citations.append({"evidence_id": eid})
        if citations:
            return citations

    # 3. Fallback: regex extraction from raw response text
    regex_ids = sorted(extract_evidence_ids(raw_text))
    if regex_ids:
        logger.warning(
            "%sNo structured evidence found in agent output — fell back to regex "
            "extraction from raw text (%d IDs found).  Check agent output format.",
            tag, len(regex_ids),
        )
    else:
        logger.warning(
            "%sNo evidence citations found anywhere — agent output has no "
            "'evidence_citations', no claims[].evidence, and no bracket IDs in "
            "raw text.  CRIT evidential_support will be degraded.", tag,
        )
    return [{"evidence_id": eid} for eid in regex_ids]


def build_reasoning_bundle(
    state: dict,
    role: str,
    round_num: int,
    memo_evidence_lookup: dict[str, str],
) -> dict | None:
    """Assemble one agent's reasoning bundle from debate state.

    Returns the structured bundle dict for CRIT evaluation, or None
    if the agent has no proposal or revision in this state.

    Critical: critiques_received includes ONLY critiques targeting this
    agent. Critiques list is reset each round, so all entries are current.
    """
    from eval.evidence import enrich_evidence_citations, expand_evidence_ids_inline, extract_evidence_ids, normalize_evidence_id
    import copy

    # Find this agent's proposal — proposals MUST exist at this point.
    proposals = state["proposals"]
    proposal = None
    for p in proposals:
        if p["role"] == role:
            proposal = p
            break
    if proposal is None:
        return None

    # Find this agent's LATEST revision (fall back to proposal if none)
    revision = None
    for r in reversed(state.get("revisions") or []):
        if r["role"] == role:
            revision = r
            break
    if revision is None:
        revision = proposal

    # Filter critiques: only those targeting this agent.
    # Also extract this agent's own self_critique.
    # LLM outputs vary in casing ("MACRO" vs "macro") and may include
    # suffixes ("MACRO agent"), so we normalize before comparing.
    critiques_received = []
    self_critique = None
    role_lower = role.lower()
    for critic in state.get("critiques") or []:
        from_role = critic["role"]
        # Capture this agent's self_critique
        if from_role == role:
            self_critique = critic["self_critique"]
        for crit in critic["critiques"]:
            target = crit.get("target_role", "").lower().split()[0] if crit.get("target_role") else ""
            if target == role_lower:
                # Critique templates output "counter_evidence" (list of
                # bracketed evidence IDs).  Convert to the structured
                # format that enrich_evidence_citations expects.
                raw_ev = crit.get("counter_evidence") or crit.get("evidence_citations") or []
                if not raw_ev:
                    logger.warning(
                        "[%s/critique] Critique from %s has no counter_evidence "
                        "and no evidence_citations — CRIT cannot evaluate "
                        "evidence quality of this critique.", role, from_role,
                    )
                structured_ev = []
                for ev in raw_ev:
                    if isinstance(ev, dict):
                        structured_ev.append(copy.deepcopy(ev))
                    elif isinstance(ev, str):
                        structured_ev.append({"evidence_id": normalize_evidence_id(ev)})
                critiques_received.append({
                    "from_role": from_role,
                    "target_claim": crit["target_claim"],
                    "critique_text": crit["objection"],
                    "evidence_citations": structured_ev,
                    "portfolio_implication": crit["portfolio_implication"],
                    "suggested_adjustment": crit["suggested_adjustment"],
                    "falsifier": crit["falsifier"],
                    "objection_confidence": crit["objection_confidence"],
                })

    # Extract action_dict content for proposal and revision — these MUST exist.
    prop_action = proposal["action_dict"]
    if not isinstance(prop_action, dict):
        raise TypeError(
            f"proposal['action_dict'] for {role} must be a dict, "
            f"got {type(prop_action).__name__}"
        )
    rev_action = revision["action_dict"]
    if not isinstance(rev_action, dict):
        raise TypeError(
            f"revision['action_dict'] for {role} must be a dict, "
            f"got {type(rev_action).__name__}"
        )

    # Build proposal bundle with embedded evidence
    prop_citations = _extract_citations(
        prop_action, proposal.get("raw_response") or "", extract_evidence_ids,
        role=role, phase="propose",
    )
    enrich_evidence_citations(prop_citations, memo_evidence_lookup)
    proposal_bundle = {
        "portfolio_allocation": prop_action["allocation"],
        "reasoning": _extract_reasoning(prop_action, normalize_evidence_id),
        "raw_response": proposal.get("raw_response") or "",
        "evidence_citations": prop_citations,
    }

    # Build revision bundle with embedded evidence
    rev_citations = _extract_citations(
        rev_action, revision.get("raw_response") or "", extract_evidence_ids,
        role=role, phase="revise",
    )
    enrich_evidence_citations(rev_citations, memo_evidence_lookup)
    revised_bundle = {
        "portfolio_allocation": rev_action["allocation"],
        "reasoning": _extract_reasoning(
            rev_action, normalize_evidence_id,
            revision_notes=revision.get("revision_notes") or "",
        ),
        "raw_response": revision.get("raw_response") or "",
        "evidence_citations": rev_citations,
    }

    # Enrich critique citations
    for crit in critiques_received:
        enrich_evidence_citations(
            crit["evidence_citations"], memo_evidence_lookup,
        )

    # Expand evidence IDs inline in text fields
    for bundle_part in (proposal_bundle, revised_bundle):
        bundle_part["reasoning"]["thesis"] = expand_evidence_ids_inline(bundle_part["reasoning"]["thesis"], memo_evidence_lookup)
        bundle_part["raw_response"] = expand_evidence_ids_inline(bundle_part["raw_response"], memo_evidence_lookup)
    for crit in critiques_received:
        crit["critique_text"] = expand_evidence_ids_inline(crit["critique_text"], memo_evidence_lookup)

    bundle = {
        "round": round_num,
        "agent_role": role,
        "proposal": proposal_bundle,
        "critiques_received": critiques_received,
        "revised_argument": revised_bundle,
    }
    if self_critique is not None:
        bundle["self_critique"] = self_critique
    return bundle


class MultiAgentRunner:
    """
    Orchestrates multi-agent debate for trading decisions.

    Wraps three LangGraph sub-graphs (pipeline, single-round, finalize)
    with a clean interface matching Deveen's SingleAgentRunner /
    MajorityVoteRunner / DebateRunner pattern.

    The iteration loop lives in the runner so that external controllers
    (e.g. PID) can intervene between rounds.
    """

    def __init__(self, config: DebateConfig | None = None):
        self.config = config or DebateConfig()

        # Auto-inject devil's advocate if adversarial mode is enabled.
        # Copy roles to avoid mutating a shared config object.
        if self.config.enable_adversarial:
            if "devils_advocate" not in self.config.roles:
                self.config.roles = list(self.config.roles) + ["devils_advocate"]

        self.pipeline_graph = compile_pipeline_graph(self.config)
        if self.config.parallel_agents:
            self.single_round_graph = compile_parallel_single_round_graph(self.config)
        else:
            self.single_round_graph = compile_single_round_graph(self.config)
        self.finalize_graph = compile_finalize_graph(self.config)

        # --- Per-phase sub-graphs (compiled only when PID enabled) ---
        self._propose_graph = None
        self._critique_graph = None
        self._revise_graph = None

        if self.config.pid_enabled or self.config.intervention_config:
            if self.config.parallel_agents:
                self._propose_graph = compile_parallel_propose_graph(self.config)
                self._critique_graph = compile_parallel_critique_graph(self.config)
                self._revise_graph = compile_parallel_revise_graph(self.config)
            else:
                self._propose_graph = compile_propose_graph(self.config)
                self._critique_graph = compile_critique_graph(self.config)
                self._revise_graph = compile_revise_graph(self.config)

        # --- PID logging flags (instance-level, not mutating global logger) ---
        self._log_metrics = self.config.pid_log_metrics
        self._log_llm = self.config.pid_log_llm_calls

        # --- CRIT scorer (uses dedicated model, default gpt-5-mini) ---
        self._crit_scorer = None
        if self.config.pid_enabled or self.config.intervention_config:
            from eval.crit import CritScorer

            crit_config = self.config.to_dict()
            crit_config["model_name"] = self.config.crit_model_name
            base_llm_fn = lambda sys, usr, **kw: _call_llm(crit_config, sys, usr, phase="crit", **kw)

            def _logging_llm_fn(system_prompt: str, user_prompt: str, **kw) -> str:
                if self._log_llm:
                    pid_llm_logger.debug(
                        "[CRIT LLM REQUEST]\n"
                        "===== SYSTEM PROMPT =====\n%s\n"
                        "===== USER PROMPT =====\n%s",
                        system_prompt, user_prompt,
                    )
                response = base_llm_fn(system_prompt, user_prompt, **kw)
                if self._log_llm:
                    pid_llm_logger.debug(
                        "[CRIT LLM RESPONSE]\n%s", response,
                    )
                return response

            def _crit_capture(role, system_prompt, user_prompt, raw_response):
                self._crit_current_captures[role] = {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "raw_response": raw_response,
                }

            self._crit_scorer = CritScorer(
                llm_fn=_logging_llm_fn, capture_fn=_crit_capture,
                crit_system_template=self.config.crit_system_template,
                crit_user_template=self.config.crit_user_template,
            )

        # --- CRIT prompt/response capture ---
        self._crit_current_captures: dict[str, dict] = {}
        self._crit_round1_captures: dict[str, dict] | None = None
        self._last_round_crit = None  # latest RoundCritResult for post-crit checkpoint

        # --- PID controller ---
        self._pid_controller = None
        self._pid_events: list[PIDEvent] = []
        self._debate_id: str = str(uuid.uuid4())
        self._pid_phase_data: list[dict] = []
        self._stable_rounds: int = 0
        self._prev_rho_bar: float | None = None
        self._original_beta = self.config.initial_beta
        self._memo_evidence_lookup: dict[str, str] = {}

        # --- Structured debate logger ---
        self._debate_logger = None
        if self.config.logging_mode != "off":
            from .debate_logger import DebateLogger
            experiment = self.config.experiment_name or "default"
            self._debate_logger = DebateLogger(self.config, experiment)

        # --- Intervention engine (intra-round retry on acute failures) ---
        self._intervention_engine = None
        if self.config.intervention_config:
            from eval.interventions import build_intervention_engine
            self._intervention_engine = build_intervention_engine(
                self.config.intervention_config
            )

        if self.config.pid_config:
            from eval.PID.controller import PIDController
            from eval.PID.stability import validate_gains

            pid_cfg = self.config.pid_config
            # TODO: re-enable after tuning gains for stability
            # validate_gains(
            #     pid_cfg.gains,
            #     pid_cfg.T_max,
            #     pid_cfg.gamma_beta,
            #     rho_star=pid_cfg.rho_star,
            #     mu=pid_cfg.mu,
            # )
            self._pid_controller = PIDController(pid_cfg, self.config.initial_beta)

    def _reset_per_invocation_state(self) -> None:
        """Reset mutable state that must not leak between run() calls.

        A single DebateAgentSystem (and thus MultiAgentRunner) is reused
        across all decision points within an episode.  PID events and
        controller state must be fresh for each decision point.
        """
        self._pid_events = []
        self._debate_id = str(uuid.uuid4())
        self._pid_phase_data = []
        self._stable_rounds = 0
        self._prev_rho_bar = None
        self._memo_evidence_lookup = {}
        self._crit_current_captures = {}
        self._crit_round1_captures = None
        self._intervention_history: list[dict] = []
        reset_registry_cache()
        if self.config.pid_config:
            from eval.PID.controller import PIDController
            self._pid_controller = PIDController(
                self.config.pid_config, self.config.initial_beta
            )

    @staticmethod
    def _latest_per_role(decisions: list[dict]) -> list[dict]:
        """Deduplicate decisions to the latest entry per role.

        LangGraph reducers append to lists, so after retries
        ``state["revisions"]`` contains entries from the original revise
        AND every retry.  This helper keeps only the last entry per role
        so JS / evidence metrics reflect the current state.
        """
        by_role: dict[str, dict] = {}
        for d in decisions:
            role = d.get("role")
            if role is None:
                raise ValueError(f"Decision dict missing 'role' key: {list(d.keys())}")
            by_role[role] = d
        return list(by_role.values())

    def _get_js_divergence(self, state: dict) -> float | None:
        """Compute JS divergence between the latest agent allocations."""
        from eval.divergence import generalized_js_divergence
        decisions = self._latest_per_role(
            state.get("revisions") or state.get("proposals", [])
        )
        if not decisions:
            return None
        allocs = [
            d.get("action_dict", {}).get("allocation", {})
            for d in decisions
        ]
        if len(allocs) < 2:
            return None
        return generalized_js_divergence(allocs)

    def _log_prompt_manifest(self, state: dict, round_num: int) -> None:
        """Log prompt file manifest once at the start of a round.

        Lists files top-to-bottom in the same order they appear in the
        rendered system prompt, then the user-prompt (phase) template.
        """
        config = state["config"]
        manifest = build_prompt_manifest(config)

        snapshot_id = _extract_snapshot_id(
            state.get("enriched_context", ""),
            state.get("observation", {}),
        )

        beta = manifest.get("beta")
        bucket = manifest.get("beta_bucket", "")
        beta_str = f"β={beta:.2f} ({bucket})" if beta is not None else "β=N/A"

        lines = [
            "=" * 72,
            f"  [Prompt Manifest] Round {round_num} | {beta_str} | {snapshot_id}",
            "-" * 72,
            "  System prompt (top → bottom):",
        ]

        # Walk the block order and emit each file in rendered order
        order = manifest.get("block_order", [])
        role_files = manifest.get("role_files", {})

        for block in order:
            if block == "causal_contract":
                cc = manifest.get("causal_contract")
                if cc:
                    lines.append(f"    1. [{block}] {cc}")
            elif block == "role_system":
                for role, fname in role_files.items():
                    lines.append(f"    2. [{block}] {fname}  ({role})")
            elif block == "phase_preamble":
                lines.append(f"    3. [{block}] [inline]")
            elif block == "tone":
                tone = manifest.get("tone", {})
                if tone:
                    for phase, fname in tone.items():
                        if fname:
                            lines.append(f"    4. [{block}] {fname}  ({phase})")

        # User-prompt templates
        lines.append("  User prompt templates:")
        phase_templates = manifest.get("phase_templates", {})
        for phase, fname in phase_templates.items():
            lines.append(f"    - {fname}  ({phase})")

        lines.append("=" * 72)
        prompt_logger.info("\n".join(lines))

    def run(self, observation: Observation) -> tuple[Action, AgentTrace]:
        """
        Run the full debate pipeline on an observation.

        Returns:
            (action, trace) -- the final Action and the full AgentTrace
        """
        self._reset_per_invocation_state()
        state = self._run_pipeline(observation)

        # Parse final action
        final_dict = state.get("final_action", {})
        action = self._parse_action(final_dict)

        # Parse trace
        trace_dict = state.get("trace", {})
        trace = self._parse_trace(trace_dict, observation, action)

        # Save to disk
        self._save_trace(trace, state)

        return action, trace

    def _run_pipeline(self, observation: Observation) -> dict:
        """Execute all three phases and return the raw state dict.

        Shared implementation for run() and run_returning_state() so
        the pipeline/loop/finalize logic is defined in exactly one place.
        """
        state = self._initialize_state(observation)

        # Phase 1: Pipeline (news digest, data analysis, context building)
        state = self.pipeline_graph.invoke(state)

        # Parse memo evidence once for CRIT citation enrichment
        if self.config.pid_enabled or self._intervention_engine:
            from eval.evidence import parse_memo_evidence
            self._memo_evidence_lookup = parse_memo_evidence(
                state.get("enriched_context", "")
            )

        # Initialize structured debate logger
        if self._debate_logger:
            self._debate_logger.init_run(
                self._debate_id,
                state.get("observation", {}),
                state.get("enriched_context", ""),
            )
            # Set prompt capture callback for debug mode
            if self.config.logging_mode == "debug":
                state["config"]["_prompt_capture"] = self._debate_logger.write_prompt

        # Set LLM lifecycle callback for Rich console display
        if self.config.console_display:
            from .terminal_display import _llm_call_start, _llm_call_end, _reset_llm_tracker
            state["config"]["_llm_lifecycle"] = (_llm_call_start, _llm_call_end, _reset_llm_tracker)

        # Phase 2: Debate rounds
        terminated_early = False
        for t in range(self.config.max_rounds):
            state["current_round"] = t + 1

            # Start round in structured logger
            if self._debate_logger:
                beta = self._pid_controller.beta if self.config.pid_enabled and self._pid_controller else self._original_beta
                self._debate_logger.start_round(t + 1, beta)

            # Prompt manifest: log file names once at round start
            # (skip when Rich console is active — avoids terminal noise)
            if state["config"].get("log_prompt_manifest") and not self.config.console_display:
                self._log_prompt_manifest(state, t + 1)

            # Write prompt manifest to structured log (first round only)
            if t == 0 and self._debate_logger:
                manifest = build_prompt_manifest(state["config"])
                self._debate_logger.write_prompt_manifest(manifest)

            # Reset critiques so operator.add (parallel graph) doesn't
            # accumulate across rounds.  Harmless for the sequential graph
            # (batch nodes replace lists entirely).
            # NOTE: revisions are NOT reset here because critique nodes need
            # to see the previous round's revisions as their source.
            if t > 0:
                state["critiques"] = []

            if self.config.pid_enabled or self._intervention_engine:
                state = self._run_round_with_pid(state, t + 1)
            else:
                state = self.run_single_round(state)
                if self.config.console_display:
                    render_portfolio_table(state.get("proposals", []), "Allocations")
                    render_portfolio_table(state.get("revisions", []), "Revisions")
                else:
                    _print_comparison_table(state.get("proposals", []), "Allocations")
                    _print_comparison_table(state.get("revisions", []), "Revisions")

                js = self._get_js_divergence(state)
                if js is not None:
                    if self.config.console_display:
                        from .terminal_display import render_divergence_metrics
                        render_divergence_metrics(js, 0.0) # Overlap not avail here
                    else:
                        print(f"\n  Round {t+1} Disagreement (JS Divergence): {js:.4f} bits")

                if self._debate_logger:
                    self._debate_logger.write_proposals(state.get("proposals", []))
                    self._debate_logger.write_critiques(state.get("critiques", []))
                    self._debate_logger.write_revisions(state.get("revisions", []))
                    metrics = {"js_divergence": js} if js is not None else None
                    self._debate_logger.write_round_state(state, t + 1, metrics=metrics)

            if self.config.parallel_agents:
                # With operator.add, revisions accumulate (previous round's
                # entries + this round's entries).  Keep only the latest
                # entry per role so the next round and judge see clean state.
                seen = set()
                deduped = []
                for rev in reversed(state.get("revisions", [])):
                    if rev["role"] not in seen:
                        seen.add(rev["role"])
                        deduped.append(rev)
                state["revisions"] = sorted(reversed(deduped), key=lambda r: r["role"])

                # Sort proposals and critiques by role for deterministic
                # ordering — parallel fan-out merges via operator.add in
                # thread-completion order, which is non-deterministic.
                state["proposals"] = sorted(state.get("proposals", []), key=lambda p: p["role"])
                state["critiques"] = sorted(state.get("critiques", []), key=lambda c: c["role"])

            terminated_early = self.should_terminate(state)
            if terminated_early:
                break

        # Emit end-of-debate PID summary (skip when Rich console is active)
        if self._pid_controller and self._pid_phase_data and self._log_metrics and not self.config.console_display:
            summary = self._build_pid_summary(terminated_early=terminated_early)
            pid_metrics_logger.info(json.dumps(summary, indent=2))

        # Phase 3: Judge + trace
        if self.config.console_display:
            render_phase_label("Judge")
            from .terminal_display import _reset_llm_tracker
            _reset_llm_tracker(1)
            print(f"  Synthesizing final allocation ({self.config.model_name})", flush=True)
        state = self.finalize_graph.invoke(state)

        # Rich console display: judge result + debate end
        if self.config.console_display:
            final_action = state.get("final_action", {})
            if final_action:
                render_judge_result(final_action)

            logged_dir = None
            if self._debate_logger:
                logged_dir = str(self._debate_logger.run_dir)

            total_rounds = len({d["round"] for d in self._pid_phase_data}) if self._pid_phase_data else self.config.max_rounds
            reason = "stable_convergence" if terminated_early else "max_rounds"
            render_debate_end(terminated_early, reason, total_rounds, logged_dir)

        # Finalize structured debate log
        if self._debate_logger:
            self._debate_logger.finalize(
                state, self._pid_phase_data, terminated_early,
                state.get("enriched_context", ""),
                crit_captures=self._crit_round1_captures,
                intervention_history=self._intervention_history,
            )
            if not self.config.console_display:
                print(f"[Logged] {self._debate_logger.run_dir}", flush=True)

        # Stamp intervention history onto state for downstream / test access
        if self._intervention_history:
            state["intervention_history"] = list(self._intervention_history)

        return state

    def _run_round_with_pid(self, state: dict, round_num: int) -> dict:
        """Run one debate round with once-per-round CRIT scoring and PID control.

        β stays constant across propose/critique/revise (from prior round's
        PID update).  CRIT runs once after revise, scoring each agent
        independently via parallel LLM calls.  PID updates β for the next round.

        Also injects ``_current_beta`` into config so the modular prompt
        registry can map beta to tone bucket without seeing the numeric value.
        """
        beta = self._pid_controller.beta if self._pid_controller else self._original_beta
        use_display = self.config.console_display

        # Start round in structured logger
        if self._debate_logger:
            self._debate_logger.start_round(round_num, beta)

        # --- Round header (terminal display) ---
        if use_display:
            universe = list(self.config.roles)
            obs_universe = state.get("observation", {}).get("universe", [])
            tone = beta_to_bucket(beta)
            render_round_header(
                round_num, self.config.max_rounds, beta, tone,
                obs_universe,
            )

        # --- Propose phase (round 1 only — round 2+ critique/revise
        #     operate on the prior round's revisions via _resolve_source) ---
        # Capture prior revisions before they're overwritten — these are the
        # effective "proposals" (input) for round 2+.
        prior_revisions = list(state.get("revisions", [])) if round_num > 1 else None
        n_agents = len(self.config.roles)
        role_names = ", ".join(r.upper() for r in self.config.roles)
        if round_num == 1:
            if use_display:
                render_phase_label("Propose")
                from .terminal_display import _reset_llm_tracker
                _reset_llm_tracker(n_agents)
                print(f"  Calling {n_agents} agents: {role_names}", flush=True)
            state["config"]["_current_beta"] = None
            state = self._propose_graph.invoke(state)

        display_source = state.get("revisions") or state.get("proposals", [])
        display_label = "Revised Allocations" if state.get("revisions") else "Allocations"
        if use_display:
            render_portfolio_table(display_source, display_label)
        else:
            _print_comparison_table(display_source, display_label)

        if round_num == 1 and self._debate_logger:
            self._debate_logger.write_proposals(state.get("proposals", []))

        # --- Propose-phase JS divergence + evidence overlap ---
        self._proposed_divergence = None
        if self._debate_logger or self._intervention_engine:
            from eval.divergence import generalized_js_divergence
            from eval.evidence import extract_agent_evidence_spans, compute_mean_overlap

            prop_decisions = self._latest_per_role(
                state.get("proposals", [])
            )
            prop_allocs = [
                d.get("action_dict", {}).get("allocation", {})
                for d in prop_decisions
            ]
            if len(prop_allocs) >= 2:
                prop_js = generalized_js_divergence(prop_allocs)
                prop_ev = extract_agent_evidence_spans(prop_decisions, allocation_mode=True)
                prop_ov = compute_mean_overlap(prop_ev)
                if prop_ov is None:
                    prop_ov = 0.0

                prop_confidences = {
                    d["role"]: d.get("action_dict", {}).get("confidence", 0.5)
                    for d in prop_decisions
                }
                prop_evidence = {role: sorted(spans) for role, spans in prop_ev.items()}

                if self._debate_logger:
                    self._debate_logger.write_divergence_metrics(
                        prop_js, prop_ov, prop_confidences, prop_evidence,
                        round_num, phase="propose",
                    )
                self._proposed_divergence = {
                    "js": prop_js,
                    "ov": prop_ov,
                    "agent_confidences": prop_confidences,
                    "agent_evidence_ids": prop_evidence,
                }

        # --- Critique phase (uses PID beta for tone) ---
        if use_display:
            render_phase_label("Critique")
            from .terminal_display import _reset_llm_tracker
            _reset_llm_tracker(n_agents)
            print(f"  Calling {n_agents} agents: {role_names}", flush=True)
        state["config"]["_current_beta"] = beta
        state = self._critique_graph.invoke(state)

        if self._debate_logger:
            self._debate_logger.write_critiques(state.get("critiques", []))

        # --- Revise phase (uses PID beta for tone) ---
        if use_display:
            render_phase_label("Revise")
            from .terminal_display import _reset_llm_tracker
            _reset_llm_tracker(n_agents)
            print(f"  Calling {n_agents} agents: {role_names}", flush=True)
        state["config"]["_current_beta"] = beta
        state = self._revise_graph.invoke(state)
        if use_display:
            render_portfolio_table(state.get("revisions", []), "Revisions")
        else:
            _print_comparison_table(state.get("revisions", []), "Revisions")

        if self._debate_logger:
            self._debate_logger.write_revisions(state.get("revisions", []))

        # --- Revise-phase JS divergence + evidence overlap ---
        # Written here (not inside _crit_and_pid_step) so it's logged
        # even if CRIT scoring fails or is disabled.
        if self._debate_logger:
            from eval.divergence import generalized_js_divergence
            from eval.evidence import extract_agent_evidence_spans, compute_mean_overlap

            rev_decisions = self._latest_per_role(state.get("revisions", []))
            rev_allocs = [
                d.get("action_dict", {}).get("allocation", {})
                for d in rev_decisions
            ]
            if len(rev_allocs) >= 2:
                rev_js = generalized_js_divergence(rev_allocs)
                rev_ev = extract_agent_evidence_spans(rev_decisions, allocation_mode=True)
                rev_ov = compute_mean_overlap(rev_ev)
                if rev_ov is None:
                    rev_ov = 0.0

                rev_confidences = {
                    d["role"]: d.get("action_dict", {}).get("confidence", 0.5)
                    for d in rev_decisions
                }
                rev_evidence = {role: sorted(spans) for role, spans in rev_ev.items()}

                self._debate_logger.write_divergence_metrics(
                    rev_js, rev_ov, rev_confidences, rev_evidence, round_num,
                )

        # --- Post-revision intervention checkpoint ---
        if self._intervention_engine:
            state = self._intervention_checkpoint(
                state, round_num, stage="post_revision",
                beta=beta, use_display=use_display,
                n_agents=n_agents, role_names=role_names,
            )

        # --- CRIT + PID (once per round, after revise) ---
        if self._crit_scorer:
            if use_display:
                n_agents = len(self.config.roles)
                render_phase_label("CRIT Scoring")
                from .terminal_display import _reset_llm_tracker
                _reset_llm_tracker(n_agents)
                print(f"  Scoring {n_agents} agents with CRIT ({self.config.crit_model_name})", flush=True)
            self._crit_and_pid_step(
                state, round_num, beta_in=beta,
                prior_revisions=prior_revisions,
            )

        # --- Post-CRIT intervention checkpoint ---
        if self._intervention_engine and self._last_round_crit:
            state = self._intervention_checkpoint(
                state, round_num, stage="post_crit",
                beta=beta, use_display=use_display,
                n_agents=n_agents, role_names=role_names,
            )

        return state

    @staticmethod
    def _enrich_nudge_with_proposal(
        nudge_text: str | dict[str, str],
        state: dict,
        revision_limits: dict | None = None,
    ) -> str | dict[str, str]:
        """Append each agent's original proposal allocation to their nudge.

        Extracts proposal allocations from state["proposals"] and appends a
        PORTFOLIO REVISION LIMITS block that shows the agent's original
        allocation and constrains how far the retry can deviate from it.

        Constraint parameters come from ``revision_limits`` (read from
        ``intervention_config.revision_limits`` in the YAML config).
        """
        if revision_limits is None:
            return nudge_text  # soft mode — nudge text only, no constraints

        proposals = state.get("proposals", [])
        proposal_by_role: dict[str, dict] = {}
        for p in proposals:
            role = p.get("role", "unknown")
            alloc = p.get("action_dict", {}).get("allocation", {})
            if alloc:
                proposal_by_role[role] = alloc

        if not proposal_by_role:
            return nudge_text

        rl = revision_limits or {}
        max_tickers = rl.get("max_changed_tickers", 2)
        max_pct = rl.get("max_change_pct", 5)

        def _format_allocation(alloc: dict) -> str:
            lines = []
            for ticker, weight in sorted(alloc.items(), key=lambda x: -x[1]):
                lines.append(f"  {ticker}: {weight:.0%}")
            return "\n".join(lines)

        def _build_limits_block() -> str:
            return (
                "\n\nPORTFOLIO REVISION LIMITS:\n"
                "Your revision must stay close to YOUR ORIGINAL PROPOSAL below.\n"
                f"• You may change AT MOST {max_tickers} tickers.\n"
                f"• Each changed ticker may move by no more than {max_pct}% of total allocation.\n"
                "• All other positions must remain exactly as they were."
            )

        def _append_proposal_block(text: str, role: str) -> str:
            alloc = proposal_by_role.get(role)
            if not alloc:
                return text
            return text + _build_limits_block() + (
                f"\n\nYOUR ORIGINAL PROPOSAL:\n{_format_allocation(alloc)}"
            )

        if isinstance(nudge_text, dict):
            return {
                role: _append_proposal_block(text, role)
                for role, text in nudge_text.items()
            }
        else:
            blocks = []
            for role, alloc in sorted(proposal_by_role.items()):
                blocks.append(f"\n{role.upper()} ORIGINAL PROPOSAL:\n{_format_allocation(alloc)}")
            return nudge_text + _build_limits_block() + "".join(blocks)

    def _intervention_checkpoint(
        self,
        state: dict,
        round_num: int,
        *,
        stage: str,
        beta: float,
        use_display: bool,
        n_agents: int,
        role_names: str,
    ) -> dict:
        """Run the intervention retry loop for a given stage.

        Evaluates all registered rules, and if any fire with a retry
        action, re-runs the revise phase with a nudge injected into
        the prompt.  Repeats until no rules fire or max_retries exhausted.

        Intervention history is tracked on self._intervention_history
        (not in the LangGraph state dict, which would be stripped by
        graph.invoke()).  The history is written to state at the end
        for downstream access.
        """
        from eval.interventions import InterventionContext

        retry_count = 0
        while True:
            js_rev = self._get_js_divergence(state)

            # Populate crit-specific fields when stage is post_crit
            _rho_bar = None
            _agent_crit_scores = None
            if stage == "post_crit" and self._last_round_crit:
                _rho_bar = self._last_round_crit.rho_bar
                _agent_crit_scores = self._last_round_crit.agent_scores

            ctx = InterventionContext(
                round_num=round_num,
                stage=stage,
                retry_count=retry_count,
                state=state,
                js_proposal=(
                    self._proposed_divergence.get("js")
                    if self._proposed_divergence else None
                ),
                js_revision=js_rev,
                ov_revision=None,
                rho_bar=_rho_bar,
                agent_crit_scores=_agent_crit_scores,
                pid_result=None,
                intervention_history=list(self._intervention_history),
            )
            results = self._intervention_engine.evaluate(ctx)
            retry_actions = [r for r in results if r.action.startswith("retry")]
            if not retry_actions:
                break

            # Pick highest-severity action (critical > warning)
            action = max(
                retry_actions,
                key=lambda r: (r.severity == "critical", r.rule_name),
            )

            # Record in intervention history (runner-owned, survives invoke)
            history_entry = {
                "round": round_num,
                "stage": stage,
                "rule": action.rule_name,
                "retry": retry_count + 1,
                "action": action.action,
                "metrics": action.metrics,
                "severity": action.severity,
                "nudge_text": action.nudge_text,
            }
            self._intervention_history.append(history_entry)

            # Log the intervention
            self._log_intervention(action, round_num, retry_count)

            # Enrich nudge with each agent's original proposal allocation
            # so the model knows the anchor point for portfolio revision limits.
            revision_limits = (self.config.intervention_config or {}).get(
                "revision_limits",
            )
            enriched_nudge = self._enrich_nudge_with_proposal(
                action.nudge_text, state,
                revision_limits=revision_limits,
            )

            # Inject nudge and target roles into state config.
            # nudge_text may be a str (broadcast) or dict (per-agent).
            nudge = enriched_nudge
            if action.target_roles:
                # Per-agent targeting: only targeted roles see the nudge
                # and only targeted roles re-run the LLM call
                state["config"]["_intervention_nudge"] = None
                state["config"]["_intervention_target_roles"] = action.target_roles
                for role in action.target_roles:
                    if isinstance(nudge, dict):
                        state["config"][f"_intervention_nudge_{role}"] = nudge.get(role, "")
                    else:
                        state["config"][f"_intervention_nudge_{role}"] = nudge
            else:
                # Broadcast: all agents see the same nudge and re-run
                state["config"]["_intervention_nudge"] = nudge if isinstance(nudge, str) else ""
                state["config"]["_intervention_target_roles"] = None

            # Console display
            if use_display:
                if action.target_roles:
                    retry_agents = ', '.join(r.upper() for r in action.target_roles)
                    retry_count_display = len(action.target_roles)
                else:
                    retry_agents = role_names
                    retry_count_display = n_agents
                render_phase_label(
                    f"Revise (RETRY {retry_count + 1} — {action.rule_name} → {retry_agents})"
                )
                from .terminal_display import _reset_llm_tracker
                _reset_llm_tracker(retry_count_display)
                print(
                    f"  Retrying {retry_count_display} agent(s): {retry_agents} "
                    f"[{action.rule_name}: {action.severity}]",
                    flush=True,
                )

            # Re-run revise (only targeted agents call the LLM)
            state["config"]["_current_beta"] = beta
            if self._debate_logger:
                self._debate_logger._prompt_retry = retry_count + 1
            state = self._revise_graph.invoke(state)
            if self._debate_logger:
                self._debate_logger._prompt_retry = None
            # Clear nudges and target roles after retry
            state["config"]["_intervention_nudge"] = None
            state["config"]["_intervention_target_roles"] = None
            if action.target_roles:
                for role in action.target_roles:
                    state["config"].pop(f"_intervention_nudge_{role}", None)

            if use_display:
                render_portfolio_table(state.get("revisions", []), "Revisions (retry)")
            else:
                _print_comparison_table(state.get("revisions", []), "Revisions (retry)")

            # Log the retry revisions
            if self._debate_logger:
                self._debate_logger.write_revisions(
                    state.get("revisions", []),
                    retry=retry_count + 1,
                )

            # Log retry JS divergence + evidence overlap
            retry_js = self._get_js_divergence(state)
            if retry_js is not None:
                from eval.evidence import extract_agent_evidence_spans, compute_mean_overlap
                retry_decisions = self._latest_per_role(state.get("revisions", []))
                retry_evidence = extract_agent_evidence_spans(
                    retry_decisions, allocation_mode=True,
                )
                retry_ov = compute_mean_overlap(retry_evidence)
                if retry_ov is None:
                    retry_ov = 0.0
                retry_confidences = {
                    d["role"]: d.get("action_dict", {}).get("confidence", 0.5)
                    for d in retry_decisions
                }
                retry_evidence_ids = {
                    role: sorted(spans) for role, spans in retry_evidence.items()
                }
                if use_display:
                    from .terminal_display import render_divergence_metrics
                    render_divergence_metrics(retry_js, retry_ov)
                if self._debate_logger:
                    self._debate_logger.write_divergence_metrics(
                        retry_js, retry_ov, retry_confidences, retry_evidence_ids,
                        round_num, phase=f"retry_{retry_count + 1:03d}",
                    )

            # Post-crit retries are expensive: re-run CRIT after revise retry
            if stage == "post_crit" and self._crit_scorer:
                if use_display:
                    render_phase_label(f"CRIT Scoring (retry {retry_count + 1})")
                    from .terminal_display import _reset_llm_tracker
                    _reset_llm_tracker(n_agents)
                    print(f"  Re-scoring {n_agents} agents with CRIT", flush=True)
                self._crit_and_pid_step(state, round_num, beta_in=beta)

            retry_count += 1

        # Write history to state for downstream access (post-hoc analysis)
        if self._intervention_history:
            state["intervention_history"] = list(self._intervention_history)

        return state

    def _log_intervention(
        self, result, round_num: int, retry_count: int,
    ) -> None:
        """Log an intervention event to both the Python logger and DebateLogger."""
        entry = {
            "type": "intervention",
            "round": round_num,
            "stage": result.action.replace("retry_", "post_"),
            "rule": result.rule_name,
            "action": result.action,
            "severity": result.severity,
            "retry": retry_count + 1,
            "metrics": result.metrics,
            "nudge_text": result.nudge_text,
        }
        logger.info("INTERVENTION TRIGGERED: %s", json.dumps(entry, indent=2))
        if self._debate_logger:
            self._debate_logger.write_intervention(entry)

    def _crit_and_pid_step(
        self, state: dict, round_num: int, *, beta_in: float,
        prior_revisions: list[dict] | None = None,
    ) -> None:
        """Run per-agent CRIT scoring + PID controller step once per round.

        Called once after revise.  Assembles reasoning bundles for each agent,
        scores them in parallel via CritScorer, computes JS + OV, and
        updates the PID controller.
        """
        from eval.divergence import generalized_js_divergence
        from eval.evidence import extract_agent_evidence_spans, compute_mean_overlap

        decisions = self._latest_per_role(
            state.get("revisions", state.get("proposals", []))
        )
        if not decisions:
            raise RuntimeError(
                f"_crit_and_pid_step called with no decisions in round {round_num}"
            )

        # --- Build reasoning bundles for each agent ---
        bundles = {}
        for role in self.config.roles:
            bundle = build_reasoning_bundle(
                state, role, round_num, self._memo_evidence_lookup,
            )
            if bundle is not None:
                bundles[role] = bundle

        if not bundles:
            raise RuntimeError(
                f"No reasoning bundles assembled for CRIT in round {round_num} "
                f"(roles: {self.config.roles})"
            )

        # --- CRIT audit (per-agent parallel scoring → RoundCritResult) ---
        self._crit_current_captures = {}
        round_crit = self._crit_scorer.score(bundles)
        if round_num == 1:
            self._crit_round1_captures = self._crit_current_captures

        # --- JS divergence from portfolio allocation vectors ---
        allocations = [
            d.get("action_dict", {}).get("allocation", {})
            for d in decisions
        ]
        js = generalized_js_divergence(allocations) if len(allocations) >= 2 else 0.0

        # --- Evidence overlap (average pairwise Jaccard) ---
        evidence_sets = extract_agent_evidence_spans(
            decisions, allocation_mode=True,
        )
        ov = compute_mean_overlap(evidence_sets)
        if ov is None:
            ov = 0.0

        if not self.config.console_display:
            print(f"\n  Round {round_num} Disagreement (JS Divergence): {js:.4f} bits")
            print(f"  Round {round_num} Evidence Overlap: {ov:.4f}")

        # Guard rho_bar against None/NaN
        rho_bar = round_crit.rho_bar
        if rho_bar is None or math.isnan(rho_bar):
            logger.warning("rho_bar is %s (round %d) — defaulting to 0.0",
                           rho_bar, round_num)
            rho_bar = 0.0

        # --- Store latest CRIT result for post-crit intervention checkpoint ---
        self._last_round_crit = round_crit

        # --- PID controller step (only when PID is active) ---
        old_beta = None
        pid_result = None
        if self._pid_controller:
            old_beta = self._pid_controller.beta
            pid_result = self._pid_controller.step(rho_bar, js, ov)

            # Record PIDEvent
            ctrl_output = ControllerOutput(new_beta=pid_result.beta_new)
            event = PIDEvent(
                round_index=round_num,
                metrics=RoundMetrics(
                    round_index=round_num,
                    rho_bar=rho_bar,
                    js_divergence=js,
                    ov_overlap=ov,
                ),
                crit_result=round_crit.model_dump(),
                pid_step={
                    "e_t": pid_result.e_t,
                    "u_t": pid_result.u_t,
                    "beta_new": pid_result.beta_new,
                    "p_term": pid_result.p_term,
                    "i_term": pid_result.i_term,
                    "d_term": pid_result.d_term,
                    "s_t": pid_result.s_t,
                    "quadrant": pid_result.quadrant,
                    "div_signal": pid_result.div_signal,
                    "qual_signal": pid_result.qual_signal,
                },
                controller_output=ctrl_output,
            )
            self._pid_events.append(event)

        # --- Structured debate logger: CRIT metrics + prompts ---
        if self._debate_logger:
            self._debate_logger.write_crit_metrics(round_crit, round_num)
            if self._crit_current_captures:
                self._debate_logger.write_crit_prompts(self._crit_current_captures)

        # --- Compute agent confidences + evidence (used by both loggers) ---
        agent_confidences = {}
        for d in decisions:
            role = d.get("role", "unknown")
            agent_confidences[role] = d.get("action_dict", {}).get("confidence", 0.5)

        agent_evidence = {
            role: sorted(spans) for role, spans in evidence_sets.items()
        }

        # NOTE: Revise-phase divergence metrics are now written in
        # _run_round_with_pid (after write_revisions), decoupled from CRIT.
        # This ensures they're logged even if CRIT fails.

        # --- Structured JSON logging ---
        if self._log_metrics or self._debate_logger:
            phase_data = {
                "type": "pid_round" if pid_result else "crit_round",
                "debate_id": self._debate_id,
                "phase_id": str(uuid.uuid4()),
                "round": round_num,
                "beta_in": beta_in,
                "tone_bucket": beta_to_bucket(pid_result.beta_new) if pid_result else beta_to_bucket(beta_in),
                "crit": {
                    "rho_bar": round_crit.rho_bar,
                    "rho_i": {
                        role: cr.rho_bar
                        for role, cr in round_crit.agent_scores.items()
                    },
                    "agents": {
                        role: {
                            "rho_i": cr.rho_bar,
                            "pillars": {
                                "LV": cr.pillar_scores.logical_validity,
                                "ES": cr.pillar_scores.evidential_support,
                                "AC": cr.pillar_scores.alternative_consideration,
                                "CA": cr.pillar_scores.causal_alignment,
                            },
                            "diagnostics": {
                                "contradictions": cr.diagnostics.contradictions_detected,
                                "unsupported_claims": cr.diagnostics.unsupported_claims_detected,
                                "ignored_critiques": cr.diagnostics.ignored_critiques_detected,
                                "premature_certainty": cr.diagnostics.premature_certainty_detected,
                                "causal_overreach": cr.diagnostics.causal_overreach_detected,
                                "conclusion_drift": cr.diagnostics.conclusion_drift_detected,
                            },
                            "explanations": {
                                "logical_validity": cr.explanations.logical_validity,
                                "evidential_support": cr.explanations.evidential_support,
                                "alternative_consideration": cr.explanations.alternative_consideration,
                                "causal_alignment": cr.explanations.causal_alignment,
                            },
                        }
                        for role, cr in round_crit.agent_scores.items()
                    },
                },
                "divergence": {
                    "js": js,
                    "ov": ov,
                    "agent_confidences": agent_confidences,
                    "agent_evidence_ids": agent_evidence,
                },
                "divergence_propose": self._proposed_divergence,
                "convergence": {
                    "stable_rounds": self._stable_rounds,
                    "delta_rho_actual": (
                        abs(rho_bar - self._prev_rho_bar)
                        if self._prev_rho_bar is not None else None
                    ),
                    "delta_rho_threshold": self.config.delta_rho,
                },
            }

            # Add PID data only when PID controller is active
            if pid_result:
                phase_data["pid"] = {
                    "e_t": pid_result.e_t,
                    "integral": self._pid_controller.state.integral,
                    "e_prev": self._pid_controller.state.e_prev,
                    "p_term": pid_result.p_term,
                    "i_term": pid_result.i_term,
                    "d_term": pid_result.d_term,
                    "u_t": pid_result.u_t,
                    "beta_old": old_beta,
                    "beta_new": pid_result.beta_new,
                    "quadrant": pid_result.quadrant,
                    "div_signal": pid_result.div_signal,
                    "qual_signal": pid_result.qual_signal,
                    "sycophancy": pid_result.s_t,
                }

            phase_data = _round_floats(phase_data)
            # Replace any existing entry for this round (post-crit retry
            # re-runs CRIT, so only keep the final scoring per round).
            self._pid_phase_data = [
                d for d in self._pid_phase_data if d.get("round") != round_num
            ]
            self._pid_phase_data.append(phase_data)

            # Structured debate logger: metrics + round state
            if self._debate_logger:
                self._debate_logger.write_pid_metrics(phase_data)
                metrics_summary = {
                    "rho_bar": round_crit.rho_bar,
                    "rho_i": {
                        role: cr.rho_bar
                        for role, cr in round_crit.agent_scores.items()
                    },
                    "js_divergence": js,
                    "evidence_overlap": ov,
                }
                if pid_result:
                    metrics_summary["beta_new"] = pid_result.beta_new
                    metrics_summary["quadrant"] = pid_result.quadrant

                crit_data = {
                    "rho_bar": round_crit.rho_bar,
                }
                crit_data.update({
                    role: {
                        "rho_i": cr.rho_bar,
                        "pillars": {
                            "LV": cr.pillar_scores.logical_validity,
                            "ES": cr.pillar_scores.evidential_support,
                            "AC": cr.pillar_scores.alternative_consideration,
                            "CA": cr.pillar_scores.causal_alignment,
                        },
                    }
                    for role, cr in round_crit.agent_scores.items()
                })

                pid_data = None
                if pid_result:
                    pid_data = {
                        "beta_in": beta_in,
                        "beta_new": pid_result.beta_new,
                        "tone_bucket": beta_to_bucket(pid_result.beta_new),
                        "e_t": pid_result.e_t,
                        "p_term": pid_result.p_term,
                        "i_term": pid_result.i_term,
                        "d_term": pid_result.d_term,
                        "u_t": pid_result.u_t,
                        "quadrant": pid_result.quadrant,
                        "sycophancy": pid_result.s_t,
                        "convergence": {
                            "stable_rounds": self._stable_rounds,
                            "delta_rho_actual": (
                                abs(rho_bar - self._prev_rho_bar)
                                if self._prev_rho_bar is not None else None
                            ),
                            "delta_rho_threshold": self.config.delta_rho,
                        },
                    }
                self._debate_logger.write_round_state(
                    state, round_num, _round_floats(metrics_summary),
                    crit_data=_round_floats(crit_data),
                    pid_data=_round_floats(pid_data) if pid_data else None,
                    prior_revisions=prior_revisions,
                )

            # Rich console display: CRIT + PID metrics
            if self.config.console_display:
                rho_star = self.config.pid_rho_star
                render_phase_metrics(phase_data, rho_star=rho_star)

            # PID metrics logger (programmatic JSON channel — skip when Rich
            # console is active to avoid terminal noise; data is already
            # rendered by render_phase_metrics and saved by debate_logger)
            if self._log_metrics and not self.config.console_display:
                pid_metrics_logger.info(json.dumps(phase_data, indent=2))

    def _build_pid_summary(self, *, terminated_early: bool) -> dict:
        """Assemble end-of-debate JSON summary from accumulated phase data."""
        pid_cfg = self._pid_controller.config
        last = self._pid_phase_data[-1] if self._pid_phase_data else {}
        total_rounds = len({d["round"] for d in self._pid_phase_data}) if self._pid_phase_data else 0
        return {
            "type": "pid_summary",
            "debate_id": self._debate_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "Kp": pid_cfg.gains.Kp,
                "Ki": pid_cfg.gains.Ki,
                "Kd": pid_cfg.gains.Kd,
                "rho_star": pid_cfg.rho_star,
                "gamma_beta": pid_cfg.gamma_beta,
                "initial_beta": self.config.initial_beta,
                "epsilon": pid_cfg.epsilon,
                "T_max": pid_cfg.T_max,
                "mu": pid_cfg.mu,
                "delta_s": pid_cfg.delta_s,
                "delta_js": pid_cfg.delta_js,
                "delta_beta": pid_cfg.delta_beta,
            },
            "phases": [rd["phase_id"] for rd in self._pid_phase_data],
            "outcome": {
                "total_rounds": total_rounds,
                "total_phase_steps": len(self._pid_phase_data),
                "terminated_early": terminated_early,
                "termination_reason": "stable_convergence" if terminated_early else "max_rounds",
                "final_beta": last.get("pid", {}).get("beta_new"),
                "final_rho_bar": last.get("crit", {}).get("rho_bar"),
                "final_js": last.get("divergence", {}).get("js"),
            },
        }

    def _initialize_state(self, observation: Observation) -> dict:
        """Build the initial state dict for the debate.

        This dict matches the DebateState TypedDict schema exactly.
        Every field is initialized to its zero-value so that nodes can
        safely read from state without KeyError, and so that LangGraph's
        reducers have a valid starting point (e.g. debate_turns starts
        as [] so operator.add can append to it).
        """
        return {
            "observation": observation.model_dump(),
            "config": self.config.to_dict(),
            "news_digest": "",
            "data_analysis": "",
            "enriched_context": "",
            "proposals": [],
            "critiques": [],
            "revisions": [],
            "current_round": 0,
            "debate_turns": [],
            "final_action": {},
            "strongest_objection": "",
            "audited_memo": "",
            "trace": {},
        }

    def run_single_round(self, state: dict) -> dict:
        """Execute one debate round via LangGraph: propose → critique → revise.

        This is the primitive that external controllers will call.  The
        caller sets state["current_round"] before invoking, then inspects
        the returned state to decide whether to continue.

        Invariants:
          - propose is idempotent (no-op if proposals exist)
          - debate_turns accumulates via LangGraph's operator.add reducer
          - critiques/revisions are replaced (plain list, no reducer)
        """
        return self.single_round_graph.invoke(state)

    def should_terminate(self, state: dict) -> bool:
        """Check if debate should end early via stable convergence.

        Requires ALL of the following for `convergence_window` consecutive
        rounds:
          1. quadrant == "converged" (high quality + low diversity)
          2. JS < epsilon (agents agree)
          3. |rho_bar(t) - rho_bar(t-1)| < delta_rho (quality plateau)

        Quality alone is insufficient — we require stability over time.
        When PID is disabled, always returns False.
        """
        if not (self._pid_controller and self._pid_events):
            return False

        last = self._pid_events[-1]
        js = last.metrics.js_divergence
        rho_bar = last.metrics.rho_bar
        quadrant = last.pid_step.get("quadrant", "")

        js_ok = js < self._pid_controller.config.epsilon
        quad_ok = quadrant == "converged"
        rho_stable = (
            self._prev_rho_bar is not None
            and abs(rho_bar - self._prev_rho_bar) < self.config.delta_rho
        )

        if js_ok and quad_ok and rho_stable:
            self._stable_rounds += 1
        else:
            self._stable_rounds = 0

        self._prev_rho_bar = rho_bar

        if self._stable_rounds >= self.config.convergence_window:
            logger.info(
                "Terminating debate: stable convergence for %d consecutive rounds "
                "(JS=%.5f, rho_bar=%.3f, quadrant=%s)",
                self._stable_rounds, js, rho_bar, quadrant,
            )
            return True
        return False

    def run_returning_state(self, observation: Observation) -> dict:
        """Run full pipeline and return raw LangGraph state dict.

        Same three-phase logic as run() but returns the raw state dict
        instead of (Action, AgentTrace).  No disk I/O — does NOT call
        _save_trace.  Used by equivalence tests to compare against the
        old monolithic compile_debate_graph().invoke() path.
        """
        self._reset_per_invocation_state()
        return self._run_pipeline(observation)

    @staticmethod
    def _coerce_variables(raw) -> list[str]:
        """Coerce LLM-returned variables to list[str].

        LLMs sometimes return a dict like {"X": "desc", "Y": "desc"}
        instead of ["X", "Y"]. Convert gracefully.
        """
        if isinstance(raw, list):
            return [str(v) for v in raw]
        if isinstance(raw, dict):
            return [str(k) for k in raw]
        return []

    def _parse_action(self, d: dict) -> Action:
        """Convert a raw dict into a validated Action model."""
        orders = []
        for o in d.get("orders", []):
            orders.append(
                Order(
                    ticker=o.get("ticker", ""),
                    side=o.get("side", "buy"),
                    size=o.get("size", 0),
                    type=o.get("type", "market"),
                )
            )

        _VALID_REASONING_TYPES = {"causal", "observational", "risk_assessment", "pattern"}

        claims = []
        for c in d.get("claims", []):
            raw_rtype = c.get("reasoning_type", "observational")
            # LLMs sometimes return pipe-separated combos like
            # "causal | risk_assessment" — extract the first valid value.
            if raw_rtype not in _VALID_REASONING_TYPES:
                for part in raw_rtype.replace("|", " ").split():
                    part = part.strip().lower()
                    if part in _VALID_REASONING_TYPES:
                        raw_rtype = part
                        break
                else:
                    raw_rtype = "observational"
            claims.append(
                Claim(
                    claim_text=c.get("claim_text", ""),
                    reasoning_type=raw_rtype,
                    assumptions=c.get("assumptions"),
                    confidence=c.get("confidence", 0.5),
                )
            )

        return Action(
            orders=orders,
            allocation=d.get("allocation"),
            justification=d.get("justification", ""),
            confidence=d.get("confidence", 0.5),
            claims=claims,
        )

    def _parse_trace(
        self,
        trace_dict: dict,
        observation: Observation,
        action: Action,
    ) -> AgentTrace:
        """Convert a raw trace dict into a validated AgentTrace model."""
        return AgentTrace(
            observation_timestamp=trace_dict.get(
                "observation_timestamp", observation.timestamp
            ),
            architecture="debate",
            what_i_saw=trace_dict.get("what_i_saw", ""),
            hypothesis=trace_dict.get("hypothesis", ""),
            decision=trace_dict.get("decision", "Hold"),
            risks_or_falsifiers=trace_dict.get("risks_or_falsifiers"),
            strongest_objection=trace_dict.get("strongest_objection"),
            debate_turns=None,  # Stored in full_state for auditability
            action=action,
            logged_at=datetime.now(timezone.utc).isoformat(),
            initial_market_state=observation.market_state,
            initial_portfolio_state=observation.portfolio_state,
            pid_events=self._pid_events if self._pid_events else None,
        )

    def _save_trace(self, trace: AgentTrace, full_state: dict) -> None:
        """Save the trace + full debate history to disk."""
        trace_dir = Path(self.config.trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime

        now = datetime.now()
        safe_ts = now.strftime("%Y-%m-%d_%I-%M-%S%p").lower()
        filename = f"debate_langgraph_{safe_ts}.json"
        filepath = trace_dir / filename

        output = {
            "trace": trace.model_dump(),
            "debate_turns": full_state.get("debate_turns", []),
            "config": full_state.get("config", {}),
        }

        filepath.write_text(json.dumps(output, indent=2, default=str))
        print(f"[MultiAgent] Trace written to {filepath}")
