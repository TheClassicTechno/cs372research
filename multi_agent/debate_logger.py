"""Structured logging for debate runs.

Writes crash-safe, per-round artifacts to a well-defined directory tree.
Called synchronously by the runner at each phase boundary.

Directory layout::

    logging/runs/<experiment>/run_<timestamp>/
    ├── manifest.json
    ├── pid_config.json
    ├── prompt_manifest.json
    ├── shared_context/memo.txt
    ├── rounds/round_001/
    │   ├── round_state.json
    │   ├── proposals/<agent>/{response.txt, portfolio.json, prompt.txt*}
    │   ├── critiques/<agent>/{response.json, prompt.txt*}
    │   ├── revisions/<agent>/{response.txt, portfolio.json, prompt.txt*}
    │   └── metrics/{crit_scores.json, js_divergence.json, evidence_overlap.json, pid_state.json}
    └── final/{final_portfolio.json, debate_diagnostic.json}

    * prompt.txt files written in debug mode only.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _write_json(path: Path, data: Any) -> None:
    """Write data as pretty-printed JSON. Crash-safe (single write_text call)."""
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _round_floats(obj: Any, ndigits: int = 4) -> Any:
    """Recursively round floats for clean JSON output."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    return obj


class DebateLogger:
    """Structured logging for debate runs.

    Instantiated by MultiAgentRunner when logging_mode is not "off".
    All public methods are no-ops when mode is "off".
    """

    def __init__(self, config: Any, experiment_name: str) -> None:
        self._mode: str = getattr(config, "logging_mode", "off")
        self._config = config
        self._experiment_name = experiment_name

        now = datetime.now()
        self._timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        self._run_id = f"run_{self._timestamp}"
        self._run_dir = Path("logging/runs") / experiment_name / self._run_id
        self._round_dir: Path | None = None
        self._current_beta: float = 0.5
        self._round_num: int = 0

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    # ------------------------------------------------------------------
    # Lifecycle: init
    # ------------------------------------------------------------------

    def init_run(
        self,
        debate_id: str,
        observation: dict,
        enriched_context: str,
    ) -> None:
        """Create directory tree and write initial metadata files."""
        if self._mode == "off":
            return

        # Create top-level directories
        self._run_dir.mkdir(parents=True, exist_ok=True)
        (self._run_dir / "shared_context").mkdir(exist_ok=True)
        (self._run_dir / "rounds").mkdir(exist_ok=True)
        (self._run_dir / "final").mkdir(exist_ok=True)

        # Write partial manifest (completed_at is null until finalize)
        manifest = {
            "experiment_name": self._experiment_name,
            "run_id": self._run_id,
            "debate_id": debate_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "model_name": getattr(self._config, "model_name", "unknown"),
            "temperature": getattr(self._config, "temperature", 0.3),
            "roles": [r.value for r in self._config.roles],
            "max_rounds": self._config.max_rounds,
            "actual_rounds": None,
            "terminated_early": None,
            "termination_reason": None,
            "agreeableness": self._config.agreeableness,
            "initial_beta": getattr(self._config, "initial_beta", 0.5),
            "final_beta": None,
            "ticker_universe": observation.get("universe", []),
            "invest_quarter": observation.get("timestamp", ""),
            "pid_enabled": self._config.pid_enabled,
            "logging_mode": self._mode,
            "parallel_agents": getattr(self._config, "parallel_agents", True),
        }
        _write_json(self._run_dir / "manifest.json", manifest)

        # Write PID config if enabled
        if self._config.pid_config is not None:
            pid_cfg = self._config.pid_config
            pid_data = {
                "Kp": pid_cfg.gains.Kp,
                "Ki": pid_cfg.gains.Ki,
                "Kd": pid_cfg.gains.Kd,
                "rho_star": pid_cfg.rho_star,
                "gamma_beta": pid_cfg.gamma_beta,
                "epsilon": pid_cfg.epsilon,
                "T_max": pid_cfg.T_max,
                "mu": pid_cfg.mu,
                "delta_s": pid_cfg.delta_s,
                "delta_js": pid_cfg.delta_js,
                "delta_beta": pid_cfg.delta_beta,
                "convergence_window": getattr(self._config, "convergence_window", 2),
                "delta_rho": getattr(self._config, "delta_rho", 0.02),
            }
            _write_json(self._run_dir / "pid_config.json", pid_data)

        # Write memo
        memo_path = self._run_dir / "shared_context" / "memo.txt"
        memo_path.write_text(enriched_context, encoding="utf-8")

    def write_prompt_manifest(self, manifest: dict) -> None:
        """Write prompt_manifest.json (once, at first round)."""
        if self._mode == "off":
            return
        _write_json(self._run_dir / "prompt_manifest.json", manifest)

    # ------------------------------------------------------------------
    # Per-round: start + artifact writers
    # ------------------------------------------------------------------

    def start_round(self, round_num: int, beta: float) -> None:
        """Create round_NNN/ directory and its subdirectories."""
        if self._mode == "off":
            return
        self._round_num = round_num
        self._current_beta = beta
        self._round_dir = self._run_dir / "rounds" / f"round_{round_num:03d}"
        self._round_dir.mkdir(parents=True, exist_ok=True)
        (self._round_dir / "proposals").mkdir(exist_ok=True)
        (self._round_dir / "critiques").mkdir(exist_ok=True)
        (self._round_dir / "revisions").mkdir(exist_ok=True)
        (self._round_dir / "metrics").mkdir(exist_ok=True)

    def write_proposals(self, proposals: list[dict]) -> None:
        """Write proposals/<agent>/response.txt + portfolio.json for each agent."""
        if self._mode == "off" or self._round_dir is None:
            return
        for p in proposals:
            role = p.get("role", "unknown")
            agent_dir = self._round_dir / "proposals" / role
            agent_dir.mkdir(parents=True, exist_ok=True)

            # response.txt — full reasoning text
            raw = p.get("raw_response", "")
            if raw is None:
                raw = ""
            (agent_dir / "response.txt").write_text(raw, encoding="utf-8")

            # portfolio.json — allocation weights only
            action = p.get("action_dict", {})
            allocation = action.get("allocation", {}) if isinstance(action, dict) else {}
            _write_json(agent_dir / "portfolio.json", allocation)

    def write_critiques(self, critiques: list[dict]) -> None:
        """Write critiques/<agent>/response.json for each agent."""
        if self._mode == "off" or self._round_dir is None:
            return
        for c in critiques:
            role = c.get("role", "unknown")
            agent_dir = self._round_dir / "critiques" / role
            agent_dir.mkdir(parents=True, exist_ok=True)

            data = {
                "critiques": c.get("critiques", []),
                "self_critique": c.get("self_critique", ""),
            }
            _write_json(agent_dir / "response.json", data)

    def write_revisions(self, revisions: list[dict]) -> None:
        """Write revisions/<agent>/response.txt + portfolio.json for each agent."""
        if self._mode == "off" or self._round_dir is None:
            return
        for r in revisions:
            role = r.get("role", "unknown")
            agent_dir = self._round_dir / "revisions" / role
            agent_dir.mkdir(parents=True, exist_ok=True)

            # response.txt — full revised reasoning text
            raw = r.get("raw_response", "") or r.get("revision_notes", "")
            if raw is None:
                raw = ""
            (agent_dir / "response.txt").write_text(raw, encoding="utf-8")

            # portfolio.json — allocation weights only
            action = r.get("action_dict", {})
            allocation = action.get("allocation", {}) if isinstance(action, dict) else {}
            _write_json(agent_dir / "portfolio.json", allocation)

    def write_crit_metrics(self, round_crit: Any, round_num: int) -> None:
        """Write metrics/crit_scores.json."""
        if self._mode == "off" or self._round_dir is None:
            return
        data: dict[str, Any] = {
            "round": round_num,
            "rho_bar": round_crit.rho_bar,
            "agent_scores": {},
        }
        for role, cr in round_crit.agent_scores.items():
            data["agent_scores"][role] = {
                "rho_i": cr.rho_bar,
                "pillar_scores": {
                    "IC": cr.pillar_scores.internal_consistency,
                    "ES": cr.pillar_scores.evidence_support,
                    "TA": cr.pillar_scores.trace_alignment,
                    "CI": cr.pillar_scores.causal_integrity,
                },
                "diagnostics": {
                    "contradictions": cr.diagnostics.contradictions_detected,
                    "unsupported_claims": cr.diagnostics.unsupported_claims_detected,
                    "conclusion_drift": cr.diagnostics.conclusion_drift_detected,
                    "causal_overreach": cr.diagnostics.causal_overreach_detected,
                },
                "explanations": {
                    "internal_consistency": cr.explanations.internal_consistency,
                    "evidence_support": cr.explanations.evidence_support,
                    "trace_alignment": cr.explanations.trace_alignment,
                    "causal_integrity": cr.explanations.causal_integrity,
                },
            }
        _write_json(
            self._round_dir / "metrics" / "crit_scores.json",
            _round_floats(data),
        )

    def write_divergence_metrics(
        self,
        js: float,
        ov: float,
        agent_confidences: dict[str, float],
        agent_evidence: dict[str, list[str]],
        round_num: int,
    ) -> None:
        """Write metrics/js_divergence.json and metrics/evidence_overlap.json."""
        if self._mode == "off" or self._round_dir is None:
            return
        metrics_dir = self._round_dir / "metrics"
        _write_json(
            metrics_dir / "js_divergence.json",
            _round_floats({
                "round": round_num,
                "js_divergence": js,
                "agent_confidences": agent_confidences,
            }),
        )
        _write_json(
            metrics_dir / "evidence_overlap.json",
            _round_floats({
                "round": round_num,
                "mean_overlap": ov,
                "agent_evidence_ids": agent_evidence,
            }),
        )

    def write_pid_metrics(self, phase_data: dict) -> None:
        """Write metrics/pid_state.json."""
        if self._mode == "off" or self._round_dir is None:
            return
        pid = phase_data.get("pid", {})
        data = {
            "round": phase_data.get("round"),
            "beta_in": phase_data.get("beta_in"),
            "beta_new": pid.get("beta_new"),
            "tone_bucket": phase_data.get("tone_bucket"),
            "error": {
                "e_t": pid.get("e_t"),
                "integral": pid.get("integral"),
                "e_prev": pid.get("e_prev"),
            },
            "gains": {
                "p_term": pid.get("p_term"),
                "i_term": pid.get("i_term"),
                "d_term": pid.get("d_term"),
            },
            "u_t": pid.get("u_t"),
            "quadrant": pid.get("quadrant"),
            "div_signal": pid.get("div_signal"),
            "qual_signal": pid.get("qual_signal"),
            "sycophancy": pid.get("sycophancy"),
            "convergence": phase_data.get("convergence", {}),
        }
        _write_json(
            self._round_dir / "metrics" / "pid_state.json",
            _round_floats(data),
        )

    def write_round_state(
        self,
        state: dict,
        round_num: int,
        metrics: dict | None = None,
    ) -> None:
        """Write round_state.json — compact round snapshot."""
        if self._mode == "off" or self._round_dir is None:
            return

        proposals_summary = {}
        for p in state.get("proposals", []):
            role = p.get("role", "unknown")
            action = p.get("action_dict", {}) if isinstance(p.get("action_dict"), dict) else {}
            proposals_summary[role] = {
                "allocation": action.get("allocation", {}),
                "confidence": action.get("confidence", 0.5),
            }

        revisions_summary = {}
        for r in state.get("revisions", []):
            role = r.get("role", "unknown")
            action = r.get("action_dict", {}) if isinstance(r.get("action_dict"), dict) else {}
            revisions_summary[role] = {
                "allocation": action.get("allocation", {}),
                "confidence": action.get("confidence", 0.5),
            }

        data = {
            "round": round_num,
            "beta": self._current_beta,
            "proposals": proposals_summary,
            "revisions": revisions_summary,
            "metrics": metrics or {},
        }
        _write_json(self._round_dir / "round_state.json", _round_floats(data))

    def write_prompt(
        self,
        phase: str,
        role: str,
        system_prompt: str,
        user_prompt: str,
    ) -> None:
        """Write <phase>/<agent>/prompt.txt (debug mode only)."""
        if self._mode != "debug" or self._round_dir is None:
            return
        # phase is "proposals", "critiques", "revisions", or "final"
        if phase == "final":
            prompt_path = self._run_dir / "final" / f"{role}_prompt.txt"
        else:
            agent_dir = self._round_dir / phase / role
            agent_dir.mkdir(parents=True, exist_ok=True)
            prompt_path = agent_dir / "prompt.txt"

        content = (
            "=== SYSTEM PROMPT ===\n"
            f"{system_prompt}\n\n"
            "=== USER PROMPT ===\n"
            f"{user_prompt}\n"
        )
        prompt_path.write_text(content, encoding="utf-8")

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    def finalize(
        self,
        state: dict,
        pid_phase_data: list[dict],
        terminated_early: bool,
        enriched_context: str,
    ) -> None:
        """Write final/ artifacts and update manifest.json."""
        if self._mode == "off":
            return

        final_dir = self._run_dir / "final"
        final_dir.mkdir(exist_ok=True)

        # final_portfolio.json
        final_action = state.get("final_action", {})
        allocation = final_action.get("allocation", {})
        _write_json(final_dir / "final_portfolio.json", allocation)

        # judge_response.txt — full judge reasoning (from debate_turns)
        for turn in reversed(state.get("debate_turns", [])):
            if turn.get("type") == "judge_decision":
                raw = turn.get("raw_response", "") or ""
                (final_dir / "judge_response.txt").write_text(raw, encoding="utf-8")
                break

        # debate_diagnostic.json
        diagnostic = self._build_diagnostic(
            state, pid_phase_data, terminated_early, enriched_context,
        )
        _write_json(final_dir / "debate_diagnostic.json", diagnostic)

        # Update manifest with completion data
        manifest_path = self._run_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = {}

        total_rounds = len({d["round"] for d in pid_phase_data}) if pid_phase_data else self._round_num
        last_pid = pid_phase_data[-1] if pid_phase_data else {}

        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["actual_rounds"] = total_rounds
        manifest["terminated_early"] = terminated_early
        manifest["termination_reason"] = (
            "stable_convergence" if terminated_early else "max_rounds"
        )
        manifest["final_beta"] = last_pid.get("pid", {}).get("beta_new")
        _write_json(manifest_path, manifest)

    # ------------------------------------------------------------------
    # Diagnostic artifact builder
    # ------------------------------------------------------------------

    def _build_diagnostic(
        self,
        state: dict,
        pid_phase_data: list[dict],
        terminated_early: bool,
        enriched_context: str,
    ) -> dict:
        """Build the debate_diagnostic.json artifact."""
        obs = state.get("observation", {})
        total_rounds = len({d["round"] for d in pid_phase_data}) if pid_phase_data else self._round_num

        # Meta
        meta = {
            "experiment": self._experiment_name,
            "run_id": self._run_id,
            "model": getattr(self._config, "model_name", "unknown"),
            "ticker_universe": obs.get("universe", []),
            "invest_quarter": obs.get("timestamp", ""),
            "rounds_completed": total_rounds,
            "terminated_early": terminated_early,
            "termination_reason": (
                "stable_convergence" if terminated_early else "max_rounds"
            ),
        }

        # Memo excerpt
        memo_excerpt = enriched_context[:2000] if enriched_context else ""

        # Context samples — two agents from round 1
        context_samples = self._build_context_samples(state, enriched_context)

        # Round summaries
        round_summaries = self._build_round_summaries(state, pid_phase_data)

        # Disagreement diagnostics
        disagreement = self._build_disagreement_diagnostics(state)

        # Final decision
        final_action = state.get("final_action", {})
        justification = final_action.get("justification", "")
        final_decision = {
            "allocation": final_action.get("allocation", {}),
            "confidence": final_action.get("confidence", 0.5),
            "justification_excerpt": justification[:300],
            "strongest_objection": state.get("strongest_objection", ""),
        }

        return _round_floats({
            "meta": meta,
            "memo_excerpt": memo_excerpt,
            "context_samples": context_samples,
            "round_summaries": round_summaries,
            "disagreement_diagnostics": disagreement,
            "final_decision": final_decision,
        })

    def _build_context_samples(
        self,
        state: dict,
        enriched_context: str,
    ) -> dict:
        """Extract prompt+response samples from two agents in round 1."""
        debate_turns = state.get("debate_turns", [])
        roles = [r.value for r in self._config.roles]
        sample_roles = roles[:2] if len(roles) >= 2 else roles

        samples = {}
        for idx, role in enumerate(sample_roles):
            sample = {"role": role}
            for phase_type, key_prefix in [
                ("proposal", "proposal"),
                ("critique", "critique"),
                ("revision", "revision"),
            ]:
                # Find the turn for this agent + phase in round 0 (proposals)
                # or round 1 (critiques/revisions)
                target_round = 0 if phase_type == "proposal" else 1
                turn = None
                for t in debate_turns:
                    t_role = t.get("role", "")
                    t_type = t.get("type", "")
                    t_round = t.get("round", -1)
                    if t_role == role and t_type == phase_type and t_round == target_round:
                        turn = t
                        break

                if turn:
                    sys_prompt = turn.get("raw_system_prompt", "") or ""
                    usr_prompt = turn.get("raw_user_prompt", "") or ""
                    full_prompt = (
                        f"=== SYSTEM PROMPT ===\n{sys_prompt}\n\n"
                        f"=== USER PROMPT ===\n{usr_prompt}"
                    )
                    full_prompt = _replace_memo_in_prompt(full_prompt, enriched_context)
                    sample[f"{key_prefix}_prompt_full"] = full_prompt
                    sample[f"{key_prefix}_response_example"] = turn.get("raw_response", "") or ""
                else:
                    sample[f"{key_prefix}_prompt_full"] = ""
                    sample[f"{key_prefix}_response_example"] = ""

            samples[f"agent_example_{idx + 1}"] = sample

        return samples

    def _build_round_summaries(
        self,
        state: dict,
        pid_phase_data: list[dict],
    ) -> list[dict]:
        """Build compact per-round summaries."""
        # Group pid_phase_data by round
        round_data = {}
        for pd in pid_phase_data:
            r = pd.get("round", 0)
            round_data[r] = pd

        summaries = []
        # Use round_data rounds or fall back to simple range
        max_round = max(round_data.keys()) if round_data else self._round_num
        proposals_list = state.get("proposals", [])
        revisions_list = state.get("revisions", [])

        for r in range(1, max_round + 1):
            pd = round_data.get(r, {})
            crit = pd.get("crit", {})
            pid = pd.get("pid", {})

            # Build proposal summaries from state (only available for last state)
            proposal_summaries = {}
            source = proposals_list if r == 1 else revisions_list
            for p in source:
                role = p.get("role", "unknown")
                action = p.get("action_dict", {}) if isinstance(p.get("action_dict"), dict) else {}
                alloc = action.get("allocation", {})
                proposal_summaries[role] = {
                    "allocation_top3": _build_allocation_top3(alloc),
                    "confidence": action.get("confidence", 0.5),
                    "thesis_excerpt": (action.get("justification", "") or "")[:200],
                }

            # Critique highlights
            critique_highlights = []
            for c in state.get("critiques", []):
                from_role = c.get("role", "?")
                for crit_item in c.get("critiques", []):
                    critique_highlights.append({
                        "from": from_role,
                        "to": crit_item.get("target_role", "?"),
                        "objection_excerpt": (crit_item.get("objection", "") or "")[:150],
                    })

            # Revision summaries
            revision_summaries = {}
            for rv in revisions_list:
                role = rv.get("role", "unknown")
                action = rv.get("action_dict", {}) if isinstance(rv.get("action_dict"), dict) else {}
                alloc = action.get("allocation", {})
                revision_summaries[role] = {
                    "allocation_top3": _build_allocation_top3(alloc),
                    "confidence": action.get("confidence", 0.5),
                    "changed": True,  # Could compare but approximation is fine
                }

            summary = {
                "round": r,
                "proposals": proposal_summaries,
                "critique_highlights": critique_highlights,
                "revisions": revision_summaries,
                "crit": {
                    "rho_bar": crit.get("rho_bar"),
                    "rho_i": crit.get("rho_i", {}),
                },
                "pid": {
                    "beta_in": pd.get("beta_in"),
                    "beta_new": pid.get("beta_new"),
                    "quadrant": pid.get("quadrant"),
                },
            }
            summaries.append(summary)

        return summaries

    def _build_disagreement_diagnostics(self, state: dict) -> dict:
        """Compute allocation variance, confidence spread, drift."""
        revisions = state.get("revisions", [])
        proposals = state.get("proposals", [])
        final_action = state.get("final_action", {})

        # Allocation variance across agents (from revisions)
        ticker_weights: dict[str, list[float]] = {}
        for r in revisions:
            action = r.get("action_dict", {}) if isinstance(r.get("action_dict"), dict) else {}
            alloc = action.get("allocation", {})
            for ticker, weight in alloc.items():
                ticker_weights.setdefault(ticker, []).append(weight)

        allocation_variance = {}
        for ticker, weights in ticker_weights.items():
            if len(weights) >= 2:
                mean = sum(weights) / len(weights)
                var = sum((w - mean) ** 2 for w in weights) / len(weights)
                allocation_variance[ticker] = var

        # Confidence spread
        confidences = []
        for r in revisions:
            action = r.get("action_dict", {}) if isinstance(r.get("action_dict"), dict) else {}
            confidences.append(action.get("confidence", 0.5))

        if confidences:
            confidence_spread = {
                "min": min(confidences),
                "max": max(confidences),
                "range": max(confidences) - min(confidences),
            }
        else:
            confidence_spread = {"min": 0, "max": 0, "range": 0}

        # Final vs initial drift
        initial_allocs: dict[str, float] = {}
        for p in proposals:
            action = p.get("action_dict", {}) if isinstance(p.get("action_dict"), dict) else {}
            alloc = action.get("allocation", {})
            for ticker, weight in alloc.items():
                initial_allocs.setdefault(ticker, []).append(weight)  # type: ignore[union-attr]

        initial_mean = {t: sum(ws) / len(ws) for t, ws in initial_allocs.items() if ws}
        final_alloc = final_action.get("allocation", {})
        drift = {}
        all_tickers = set(initial_mean.keys()) | set(final_alloc.keys())
        for t in all_tickers:
            drift[t] = final_alloc.get(t, 0.0) - initial_mean.get(t, 0.0)

        # Persistent disagreements
        persistent = []
        for ticker, weights in ticker_weights.items():
            if len(weights) >= 2:
                spread = max(weights) - min(weights)
                if spread > 0.1:  # >10% disagreement
                    parts = []
                    for rv in revisions:
                        role = rv.get("role", "?")
                        action = rv.get("action_dict", {}) if isinstance(rv.get("action_dict"), dict) else {}
                        w = action.get("allocation", {}).get(ticker)
                        if w is not None:
                            parts.append(f"{role} wants {w:.0%}")
                    persistent.append(f"{ticker} weight: {', '.join(parts)}")

        return _round_floats({
            "allocation_variance": allocation_variance,
            "confidence_spread": confidence_spread,
            "persistent_disagreements": persistent,
            "final_vs_initial_drift": drift,
        })


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _replace_memo_in_prompt(prompt_text: str, enriched_context: str) -> str:
    """Replace memo content in prompt text with a placeholder.

    Detects the memo body using the same markers as the enriched context
    and replaces it with ``<<MEMO CONTENT INSERTED HERE>>``.
    """
    if not enriched_context or len(enriched_context) < 100:
        return prompt_text

    # Find the memo start marker
    memo_start_marker = "[INFO] QUARTERLY SNAPSHOT MEMO"
    memo_start = prompt_text.find(memo_start_marker)
    if memo_start == -1:
        # Try matching the enriched_context body directly (first 200 chars)
        # as a fallback when the marker isn't present
        snippet = enriched_context[100:300].strip()
        if snippet and snippet in prompt_text:
            # Replace the entire enriched_context block
            return prompt_text.replace(
                enriched_context,
                "<<MEMO CONTENT INSERTED HERE>>",
            )
        return prompt_text

    # Find the end of the memo using known section markers
    end_markers = [
        "CRITICAL — Evidence citation rules:",
        "Using the data above",
        "## Causal Claim Requirements",
        "## Mandatory Uncertainty Disclosure",
        "## Causal Reasoning Traps",
        "## Your Task",
        "## Financial Context",
        "Respond with valid JSON",
    ]
    memo_end = len(prompt_text)
    for marker in end_markers:
        idx = prompt_text.find(marker, memo_start)
        if idx != -1 and idx < memo_end:
            memo_end = idx

    # Replace the memo region
    before = prompt_text[:memo_start]
    after = prompt_text[memo_end:]
    return before + "<<MEMO CONTENT INSERTED HERE>>\n\n" + after


def _build_allocation_top3(allocation: dict[str, float]) -> str:
    """Build a compact top-3 allocation string like 'NVDA:25%, AAPL:15%, MSFT:12%'."""
    if not allocation:
        return ""
    sorted_alloc = sorted(allocation.items(), key=lambda x: -x[1])
    top3 = sorted_alloc[:3]
    return ", ".join(f"{t}:{w:.0%}" for t, w in top3)
