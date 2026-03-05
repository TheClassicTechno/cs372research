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
    └── final/{final_portfolio.json, debate_diagnostic.txt}

    * prompt.txt files written in debug mode only.
"""

from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ------------------------------------------------------------------
# Diagnostic token budget — trim memo if casefile exceeds this
# ------------------------------------------------------------------

_DIAGNOSTIC_MAX_TOKENS = 50_000
_CHARS_PER_TOKEN = 4  # conservative heuristic for mixed English/code text


# ------------------------------------------------------------------
# Diagnostic scaffold — inserted verbatim at top of casefile
# ------------------------------------------------------------------

_DIAGNOSTIC_SCAFFOLD = """\
LLM DEBATE DIAGNOSIS SCAFFOLD
=============================

You are diagnosing the behavior of a multi-agent debate system used for portfolio allocation.

You will receive a structured debate diagnostic report describing how a debate evolved across multiple rounds, including agent allocations, reasoning quality metrics (CRIT), PID controller behavior, disagreement metrics, evidence usage, debate dynamics, and the prompts given to each agent.

Your task is to carefully analyze the debate and determine what likely went wrong or what dynamics occurred.

Follow the analysis steps below in order.

Do NOT skip steps.
Do NOT jump to conclusions before completing the analysis.

Focus on careful reasoning grounded in the provided artifact.


---------------------------------------------------------------------

STEP 1 — Debate Reconstruction

First reconstruct what happened in the debate.

Summarize:

1. What decision the agents were trying to make.
2. How many agents participated and their roles.
3. How many rounds occurred.
4. What the final portfolio allocations were.
5. Whether agents broadly agreed or disagreed at the end.

This step should only restate the facts of the debate.

Do not interpret yet.


---------------------------------------------------------------------

STEP 2 — Prompt Architecture Analysis

Carefully examine the prompts given to each agent.

Your goal is to determine whether the **prompt design itself may have caused the observed debate behavior**.

Analyze the following:

1. Role differentiation
   - Do the agent prompts meaningfully differ in goals, reasoning style, or decision criteria?
   - Or are they mostly identical with only minor cosmetic differences?

2. Shared reasoning scaffolding
   - Do the prompts force agents to follow the same reasoning structure?
   - Do they require identical causal templates or argument structures?

3. Evidence usage constraints
   - Are agents required to use the same types of evidence?
   - Are the evidence heuristics too similar across roles?

4. Over-constrained causal scaffolding
   - Are the prompts heavily structured in a way that restricts agents reasoning paths?
   - Do the prompts implicitly guide all agents toward similar conclusions?

5. Role collapse risk
   Determine whether the prompts make it difficult for agents to produce meaningfully different allocations.

6. Tone or PID effects
   - Does the tone configuration (e.g., adversarial vs balanced) encourage diversity or suppress disagreement?
   - Could tone changes be reducing debate diversity?

Explain whether prompt design likely **reduced agent diversity**.


---------------------------------------------------------------------

STEP 3 — Allocation Behavior

Analyze the portfolio allocations across agents.

Identify:

1. Which agents had the most similar portfolios.
2. Which agents disagreed the most.
3. Whether allocations changed meaningfully across rounds.
4. Whether the debate converged quickly or slowly.

Explain what these patterns imply about the behavior of the debate.


---------------------------------------------------------------------

STEP 4 — Reasoning Quality (CRIT)

Examine the reasoning quality metrics.

Focus on:

- overall reasoning quality (rho_bar)
- per-agent reasoning scores

Then analyze the CRIT pillars:

LV — Logical Validity
ES — Evidential Support
AC — Alternative Consideration
CA — Causal Alignment

Identify:

1. Which agents had the weakest reasoning.
2. Which pillar was weakest overall.
3. What types of reasoning failures occurred.


---------------------------------------------------------------------

STEP 5 — Debate Dynamics

Analyze how the debate evolved across rounds.

Determine:

1. Did disagreement decrease, remain stable, or increase?
2. Did reasoning quality improve or degrade?
3. Did agents revise their allocations meaningfully?
4. Did the debate stabilize or stagnate?

Classify the debate as one of the following:

- productive convergence
- premature convergence
- oscillation
- stagnation
- unresolved disagreement

Explain your reasoning.


---------------------------------------------------------------------

STEP 6 — PID Controller Behavior

Analyze the PID controller output.

Determine:

1. How beta changed across rounds.
2. Whether the controller stabilized the debate or caused oscillations.
3. Whether the controller reacted appropriately to reasoning signals.

Explain whether the controller improved debate quality or contributed to failure.


---------------------------------------------------------------------

STEP 7 — Evidence Usage

Analyze how evidence was used in the debate.

Determine:

1. Whether agents relied on agents evidence.
2. Whether a small number of evidence items dominated reasoning.
3. Whether claims were grounded in cited evidence.

Explain whether the debate relied on strong evidence or weak unsupported claims.


---------------------------------------------------------------------

STEP 8 — Failure Diagnosis

Based on the previous analysis, determine the most likely failure mode of the debate.

Possible failure modes include:

- Premature convergence
- Debate stagnation
- Evidence misuse
- Unsupported claims
- Weak causal reasoning
- Controller misregulation
- Overconfidence convergence
- Structural disagreement
- Prompt-induced role collapse

Explain your diagnosis and support it using evidence from the artifact.


---------------------------------------------------------------------

STEP 9 — Suggested Improvements

Propose concrete changes that could improve the debate system.

Possible areas include:

- prompt design
- critique aggressiveness
- evidence requirements
- PID parameter tuning
- debate structure
- agent diversity

Explain why each suggested change would improve the system.
"""


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
            "run_command": getattr(self._config, "run_command", None),
            "config_paths": getattr(self._config, "config_paths", []),
            "model_name": getattr(self._config, "model_name", "unknown"),
            "crit_model_name": getattr(self._config, "crit_model_name", "gpt-5-mini"),
            "temperature": getattr(self._config, "temperature", 0.3),
            "roles": [r.value for r in self._config.roles],
            "max_rounds": self._config.max_rounds,
            "actual_rounds": None,
            "terminated_early": None,
            "termination_reason": None,
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
        (self._round_dir / "CRIT").mkdir(exist_ok=True)
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

    def write_crit_prompts(self, captures: dict[str, dict]) -> None:
        """Write CRIT/<agent>/prompt.txt + response.txt for each agent."""
        if self._mode == "off" or self._round_dir is None:
            return
        for role, cap in captures.items():
            agent_dir = self._round_dir / "CRIT" / role
            agent_dir.mkdir(parents=True, exist_ok=True)
            prompt_content = (
                "=== SYSTEM PROMPT ===\n"
                f"{cap['system_prompt']}\n\n"
                "=== USER PROMPT ===\n"
                f"{cap['user_prompt']}\n"
            )
            (agent_dir / "prompt.txt").write_text(prompt_content, encoding="utf-8")
            (agent_dir / "response.txt").write_text(
                cap.get("raw_response", ""), encoding="utf-8",
            )

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
        crit_data: dict | None = None,
        pid_data: dict | None = None,
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
            "crit": crit_data or {},
            "pid": pid_data or {},
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
        crit_captures: dict[str, dict] | None = None,
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

        # Copy config file(s) used for this run
        config_paths = getattr(self._config, "config_paths", [])
        for cfg_path_str in config_paths:
            cfg_path = Path(cfg_path_str)
            if cfg_path.exists():
                shutil.copy2(cfg_path, final_dir / cfg_path.name)

        # pid_crit_all_rounds.json — consolidated PID/CRIT data for all rounds
        if pid_phase_data:
            _write_json(final_dir / "pid_crit_all_rounds.json", pid_phase_data)

        # debate_diagnostic.txt — plaintext casefile for LLM diagnosis
        casefile = self._build_diagnostic_casefile(
            state, pid_phase_data, terminated_early, enriched_context,
            crit_captures=crit_captures,
        )
        (final_dir / "debate_diagnostic.txt").write_text(casefile, encoding="utf-8")

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
    # Diagnostic casefile builder
    # ------------------------------------------------------------------

    def _build_diagnostic_casefile(
        self,
        state: dict,
        pid_phase_data: list[dict],
        terminated_early: bool,
        enriched_context: str,
        crit_captures: dict[str, dict] | None = None,
    ) -> str:
        """Build the plaintext debate diagnostic casefile.

        Returns a string with two parts:
        1. Fixed diagnostic scaffold (instructions for the diagnosing LLM)
        2. Structured debate data in three layers: SIGNAL / PROMPT / TRACE
        """
        sep = "=" * 80

        sections = [
            f"\n{sep}\nPART 2 — STRUCTURED DEBATE DATA\n{sep}\n",
            # --- SIGNAL LAYER (most important for diagnosis) ---
            f"\n{sep}\nSECTION 0 — META\n{sep}\n\n"
            + self._section_meta(state, pid_phase_data, terminated_early),
            f"\n{sep}\nSECTION 1 — TASK\n{sep}\n\n"
            + self._section_task_statement(enriched_context),
            f"\n{sep}\nSECTION 2 — SHARED INVESTMENT MEMO\n{sep}\n\n"
            + self._section_memo(enriched_context),
            f"\n{sep}\nSECTION 3 — SHARED PROMPT CONTRACT\n{sep}\n\n"
            + self._section_shared_contract(state),
            # --- PROMPT LAYER ---
            f"\n{sep}\nSECTION 4 — ROLE PROMPTS\n{sep}\n\n"
            + self._section_role_prompts(state),
            # --- SIGNAL LAYER (continued) ---
            f"\n{sep}\nSECTION 5 — ROUND HISTORY\n{sep}\n\n"
            + self._section_round_history(state),
            f"\n{sep}\nSECTION 6 — REASONING QUALITY (CRIT)\n{sep}\n\n"
            + self._section_crit_signal(pid_phase_data),
            f"\n{sep}\nSECTION 7 — PID CONTROLLER\n{sep}\n\n"
            + self._section_pid_signal(pid_phase_data),
            f"\n{sep}\nSECTION 8 — DISAGREEMENT METRICS\n{sep}\n\n"
            + self._section_disagreement_signal(pid_phase_data),
            # --- TRACE LAYER ---
            f"\n{sep}\nSECTION 9 — FULL PROPOSALS (ROUND 1 ONLY)\n{sep}\n\n"
            + self._section_full_proposals_r1(state),
        ]

        casefile = _DIAGNOSTIC_SCAFFOLD + "\n".join(sections) + "\n"

        # Trim memo if casefile exceeds token budget
        estimated_tokens = len(casefile) // _CHARS_PER_TOKEN
        if estimated_tokens > _DIAGNOSTIC_MAX_TOKENS:
            excess_chars = (estimated_tokens - _DIAGNOSTIC_MAX_TOKENS) * _CHARS_PER_TOKEN
            casefile = self._trim_memo_to_fit(casefile, excess_chars)

        return casefile

    # --- Token budget trimming ---

    @staticmethod
    def _trim_memo_to_fit(casefile: str, excess_chars: int) -> str:
        """Trim the memo (Section 2) from the bottom to shed *excess_chars*.

        Preserves complete lines and appends a truncation notice.
        Returns the casefile unchanged if memo markers aren't found.
        """
        memo_start = "SHARED INVESTMENT MEMO"
        memo_end = "END MEMO"

        start = casefile.find(memo_start)
        end = casefile.find(memo_end)
        if start < 0 or end < 0:
            return casefile

        memo_text = casefile[start:end]
        target_len = max(len(memo_text) - excess_chars, 200)
        trimmed = memo_text[:target_len]

        # Snap to last complete line
        last_newline = trimmed.rfind("\n")
        if last_newline > 0:
            trimmed = trimmed[:last_newline]

        trimmed += "\n\n[... MEMO TRIMMED — full memo in shared_context/memo.txt ...]\n"
        return casefile[:start] + trimmed + casefile[end:]

    # --- Section builders ---

    def _section_meta(
        self,
        state: dict,
        pid_phase_data: list[dict],
        terminated_early: bool,
    ) -> str:
        obs = state.get("observation", {})
        config = state.get("config", {})
        total_rounds = (
            len({d["round"] for d in pid_phase_data})
            if pid_phase_data
            else self._round_num
        )
        universe = obs.get("universe", [])
        roles = config.get("roles", [r.value for r in self._config.roles])
        reason = "stable_convergence" if terminated_early else "max_rounds"

        lines = [
            f"experiment: {self._experiment_name}",
            f"run_id: {self._run_id}",
            f"model: {getattr(self._config, 'model_name', 'unknown')}",
            f"as_of_date: {obs.get('timestamp', '')}",
            f"investment_quarter: {obs.get('timestamp', '')}",
            f"universe: {', '.join(universe)}",
            f"agents: [{' '.join(roles)}]",
            f"rounds_completed: {total_rounds}",
            f"termination_reason: {reason}",
        ]
        return "\n".join(lines) + "\n"

    def _section_task_statement(self, enriched_context: str) -> str:
        marker = "[INFO] QUARTERLY SNAPSHOT MEMO"
        idx = enriched_context.find(marker)
        if idx >= 0:
            task = enriched_context[:idx].strip()
            return (task + "\n") if task else "(no task header found before memo)\n"
        return (enriched_context[:500].strip() + "\n") if enriched_context else ""

    def _section_memo(self, enriched_context: str) -> str:
        marker = "[INFO] QUARTERLY SNAPSHOT MEMO"
        idx = enriched_context.find(marker)
        memo_body = enriched_context[idx:] if idx >= 0 else enriched_context

        lines = [
            "SHARED INVESTMENT MEMO",
            "(all agents receive this identical memo)",
            "----------------------",
            "",
            memo_body,
            "",
            "END MEMO",
        ]
        return "\n".join(lines) + "\n"

    def _section_shared_contract(self, state: dict) -> str:
        """Extract shared reasoning contract from agent prompts.

        Outputs the causal scaffolding, uncertainty disclosure, and trap
        awareness text that is common across all agents.  Shown once here
        instead of being duplicated inside every agent prompt dump.
        """
        # Extract scaffolding from the first proposal user prompt
        debate_turns = state.get("debate_turns", [])
        user_prompt = ""
        for turn in debate_turns:
            if turn.get("type") == "proposal" and turn.get("round", -1) == 0:
                user_prompt = turn.get("raw_user_prompt", "") or ""
                break

        # Also check if system causal contract was used
        sys_prompt = ""
        for turn in debate_turns:
            if turn.get("type") == "proposal" and turn.get("round", -1) == 0:
                sys_prompt = turn.get("raw_system_prompt", "") or ""
                break

        parts: list[str] = []
        parts.append(
            "(This reasoning contract is shared across all agents.\n"
            " It appears once here instead of being repeated per-agent.)\n"
        )

        # Extract system-level causal contract if present
        try:
            from ..prompts import SYSTEM_CAUSAL_CONTRACT
            if SYSTEM_CAUSAL_CONTRACT and SYSTEM_CAUSAL_CONTRACT.strip()[:60] in sys_prompt:
                parts.append("--- SYSTEM CAUSAL CONTRACT ---")
                parts.append(SYSTEM_CAUSAL_CONTRACT.strip())
                parts.append("")
        except ImportError:
            pass

        # Extract user-prompt scaffolding blocks
        _SCAFFOLDING_MARKERS = [
            ("CAUSAL CLAIM REQUIREMENTS", "## Causal Claim", "## Mandatory Uncertainty"),
            ("UNCERTAINTY DISCLOSURE", "## Mandatory Uncertainty", "## Causal Reasoning Traps"),
            ("TRAP AWARENESS", "## Causal Reasoning Traps", None),
        ]
        for label, start_marker, end_marker in _SCAFFOLDING_MARKERS:
            start_idx = user_prompt.find(start_marker)
            if start_idx == -1:
                continue
            if end_marker:
                end_idx = user_prompt.find(end_marker, start_idx)
                if end_idx == -1:
                    end_idx = len(user_prompt)
            else:
                # For the last block, find the next section boundary
                end_idx = len(user_prompt)
                for boundary in ["## Your Task", "Respond with valid JSON",
                                 "## Financial Context", "Using the data above"]:
                    bidx = user_prompt.find(boundary, start_idx + len(start_marker))
                    if bidx != -1 and bidx < end_idx:
                        end_idx = bidx
            block = user_prompt[start_idx:end_idx].strip()
            if block:
                parts.append(f"--- {label} ---")
                parts.append(block)
                parts.append("")

        if len(parts) <= 1:
            # Fallback: load scaffolding from module files directly
            try:
                from ..prompts import CAUSAL_CLAIM_FORMAT, FORCED_UNCERTAINTY, TRAP_AWARENESS
                if CAUSAL_CLAIM_FORMAT.strip():
                    parts.append("--- CAUSAL CLAIM REQUIREMENTS ---")
                    parts.append(CAUSAL_CLAIM_FORMAT.strip())
                    parts.append("")
                if FORCED_UNCERTAINTY.strip():
                    parts.append("--- UNCERTAINTY DISCLOSURE ---")
                    parts.append(FORCED_UNCERTAINTY.strip())
                    parts.append("")
                if TRAP_AWARENESS.strip():
                    parts.append("--- TRAP AWARENESS ---")
                    parts.append(TRAP_AWARENESS.strip())
                    parts.append("")
            except ImportError:
                parts.append("(scaffolding modules not available)")

        return "\n".join(parts) + "\n"

    def _section_role_prompts(self, state: dict) -> str:
        """Extract role-specific system prompts only (no shared contract).

        Each agent's role identity is shown once.  The shared causal
        contract and scaffolding are in Section 3 and not repeated here.
        """
        debate_turns = state.get("debate_turns", [])
        parts: list[str] = []

        # Collect unique role system prompts from round-0 proposals
        seen_roles: set[str] = set()
        for turn in debate_turns:
            if turn.get("type") != "proposal" or turn.get("round", -1) != 0:
                continue
            role = turn.get("role", "unknown")
            if role in seen_roles:
                continue
            seen_roles.add(role)

            sys_prompt = turn.get("raw_system_prompt", "") or ""

            # Strip the system causal contract prefix if present
            role_text = sys_prompt
            try:
                from ..prompts import SYSTEM_CAUSAL_CONTRACT
                contract = SYSTEM_CAUSAL_CONTRACT.strip()
                if contract and role_text.strip().startswith(contract[:60]):
                    # Remove the contract block
                    idx = role_text.find(contract)
                    if idx != -1:
                        role_text = role_text[idx + len(contract):].strip()
            except ImportError:
                pass

            parts.append(f"{role.upper()}\n")
            parts.append(role_text.strip())
            parts.append("")

        if not parts:
            return "(no proposal turns found — role prompts unavailable)\n"

        parts.insert(0, "(Shared reasoning contract omitted — see Section 3)\n")
        return "\n".join(parts) + "\n"

    def _section_round_history(self, state: dict) -> str:
        """Compact allocation vectors per agent per round."""
        debate_turns = state.get("debate_turns", [])

        # Group turns by round, keeping proposals and revisions
        rounds: dict[int, list[dict]] = defaultdict(list)
        for turn in debate_turns:
            t_type = turn.get("type", "")
            if t_type in ("proposal", "revision"):
                r = turn.get("round", 0)
                rounds[r].append(turn)

        if not rounds:
            return "(no allocation data found)\n"

        parts: list[str] = []
        for r in sorted(rounds.keys()):
            # Proposals use round=0 (0-indexed); revisions use 1-indexed round
            # numbers. Normalize to 1-indexed display labels.
            display_round = 1 if r == 0 else r
            label = f"ROUND {display_round}"
            parts.append(label)
            parts.append("")
            for turn in rounds[r]:
                role = turn.get("role", "unknown")
                content = turn.get("content", {})
                if not isinstance(content, dict):
                    content = {}
                alloc = content.get("allocation", {})
                if not alloc:
                    parts.append(f"{role}:\n(no allocation)\n")
                    continue
                weights = " ".join(
                    f"{t}={w:.2f}"
                    for t, w in sorted(alloc.items(), key=lambda x: -x[1])
                    if w > 0
                )
                parts.append(f"{role}:\n[{weights}]\n")
            parts.append("")

        return "\n".join(parts)

    def _section_crit_signal(self, pid_phase_data: list[dict]) -> str:
        """CRIT signal: per-round rho_bar + per-agent pillar table + per-round detail."""
        if not pid_phase_data:
            return "(no CRIT data — PID was not enabled)\n"

        parts: list[str] = []
        pillar_abbrs = ("LV", "ES", "AC", "CA")

        # Collect all agent roles across rounds (stable order)
        all_roles: list[str] = []
        for pd in pid_phase_data:
            for role in pd.get("crit", {}).get("agents", {}):
                if role not in all_roles:
                    all_roles.append(role)
        all_roles.sort()

        # --- Summary table: round | rho_bar | per-agent rho_i + pillars ---
        agent_cols = "   ".join(
            f"{r:<24}" for r in all_roles
        )
        header = f"{'round':<8}{'rho_bar':<10}{agent_cols}"
        parts.append(header)

        sub_labels = "   ".join(
            f"{'rho_i':<6} {'LV':>4} {'ES':>4} {'AC':>4} {'CA':>4}" for _ in all_roles
        )
        parts.append(f"{'':8}{'':10}{sub_labels}")

        for pd in pid_phase_data:
            r = pd.get("round", "?")
            crit = pd.get("crit", {})
            rho = crit.get("rho_bar")
            rho_str = f"{rho:.2f}" if isinstance(rho, (int, float)) else "—"
            agents = crit.get("agents", {})

            agent_vals: list[str] = []
            for role in all_roles:
                ad = agents.get(role, {})
                rho_i = ad.get("rho_i")
                pillars = ad.get("pillars", {})
                ri_str = f"{rho_i:.2f}" if isinstance(rho_i, (int, float)) else "  — "
                pv = [
                    f"{pillars.get(k, 0.0):.2f}" if isinstance(pillars.get(k), (int, float)) else " —  "
                    for k in pillar_abbrs
                ]
                agent_vals.append(f"{ri_str:<6} {pv[0]:>4} {pv[1]:>4} {pv[2]:>4} {pv[3]:>4}")

            line = f"{r:<8}{rho_str:<10}" + "   ".join(agent_vals)
            parts.append(line)

        parts.append("")

        # --- Per-round detail: diagnostics + explanations ---
        weakest_pillar = ""
        weakest_score = 2.0

        for pd in pid_phase_data:
            r = pd.get("round", "?")
            agents = pd.get("crit", {}).get("agents", {})
            if not agents:
                continue

            parts.append(f"--- Round {r} per-agent detail ---")
            parts.append("")

            for role in sorted(agents.keys()):
                agent_data = agents[role]
                pillars = agent_data.get("pillars", {})
                rho_i = agent_data.get("rho_i")
                rho_i_str = f" (rho_i={rho_i:.2f})" if isinstance(rho_i, (int, float)) else ""
                parts.append(f"{role}{rho_i_str}")
                for abbr in pillar_abbrs:
                    val = pillars.get(abbr)
                    val_str = f"{val:.2f}" if isinstance(val, (int, float)) else str(val)
                    parts.append(f"  {abbr} {val_str}")
                    if isinstance(val, (int, float)) and val < weakest_score:
                        weakest_score = val
                        weakest_pillar = f"{role}/{abbr}"

                # Diagnostics
                diag = agent_data.get("diagnostics", {})
                fired = [k for k, v in diag.items() if v]
                if fired:
                    parts.append(f"  flags: {', '.join(fired)}")

                # Explanations
                explanations = agent_data.get("explanations", {})
                if explanations:
                    for pillar_key, short_name in [
                        ("logical_validity", "LV"),
                        ("evidential_support", "ES"),
                        ("alternative_consideration", "AC"),
                        ("causal_alignment", "CA"),
                    ]:
                        expl = explanations.get(pillar_key)
                        if expl:
                            parts.append(f"    {short_name}: {expl}")

                parts.append("")

        if weakest_pillar:
            parts.append(f"weakest_pillar: {weakest_pillar} ({weakest_score:.2f})")
            parts.append("")

        return "\n".join(parts)

    def _section_pid_signal(self, pid_phase_data: list[dict]) -> str:
        """Compact PID signal: target_rho + beta table + controller params."""
        if not pid_phase_data:
            return "(no PID data — PID was not enabled)\n"

        parts: list[str] = []

        # target_rho from config
        rho_star = getattr(self._config, "pid_rho_star", None)
        if rho_star is None and self._config.pid_config is not None:
            rho_star = self._config.pid_config.rho_star
        if rho_star is not None:
            parts.append(f"target_rho: {rho_star}")
            parts.append("")

        # --- Beta table ---
        parts.append(f"{'round':<8}{'beta':<10}")
        for pd in pid_phase_data:
            r = pd.get("round", "?")
            beta = pd.get("pid", {}).get("beta_new", pd.get("beta_in"))
            beta_str = f"{beta:.2f}" if isinstance(beta, (int, float)) else str(beta)
            parts.append(f"{r:<8}{beta_str:<10}")
        parts.append("")

        # --- Controller parameters (compact) ---
        if self._config.pid_config is not None:
            gains = self._config.pid_config.gains
            parts.append("controller_params:")
            parts.append(f"  Kp={gains.Kp}  Ki={gains.Ki}  Kd={gains.Kd}")
            parts.append("")

        return "\n".join(parts)

    def _section_disagreement_signal(self, pid_phase_data: list[dict]) -> str:
        """Compact disagreement tables: JS divergence + evidence overlap."""
        if not pid_phase_data:
            return "(no divergence data — PID was not enabled)\n"

        parts: list[str] = []

        # --- JS divergence table ---
        parts.append(f"{'round':<8}{'JS':<10}")
        for pd in pid_phase_data:
            r = pd.get("round", "?")
            js = pd.get("divergence", {}).get("js")
            js_str = f"{js:.2f}" if isinstance(js, (int, float)) else str(js)
            parts.append(f"{r:<8}{js_str:<10}")
        parts.append("")

        # --- Evidence overlap table ---
        parts.append(f"{'round':<8}{'overlap':<10}")
        for pd in pid_phase_data:
            r = pd.get("round", "?")
            ov = pd.get("divergence", {}).get("ov")
            ov_str = f"{ov:.2f}" if isinstance(ov, (int, float)) else str(ov)
            parts.append(f"{r:<8}{ov_str:<10}")
        parts.append("")

        return "\n".join(parts)

    def _section_full_proposals_r1(self, state: dict) -> str:
        """Full proposal output for round 1 only — reasoning, claims, evidence."""
        debate_turns = state.get("debate_turns", [])
        parts: list[str] = []

        for turn in debate_turns:
            if turn.get("type") != "proposal" or turn.get("round", -1) != 0:
                continue
            role = turn.get("role", "unknown")
            raw = turn.get("raw_response", "") or ""
            parts.append(f"=== {role.upper()} ===\n")
            parts.append(raw.strip())
            parts.append("")

        return "\n".join(parts) if parts else "(no proposal turns found)\n"


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


def _fmt_dict(d: dict, ndigits: int = 4) -> str:
    """Format a dict as compact key=value pairs."""
    parts = []
    for k, v in sorted(d.items()):
        if isinstance(v, float):
            parts.append(f"{k}={round(v, ndigits)}")
        else:
            parts.append(f"{k}={v}")
    return "  ".join(parts)
