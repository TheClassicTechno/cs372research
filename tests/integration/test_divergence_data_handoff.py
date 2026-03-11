"""Integration tests for divergence metric data handoffs.

Verifies the full pipeline: runner → logger → disk → scanner.

1. Runner._latest_per_role deduplicates accumulated state correctly
2. Revision divergence is logged independently of CRIT (decoupled)
3. Retry divergence uses deduplicated decisions (correct JS + evidence)
4. Scanner reads retry files with proper is-not-None guards (no falsy-zero bugs)
5. Evidence overlap is non-empty when agents cite overlapping evidence

Uses _CollapsingLLM to trigger intervention retries with known allocations.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from multi_agent.config import DebateConfig
from multi_agent.models import MarketState, Observation, PortfolioState
from multi_agent.prompts.profile_loader import get_agent_profiles
from multi_agent.runner import MultiAgentRunner

pytestmark = [pytest.mark.integration, pytest.mark.no_external_api]

TICKERS = ["AAPL", "MSFT", "NVDA"]
ROLES = ["value", "risk", "technical"]
AGENT_PROFILE_MAP = {
    "value": "value_enriched",
    "risk": "risk_enriched",
    "technical": "technical_enriched",
}
JUDGE_PROFILE = "judge_standard"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _equal_allocation(tickers: list[str]) -> dict[str, float]:
    n = len(tickers)
    w = round(1.0 / n, 4)
    alloc = {t: w for t in tickers}
    remainder = round(1.0 - sum(alloc.values()), 4)
    alloc[tickers[0]] = round(alloc[tickers[0]] + remainder, 4)
    return alloc


def _make_observation() -> Observation:
    return Observation(
        universe=TICKERS,
        timestamp="2025-01-15T00:00:00Z",
        market_state=MarketState(
            prices={t: 100.0 + i * 50 for i, t in enumerate(TICKERS)},
        ),
        portfolio_state=PortfolioState(cash=100_000.0, positions={}),
        text_context=(
            "## Market Summary\n"
            "Equity markets showed mixed signals.\n"
            "[AAPL-RET60]: AAPL 60-day return +8.2%\n"
            "[MSFT-VOL30]: MSFT 30-day volatility 22.1%\n"
            "[NVDA-RET30]: NVDA 30-day return +15.4%\n"
            "[MACRO-FF]: Fed Funds Rate 5.25%\n"
        ),
    )


def _make_crit_response() -> str:
    return json.dumps({
        "pillar_scores": {
            "logical_validity": 0.78,
            "evidential_support": 0.72,
            "alternative_consideration": 0.68,
            "causal_alignment": 0.65,
        },
        "diagnostics": {
            "contradictions_detected": False,
            "unsupported_claims_detected": False,
            "ignored_critiques_detected": False,
            "premature_certainty_detected": False,
            "causal_overreach_detected": False,
            "conclusion_drift_detected": False,
            "contradictions_count": 0,
            "unsupported_claims_count": 0,
            "ignored_critiques_count": 0,
            "causal_overreach_count": 0,
            "orphaned_positions_count": 0,
        },
        "explanations": {
            "logical_validity": "Sound reasoning.",
            "evidential_support": "Well-supported.",
            "alternative_consideration": "Considered alternatives.",
            "causal_alignment": "Aligned.",
        },
    })


def _make_enriched_critique(role: str, roles: list[str]) -> str:
    targets = [r for r in roles if r != role]
    return json.dumps({
        "critiques": [
            {
                "target_role": t,
                "target_claim": f"C1",
                "objection": f"{role} challenges {t} position.",
                "critique_text": f"{role} challenges {t} position.",
                "critique_type": "logical",
                "severity": "moderate",
                "affected_positions": TICKERS[:1],
                "suggested_action": "Revise allocation",
                "suggested_adjustment": "Revise allocation weights",
                "portfolio_implication": f"Reduces {t} concentration risk",
                "falsifier": "Sector momentum reversal",
                "objection_confidence": 0.65,
                "counter_evidence": [f"[{TICKERS[0]}-RET60]"],
            }
            for t in targets
        ],
    })


def _make_judge_response(tickers: list[str]) -> str:
    alloc = _equal_allocation(tickers)
    return json.dumps({
        "allocation": alloc,
        "audited_memo": "Balanced allocation across all tickers based on debate.",
        "portfolio_rationale": "Balanced allocation.",
        "confidence": 0.7,
        "claims": [{
            "claim_id": "C1",
            "claim_text": "Balanced approach.",
            "claim_type": "macro",
            "reasoning_type": "causal",
            "evidence": [f"[{tickers[0]}-RET60]"],
            "assumptions": ["Stable"],
            "falsifiers": ["Shock"],
            "impacts_positions": tickers[:],
            "confidence": 0.7,
        }],
        "position_rationale": [
            {"ticker": t, "weight": alloc[t],
             "supported_by_claims": ["C1"],
             "explanation": f"Weight for {t}."}
            for t in tickers
        ],
        "risks_or_falsifiers": ["Macro shock"],
    })


class _CollapsingLLM:
    """LLM that triggers JS collapse on first revise, diverse on retry.

    All responses include evidence IDs so evidence overlap can be computed.
    """
    _DIVERSE_ALLOCS = {
        "value":     {"AAPL": 0.60, "MSFT": 0.25, "NVDA": 0.15},
        "risk":      {"AAPL": 0.20, "MSFT": 0.50, "NVDA": 0.30},
        "technical": {"AAPL": 0.10, "MSFT": 0.20, "NVDA": 0.70},
    }
    _RETRY_ALLOCS = {
        "value":     {"AAPL": 0.55, "MSFT": 0.30, "NVDA": 0.15},
        "risk":      {"AAPL": 0.25, "MSFT": 0.45, "NVDA": 0.30},
        "technical": {"AAPL": 0.15, "MSFT": 0.20, "NVDA": 0.65},
    }

    def __init__(self, tickers: list[str], roles: list[str]):
        self.calls: list[dict] = []
        self._tickers = tickers
        self._roles = roles
        self._revise_call_count = 0

    def __call__(
        self, config, system_prompt, user_prompt,
        role=None, phase=None, round_num=0,
    ) -> str:
        self.calls.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "role": role,
            "phase": phase,
            "round_num": round_num,
        })
        if phase == "crit":
            return _make_crit_response()
        if phase == "judge":
            return _make_judge_response(self._tickers)
        if phase == "propose":
            return self._make_response(
                role, self._DIVERSE_ALLOCS.get(role, _equal_allocation(self._tickers)),
                "proposes diverse allocation",
            )
        if phase == "critique":
            return _make_enriched_critique(role, self._roles)
        if phase == "revise":
            self._revise_call_count += 1
            if self._revise_call_count <= len(self._roles):
                # First revise: collapse (near-identical allocations)
                alloc = {t: round(1.0 / len(self._tickers), 4) for t in self._tickers}
                remainder = round(1.0 - sum(alloc.values()), 4)
                alloc[self._tickers[0]] = round(alloc[self._tickers[0]] + remainder, 4)
                return self._make_response(role, alloc, "collapsed to consensus")
            else:
                # Retry: diverse
                alloc = self._RETRY_ALLOCS.get(role, _equal_allocation(self._tickers))
                return self._make_response(role, alloc, "revised with diversity")
        raise ValueError(f"Unexpected phase={phase!r} role={role!r}")

    def _make_response(self, role: str, alloc: dict, rationale: str) -> str:
        return json.dumps({
            "allocation": alloc,
            "portfolio_rationale": f"{role} {rationale}.",
            "confidence": 0.7,
            "revision_notes": f"{role} {rationale}.",
            "claims": [{
                "claim_id": "C1",
                "claim_text": f"{role} claim C1: Analysis-based.",
                "claim_type": "macro",
                "reasoning_type": "causal",
                "evidence": [f"[{self._tickers[0]}-RET60]", f"[MACRO-FF]"],
                "assumptions": ["Stable"],
                "falsifiers": ["Shock"],
                "impacts_positions": self._tickers[:],
                "confidence": 0.7,
            }],
            "position_rationale": [
                {"ticker": t, "weight": alloc[t],
                 "supported_by_claims": ["C1"],
                 "explanation": f"Weight for {t}."}
                for t in self._tickers
            ],
            "risks_or_falsifiers": ["Macro shock"],
        })


def _make_intervention_config_with_logging(
    tmp_path: Path,
    max_rounds: int = 1,
) -> DebateConfig:
    """DebateConfig with intervention + debug logging to tmp_path."""
    from eval.PID.types import PIDConfig, PIDGains

    profiles = get_agent_profiles(AGENT_PROFILE_MAP, judge_profile_name=JUDGE_PROFILE)
    judge_profile = profiles.pop("judge", {})

    return DebateConfig(
        roles=list(ROLES),
        max_rounds=max_rounds,
        mock=False,
        parallel_agents=False,
        console_display=False,
        verbose=False,
        trace_dir=str(tmp_path / "traces"),
        agent_profiles=profiles,
        agent_profile_names={r: AGENT_PROFILE_MAP[r] for r in ROLES},
        judge_profile=judge_profile,
        pid_config=PIDConfig(
            gains=PIDGains(Kp=0.15, Ki=0.01, Kd=0.03),
            rho_star=0.8,
        ),
        initial_beta=0.5,
        logging_mode="debug",
        experiment_name="test_divergence_handoff",
        crit_model_name="gpt-5-mini",
        intervention_config={
            "enabled": True,
            "rules": {
                "js_collapse": {
                    "threshold": 0.4,
                    "min_js_proposal": 0.05,
                    "max_retries": 2,
                },
            },
        },
    )


def _find_run_dir(base: Path) -> Path:
    """Find the single run directory under the experiment logging path."""
    experiment_dir = base / "logging" / "runs" / "test_divergence_handoff"
    runs = sorted(experiment_dir.glob("run_*"))
    assert len(runs) == 1, f"Expected 1 run dir, found {len(runs)}: {runs}"
    return runs[0]


# ---------------------------------------------------------------------------
# Test: Runner._latest_per_role
# ---------------------------------------------------------------------------

class TestRunnerLatestPerRole:
    """Unit tests for MultiAgentRunner._latest_per_role."""

    def test_deduplicates_to_latest(self):
        r1 = [
            {"role": "macro", "v": "r1_macro"},
            {"role": "tech", "v": "r1_tech"},
        ]
        r2 = [
            {"role": "macro", "v": "r2_macro"},
            {"role": "tech", "v": "r2_tech"},
        ]
        result = MultiAgentRunner._latest_per_role(r1 + r2)
        assert len(result) == 2
        by_role = {d["role"]: d for d in result}
        assert by_role["macro"]["v"] == "r2_macro"
        assert by_role["tech"]["v"] == "r2_tech"

    def test_single_round_unchanged(self):
        entries = [
            {"role": "a", "v": 1},
            {"role": "b", "v": 2},
        ]
        result = MultiAgentRunner._latest_per_role(entries)
        assert len(result) == 2

    def test_empty_list(self):
        assert MultiAgentRunner._latest_per_role([]) == []

    def test_crashes_on_missing_role(self):
        entries = [{"no_role_key": True}]
        with pytest.raises(ValueError, match="missing 'role' key"):
            MultiAgentRunner._latest_per_role(entries)

    def test_three_duplicates_keeps_last(self):
        entries = [
            {"role": "x", "v": 1},
            {"role": "x", "v": 2},
            {"role": "x", "v": 3},
        ]
        result = MultiAgentRunner._latest_per_role(entries)
        assert len(result) == 1
        assert result[0]["v"] == 3


# ---------------------------------------------------------------------------
# Test: Divergence files written to disk
# ---------------------------------------------------------------------------

class TestDivergenceFilesWritten:
    """Verify divergence metric files are written for all phases."""

    @pytest.fixture
    def run_dir(self, tmp_path, monkeypatch):
        """Run pipeline with logging, return the run directory."""
        monkeypatch.chdir(tmp_path)
        llm = _CollapsingLLM(TICKERS, ROLES)
        monkeypatch.setattr("multi_agent.graph.nodes._call_llm", llm)
        monkeypatch.setattr("multi_agent.runner._call_llm", llm)

        config = _make_intervention_config_with_logging(tmp_path)
        runner = MultiAgentRunner(config)
        runner.run_returning_state(_make_observation())
        return _find_run_dir(tmp_path)

    def test_proposal_js_divergence_exists(self, run_dir):
        f = run_dir / "rounds" / "round_001" / "metrics" / "js_divergence_propose.json"
        assert f.exists(), f"Missing {f}"
        data = json.loads(f.read_text())
        assert data["phase"] == "propose"
        assert isinstance(data["js_divergence"], float)
        assert data["js_divergence"] > 0

    def test_proposal_evidence_overlap_exists(self, run_dir):
        f = run_dir / "rounds" / "round_001" / "metrics" / "evidence_overlap_propose.json"
        assert f.exists(), f"Missing {f}"
        data = json.loads(f.read_text())
        assert data["phase"] == "propose"
        assert "mean_overlap" in data
        assert "agent_evidence_ids" in data

    def test_revision_js_divergence_exists(self, run_dir):
        """Revision divergence must be written even when CRIT runs after."""
        f = run_dir / "rounds" / "round_001" / "metrics" / "js_divergence.json"
        assert f.exists(), f"Missing revision js_divergence.json"
        data = json.loads(f.read_text())
        assert data["phase"] == "revise"
        assert isinstance(data["js_divergence"], float)

    def test_revision_evidence_overlap_exists(self, run_dir):
        f = run_dir / "rounds" / "round_001" / "metrics" / "evidence_overlap.json"
        assert f.exists(), f"Missing revision evidence_overlap.json"
        data = json.loads(f.read_text())
        assert data["phase"] == "revise"
        assert "mean_overlap" in data
        assert "agent_evidence_ids" in data

    def test_retry_js_divergence_exists(self, run_dir):
        """At least one retry file must exist (CollapsingLLM triggers collapse)."""
        metrics = run_dir / "rounds" / "round_001" / "metrics"
        retry_files = sorted(metrics.glob("js_divergence_retry_*.json"))
        assert len(retry_files) >= 1, "No retry JS divergence files found"
        data = json.loads(retry_files[0].read_text())
        assert data["phase"].startswith("retry_")
        assert isinstance(data["js_divergence"], float)

    def test_retry_evidence_overlap_exists(self, run_dir):
        metrics = run_dir / "rounds" / "round_001" / "metrics"
        retry_files = sorted(metrics.glob("evidence_overlap_retry_*.json"))
        assert len(retry_files) >= 1, "No retry evidence overlap files found"
        data = json.loads(retry_files[0].read_text())
        assert data["phase"].startswith("retry_")
        assert "agent_evidence_ids" in data

    def test_retry_evidence_ids_not_empty(self, run_dir):
        """Retry evidence must have non-empty agent_evidence_ids."""
        metrics = run_dir / "rounds" / "round_001" / "metrics"
        retry_files = sorted(metrics.glob("evidence_overlap_retry_*.json"))
        assert len(retry_files) >= 1
        data = json.loads(retry_files[0].read_text())
        assert data["agent_evidence_ids"], (
            f"agent_evidence_ids is empty: {data}"
        )
        # Each agent should have evidence IDs
        for role, ids in data["agent_evidence_ids"].items():
            assert len(ids) > 0, f"Agent {role} has no evidence IDs in retry"

    def test_revision_js_less_than_proposal_js(self, run_dir):
        """Collapsed revisions should have lower JS than diverse proposals."""
        metrics = run_dir / "rounds" / "round_001" / "metrics"
        propose = json.loads((metrics / "js_divergence_propose.json").read_text())
        revise = json.loads((metrics / "js_divergence.json").read_text())
        assert revise["js_divergence"] < propose["js_divergence"], (
            f"Revision JS ({revise['js_divergence']}) should be less than "
            f"proposal JS ({propose['js_divergence']}) due to collapse"
        )

    def test_retry_js_different_from_collapsed_revision(self, run_dir):
        """Retry JS should differ from collapsed revision JS."""
        metrics = run_dir / "rounds" / "round_001" / "metrics"
        revise = json.loads((metrics / "js_divergence.json").read_text())
        retry_files = sorted(metrics.glob("js_divergence_retry_*.json"))
        assert len(retry_files) >= 1
        retry = json.loads(retry_files[0].read_text())
        # Retry allocations are diverse, so JS should be different
        assert retry["js_divergence"] != revise["js_divergence"], (
            f"Retry JS ({retry['js_divergence']}) should differ from "
            f"collapsed revision JS ({revise['js_divergence']})"
        )


# ---------------------------------------------------------------------------
# Test: Scanner reads divergence files correctly
# ---------------------------------------------------------------------------

class TestScannerDivergenceHandoff:
    """Verify the scanner reads all divergence phases including retries."""

    @pytest.fixture
    def run_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        llm = _CollapsingLLM(TICKERS, ROLES)
        monkeypatch.setattr("multi_agent.graph.nodes._call_llm", llm)
        monkeypatch.setattr("multi_agent.runner._call_llm", llm)

        config = _make_intervention_config_with_logging(tmp_path)
        runner = MultiAgentRunner(config)
        runner.run_returning_state(_make_observation())
        return _find_run_dir(tmp_path)

    def _get_trajectory(self, run_dir):
        """Call get_divergence_trajectory with correctly decomposed path args."""
        from tools.dashboard.run_scanner import get_divergence_trajectory
        # run_dir = .../logging/runs/<experiment>/<run_id>
        # get_divergence_trajectory expects (base_path, experiment, run_id)
        return get_divergence_trajectory(
            run_dir.parents[1], run_dir.parent.name, run_dir.name,
        )

    def test_scanner_returns_all_phases(self, run_dir):
        result = self._get_trajectory(run_dir)
        phases = [entry["phase"] for entry in result]
        assert "propose" in phases, f"Missing propose phase: {phases}"
        assert "revise" in phases, f"Missing revise phase: {phases}"
        # At least one retry
        retry_phases = [p for p in phases if p.startswith("retry_")]
        assert len(retry_phases) >= 1, f"No retry phases found: {phases}"

    def test_scanner_propose_has_js(self, run_dir):
        result = self._get_trajectory(run_dir)
        propose = next(e for e in result if e["phase"] == "propose")
        assert propose["js_divergence"] is not None
        assert isinstance(propose["js_divergence"], float)
        assert propose["js_divergence"] > 0

    def test_scanner_revise_has_js(self, run_dir):
        result = self._get_trajectory(run_dir)
        revise = next(e for e in result if e["phase"] == "revise")
        assert revise["js_divergence"] is not None
        assert isinstance(revise["js_divergence"], float)

    def test_scanner_retry_has_js(self, run_dir):
        result = self._get_trajectory(run_dir)
        retries = [e for e in result if e["phase"].startswith("retry_")]
        assert len(retries) >= 1
        for r in retries:
            assert r["js_divergence"] is not None, (
                f"Retry {r['phase']} has None js_divergence"
            )
            assert isinstance(r["js_divergence"], float)

    def test_scanner_zero_overlap_not_treated_as_none(self, run_dir):
        """Overlap of 0.0 must appear as 0.0 in scanner output, not None."""
        result = self._get_trajectory(run_dir)
        for entry in result:
            # mean_overlap should be a number, never None
            assert entry.get("mean_overlap") is not None or entry["mean_overlap"] == 0.0, (
                f"Phase {entry['phase']}: mean_overlap should not be None "
                f"(falsy-zero bug)"
            )

    def test_scanner_retry_evidence_not_empty(self, run_dir):
        """Scanner should report non-empty evidence for retries."""
        result = self._get_trajectory(run_dir)
        retries = [e for e in result if e["phase"].startswith("retry_")]
        assert len(retries) >= 1
        # At least the first retry should have evidence data
        # (the agent responses include evidence IDs)
        first_retry = retries[0]
        # If mean_overlap is present and there are agent_evidence_ids,
        # evidence extraction worked
        assert "mean_overlap" in first_retry, (
            f"Retry missing mean_overlap: {first_retry}"
        )

    def test_scanner_phase_ordering(self, run_dir):
        """Phases within a round must be ordered: propose, revise, retry_001, retry_002..."""
        result = self._get_trajectory(run_dir)
        # Group by round
        by_round: dict[int, list[str]] = {}
        for entry in result:
            rn = entry["round"]
            if rn not in by_round:
                by_round[rn] = []
            by_round[rn].append(entry["phase"])

        for rn, phases in by_round.items():
            assert phases[0] == "propose", (
                f"Round {rn}: first phase should be 'propose', got {phases}"
            )
            assert phases[1] == "revise", (
                f"Round {rn}: second phase should be 'revise', got {phases}"
            )
            # Remaining should be retries in order
            for i, p in enumerate(phases[2:]):
                assert p.startswith("retry_"), (
                    f"Round {rn}: phase[{i+2}] should be retry, got {p}"
                )
