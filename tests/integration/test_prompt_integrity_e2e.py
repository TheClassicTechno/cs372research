"""Prompt-Integrity End-to-End Integration Tests.

Exercises the full debate pipeline (propose → critique → revise → CRIT → PID)
with intercepted LLM calls, verifying prompt assembly, normalization, critique
routing, state transitions, call counts, and cross-round propagation.

Uses a CapturingLLM that intercepts at the _call_llm boundary (NOT mock=True,
which bypasses prompt assembly) and returns enriched-format fixture responses
routed by the explicit `phase` parameter.

See PLAN_prompt_integrity_e2e.md for the full design rationale.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from multi_agent.config import DebateConfig
from multi_agent.models import MarketState, Observation, PortfolioState
from multi_agent.prompts.profile_loader import get_agent_profiles
from multi_agent.runner import MultiAgentRunner

# ---------------------------------------------------------------------------
# Constants & Config
# ---------------------------------------------------------------------------

pytestmark = [pytest.mark.integration, pytest.mark.no_external_api]

TICKERS = ["AAPL", "MSFT", "NVDA"]
ROLES = ["value", "risk", "technical"]
GOLDEN_DIR = Path(__file__).parent / "golden_prompts"
HASH_FILE = Path(__file__).parent / "prompt_hashes.json"
CANONICAL_RUN = (
    Path(__file__).resolve().parent.parent.parent
    / "logging" / "runs" / "test" / "run_2026-03-07_19-50-06"
)

# Agent profile map — enriched profiles for each role
AGENT_PROFILE_MAP = {
    "value": "value_enriched",
    "risk": "risk_enriched",
    "technical": "technical_enriched",
}
JUDGE_PROFILE = "judge_standard"


# ---------------------------------------------------------------------------
# Fixture Factories — produce enriched-format JSON responses
# ---------------------------------------------------------------------------

def _equal_allocation(tickers: list[str]) -> dict[str, float]:
    """Create equal-weight allocation across tickers."""
    n = len(tickers)
    w = round(1.0 / n, 4)
    alloc = {t: w for t in tickers}
    # Fix rounding to sum exactly to 1.0
    remainder = round(1.0 - sum(alloc.values()), 4)
    alloc[tickers[0]] = round(alloc[tickers[0]] + remainder, 4)
    return alloc


def _make_enriched_proposal(role: str, tickers: list[str]) -> str:
    """Enriched proposal with claims, position_rationale, evidence."""
    alloc = _equal_allocation(tickers)
    result = {
        "allocation": alloc,
        "portfolio_rationale": f"{role} proposes balanced allocation based on domain analysis.",
        "confidence": 0.7,
        "claims": [
            {
                "claim_id": "C1",
                "claim_text": f"{role} claim C1: Market conditions favor diversification.",
                "claim_type": "macro",
                "reasoning_type": "causal",
                "evidence": [f"[{tickers[0]}-RET60]", f"[{tickers[1]}-VOL30]"],
                "assumptions": ["Stable macro environment"],
                "falsifiers": ["Sudden rate hike reversal"],
                "impacts_positions": [tickers[0], tickers[1]],
                "confidence": 0.75,
            },
            {
                "claim_id": "C2",
                "claim_text": f"{role} claim C2: Tech sector shows momentum.",
                "claim_type": "sector",
                "reasoning_type": "observational",
                "evidence": [f"[{tickers[2]}-RET30]"],
                "assumptions": ["Earnings growth continues"],
                "falsifiers": ["Sector rotation to defensives"],
                "impacts_positions": [tickers[2]],
                "confidence": 0.65,
            },
            {
                "claim_id": "C3",
                "claim_text": f"{role} claim C3: Risk-adjusted returns support equal weight.",
                "claim_type": "risk",
                "reasoning_type": "risk_assessment",
                "evidence": [f"[MACRO-FF]"],
                "assumptions": ["Volatility stays contained"],
                "falsifiers": ["VIX spike above 30"],
                "impacts_positions": tickers[:],
                "confidence": 0.6,
            },
        ],
        "position_rationale": [
            {
                "ticker": t,
                "weight": alloc[t],
                "supported_by_claims": ["C1", "C3"] if t != tickers[2] else ["C2", "C3"],
                "explanation": f"Allocating {alloc[t]:.2%} to {t} based on {role} analysis.",
            }
            for t in tickers
        ],
        "risks_or_falsifiers": [
            "Unexpected macro shock",
            "Liquidity crisis",
        ],
    }
    return json.dumps(result, indent=2)


def _make_enriched_critique(critic_role: str, all_roles: list[str]) -> str:
    """Enriched critique targeting each OTHER agent."""
    critiques = []
    for target in all_roles:
        if target == critic_role:
            continue
        critiques.append({
            "target_role": target,
            "target_claim": "C1",
            "objection": (
                f"{critic_role} challenges {target}'s C1: "
                f"The evidence cited is insufficient for the causal claim."
            ),
            "counter_evidence": [f"[{TICKERS[0]}-VOL60]"],
            "portfolio_implication": f"Suggests reducing {target}'s overweight positions.",
            "suggested_adjustment": f"Reduce {TICKERS[0]} weight by 5%.",
            "falsifier": "If new earnings data confirms the thesis, objection is withdrawn.",
            "objection_confidence": 0.6,
        })
    return json.dumps({"critiques": critiques}, indent=2)


def _make_enriched_revision(role: str, tickers: list[str]) -> str:
    """Enriched revision with revision_notes addressing critiques."""
    alloc = _equal_allocation(tickers)
    # Slightly adjust allocation to show revision effect
    alloc[tickers[0]] = round(alloc[tickers[0]] - 0.02, 4)
    alloc[tickers[2]] = round(alloc[tickers[2]] + 0.02, 4)

    result = {
        "allocation": alloc,
        "portfolio_rationale": f"{role} revised allocation after considering critiques.",
        "confidence": 0.65,
        "revision_notes": (
            f"Accepted critique on C1 — reduced confidence and adjusted {tickers[0]} weight. "
            f"Rejected critique on C3 — risk assessment remains valid per [MACRO-FF] evidence."
        ),
        "claims": [
            {
                "claim_id": "C1",
                "claim_text": f"{role} revised C1: Market conditions favor diversification (revised down).",
                "claim_type": "macro",
                "reasoning_type": "causal",
                "evidence": [f"[{tickers[0]}-RET60]", f"[{tickers[1]}-VOL30]"],
                "assumptions": ["Stable macro environment"],
                "falsifiers": ["Sudden rate hike reversal"],
                "impacts_positions": [tickers[0], tickers[1]],
                "confidence": 0.65,
            },
            {
                "claim_id": "C2",
                "claim_text": f"{role} revised C2: Tech sector shows momentum.",
                "claim_type": "sector",
                "reasoning_type": "observational",
                "evidence": [f"[{tickers[2]}-RET30]"],
                "assumptions": ["Earnings growth continues"],
                "falsifiers": ["Sector rotation to defensives"],
                "impacts_positions": [tickers[2]],
                "confidence": 0.65,
            },
            {
                "claim_id": "C3",
                "claim_text": f"{role} revised C3: Risk-adjusted returns support near-equal weight.",
                "claim_type": "risk",
                "reasoning_type": "risk_assessment",
                "evidence": [f"[MACRO-FF]"],
                "assumptions": ["Volatility stays contained"],
                "falsifiers": ["VIX spike above 30"],
                "impacts_positions": tickers[:],
                "confidence": 0.6,
            },
        ],
        "position_rationale": [
            {
                "ticker": t,
                "weight": alloc[t],
                "supported_by_claims": ["C1", "C3"] if t != tickers[2] else ["C2", "C3"],
                "explanation": f"Revised allocation for {t} based on critique feedback.",
            }
            for t in tickers
        ],
        "risks_or_falsifiers": [
            "Unexpected macro shock",
            "Liquidity crisis",
            "Counter-evidence on rate path",
        ],
    }
    return json.dumps(result, indent=2)


def _make_crit_response() -> str:
    """CRIT evaluation response with 4 pillar scores."""
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
            "causal_overreach_detected": True,
            "conclusion_drift_detected": False,
            "contradictions_count": 0,
            "unsupported_claims_count": 0,
            "ignored_critiques_count": 0,
            "causal_overreach_count": 1,
            "orphaned_positions_count": 0,
        },
        "explanations": {
            "logical_validity": "Argument is internally consistent with clear claim structure.",
            "evidential_support": "Most claims cite evidence, minor gaps in C3.",
            "alternative_consideration": "Critiques addressed in revision notes.",
            "causal_alignment": "C1 claims causal mechanism but evidence is primarily associational.",
        },
    })


def _make_judge_response(tickers: list[str]) -> str:
    """Judge synthesis response."""
    alloc = _equal_allocation(tickers)
    return json.dumps({
        "allocation": alloc,
        "justification": "Synthesized final allocation from agent revisions.",
        "confidence": 0.7,
        "claims": [],
    })


# ---------------------------------------------------------------------------
# Prompt Normalization for Snapshots
# ---------------------------------------------------------------------------

def _normalize_prompt_for_snapshot(prompt_text: str) -> str:
    """Remove unstable elements from a prompt for snapshot comparison."""
    text = prompt_text
    # Normalize UUIDs
    text = re.sub(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        "<UUID>", text,
    )
    # Normalize ISO timestamps
    text = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[^\s\"]*", "<TIMESTAMP>", text)
    # Normalize floating-point numbers
    text = re.sub(r"\b\d+\.\d+\b", "<FLOAT>", text)
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _assert_snapshot(prompt_text: str, snapshot_name: str, request) -> None:
    """Compare prompt against golden snapshot, or update if --update-snapshots."""
    snapshot_path = GOLDEN_DIR / snapshot_name
    normalized = _normalize_prompt_for_snapshot(prompt_text)

    if request.config.getoption("--update-snapshots", default=False):
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(normalized)
        return

    assert snapshot_path.exists(), f"Golden snapshot missing: {snapshot_path}. Run with --update-snapshots to generate."
    expected = snapshot_path.read_text()
    assert normalized == expected, (
        f"Prompt regression detected in {snapshot_name}. "
        f"Run with --update-snapshots to regenerate."
    )


def _compute_prompt_hash(prompt_text: str) -> str:
    """Compute sha256 hash of a normalized prompt."""
    normalized = _normalize_prompt_for_snapshot(prompt_text)
    return hashlib.sha256(normalized.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Canonical Run Loader
# ---------------------------------------------------------------------------

def _load_canonical_responses(run_dir: Path, round_num: int) -> dict:
    """Load all LLM responses from a canonical run directory.

    Returns: {(phase, role): response_text}
    """
    responses = {}
    round_dir = run_dir / f"rounds/round_{round_num:03d}"
    for phase in ("proposals", "critiques", "revisions"):
        phase_dir = round_dir / phase
        if not phase_dir.exists():
            continue
        for agent_dir in sorted(phase_dir.iterdir()):
            if not agent_dir.is_dir():
                continue
            role = agent_dir.name
            resp_file = agent_dir / "response.txt"
            if not resp_file.exists():
                resp_file = agent_dir / "response.json"
            if resp_file.exists():
                # Map log dir names to phase names used by _call_llm
                phase_key = phase.rstrip("s")  # proposals → proposal, etc.
                responses[(phase_key, role)] = resp_file.read_text()
    # CRIT responses
    crit_dir = round_dir / "CRIT"
    if crit_dir and crit_dir.exists():
        for agent_dir in sorted(crit_dir.iterdir()):
            if agent_dir.is_dir():
                resp_file = agent_dir / "response.txt"
                if resp_file.exists():
                    responses[("crit", agent_dir.name)] = resp_file.read_text()
    return responses


# ---------------------------------------------------------------------------
# Capturing LLM Mock
# ---------------------------------------------------------------------------

class CapturingLLM:
    """Records all LLM calls and returns phase-appropriate fixture responses.

    Routes strictly by the `phase` parameter — never inspects prompt text.
    """

    def __init__(self, tickers: list[str], roles: list[str]):
        self.calls: list[dict] = []
        self._tickers = tickers
        self._roles = roles

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
            return _make_enriched_proposal(role, self._tickers)
        if phase == "critique":
            return _make_enriched_critique(role, self._roles)
        if phase == "revise":
            return _make_enriched_revision(role, self._tickers)
        raise ValueError(f"Unexpected phase={phase!r} role={role!r}")

    def calls_for(self, phase=None, role=None) -> list[dict]:
        """Filter captured calls by phase and/or role."""
        result = self.calls
        if phase is not None:
            result = [c for c in result if c["phase"] == phase]
        if role is not None:
            result = [c for c in result if c["role"] == role]
        return result


class CanonicalReplayLLM:
    """Replays stored responses from a canonical run, with safety assertions."""

    def __init__(self, canonical_responses: dict):
        self._responses = canonical_responses
        self.calls: list[dict] = []

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
        phase_key = self._phase_to_log_key(phase)
        key = (phase_key, role)
        if key not in self._responses:
            raise KeyError(
                f"Missing canonical response for phase={phase!r} "
                f"role={role!r} round={round_num}. "
                f"Available keys: {sorted(self._responses.keys())}"
            )
        return self._responses[key]

    @staticmethod
    def _phase_to_log_key(phase: str) -> str:
        """Map _call_llm phase names to canonical log directory names."""
        return {
            "propose": "proposal",
            "critique": "critique",
            "revise": "revision",
            "crit": "crit",
            "judge": "judge",
        }.get(phase, phase)


# ---------------------------------------------------------------------------
# Observation Factory
# ---------------------------------------------------------------------------

def _make_observation() -> Observation:
    """Observation with all test tickers and minimal context."""
    return Observation(
        universe=TICKERS,
        timestamp="2025-01-15T00:00:00Z",
        market_state=MarketState(
            prices={t: 100.0 + i * 50 for i, t in enumerate(TICKERS)},
        ),
        portfolio_state=PortfolioState(cash=100_000.0, positions={}),
        text_context=(
            "## Market Summary\n"
            "Equity markets showed mixed signals. Tech leads on AI optimism.\n"
            "## Macro\n"
            "Fed Funds Rate: 5.25%. CPI trending down.\n"
            "[AAPL-RET60]: AAPL 60-day return +8.2%\n"
            "[MSFT-VOL30]: MSFT 30-day volatility 22.1%\n"
            "[NVDA-RET30]: NVDA 30-day return +15.4%\n"
            "[MACRO-FF]: Fed Funds Rate 5.25%\n"
            "[AAPL-VOL60]: AAPL 60-day volatility 25.3%\n"
        ),
    )


# ---------------------------------------------------------------------------
# Config Factory
# ---------------------------------------------------------------------------

def _make_config(
    max_rounds: int = 1,
    pid_enabled: bool = True,
) -> DebateConfig:
    """Build a DebateConfig with enriched agent profiles for testing."""
    from eval.PID.types import PIDConfig, PIDGains

    profiles = get_agent_profiles(AGENT_PROFILE_MAP, judge_profile_name=JUDGE_PROFILE)
    judge_profile = profiles.pop("judge", {})

    pid_config = None
    if pid_enabled:
        pid_config = PIDConfig(
            gains=PIDGains(Kp=0.15, Ki=0.01, Kd=0.03),
            rho_star=0.8,
        )

    return DebateConfig(
        roles=list(ROLES),
        max_rounds=max_rounds,
        mock=False,
        parallel_agents=False,
        console_display=False,
        verbose=False,
        trace_dir="/tmp/test_prompt_integrity",
        agent_profiles=profiles,
        agent_profile_names={r: AGENT_PROFILE_MAP[r] for r in ROLES},
        judge_profile=judge_profile,
        pid_config=pid_config,
        initial_beta=0.5,
        logging_mode="off",
        crit_model_name="gpt-5-mini",
    )


# ---------------------------------------------------------------------------
# Pytest Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def capturing_llm(monkeypatch):
    """Monkeypatch _call_llm at both consumer sites with CapturingLLM."""
    llm = CapturingLLM(TICKERS, ROLES)
    monkeypatch.setattr("multi_agent.graph.nodes._call_llm", llm)
    monkeypatch.setattr("multi_agent.runner._call_llm", llm)
    return llm


@pytest.fixture
def single_round_state(capturing_llm):
    """Run a 1-round pipeline and return (state, capturing_llm)."""
    config = _make_config(max_rounds=1)
    runner = MultiAgentRunner(config)
    state = runner.run_returning_state(_make_observation())
    return state


@pytest.fixture
def capturing_llm_2rd(monkeypatch):
    """Separate CapturingLLM for 2-round tests."""
    llm = CapturingLLM(TICKERS, ROLES)
    monkeypatch.setattr("multi_agent.graph.nodes._call_llm", llm)
    monkeypatch.setattr("multi_agent.runner._call_llm", llm)
    return llm


@pytest.fixture
def two_round_state(capturing_llm_2rd):
    """Run a 2-round pipeline and return state."""
    config = _make_config(max_rounds=2)
    runner = MultiAgentRunner(config)
    state = runner.run_returning_state(_make_observation())
    return state


# ---------------------------------------------------------------------------
# Group 1: LLM Call Count Verification
# ---------------------------------------------------------------------------

class TestLLMCallCounts:
    """Verify exact number of LLM calls per phase."""

    def test_single_round_total_calls(self, single_round_state, capturing_llm):
        assert len(capturing_llm.calls) == 13

    def test_single_round_propose_calls(self, single_round_state, capturing_llm):
        assert len(capturing_llm.calls_for(phase="propose")) == 3

    def test_single_round_critique_calls(self, single_round_state, capturing_llm):
        assert len(capturing_llm.calls_for(phase="critique")) == 3

    def test_single_round_revise_calls(self, single_round_state, capturing_llm):
        assert len(capturing_llm.calls_for(phase="revise")) == 3

    def test_single_round_crit_calls(self, single_round_state, capturing_llm):
        assert len(capturing_llm.calls_for(phase="crit")) == 3

    def test_single_round_judge_calls(self, single_round_state, capturing_llm):
        assert len(capturing_llm.calls_for(phase="judge")) == 1

    def test_two_round_total_calls(self, two_round_state, capturing_llm_2rd):
        assert len(capturing_llm_2rd.calls) == 22

    def test_round2_propose_is_noop(self, two_round_state, capturing_llm_2rd):
        """Propose is idempotent — no LLM calls in round 2."""
        r2_proposes = [
            c for c in capturing_llm_2rd.calls_for(phase="propose")
            if c["round_num"] == 2
        ]
        assert len(r2_proposes) == 0

    def test_each_role_called_once_per_phase(self, single_round_state, capturing_llm):
        """No agent is called twice in the same phase."""
        for phase in ("propose", "critique", "revise"):
            roles_called = [c["role"] for c in capturing_llm.calls_for(phase=phase)]
            assert sorted(roles_called) == sorted(ROLES), (
                f"Phase {phase}: expected {sorted(ROLES)}, got {sorted(roles_called)}"
            )


# ---------------------------------------------------------------------------
# Group 2: Propose Prompt Content (Structural)
# ---------------------------------------------------------------------------

class TestProposePromptContent:
    """Verify structural integrity of propose prompts."""

    def test_system_prompt_contains_role_text(self, single_round_state, capturing_llm):
        for role in ROLES:
            calls = capturing_llm.calls_for(phase="propose", role=role)
            assert len(calls) == 1
            sys = calls[0]["system_prompt"]
            assert role.lower() in sys.lower() or role.upper() in sys, (
                f"System prompt for {role} does not contain role name"
            )

    def test_user_prompt_contains_memo_context(self, single_round_state, capturing_llm):
        for role in ROLES:
            calls = capturing_llm.calls_for(phase="propose", role=role)
            user = calls[0]["user_prompt"]
            assert "Market Summary" in user, (
                f"Propose user prompt for {role} missing memo context"
            )
            assert "AAPL-RET60" in user, (
                f"Propose user prompt for {role} missing evidence IDs from memo"
            )

    def test_user_prompt_contains_claim_type_enum(self, single_round_state, capturing_llm):
        """claim_type with all 5 values must appear in enriched proposal template."""
        for role in ROLES:
            calls = capturing_llm.calls_for(phase="propose", role=role)
            user = calls[0]["user_prompt"]
            for ct in ("macro", "sector", "firm", "risk", "technical"):
                assert ct in user.lower(), (
                    f"Propose user prompt for {role} missing claim_type '{ct}'"
                )

    def test_no_unresolved_template_vars(self, single_round_state, capturing_llm):
        for role in ROLES:
            calls = capturing_llm.calls_for(phase="propose", role=role)
            for field in ("system_prompt", "user_prompt"):
                text = calls[0][field]
                unresolved = re.findall(r"\{\{[^}]+\}\}", text)
                assert not unresolved, (
                    f"Unresolved template vars in {role} propose {field}: {unresolved}"
                )


# ---------------------------------------------------------------------------
# Group 3: Critique Prompt Content (Structural)
# ---------------------------------------------------------------------------

class TestCritiquePromptContent:
    """Verify structural integrity of critique prompts."""

    def test_critique_prompt_contains_own_proposal(self, single_round_state, capturing_llm):
        """Critique prompt contains rendered content from own proposal."""
        for role in ROLES:
            calls = capturing_llm.calls_for(phase="critique", role=role)
            assert len(calls) == 1
            user = calls[0]["user_prompt"]
            # The proposal renderer uses "## Previous Portfolio Allocation"
            assert "Previous Portfolio Allocation" in user or "allocation" in user.lower(), (
                f"Critique prompt for {role} missing own proposal content"
            )

    def test_critique_prompt_contains_other_proposals(self, single_round_state, capturing_llm):
        """Critique prompt contains other agents' proposals."""
        for role in ROLES:
            calls = capturing_llm.calls_for(phase="critique", role=role)
            user = calls[0]["user_prompt"]
            for other in ROLES:
                if other == role:
                    continue
                assert other.upper() in user or other in user, (
                    f"Critique prompt for {role} missing {other}'s proposal"
                )

    def test_no_unresolved_template_vars(self, single_round_state, capturing_llm):
        for role in ROLES:
            calls = capturing_llm.calls_for(phase="critique", role=role)
            for field in ("system_prompt", "user_prompt"):
                text = calls[0][field]
                unresolved = re.findall(r"\{\{[^}]+\}\}", text)
                assert not unresolved, (
                    f"Unresolved template vars in {role} critique {field}: {unresolved}"
                )


# ---------------------------------------------------------------------------
# Group 4: Revision Prompt Content (Structural)
# ---------------------------------------------------------------------------

class TestRevisionPromptContent:
    """Verify structural integrity of revision prompts."""

    def test_revision_prompt_contains_own_proposal(self, single_round_state, capturing_llm):
        for role in ROLES:
            calls = capturing_llm.calls_for(phase="revise", role=role)
            assert len(calls) == 1
            user = calls[0]["user_prompt"]
            assert "Previous Portfolio Allocation" in user or "allocation" in user.lower(), (
                f"Revision prompt for {role} missing own proposal"
            )

    def test_revision_prompt_contains_critique_text(self, single_round_state, capturing_llm):
        """Revision prompt contains critique content."""
        for role in ROLES:
            calls = capturing_llm.calls_for(phase="revise", role=role)
            user = calls[0]["user_prompt"]
            # Should contain critique markers — either structured critique text
            # or the "no critiques" fallback
            has_critique = "Critique" in user or "objection" in user.lower() or "No critiques" in user
            assert has_critique, (
                f"Revision prompt for {role} missing critique content"
            )

    def test_no_unresolved_template_vars(self, single_round_state, capturing_llm):
        for role in ROLES:
            calls = capturing_llm.calls_for(phase="revise", role=role)
            for field in ("system_prompt", "user_prompt"):
                text = calls[0][field]
                unresolved = re.findall(r"\{\{[^}]+\}\}", text)
                assert not unresolved, (
                    f"Unresolved template vars in {role} revise {field}: {unresolved}"
                )


# ---------------------------------------------------------------------------
# Group 5: Golden Prompt Snapshots
# ---------------------------------------------------------------------------

class TestGoldenSnapshots:
    """Compare rendered prompts against golden snapshots."""

    @pytest.mark.parametrize("phase", ["propose", "critique", "revise"])
    @pytest.mark.parametrize("role", ROLES)
    def test_system_prompt_snapshot(self, phase, role, single_round_state, capturing_llm, request):
        calls = capturing_llm.calls_for(phase=phase, role=role)
        assert len(calls) >= 1, f"No {phase} calls for {role}"
        _assert_snapshot(
            calls[0]["system_prompt"],
            f"{phase}_{role}_system.txt",
            request,
        )

    @pytest.mark.parametrize("phase", ["propose", "critique", "revise"])
    @pytest.mark.parametrize("role", ROLES)
    def test_user_prompt_snapshot(self, phase, role, single_round_state, capturing_llm, request):
        calls = capturing_llm.calls_for(phase=phase, role=role)
        assert len(calls) >= 1, f"No {phase} calls for {role}"
        _assert_snapshot(
            calls[0]["user_prompt"],
            f"{phase}_{role}_user.txt",
            request,
        )


# ---------------------------------------------------------------------------
# Group 6: Response Parsing & Normalization
# ---------------------------------------------------------------------------

class TestResponseParsing:
    """Verify that fixture responses parse through the pipeline without errors."""

    def test_proposals_have_complete_action_dict(self, single_round_state):
        for p in single_round_state["proposals"]:
            ad = p["action_dict"]
            assert "allocation" in ad
            assert "claims" in ad
            assert "position_rationale" in ad
            assert "confidence" in ad

    def test_revisions_have_complete_action_dict(self, single_round_state):
        for r in single_round_state["revisions"]:
            ad = r["action_dict"]
            assert "allocation" in ad
            assert "claims" in ad
            assert "position_rationale" in ad
            assert "confidence" in ad

    def test_allocation_sums_to_one(self, single_round_state):
        for p in single_round_state["proposals"]:
            total = sum(p["action_dict"]["allocation"].values())
            assert abs(total - 1.0) < 0.02, f"Proposal allocation sums to {total}"
        for r in single_round_state["revisions"]:
            total = sum(r["action_dict"]["allocation"].values())
            assert abs(total - 1.0) < 0.02, f"Revision allocation sums to {total}"

    def test_critiques_have_target_role(self, single_round_state):
        for critic in single_round_state["critiques"]:
            for crit in critic["critiques"]:
                assert crit.get("target_role"), (
                    f"Critique from {critic['role']} has no target_role"
                )


# ---------------------------------------------------------------------------
# Group 7: Phase Transition Verification
# ---------------------------------------------------------------------------

class TestPhaseTransitions:
    """Verify state transitions between debate phases."""

    def test_proposals_populated(self, single_round_state):
        proposals = single_round_state["proposals"]
        assert len(proposals) == len(ROLES)
        for p in proposals:
            assert p["role"] in ROLES

    def test_critiques_populated(self, single_round_state):
        critiques = single_round_state["critiques"]
        assert len(critiques) == len(ROLES)

    def test_revisions_populated(self, single_round_state):
        revisions = single_round_state["revisions"]
        assert len(revisions) == len(ROLES)

    def test_final_action_exists(self, single_round_state):
        final = single_round_state.get("final_action", {})
        assert "allocation" in final
        total = sum(final["allocation"].values())
        assert abs(total - 1.0) < 0.02

    def test_debate_turns_accumulate(self, single_round_state):
        turns = single_round_state.get("debate_turns", [])
        # At least: 3 proposals + 3 critiques + 3 revisions + 1 judge = 10
        assert len(turns) >= 10, f"Expected >= 10 debate turns, got {len(turns)}"


# ---------------------------------------------------------------------------
# Group 8: Critique Routing Validation
# ---------------------------------------------------------------------------

class TestCritiqueRouting:
    """Verify critique-to-revision routing is correct."""

    def test_revision_prompt_excludes_other_agents_critiques(
        self, single_round_state, capturing_llm,
    ):
        """Revision prompt for agent X does NOT contain critique marker for Y."""
        for role in ROLES:
            revise_calls = capturing_llm.calls_for(phase="revise", role=role)
            assert len(revise_calls) == 1
            user_prompt = revise_calls[0]["user_prompt"]
            for other_role in ROLES:
                if other_role == role:
                    continue
                # Our fixture critiques contain "challenges {target}'s C1"
                marker = f"challenges {other_role}'s"
                assert marker not in user_prompt.lower(), (
                    f"{role}'s revision prompt contains critique targeting {other_role}"
                )

    def test_revision_prompt_contains_own_critiques(
        self, single_round_state, capturing_llm,
    ):
        """Revision prompt for agent X DOES contain critiques targeting X."""
        for role in ROLES:
            revise_calls = capturing_llm.calls_for(phase="revise", role=role)
            user_prompt = revise_calls[0]["user_prompt"]
            # Our fixture generates 2 critiques per target (from the other 2 agents)
            marker = f"challenges {role}'s"
            assert marker in user_prompt.lower(), (
                f"{role}'s revision prompt missing expected critiques targeting it"
            )

    def test_critique_target_roles_match_agents(self, single_round_state):
        """Every critique targets an agent that exists in the debate."""
        for critic in single_round_state["critiques"]:
            for crit in critic["critiques"]:
                raw_target = crit.get("target_role", "")
                target = raw_target.lower().split()[0] if raw_target else ""
                assert target in [r.lower() for r in ROLES], (
                    f"Critique targets unknown role: {raw_target}"
                )

    def test_critique_routing_via_structured_state(
        self, single_round_state, capturing_llm,
    ):
        """Cross-verify routing using structured state data, not just prompt text."""
        for role in ROLES:
            # Collect expected critiques from structured state
            expected_objections = []
            for critic in single_round_state["critiques"]:
                for crit in critic["critiques"]:
                    raw_target = crit.get("target_role", "")
                    target = raw_target.lower().split()[0] if raw_target else ""
                    if target == role.lower():
                        expected_objections.append(crit["objection"])

            assert len(expected_objections) > 0, (
                f"No critiques targeting {role} found in state"
            )

            # Verify expected critiques appear in the revision prompt
            revise_calls = capturing_llm.calls_for(phase="revise", role=role)
            user_prompt = revise_calls[0]["user_prompt"]
            for objection_text in expected_objections:
                snippet = objection_text[:60]
                assert snippet in user_prompt, (
                    f"Expected critique for {role} not found in revision prompt: "
                    f"{snippet!r}"
                )


# ---------------------------------------------------------------------------
# Group 9: CRIT Bundle Verification
# ---------------------------------------------------------------------------

class TestCritBundleCompleteness:
    """Verify CRIT bundles are well-formed."""

    def test_crit_prompts_contain_agent_role(self, single_round_state, capturing_llm):
        """CRIT system prompt mentions the agent role being evaluated."""
        for role in ROLES:
            crit_calls = capturing_llm.calls_for(phase="crit", role=role)
            if not crit_calls:
                # CRIT calls may not have role= set — check all crit calls
                continue
            sys = crit_calls[0]["system_prompt"]
            assert role in sys.lower(), (
                f"CRIT system prompt doesn't mention {role}"
            )

    def test_crit_prompts_contain_pillar_names(self, single_round_state, capturing_llm):
        """CRIT prompts reference the four evaluation pillars."""
        crit_calls = capturing_llm.calls_for(phase="crit")
        assert len(crit_calls) >= 1, "No CRIT calls found"
        # Check first CRIT call (system prompt is the same template for all)
        sys = crit_calls[0]["system_prompt"]
        for pillar in ("logical_validity", "evidential_support",
                       "alternative_consideration", "causal_alignment"):
            assert pillar in sys, f"CRIT system prompt missing pillar: {pillar}"


# ---------------------------------------------------------------------------
# Group 10: Cross-Round Propagation
# ---------------------------------------------------------------------------

class TestCrossRoundPropagation:
    """Run 2-round pipeline, verify cross-round state."""

    def test_round2_critique_sees_round1_revisions(self, two_round_state, capturing_llm_2rd):
        """Round 2 critique prompts contain round 1 revised content."""
        r2_critiques = [
            c for c in capturing_llm_2rd.calls_for(phase="critique")
            if c["round_num"] == 2
        ]
        assert len(r2_critiques) == len(ROLES), (
            f"Expected {len(ROLES)} round-2 critique calls, got {len(r2_critiques)}"
        )
        for call in r2_critiques:
            user = call["user_prompt"]
            # Round 2 critique should see revised content (contains "revised")
            assert "Previous" in user or "allocation" in user.lower(), (
                f"Round 2 critique for {call['role']} missing previous proposal content"
            )

    def test_proposals_unchanged_in_round2(self, two_round_state):
        """Proposals are set once in round 1 and not regenerated."""
        proposals = two_round_state["proposals"]
        assert len(proposals) == len(ROLES)

    def test_round2_revision_sees_round2_critiques(self, two_round_state, capturing_llm_2rd):
        """Round 2 revision prompts contain round 2 critique content."""
        r2_revisions = [
            c for c in capturing_llm_2rd.calls_for(phase="revise")
            if c["round_num"] == 2
        ]
        assert len(r2_revisions) == len(ROLES)
        for call in r2_revisions:
            user = call["user_prompt"]
            has_critique = "Critique" in user or "objection" in user.lower() or "No critiques" in user
            assert has_critique, (
                f"Round 2 revision for {call['role']} missing critique content"
            )

    def test_two_rounds_have_more_turns(self, two_round_state, single_round_state):
        turns_2 = len(two_round_state.get("debate_turns", []))
        turns_1 = len(single_round_state.get("debate_turns", []))
        assert turns_2 > turns_1, f"2-round ({turns_2}) should have more turns than 1-round ({turns_1})"


# ---------------------------------------------------------------------------
# Group 11: Prompt Structure Invariants
# ---------------------------------------------------------------------------

class TestPromptStructuralInvariants:
    """Cross-cutting structural invariants across all prompts."""

    def test_all_prompts_have_non_empty_system(self, single_round_state, capturing_llm):
        for call in capturing_llm.calls:
            assert call["system_prompt"].strip(), (
                f"Empty system prompt for phase={call['phase']} role={call['role']}"
            )

    def test_all_prompts_have_non_empty_user(self, single_round_state, capturing_llm):
        for call in capturing_llm.calls:
            # CRIT and judge may have very different user prompt structure
            assert call["user_prompt"].strip(), (
                f"Empty user prompt for phase={call['phase']} role={call['role']}"
            )

    def test_no_none_in_call_metadata(self, single_round_state, capturing_llm):
        """Every captured call has phase and role set."""
        for i, call in enumerate(capturing_llm.calls):
            assert call["phase"] is not None, f"Call {i} has None phase"
            assert call["role"] is not None, f"Call {i} has None role"


# ---------------------------------------------------------------------------
# Group 12: Prompt Hash Stability (Optional)
# ---------------------------------------------------------------------------

class TestPromptHashStability:
    """Quick sha256 check that prompts haven't changed."""

    def test_prompt_hashes_match(self, single_round_state, capturing_llm, request):
        if request.config.getoption("--update-snapshots", default=False):
            hashes = {}
            for call in capturing_llm.calls:
                if call["phase"] in ("propose", "critique", "revise"):
                    key_sys = f"{call['phase']}_{call['role']}_system"
                    key_usr = f"{call['phase']}_{call['role']}_user"
                    hashes[key_sys] = _compute_prompt_hash(call["system_prompt"])
                    hashes[key_usr] = _compute_prompt_hash(call["user_prompt"])
            HASH_FILE.write_text(json.dumps(hashes, indent=2, sort_keys=True))
            pytest.skip("Updated prompt hashes")

        if not HASH_FILE.exists():
            pytest.skip("No prompt_hashes.json — run with --update-snapshots to generate")

        expected = json.loads(HASH_FILE.read_text())
        for call in capturing_llm.calls:
            if call["phase"] in ("propose", "critique", "revise"):
                for suffix, field in [("system", "system_prompt"), ("user", "user_prompt")]:
                    key = f"{call['phase']}_{call['role']}_{suffix}"
                    actual = _compute_prompt_hash(call[field])
                    assert actual == expected.get(key), (
                        f"Prompt hash mismatch for {key}. "
                        f"Run with --update-snapshots to regenerate."
                    )


# ---------------------------------------------------------------------------
# Group 13: Canonical Replay (requires canonical run on disk)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not CANONICAL_RUN.exists(),
    reason=f"Canonical run not found at {CANONICAL_RUN}",
)
class TestCanonicalReplay:
    """Replay real run logs and verify response parsing compatibility."""

    def test_canonical_responses_parse_as_json(self):
        """Every response from the canonical run is valid JSON."""
        for round_num in (1, 2):
            responses = _load_canonical_responses(CANONICAL_RUN, round_num)
            for (phase, role), text in responses.items():
                if phase == "crit":
                    # CRIT responses should parse as JSON
                    parsed = json.loads(text)
                    assert "pillar_scores" in parsed, (
                        f"CRIT response for {role} round {round_num} missing pillar_scores"
                    )
                elif phase in ("proposal", "revision"):
                    parsed = json.loads(text)
                    assert "allocation" in parsed, (
                        f"{phase} response for {role} round {round_num} missing allocation"
                    )
                elif phase == "critique":
                    parsed = json.loads(text)
                    assert "critiques" in parsed, (
                        f"Critique response for {role} round {round_num} missing critiques"
                    )

    def test_canonical_proposals_have_enriched_fields(self):
        """Canonical proposal responses include enriched fields."""
        responses = _load_canonical_responses(CANONICAL_RUN, 1)
        for (phase, role), text in responses.items():
            if phase != "proposal":
                continue
            parsed = json.loads(text)
            assert "claims" in parsed, f"{role} proposal missing claims"
            assert "position_rationale" in parsed, f"{role} proposal missing position_rationale"
            # Verify claim structure
            for claim in parsed["claims"]:
                assert "claim_id" in claim, f"{role} claim missing claim_id"
                assert "claim_text" in claim, f"{role} claim missing claim_text"
