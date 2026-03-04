"""End-to-end smoke tests: sector constraints survive the full simulation pipeline.

Tests the complete path with sector configuration enabled:

    YAML config on disk (with sectors / sector_limits / agent_sector_permissions)
        → SimulationConfig.from_yaml()
        → AsyncSimulationRunner.run()
            → load_case_templates() from disk
            → DebateAgentSystem creation (sector_config threaded through)
            → agent.invoke() → debate graph → sector enforcement in nodes
            → broker.execute_decision()
            → EpisodeLog → disk output

Verifies:
  - Sector limits are enforced on the final (judge) allocation
  - Config threading: sector config survives YAML → output
  - Pipeline completes without errors
  - Backward compat: limits work without permissions

All tests use mock=True — no real API calls.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from models.config import SimulationConfig
from simulation.runner import AsyncSimulationRunner


# ---------------------------------------------------------------------------
# Mock CRIT response (same pattern as test_run_simulation_pid.py)
# ---------------------------------------------------------------------------

_CRIT_ENTRY = {
    "pillar_scores": {
        "internal_consistency": 0.8,
        "evidence_support": 0.7,
        "trace_alignment": 0.9,
        "causal_integrity": 0.6,
    },
    "diagnostics": {
        "contradictions_detected": False,
        "unsupported_claims_detected": False,
        "conclusion_drift_detected": False,
        "causal_overreach_detected": False,
    },
    "explanations": {
        "internal_consistency": "ok",
        "evidence_support": "ok",
        "trace_alignment": "ok",
        "causal_integrity": "ok",
    },
}

MOCK_CRIT_RESPONSE = json.dumps(_CRIT_ENTRY)


def _mock_call_llm(config: dict, system_prompt: str, user_prompt: str) -> str:
    """Drop-in replacement for ``multi_agent.runner._call_llm``.

    Debate nodes short-circuit via mock helpers in mock mode.
    Only the CRIT scorer calls _call_llm.
    """
    return MOCK_CRIT_RESPONSE


# ---------------------------------------------------------------------------
# Snapshot JSON (4 tickers across 3 sectors)
# ---------------------------------------------------------------------------

SNAPSHOT_2024_Q4 = {
    "as_of_date": "2024-12-31",
    "ticker_data": {
        "AAPL": {"asset_features": {"close": 200.0}},
        "NVDA": {"asset_features": {"close": 150.0}},
        "XOM": {"asset_features": {"close": 100.0}},
        "JPM": {"asset_features": {"close": 180.0}},
    },
}

# ---------------------------------------------------------------------------
# Sector definitions (shared by YAML templates)
# ---------------------------------------------------------------------------
# tech: AAPL, NVDA (2 tickers)  → mock equal-weight = 0.50
# energy: XOM (1 ticker)         → mock equal-weight = 0.25
# financials: JPM (1 ticker)     → mock equal-weight = 0.25

SECTOR_MAP = {
    "tech": ["AAPL", "NVDA"],
    "energy": ["XOM"],
    "financials": ["JPM"],
}

SECTOR_LIMITS = {
    "tech": {"min": 0.30, "max": 0.60},
    "energy": {"min": 0.10, "max": 0.35},
    "financials": {"min": 0.10, "max": 0.35},
}


# ---------------------------------------------------------------------------
# YAML config templates
# ---------------------------------------------------------------------------

YAML_SECTORS_FULL = """\
dataset_path: "{dataset_path}"
tickers: [AAPL, NVDA, XOM, JPM]
invest_quarter: "2025Q1"
memo_format: json
num_episodes: 1
broker:
  initial_cash: 100000.0
allocation_constraints:
  max_weight: 0.50
  min_holdings: 2
sectors:
  tech: [AAPL, NVDA]
  energy: [XOM]
  financials: [JPM]
sector_limits:
  tech:
    min: 0.30
    max: 0.60
  energy:
    min: 0.10
    max: 0.35
  financials:
    min: 0.10
    max: 0.35
agent_sector_permissions:
  macro: [energy, financials]
  value: [tech, financials]
  risk: ["*"]
  technical: [tech, energy]
agent:
  agent_system: multi_agent_debate
  llm_provider: openai
  llm_model: gpt-4o-mini
  temperature: 0.3
  system_prompt_override: "mock"
"""

YAML_MAX_SECTOR_WEIGHT = """\
dataset_path: "{dataset_path}"
tickers: [AAPL, NVDA, XOM, JPM]
invest_quarter: "2025Q1"
memo_format: json
num_episodes: 1
broker:
  initial_cash: 100000.0
allocation_constraints:
  max_weight: 0.50
  min_holdings: 2
sectors:
  tech: [AAPL, NVDA]
  energy: [XOM]
  financials: [JPM]
max_sector_weight: 0.40
agent:
  agent_system: multi_agent_debate
  llm_provider: openai
  llm_model: gpt-4o-mini
  temperature: 0.3
  system_prompt_override: "mock"
"""

YAML_MAX_SECTOR_WEIGHT_WITH_LIMITS = """\
dataset_path: "{dataset_path}"
tickers: [AAPL, NVDA, XOM, JPM]
invest_quarter: "2025Q1"
memo_format: json
num_episodes: 1
broker:
  initial_cash: 100000.0
allocation_constraints:
  max_weight: 0.50
  min_holdings: 2
sectors:
  tech: [AAPL, NVDA]
  energy: [XOM]
  financials: [JPM]
sector_limits:
  tech:
    min: 0.20
    max: 0.60
  energy:
    min: 0.10
    max: 0.35
  financials:
    min: 0.10
    max: 0.35
max_sector_weight: 0.40
agent:
  agent_system: multi_agent_debate
  llm_provider: openai
  llm_model: gpt-4o-mini
  temperature: 0.3
  system_prompt_override: "mock"
"""

YAML_SECTORS_LIMITS_ONLY = """\
dataset_path: "{dataset_path}"
tickers: [AAPL, NVDA, XOM, JPM]
invest_quarter: "2025Q1"
memo_format: json
num_episodes: 1
broker:
  initial_cash: 100000.0
allocation_constraints:
  max_weight: 0.50
  min_holdings: 2
sectors:
  tech: [AAPL, NVDA]
  energy: [XOM]
  financials: [JPM]
sector_limits:
  tech:
    min: 0.30
    max: 0.60
  energy:
    min: 0.10
    max: 0.35
  financials:
    min: 0.10
    max: 0.35
agent:
  agent_system: multi_agent_debate
  llm_provider: openai
  llm_model: gpt-4o-mini
  temperature: 0.3
  system_prompt_override: "mock"
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simulation_dir():
    """Create a tmpdir with dataset + config dirs on disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Snapshot JSON directory
        json_dir = root / "dataset" / "json_data"
        json_dir.mkdir(parents=True)
        snapshot_file = json_dir / "snapshot_2024_Q4.json"
        snapshot_file.write_text(json.dumps(SNAPSHOT_2024_Q4), encoding="utf-8")

        # Output directory
        (root / "results").mkdir()

        yield root


@pytest.fixture
def sector_full_config_path(simulation_dir):
    """Write full sector config (limits + permissions) YAML and return path."""
    dataset_path = str(simulation_dir / "dataset")
    yaml_content = YAML_SECTORS_FULL.format(dataset_path=dataset_path)
    config_path = simulation_dir / "config_sectors_full.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")
    return str(config_path)


@pytest.fixture
def sector_limits_only_config_path(simulation_dir):
    """Write sector config with limits but NO permissions, return path."""
    dataset_path = str(simulation_dir / "dataset")
    yaml_content = YAML_SECTORS_LIMITS_ONLY.format(dataset_path=dataset_path)
    config_path = simulation_dir / "config_sectors_limits.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")
    return str(config_path)


@pytest.fixture
def max_sector_weight_config_path(simulation_dir):
    """Write config with max_sector_weight only (no sector_limits)."""
    dataset_path = str(simulation_dir / "dataset")
    yaml_content = YAML_MAX_SECTOR_WEIGHT.format(dataset_path=dataset_path)
    config_path = simulation_dir / "config_max_sw.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")
    return str(config_path)


@pytest.fixture
def max_sector_weight_with_limits_config_path(simulation_dir):
    """Write config with both max_sector_weight AND sector_limits."""
    dataset_path = str(simulation_dir / "dataset")
    yaml_content = YAML_MAX_SECTOR_WEIGHT_WITH_LIMITS.format(dataset_path=dataset_path)
    config_path = simulation_dir / "config_max_sw_limits.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")
    return str(config_path)


def _run_simulation(config_path: str, output_dir: str) -> Path:
    """Run the full simulation pipeline and return the run output directory."""
    config = SimulationConfig.from_yaml(config_path)
    runner = AsyncSimulationRunner(
        config,
        config_yaml_path=config_path,
        output_dir=output_dir,
    )
    asyncio.run(runner.run())
    results_root = Path(output_dir)
    run_dirs = [d for d in results_root.iterdir() if d.is_dir()]
    assert len(run_dirs) == 1, f"Expected 1 run dir, found {len(run_dirs)}"
    return run_dirs[0]


def _get_final_allocation(run_dir: Path) -> dict[str, float]:
    """Extract the final judge allocation from episode_log.json."""
    ep_log_path = run_dir / "episodes" / "ep_000" / "episode_log.json"
    ep_log = json.loads(ep_log_path.read_text())
    agent_output = ep_log["decision_point_logs"][0]["agent_output"]
    return agent_output["debate_action"]["allocation"]


def _compute_sector_totals(
    allocation: dict[str, float],
    sector_map: dict[str, list[str]],
) -> dict[str, float]:
    """Sum allocation weights by sector."""
    totals: dict[str, float] = {}
    for sector, tickers in sector_map.items():
        totals[sector] = sum(allocation.get(t, 0.0) for t in tickers)
    return totals


# ---------------------------------------------------------------------------
# Tests: Sector limits enforced on final allocation
# ---------------------------------------------------------------------------

class TestSectorLimitsEnforced:
    """Verify the judge's final allocation respects sector bounds."""

    def test_final_allocation_within_sector_bounds(
        self, simulation_dir, sector_full_config_path, monkeypatch
    ):
        """Each sector's total weight is within [min, max]."""
        monkeypatch.setattr("multi_agent.runner._call_llm", _mock_call_llm)
        output_dir = str(simulation_dir / "results")
        run_dir = _run_simulation(sector_full_config_path, output_dir)

        alloc = _get_final_allocation(run_dir)
        totals = _compute_sector_totals(alloc, SECTOR_MAP)

        tol = 0.01  # 1% tolerance for floating point
        for sector, total in totals.items():
            lo = SECTOR_LIMITS[sector]["min"]
            hi = SECTOR_LIMITS[sector]["max"]
            assert total >= lo - tol, (
                f"Sector {sector} total {total:.4f} below min {lo}"
            )
            assert total <= hi + tol, (
                f"Sector {sector} total {total:.4f} above max {hi}"
            )

    def test_allocation_sums_to_one(
        self, simulation_dir, sector_full_config_path, monkeypatch
    ):
        """Final allocation weights sum to 1.0."""
        monkeypatch.setattr("multi_agent.runner._call_llm", _mock_call_llm)
        output_dir = str(simulation_dir / "results")
        run_dir = _run_simulation(sector_full_config_path, output_dir)

        alloc = _get_final_allocation(run_dir)
        total = sum(alloc.values())
        assert abs(total - 1.0) < 0.01, f"Allocation sums to {total}, expected 1.0"

    def test_all_tickers_present(
        self, simulation_dir, sector_full_config_path, monkeypatch
    ):
        """All 4 tickers appear in the final allocation with positive weight."""
        monkeypatch.setattr("multi_agent.runner._call_llm", _mock_call_llm)
        output_dir = str(simulation_dir / "results")
        run_dir = _run_simulation(sector_full_config_path, output_dir)

        alloc = _get_final_allocation(run_dir)
        for ticker in ["AAPL", "NVDA", "XOM", "JPM"]:
            assert ticker in alloc, f"Ticker {ticker} missing from allocation"
            assert alloc[ticker] > 0, f"Ticker {ticker} has zero weight"


# ---------------------------------------------------------------------------
# Tests: Config threading survives full pipeline
# ---------------------------------------------------------------------------

class TestSectorConfigSurvivesPipeline:
    """Verify sector config is threaded from YAML all the way through."""

    def test_simulation_completes_without_errors(
        self, simulation_dir, sector_full_config_path, monkeypatch
    ):
        """Full pipeline completes successfully with sectors enabled."""
        monkeypatch.setattr("multi_agent.runner._call_llm", _mock_call_llm)
        output_dir = str(simulation_dir / "results")
        run_dir = _run_simulation(sector_full_config_path, output_dir)

        sim_log_path = run_dir / "simulation_log.json"
        assert sim_log_path.exists()

        sim_log = json.loads(sim_log_path.read_text())
        assert len(sim_log["errors"]) == 0
        assert len(sim_log["episode_logs"]) == 1

    def test_config_yaml_contains_sector_config(
        self, simulation_dir, sector_full_config_path, monkeypatch
    ):
        """config.yaml in output contains sector definitions."""
        monkeypatch.setattr("multi_agent.runner._call_llm", _mock_call_llm)
        output_dir = str(simulation_dir / "results")
        run_dir = _run_simulation(sector_full_config_path, output_dir)

        config_copy = run_dir / "config.yaml"
        assert config_copy.exists()
        content = config_copy.read_text()
        assert "sectors:" in content
        assert "sector_limits:" in content
        assert "agent_sector_permissions:" in content

    def test_summary_json_valid(
        self, simulation_dir, sector_full_config_path, monkeypatch
    ):
        """summary.json has valid episode summary with financial metrics."""
        monkeypatch.setattr("multi_agent.runner._call_llm", _mock_call_llm)
        output_dir = str(simulation_dir / "results")
        run_dir = _run_simulation(sector_full_config_path, output_dir)

        summary_path = run_dir / "summary.json"
        assert summary_path.exists()

        summary = json.loads(summary_path.read_text())
        assert summary["num_episodes"] == 1
        assert len(summary["episode_summaries"]) == 1

        ep_summary = summary["episode_summaries"][0]
        assert "initial_cash" in ep_summary
        assert "final_cash" in ep_summary
        assert ep_summary["initial_cash"] == 100000.0

    def test_decision_was_executed(
        self, simulation_dir, sector_full_config_path, monkeypatch
    ):
        """The debate produced a decision and the broker executed it."""
        monkeypatch.setattr("multi_agent.runner._call_llm", _mock_call_llm)
        output_dir = str(simulation_dir / "results")
        run_dir = _run_simulation(sector_full_config_path, output_dir)

        ep_log_path = run_dir / "episodes" / "ep_000" / "episode_log.json"
        ep_log = json.loads(ep_log_path.read_text())
        dp = ep_log["decision_point_logs"][0]

        decision = dp["extracted_decision"]
        assert "orders" in decision
        assert dp["portfolio_before"]["cash"] == 100000.0
        assert "portfolio_after" in dp


# ---------------------------------------------------------------------------
# Tests: Sector limits without permissions (backward compat)
# ---------------------------------------------------------------------------

class TestSectorLimitsWithoutPermissions:
    """Verify limits work when agent_sector_permissions is not specified."""

    def test_limits_enforced_without_permissions(
        self, simulation_dir, sector_limits_only_config_path, monkeypatch
    ):
        """Sector limits are enforced even without permissions config."""
        monkeypatch.setattr("multi_agent.runner._call_llm", _mock_call_llm)
        output_dir = str(simulation_dir / "results")
        run_dir = _run_simulation(sector_limits_only_config_path, output_dir)

        alloc = _get_final_allocation(run_dir)
        totals = _compute_sector_totals(alloc, SECTOR_MAP)

        tol = 0.01
        for sector, total in totals.items():
            lo = SECTOR_LIMITS[sector]["min"]
            hi = SECTOR_LIMITS[sector]["max"]
            assert total >= lo - tol, (
                f"Sector {sector} total {total:.4f} below min {lo}"
            )
            assert total <= hi + tol, (
                f"Sector {sector} total {total:.4f} above max {hi}"
            )

    def test_pipeline_completes_without_permissions(
        self, simulation_dir, sector_limits_only_config_path, monkeypatch
    ):
        """Full pipeline completes successfully with limits but no permissions."""
        monkeypatch.setattr("multi_agent.runner._call_llm", _mock_call_llm)
        output_dir = str(simulation_dir / "results")
        run_dir = _run_simulation(sector_limits_only_config_path, output_dir)

        sim_log_path = run_dir / "simulation_log.json"
        sim_log = json.loads(sim_log_path.read_text())
        assert len(sim_log["errors"]) == 0
        assert len(sim_log["episode_logs"]) == 1


# ---------------------------------------------------------------------------
# Tests: max_sector_weight enforcement
# ---------------------------------------------------------------------------

MAX_SECTOR_WEIGHT = 0.40


class TestMaxSectorWeight:
    """Verify max_sector_weight caps every sector's total weight."""

    def test_max_sector_weight_enforced(
        self, simulation_dir, max_sector_weight_config_path, monkeypatch
    ):
        """No sector exceeds max_sector_weight in the final allocation.

        Mock equal-weight produces 50% tech (2/4 tickers), which must be
        capped to 40%.
        """
        monkeypatch.setattr("multi_agent.runner._call_llm", _mock_call_llm)
        output_dir = str(simulation_dir / "results")
        run_dir = _run_simulation(max_sector_weight_config_path, output_dir)

        alloc = _get_final_allocation(run_dir)
        totals = _compute_sector_totals(alloc, SECTOR_MAP)

        tol = 0.01
        for sector, total in totals.items():
            assert total <= MAX_SECTOR_WEIGHT + tol, (
                f"Sector {sector} total {total:.4f} exceeds "
                f"max_sector_weight {MAX_SECTOR_WEIGHT}"
            )

    def test_allocation_sums_to_one_with_max_sector_weight(
        self, simulation_dir, max_sector_weight_config_path, monkeypatch
    ):
        """Allocation still sums to 1.0 after max_sector_weight enforcement."""
        monkeypatch.setattr("multi_agent.runner._call_llm", _mock_call_llm)
        output_dir = str(simulation_dir / "results")
        run_dir = _run_simulation(max_sector_weight_config_path, output_dir)

        alloc = _get_final_allocation(run_dir)
        total = sum(alloc.values())
        assert abs(total - 1.0) < 0.01, f"Allocation sums to {total}, expected 1.0"

    def test_pipeline_completes_with_max_sector_weight(
        self, simulation_dir, max_sector_weight_config_path, monkeypatch
    ):
        """Full pipeline completes without errors using max_sector_weight."""
        monkeypatch.setattr("multi_agent.runner._call_llm", _mock_call_llm)
        output_dir = str(simulation_dir / "results")
        run_dir = _run_simulation(max_sector_weight_config_path, output_dir)

        sim_log_path = run_dir / "simulation_log.json"
        sim_log = json.loads(sim_log_path.read_text())
        assert len(sim_log["errors"]) == 0

    def test_max_sector_weight_combined_with_sector_limits(
        self, simulation_dir, max_sector_weight_with_limits_config_path, monkeypatch
    ):
        """When both are set, the stricter bound applies.

        sector_limits tech max=0.60, but max_sector_weight=0.40 → effective
        tech max is 0.40.
        """
        monkeypatch.setattr("multi_agent.runner._call_llm", _mock_call_llm)
        output_dir = str(simulation_dir / "results")
        run_dir = _run_simulation(max_sector_weight_with_limits_config_path, output_dir)

        alloc = _get_final_allocation(run_dir)
        totals = _compute_sector_totals(alloc, SECTOR_MAP)

        tol = 0.01
        # max_sector_weight (0.40) is stricter than tech's sector_limits max (0.60)
        for sector, total in totals.items():
            assert total <= MAX_SECTOR_WEIGHT + tol, (
                f"Sector {sector} total {total:.4f} exceeds "
                f"max_sector_weight {MAX_SECTOR_WEIGHT}"
            )
