"""Unit tests for AllocationConstraints and memo-mode config validation."""

import pytest
from pydantic import ValidationError

from models.config import AllocationConstraints, AgentConfig, SimulationConfig


# ── helpers ──────────────────────────────────────────────────────────────────

def _base_agent(**overrides) -> dict:
    """Minimal AgentConfig kwargs."""
    d = {
        "agent_system": "multi_agent_debate",
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "temperature": 0.3,
    }
    d.update(overrides)
    return d


def _sim_config(**overrides) -> SimulationConfig:
    """Build a SimulationConfig with sensible defaults."""
    d = {
        "dataset_path": "data/cases",
        "tickers": ["AAPL", "MSFT", "GOOG"],
        "invest_quarter": "2025Q1",
        "debate_setup": _base_agent(),
    }
    d.update(overrides)
    return SimulationConfig(**d)


# ── AllocationConstraints ────────────────────────────────────────────────────


class TestAllocationConstraints:
    def test_defaults(self):
        ac = AllocationConstraints()
        assert ac.max_weight == 0.40
        assert ac.min_holdings == 3
        assert ac.fully_invested is True
        assert ac.max_tickers == 10

    def test_max_weight_zero_raises(self):
        with pytest.raises(ValidationError):
            AllocationConstraints(max_weight=0.0)

    def test_max_weight_one_passes(self):
        ac = AllocationConstraints(max_weight=1.0)
        assert ac.max_weight == 1.0

    def test_max_weight_over_one_raises(self):
        with pytest.raises(ValidationError):
            AllocationConstraints(max_weight=1.1)

    def test_min_holdings_zero_raises(self):
        with pytest.raises(ValidationError):
            AllocationConstraints(min_holdings=0)

    def test_min_holdings_one_passes(self):
        ac = AllocationConstraints(min_holdings=1)
        assert ac.min_holdings == 1

    def test_roundtrip(self):
        ac = AllocationConstraints(max_weight=0.25, min_holdings=5)
        restored = AllocationConstraints(**ac.model_dump())
        assert restored == ac


# ── SimulationConfig._validate_memo_mode ─────────────────────────────────────


class TestValidateMemoMode:
    def test_requires_invest_quarter_on_validate_ready(self):
        """invest_quarter=None is allowed at construction but fails validate_ready()."""
        cfg = _sim_config(
            invest_quarter=None,
            dataset_path="data-pipeline/final_snapshots",
        )
        with pytest.raises(ValueError, match="invest_quarter is required"):
            cfg.validate_ready()

    def test_ticker_count_check(self):
        tickers = [f"T{i}" for i in range(11)]  # 11 > default max_tickers=10
        with pytest.raises(ValueError, match="Too many tickers"):
            _sim_config(
                invest_quarter="2025Q1",
                dataset_path="data-pipeline/final_snapshots",
                tickers=tickers,
            )

    def test_ticker_count_at_limit(self):
        tickers = [f"T{i}" for i in range(10)]
        cfg = _sim_config(
            invest_quarter="2025Q1",
            dataset_path="data-pipeline/final_snapshots",
            tickers=tickers,
        )
        assert len(cfg.tickers) == 10

    def test_custom_allocation_constraints(self):
        cfg = _sim_config(
            invest_quarter="2025Q1",
            dataset_path="data-pipeline/final_snapshots",
            allocation_constraints={"max_weight": 0.25, "min_holdings": 5},
        )
        assert cfg.allocation_constraints.max_weight == 0.25
        assert cfg.allocation_constraints.min_holdings == 5
