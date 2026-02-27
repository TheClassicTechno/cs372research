"""Tests for the quarters filtering feature.

Covers:
- SimulationConfig.quarters field (parsing from YAML, defaults)
- Quarter filtering logic in load_case_templates()
- Filter ordering (ticker -> quarter -> top_n_news)
- Edge cases (empty quarters list, no matches, partial matches)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from models.case import Case, CaseData, CaseDataItem, ClosePricePoint, StockData
from models.config import AgentConfig, BrokerConfig, SimulationConfig
from simulation.case_loader import load_case_templates


# =============================================================================
# HELPERS
# =============================================================================


def _minimal_stock_data(ticker: str = "NVDA") -> dict[str, StockData]:
    """Build a minimal stock_data dict for a single ticker."""
    return {
        ticker: StockData(
            ticker=ticker,
            current_price=100.0,
            daily_bars=[ClosePricePoint(timestamp="2025-01-01", close=100.0)],
        )
    }


def _write_case_file(directory: Path, rel_path: str, ticker: str = "NVDA", n_items: int = 0) -> None:
    """Write a minimal Case JSON file at directory/rel_path."""
    items = [
        CaseDataItem(kind="news", content=f"item {i}", impact_score=float(n_items - i))
        for i in range(n_items)
    ]
    case = Case(
        case_data=CaseData(items=items),
        stock_data=_minimal_stock_data(ticker),
    )
    path = directory / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(case.model_dump_json(), encoding="utf-8")


# =============================================================================
# SimulationConfig.quarters field
# =============================================================================


class TestSimulationConfigQuarters:
    """Tests for the quarters field on SimulationConfig."""

    def test_quarters_defaults_to_none(self):
        config = SimulationConfig(
            dataset_path="data/cases",
            agent=AgentConfig(agent_system="single_llm", llm_provider="openai", llm_model="gpt-4o"),
            tickers=["NVDA"],
        )
        assert config.quarters is None

    def test_quarters_accepts_list(self):
        config = SimulationConfig(
            dataset_path="data/cases",
            agent=AgentConfig(agent_system="single_llm", llm_provider="openai", llm_model="gpt-4o"),
            tickers=["NVDA"],
            quarters=["Q1", "Q3"],
        )
        assert config.quarters == ["Q1", "Q3"]

    def test_quarters_parsed_from_yaml(self, tmp_path):
        yaml_content = {
            "dataset_path": "data/cases",
            "tickers": ["NVDA"],
            "quarters": ["Q2", "Q4"],
            "agent": {"agent_system": "single_llm", "llm_provider": "openai", "llm_model": "gpt-4o"},
        }
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml.dump(yaml_content), encoding="utf-8")

        config = SimulationConfig.from_yaml(yaml_path)
        assert config.quarters == ["Q2", "Q4"]

    def test_quarters_omitted_from_yaml_is_none(self, tmp_path):
        yaml_content = {
            "dataset_path": "data/cases",
            "tickers": ["NVDA"],
            "agent": {"agent_system": "single_llm", "llm_provider": "openai", "llm_model": "gpt-4o"},
        }
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml.dump(yaml_content), encoding="utf-8")

        config = SimulationConfig.from_yaml(yaml_path)
        assert config.quarters is None


# =============================================================================
# Quarter filtering in load_case_templates
# =============================================================================


class TestQuarterFiltering:
    """Tests for the quarters parameter in load_case_templates."""

    @pytest.fixture(autouse=True)
    def setup_dataset(self, tmp_path):
        """Create a dataset with 4 NVDA quarters and 2 AAPL quarters."""
        self.dataset = tmp_path / "cases"
        self.dataset.mkdir()
        _write_case_file(self.dataset, "NVDA/2025_Q1.json", ticker="NVDA")
        _write_case_file(self.dataset, "NVDA/2025_Q2.json", ticker="NVDA")
        _write_case_file(self.dataset, "NVDA/2025_Q3.json", ticker="NVDA")
        _write_case_file(self.dataset, "NVDA/2025_Q4.json", ticker="NVDA")
        _write_case_file(self.dataset, "AAPL/2025_Q1.json", ticker="AAPL")
        _write_case_file(self.dataset, "AAPL/2025_Q3.json", ticker="AAPL")

    def test_quarters_none_loads_all(self):
        """When quarters is None, all cases are loaded."""
        cases = load_case_templates(str(self.dataset), quarters=None)
        assert len(cases) == 6

    def test_single_quarter_filter(self):
        """Filtering to Q1 returns only Q1 cases."""
        cases = load_case_templates(str(self.dataset), quarters=["Q1"])
        assert len(cases) == 2
        assert all("Q1" in c.case_id for c in cases)

    def test_multiple_quarters_filter(self):
        """Filtering to Q1 and Q3 returns both."""
        cases = load_case_templates(str(self.dataset), quarters=["Q1", "Q3"])
        assert len(cases) == 4
        for c in cases:
            assert "Q1" in c.case_id or "Q3" in c.case_id

    def test_quarter_filter_no_matches(self):
        """Filtering to a quarter with no cases returns empty list."""
        cases = load_case_templates(str(self.dataset), quarters=["Q5"])
        assert len(cases) == 0

    def test_empty_quarters_list_returns_nothing(self):
        """An empty quarters list matches nothing (explicit empty set)."""
        cases = load_case_templates(str(self.dataset), quarters=[])
        assert len(cases) == 0

    def test_quarter_filter_combined_with_ticker(self):
        """Ticker + quarter filters compose: NVDA Q1 only."""
        cases = load_case_templates(
            str(self.dataset),
            ticker_filter=["NVDA"],
            quarters=["Q1"],
        )
        assert len(cases) == 1
        assert cases[0].case_id == "NVDA/2025_Q1"

    def test_quarter_filter_combined_with_ticker_multiple(self):
        """Ticker AAPL + quarters Q1,Q3 returns both AAPL cases."""
        cases = load_case_templates(
            str(self.dataset),
            ticker_filter=["AAPL"],
            quarters=["Q1", "Q3"],
        )
        assert len(cases) == 2
        assert all("AAPL" in c.case_id for c in cases)

    def test_ticker_filter_runs_before_quarter(self):
        """Ticker filter narrows the set before quarter filter operates."""
        # Filter to AAPL (2 cases), then Q4 (0 AAPL cases) -> 0
        cases = load_case_templates(
            str(self.dataset),
            ticker_filter=["AAPL"],
            quarters=["Q4"],
        )
        assert len(cases) == 0

    def test_quarter_filter_preserves_order(self):
        """Filtered cases maintain their original sort order."""
        cases = load_case_templates(str(self.dataset), quarters=["Q3", "Q1"])
        case_ids = [c.case_id for c in cases]
        # Directory loading sorts by relative path: AAPL before NVDA, Q1 before Q3
        assert case_ids == ["AAPL/2025_Q1", "AAPL/2025_Q3", "NVDA/2025_Q1", "NVDA/2025_Q3"]


class TestQuarterFilterWithTopNNews:
    """Verify filter ordering: ticker -> quarter -> top_n_news."""

    @pytest.fixture(autouse=True)
    def setup_dataset(self, tmp_path):
        self.dataset = tmp_path / "cases"
        self.dataset.mkdir()
        # 10 news items per case; top_n_news will trim them
        _write_case_file(self.dataset, "NVDA/2025_Q1.json", ticker="NVDA", n_items=10)
        _write_case_file(self.dataset, "NVDA/2025_Q2.json", ticker="NVDA", n_items=10)
        _write_case_file(self.dataset, "NVDA/2025_Q3.json", ticker="NVDA", n_items=10)

    def test_all_three_filters_compose(self):
        """ticker + quarters + top_n_news all apply in sequence."""
        cases = load_case_templates(
            str(self.dataset),
            ticker_filter=["NVDA"],
            quarters=["Q1", "Q2"],
            top_n_news=3,
        )
        assert len(cases) == 2
        for c in cases:
            assert "Q1" in c.case_id or "Q2" in c.case_id
            assert len(c.case_data.items) == 3

    def test_top_n_news_applied_after_quarter_filter(self):
        """top_n_news does not affect which cases are selected, only their content."""
        cases = load_case_templates(
            str(self.dataset),
            quarters=["Q3"],
            top_n_news=2,
        )
        assert len(cases) == 1
        assert "Q3" in cases[0].case_id
        assert len(cases[0].case_data.items) == 2
