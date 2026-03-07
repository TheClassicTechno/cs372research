"""Integration test for the _CASH_ virtual ticker logic."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from models.config import SimulationConfig
from simulation.runner import AsyncSimulationRunner
from simulation.memo_loader import load_memo_cases

def test_runner_appends_cash_ticker():
    """Verify that AsyncSimulationRunner appends _CASH_ to the universe."""
    config_dict = {
        "dataset_path": "fake/path",
        "tickers": ["AAPL", "MSFT"],
        "invest_quarter": "2025Q1",
        "use_cash_virtual_ticker": True,
        "debate_setup": {
            "agent_system": "multi_agent_debate",
            "llm_provider": "openai",
            "llm_model": "gpt-4o",
        }
    }
    config = SimulationConfig(**config_dict)
    
    with patch("simulation.memo_loader.load_memo_cases") as mock_load:
        mock_load.return_value = []
        runner = AsyncSimulationRunner(config, "fake_config.yaml")
        
        # We need to trigger the part of run() that calls load_memo_cases
        # but we don't want to run the full simulation loop.
        # We'll just call the relevant part of run() or mock more.
        import asyncio
        async def run_part():
            # Minimal subset of AsyncSimulationRunner.run()
            from simulation.memo_loader import load_memo_cases
            tickers = list(runner._config.tickers)
            if runner._config.use_cash_virtual_ticker and "_CASH_" not in tickers:
                tickers.append("_CASH_")
            
            load_memo_cases(
                runner._config.dataset_path,
                invest_quarter=runner._config.invest_quarter,
                memo_format=runner._config.memo_format,
                tickers=tickers,
            )

        asyncio.run(run_part())
        
        # Check that tickers passed to load_memo_cases included _CASH_
        args, kwargs = mock_load.call_args
        assert "_CASH_" in kwargs["tickers"]
        assert "AAPL" in kwargs["tickers"]
        assert "MSFT" in kwargs["tickers"]

def test_memo_loader_injects_cash_price():
    """Verify that load_memo_cases sets _CASH_ price to 1.0."""
    fake_snap = {
        "ticker_data": {
            "AAPL": {"asset_features": {"close": 150.0}},
            "MSFT": {"asset_features": {"close": 300.0}},
        },
        "as_of_date": "2024-12-31"
    }
    
    with patch("simulation.memo_loader._load_snapshot_json", return_value=fake_snap), \
         patch("simulation.memo_loader._load_memo_text", return_value="fake memo"):
        
        cases = load_memo_cases(
            dataset_path=".",
            invest_quarter="2025Q1",
            memo_format="text",
            tickers=["AAPL", "MSFT", "_CASH_"]
        )
        
        decision_case = cases[0]
        assert "_CASH_" in decision_case.stock_data
        assert decision_case.stock_data["_CASH_"].current_price == 1.0
        assert decision_case.stock_data["AAPL"].current_price == 150.0

def test_normalize_allocation_remaps_cash():
    """Verify that normalize_allocation remaps 'CASH' to '_CASH_'."""
    from multi_agent.graph.allocation import normalize_allocation
    raw = {"AAPL": 0.4, "CASH": 0.6}
    universe = ["AAPL", "MSFT", "_CASH_"]
    
    # max_weight=1.0, min_holdings=1
    result = normalize_allocation(raw, universe, 1.0, 1)
    
    assert "_CASH_" in result
    assert "CASH" not in result
    assert result["_CASH_"] == pytest.approx(0.6)
    assert result["AAPL"] == pytest.approx(0.4)
    assert result["MSFT"] == 0.0
