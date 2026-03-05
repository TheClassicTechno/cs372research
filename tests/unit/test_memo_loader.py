"""Unit tests for memo_loader: prev_quarter, _parse_invest_quarter, load_memo_cases."""

import json

import pytest

from simulation.memo_loader import (
    prev_quarter,
    _parse_invest_quarter,
    load_memo_cases,
)


# ── prev_quarter ─────────────────────────────────────────────────────────────


class TestPrevQuarter:
    @pytest.mark.parametrize(
        "year,qtr,exp_year,exp_qtr",
        [
            (2025, "Q1", 2024, "Q4"),
            (2025, "Q2", 2025, "Q1"),
            (2025, "Q3", 2025, "Q2"),
            (2025, "Q4", 2025, "Q3"),
            (2024, "Q1", 2023, "Q4"),
        ],
    )
    def test_quarter_transitions(self, year, qtr, exp_year, exp_qtr):
        assert prev_quarter(year, qtr) == (exp_year, exp_qtr)


# ── _parse_invest_quarter ────────────────────────────────────────────────────


class TestParseInvestQuarter:
    @pytest.mark.parametrize(
        "iq,exp_year,exp_q",
        [
            ("2025Q1", 2025, "Q1"),
            ("2024Q4", 2024, "Q4"),
            ("2023Q2", 2023, "Q2"),
        ],
    )
    def test_valid(self, iq, exp_year, exp_q):
        assert _parse_invest_quarter(iq) == (exp_year, exp_q)

    def test_invalid_quarter_q5(self):
        with pytest.raises(ValueError, match="Invalid quarter"):
            _parse_invest_quarter("2025Q5")

    def test_invalid_format(self):
        with pytest.raises((ValueError, IndexError)):
            _parse_invest_quarter("Q12025")


# ── load_memo_cases ──────────────────────────────────────────────────────────


def _make_snapshot(tickers_prices: dict, as_of: str = "2024-12-31") -> dict:
    """Build a minimal snapshot JSON dict."""
    ticker_data = {}
    for t, price in tickers_prices.items():
        ticker_data[t] = {"asset_features": {"close": price}}
    return {"ticker_data": ticker_data, "as_of_date": as_of}


@pytest.fixture
def memo_fs(tmp_path):
    """Create a temporary filesystem for memo loader tests.

    Layout:
      <tmp_path>/
        json_data/snapshot_2024_Q4.json   (prior quarter)
        json_data/snapshot_2025_Q1.json   (invest quarter)
        memo_data/memo_2024_Q4.txt        (prior quarter memo)
    """
    (tmp_path / "json_data").mkdir()
    (tmp_path / "memo_data").mkdir()

    prior_snap = _make_snapshot(
        {"AAPL": 185.0, "MSFT": 390.0}, as_of="2024-12-31"
    )
    (tmp_path / "json_data" / "snapshot_2024_Q4.json").write_text(
        json.dumps(prior_snap)
    )

    invest_snap = _make_snapshot(
        {"AAPL": 195.0, "MSFT": 400.0}, as_of="2025-03-31"
    )
    (tmp_path / "json_data" / "snapshot_2025_Q1.json").write_text(
        json.dumps(invest_snap)
    )

    (tmp_path / "memo_data" / "memo_2024_Q4.txt").write_text(
        "Q4 2024 analysis: macro steady, tech strong."
    )

    return tmp_path


class TestLoadMemoCases:
    def test_returns_decision_and_mtm_cases(self, memo_fs):
        cases = load_memo_cases(
            str(memo_fs), invest_quarter="2025Q1",
            memo_format="text", tickers=["AAPL", "MSFT"],
        )
        assert len(cases) == 2
        assert cases[0].case_id == "memo/2025Q1"
        assert cases[1].case_id == "mtm/2025Q1"

    def test_returns_only_decision_when_no_mtm_snapshot(self, memo_fs):
        # Remove invest-quarter snapshot
        (memo_fs / "json_data" / "snapshot_2025_Q1.json").unlink()
        cases = load_memo_cases(
            str(memo_fs), invest_quarter="2025Q1",
            memo_format="text", tickers=["AAPL", "MSFT"],
        )
        assert len(cases) == 1
        assert cases[0].case_id == "memo/2025Q1"

    def test_decision_case_has_all_tickers(self, memo_fs):
        cases = load_memo_cases(
            str(memo_fs), invest_quarter="2025Q1",
            memo_format="text", tickers=["AAPL", "MSFT"],
        )
        assert set(cases[0].stock_data.keys()) == {"AAPL", "MSFT"}

    def test_decision_case_has_memo_content(self, memo_fs):
        cases = load_memo_cases(
            str(memo_fs), invest_quarter="2025Q1",
            memo_format="text", tickers=["AAPL", "MSFT"],
        )
        item = cases[0].case_data.items[0]
        assert "Q4 2024 analysis" in item.content

    def test_json_format_loads_json_context(self, memo_fs):
        cases = load_memo_cases(
            str(memo_fs), invest_quarter="2025Q1",
            memo_format="json", tickers=["AAPL", "MSFT"],
        )
        content = cases[0].case_data.items[0].content
        parsed = json.loads(content)
        assert "ticker_data" in parsed

    def test_entry_prices_correct(self, memo_fs):
        cases = load_memo_cases(
            str(memo_fs), invest_quarter="2025Q1",
            memo_format="text", tickers=["AAPL", "MSFT"],
        )
        assert cases[0].stock_data["AAPL"].current_price == pytest.approx(185.0)
        assert cases[0].stock_data["MSFT"].current_price == pytest.approx(390.0)

    def test_mtm_exit_prices_correct(self, memo_fs):
        cases = load_memo_cases(
            str(memo_fs), invest_quarter="2025Q1",
            memo_format="text", tickers=["AAPL", "MSFT"],
        )
        assert cases[1].stock_data["AAPL"].current_price == pytest.approx(195.0)
        assert cases[1].stock_data["MSFT"].current_price == pytest.approx(400.0)

    def test_missing_prior_snapshot_raises(self, memo_fs):
        (memo_fs / "json_data" / "snapshot_2024_Q4.json").unlink()
        with pytest.raises(FileNotFoundError):
            load_memo_cases(
                str(memo_fs), invest_quarter="2025Q1",
                memo_format="text", tickers=["AAPL"],
            )

    def test_missing_prior_memo_text_raises(self, memo_fs):
        (memo_fs / "memo_data" / "memo_2024_Q4.txt").unlink()
        with pytest.raises(FileNotFoundError):
            load_memo_cases(
                str(memo_fs), invest_quarter="2025Q1",
                memo_format="text", tickers=["AAPL"],
            )

    def test_mtm_case_has_empty_case_data(self, memo_fs):
        cases = load_memo_cases(
            str(memo_fs), invest_quarter="2025Q1",
            memo_format="text", tickers=["AAPL", "MSFT"],
        )
        assert cases[1].case_data.items == []

    def test_cutoff_timestamp_set(self, memo_fs):
        cases = load_memo_cases(
            str(memo_fs), invest_quarter="2025Q1",
            memo_format="text", tickers=["AAPL"],
        )
        assert cases[0].information_cutoff_timestamp == "2024-12-31"
