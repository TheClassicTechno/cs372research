"""
Fiscal year alignment tests for non-standard FYE companies.

Verifies:
  1. Configuration: all tickers have fiscal_year_end in supported_tickers.yaml
  2. Filing placement: summaries are stored in correct calendar quarter dirs
  3. No leakage: filing_dates never exceed snapshot as_of_date
  4. Fiscal period consistency: 10-K → annual, 10-Q → quarterly
  5. Memo incorporation: filings appear correctly in memos
  6. filter_filings() unit tests for non-standard fiscal periods

Non-standard FYE tickers:
  NVDA: Jan 26    WMT: Jan 31    AAPL: Sep 27    COST: Aug 31    MSFT: Jun 30

Run:  cd data-pipeline && python -m pytest tests/test_fiscal_alignment.py -v
"""

import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Make pipeline modules importable
# ---------------------------------------------------------------------------

_PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PIPELINE_DIR))
sys.path.insert(0, str(_PIPELINE_DIR / "EDGAR"))
sys.path.insert(0, str(_PIPELINE_DIR / "final_snapshots"))

from get_sec_data import filter_filings, quarter_from_month

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_YAML_PATH = _PIPELINE_DIR / "supported_tickers.yaml"
_SUMMARIES_DIR = _PIPELINE_DIR / "EDGAR" / "finished_summaries"
_SNAPSHOT_DIR = _PIPELINE_DIR / "final_snapshots" / "json_data"
_MEMO_DIR = _PIPELINE_DIR / "final_snapshots" / "memo_data"

NON_STANDARD_FYE = {
    "NVDA": "01-26",
    "WMT": "01-31",
    "AAPL": "09-27",
    "COST": "08-31",
    "MSFT": "06-30",
}

STANDARD_FYE_TICKERS = [
    "CAT", "DAL", "JPM", "GS", "BAC", "UNH", "LLY", "JNJ",
    "AMT", "XOM", "GOOG", "AMZN", "META", "TSLA", "NFLX",
]

QUARTER_ENDS = {
    "Q1": (3, 31),
    "Q2": (6, 30),
    "Q3": (9, 30),
    "Q4": (12, 31),
}


def quarter_end_date(year: int, quarter: str) -> dt.date:
    month, day = QUARTER_ENDS[quarter]
    return dt.date(year, month, day)


def _load_yaml():
    with open(_YAML_PATH, "r") as f:
        return yaml.safe_load(f)


def _load_all_summaries(ticker: str):
    """Load all finished_summaries for a ticker. Returns list of (path, dict)."""
    ticker_dir = _SUMMARIES_DIR / ticker
    if not ticker_dir.exists():
        return []
    results = []
    for json_file in sorted(ticker_dir.rglob("*.json")):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            results.append((json_file, data))
        except (json.JSONDecodeError, OSError):
            continue
    return results


def _load_snapshot(year: int, quarter: str):
    """Load a snapshot JSON. Returns None if not found."""
    path = _SNAPSHOT_DIR / f"snapshot_{year}_{quarter}.json"
    if not path.exists():
        # Try alternate location
        alt = _PIPELINE_DIR / "final_snapshots" / f"snapshot_{year}_{quarter}.json"
        if alt.exists():
            path = alt
        else:
            return None
    with open(path, "r") as f:
        return json.load(f)


def _available_snapshot_quarters():
    """Return list of (year, quarter) for which snapshots exist."""
    results = []
    for d in [_SNAPSHOT_DIR, _PIPELINE_DIR / "final_snapshots"]:
        if not d.exists():
            continue
        for f in sorted(d.glob("snapshot_*_Q*.json")):
            m = re.match(r"snapshot_(\d{4})_(Q[1-4])\.json", f.name)
            if m:
                results.append((int(m.group(1)), m.group(2)))
    return sorted(set(results))


def _make_filings(entries):
    """Build a mock SEC filings dict."""
    recent = {
        "form": [],
        "filingDate": [],
        "accessionNumber": [],
        "primaryDocument": [],
        "reportDate": [],
    }
    for e in entries:
        recent["form"].append(e["form"])
        recent["filingDate"].append(e["filingDate"])
        recent["accessionNumber"].append(e["accessionNumber"])
        recent["primaryDocument"].append(e["primaryDocument"])
        recent["reportDate"].append(e.get("reportDate", e["filingDate"]))
    return {"filings": {"recent": recent}}


# ===================================================================
# 1. FISCAL YEAR-END CONFIGURATION
# ===================================================================


class TestFiscalYearEndConfig:
    def test_all_tickers_have_fiscal_year_end(self):
        data = _load_yaml()
        for entry in data["supported_tickers"]:
            fye = entry.get("fiscal_year_end")
            assert fye is not None, f"{entry['symbol']} missing fiscal_year_end"
            assert re.match(r"^\d{2}-\d{2}$", fye), f"{entry['symbol']} bad FYE format: {fye}"

    def test_known_non_standard_fye(self):
        data = _load_yaml()
        fye_map = {e["symbol"]: e["fiscal_year_end"] for e in data["supported_tickers"]}
        for ticker, expected in NON_STANDARD_FYE.items():
            assert fye_map.get(ticker) == expected, f"{ticker}: expected {expected}, got {fye_map.get(ticker)}"

    def test_standard_fye_tickers(self):
        data = _load_yaml()
        fye_map = {e["symbol"]: e["fiscal_year_end"] for e in data["supported_tickers"]}
        for ticker in STANDARD_FYE_TICKERS:
            assert fye_map.get(ticker) == "12-31", f"{ticker}: expected 12-31, got {fye_map.get(ticker)}"


# ===================================================================
# 2. FILING PLACEMENT (summaries in correct calendar quarter dirs)
# ===================================================================


class TestNonStandardFYEFilingPlacement:
    """Verify summaries for non-standard FYE tickers have consistent metadata."""

    @pytest.mark.parametrize("ticker", list(NON_STANDARD_FYE.keys()))
    def test_summaries_have_fiscal_period(self, ticker):
        summaries = _load_all_summaries(ticker)
        if not summaries:
            pytest.skip(f"No summaries for {ticker}")
        for path, data in summaries:
            assert "fiscal_period" in data, f"{path.name} missing fiscal_period"
            assert data["fiscal_period"], f"{path.name} empty fiscal_period"

    @pytest.mark.parametrize("ticker", list(NON_STANDARD_FYE.keys()))
    def test_summaries_have_period_type(self, ticker):
        summaries = _load_all_summaries(ticker)
        if not summaries:
            pytest.skip(f"No summaries for {ticker}")
        for path, data in summaries:
            assert data.get("period_type") in ("annual", "quarterly"), \
                f"{path.name} bad period_type: {data.get('period_type')}"


# ===================================================================
# 3. NO LEAKAGE — filing_date never exceeds snapshot as_of_date
# ===================================================================


class TestNoLeakageNonStandardFYE:
    """Core safety: no filing in any snapshot has filing_date > as_of_date."""

    @pytest.mark.parametrize("ticker", list(NON_STANDARD_FYE.keys()))
    def test_periodic_filing_date_before_rebalance(self, ticker):
        quarters = _available_snapshot_quarters()
        if not quarters:
            pytest.skip("No snapshots available")

        for year, quarter in quarters:
            doc = _load_snapshot(year, quarter)
            if not doc:
                continue

            td = doc.get("ticker_data", {}).get(ticker)
            if not td:
                continue

            fs = td.get("filing_summary", {})
            periodic = fs.get("periodic") if isinstance(fs, dict) else None
            if not periodic:
                continue

            rebal = quarter_end_date(year, quarter)
            fd = periodic.get("filing_date", "")
            assert fd, f"{ticker} {year}{quarter}: periodic filing has no filing_date"
            assert fd <= rebal.isoformat(), \
                f"{ticker} {year}{quarter}: filing_date {fd} > rebalance {rebal.isoformat()}"

    @pytest.mark.parametrize("ticker", list(NON_STANDARD_FYE.keys()))
    def test_event_filings_within_window(self, ticker):
        quarters = _available_snapshot_quarters()
        if not quarters:
            pytest.skip("No snapshots available")

        for year, quarter in quarters:
            doc = _load_snapshot(year, quarter)
            if not doc:
                continue

            td = doc.get("ticker_data", {}).get(ticker)
            if not td:
                continue

            fs = td.get("filing_summary", {})
            events = fs.get("event_filings", []) if isinstance(fs, dict) else []
            rebal = quarter_end_date(year, quarter)
            cutoff_90 = (rebal - dt.timedelta(days=90)).isoformat()

            for ev in events:
                fd = ev.get("filing_date", "")
                assert fd <= rebal.isoformat(), \
                    f"{ticker} {year}{quarter}: 8-K filing_date {fd} > rebalance"
                assert fd >= cutoff_90, \
                    f"{ticker} {year}{quarter}: 8-K filing_date {fd} outside 90-day window"

    def test_no_future_fiscal_period_content(self):
        """For annual filings of non-standard FYE tickers,
        verify the fiscal year-end date <= rebalance_date."""
        quarters = _available_snapshot_quarters()
        if not quarters:
            pytest.skip("No snapshots available")

        for year, quarter in quarters:
            doc = _load_snapshot(year, quarter)
            if not doc:
                continue

            rebal = quarter_end_date(year, quarter)

            for ticker, fye in NON_STANDARD_FYE.items():
                td = doc.get("ticker_data", {}).get(ticker)
                if not td:
                    continue

                fs = td.get("filing_summary", {})
                periodic = fs.get("periodic") if isinstance(fs, dict) else None
                if not periodic:
                    continue

                if periodic.get("period_type") != "annual":
                    continue

                fiscal_period = periodic.get("fiscal_period", "")
                if not fiscal_period.startswith("FY"):
                    continue

                fy_year = int(fiscal_period[2:])
                fye_month = int(fye[:2])
                fye_day = int(fye[3:])
                fye_date = dt.date(fy_year, fye_month, fye_day)

                assert fye_date <= rebal, \
                    f"{ticker} {year}{quarter}: fiscal year-end {fye_date} > " \
                    f"rebalance {rebal} — potential content leakage"


# ===================================================================
# 4. FISCAL PERIOD CONSISTENCY
# ===================================================================


class TestFiscalPeriodConsistency:
    """Verify fiscal_period makes sense for each filing's form and period_type."""

    @pytest.mark.parametrize("ticker", list(NON_STANDARD_FYE.keys()))
    def test_10k_has_annual_fiscal_period(self, ticker):
        summaries = _load_all_summaries(ticker)
        for path, data in summaries:
            form = data.get("form", "")
            if form not in ("10-K", "10-K/A"):
                continue
            assert data.get("period_type") == "annual", \
                f"{path.name}: 10-K should be annual, got {data.get('period_type')}"
            assert data.get("fiscal_period", "").startswith("FY"), \
                f"{path.name}: 10-K fiscal_period should start with FY, got {data.get('fiscal_period')}"

    @pytest.mark.parametrize("ticker", list(NON_STANDARD_FYE.keys()))
    def test_10q_has_quarterly_fiscal_period(self, ticker):
        summaries = _load_all_summaries(ticker)
        for path, data in summaries:
            form = data.get("form", "")
            if form not in ("10-Q", "10-Q/A"):
                continue
            assert data.get("period_type") == "quarterly", \
                f"{path.name}: 10-Q should be quarterly, got {data.get('period_type')}"
            fp = data.get("fiscal_period", "")
            assert re.match(r"^\d{4}-Q[1-4]$", fp), \
                f"{path.name}: 10-Q fiscal_period should be YYYY-Q#, got {fp}"

    @pytest.mark.parametrize("ticker", list(NON_STANDARD_FYE.keys()))
    def test_fiscal_period_year_plausible(self, ticker):
        """Fiscal period year should be within ±1 of the directory year."""
        summaries = _load_all_summaries(ticker)
        for path, data in summaries:
            fp = data.get("fiscal_period", "")
            # Extract directory year
            parts = path.relative_to(_SUMMARIES_DIR / ticker).parts
            if len(parts) >= 1:
                dir_year = int(parts[0])
            else:
                continue

            # Extract fiscal year
            if fp.startswith("FY"):
                fy = int(fp[2:])
            elif re.match(r"^\d{4}-Q", fp):
                fy = int(fp[:4])
            else:
                continue

            assert abs(fy - dir_year) <= 1, \
                f"{path.name}: fiscal year {fy} too far from directory year {dir_year}"


# ===================================================================
# 5. MEMO INCORPORATION
# ===================================================================


class TestMemoIncorporation:
    """Verify non-standard FYE filings appear correctly in memos."""

    @pytest.mark.parametrize("ticker", list(NON_STANDARD_FYE.keys()))
    def test_memo_contains_filing(self, ticker):
        """Check that the memo contains a filing summary line for this ticker."""
        quarters = _available_snapshot_quarters()
        if not quarters:
            pytest.skip("No snapshots available")

        found_any = False
        for year, quarter in quarters:
            memo_path = _MEMO_DIR / f"memo_{year}_{quarter}.txt"
            if not memo_path.exists():
                continue

            memo_text = memo_path.read_text()
            # Find this ticker's section
            ticker_pattern = f"TICKER: {ticker}"
            if ticker_pattern not in memo_text:
                continue

            found_any = True

            # Get the section for this ticker
            idx = memo_text.index(ticker_pattern)
            # Find the next TICKER: line or END OF MEMO
            next_ticker = memo_text.find("TICKER:", idx + len(ticker_pattern))
            end_memo = memo_text.find("END OF MEMO", idx)
            section_end = min(
                x for x in [next_ticker, end_memo, len(memo_text)] if x > 0
            )
            section = memo_text[idx:section_end]

            # Check for Filing Summary line
            assert "Filing Summary" in section or "Filing summary: not available" in section, \
                f"{ticker} {year}{quarter}: no filing info in memo section"

            # If a filing exists, verify filing_date appears
            summaries = _load_all_summaries(ticker)
            if summaries:
                doc = _load_snapshot(year, quarter)
                if doc:
                    td = doc.get("ticker_data", {}).get(ticker, {})
                    fs = td.get("filing_summary", {})
                    periodic = fs.get("periodic") if isinstance(fs, dict) else None
                    if periodic:
                        fd = periodic.get("filing_date", "")
                        if fd:
                            assert fd in section, \
                                f"{ticker} {year}{quarter}: filing_date {fd} not in memo"

        if not found_any:
            pytest.skip(f"No memos found containing {ticker}")

    def test_memo_fiscal_annotation_for_non_standard_fye(self):
        """Non-standard FYE tickers should have FYE annotation in memo.

        This test validates that memos generated with the updated
        generate_quarterly_memo.py include fiscal period and FYE annotations
        for non-standard FYE companies. Existing memos pre-dating the change
        will fail until regenerated.
        """
        quarters = _available_snapshot_quarters()
        if not quarters:
            pytest.skip("No snapshots available")

        for year, quarter in quarters:
            memo_path = _MEMO_DIR / f"memo_{year}_{quarter}.txt"
            if not memo_path.exists():
                continue

            memo_text = memo_path.read_text()

            for ticker, fye in NON_STANDARD_FYE.items():
                ticker_pattern = f"TICKER: {ticker}"
                if ticker_pattern not in memo_text:
                    continue

                idx = memo_text.index(ticker_pattern)
                next_ticker = memo_text.find("TICKER:", idx + len(ticker_pattern))
                end_memo = memo_text.find("END OF MEMO", idx)
                section_end = min(
                    x for x in [next_ticker, end_memo, len(memo_text)] if x > 0
                )
                section = memo_text[idx:section_end]

                # If there's a Filing Summary line (not "not available"), check for FYE annotation
                if "Filing Summary (" in section and "not available" not in section.split("Filing Summary")[1][:30]:
                    assert "FYE:" in section, \
                        f"{ticker} {year}{quarter}: non-standard FYE ticker missing FYE annotation in memo"


# ===================================================================
# 6. FILTER_FILINGS UNIT TESTS — NON-STANDARD FYE
# ===================================================================


class TestFilterFilingsNonStandardFYE:
    """Unit tests for filter_filings() with non-standard fiscal periods."""

    def test_nvda_10k_jan_fye_maps_to_q1(self):
        filings = _make_filings([{
            "form": "10-K",
            "filingDate": "2025-02-26",
            "accessionNumber": "nvda-10k-fy25",
            "primaryDocument": "nvda-20250126.htm",
            "reportDate": "2025-01-26",
        }])
        results = filter_filings(filings, [(2025, "Q1")], {"10-K"})
        assert len(results) == 1
        assert results[0]["matched_year"] == 2025
        assert results[0]["matched_quarter"] == "Q1"

    def test_wmt_10k_jan_fye_maps_to_q1(self):
        filings = _make_filings([{
            "form": "10-K",
            "filingDate": "2025-03-14",
            "accessionNumber": "wmt-10k-fy25",
            "primaryDocument": "wmt-20250131.htm",
            "reportDate": "2025-01-31",
        }])
        results = filter_filings(filings, [(2025, "Q1")], {"10-K"})
        assert len(results) == 1
        assert results[0]["matched_year"] == 2025
        assert results[0]["matched_quarter"] == "Q1"

    def test_aapl_10k_sep_fye_maps_to_q3(self):
        filings = _make_filings([{
            "form": "10-K",
            "filingDate": "2024-11-01",
            "accessionNumber": "aapl-10k-fy24",
            "primaryDocument": "aapl-20240928.htm",
            "reportDate": "2024-09-28",
        }])
        results = filter_filings(filings, [(2024, "Q3")], {"10-K"})
        assert len(results) == 1
        assert results[0]["matched_year"] == 2024
        assert results[0]["matched_quarter"] == "Q3"

    def test_cost_10k_aug_fye_maps_to_q3(self):
        filings = _make_filings([{
            "form": "10-K",
            "filingDate": "2024-10-09",
            "accessionNumber": "cost-10k-fy24",
            "primaryDocument": "cost-20240901.htm",
            "reportDate": "2024-09-01",  # COST FYE Aug 31, reportDate ~ Sep 1
        }])
        results = filter_filings(filings, [(2024, "Q3")], {"10-K"})
        assert len(results) == 1
        assert results[0]["matched_year"] == 2024
        assert results[0]["matched_quarter"] == "Q3"

    def test_msft_10k_jun_fye_maps_to_q2(self):
        filings = _make_filings([{
            "form": "10-K",
            "filingDate": "2025-07-30",
            "accessionNumber": "msft-10k-fy25",
            "primaryDocument": "msft-20250630.htm",
            "reportDate": "2025-06-30",
        }])
        results = filter_filings(filings, [(2025, "Q2")], {"10-K"})
        assert len(results) == 1
        assert results[0]["matched_year"] == 2025
        assert results[0]["matched_quarter"] == "Q2"

    def test_nvda_spillover_to_prior_q4_when_q1_not_targeted(self):
        """When only Q4 target exists, NVDA's Jan FYE 10-K should spillover."""
        filings = _make_filings([{
            "form": "10-K",
            "filingDate": "2025-02-26",
            "accessionNumber": "nvda-10k-fy25",
            "primaryDocument": "nvda-20250126.htm",
            "reportDate": "2025-01-26",
        }])
        results = filter_filings(filings, [(2024, "Q4")], {"10-K"})
        assert len(results) == 1
        assert results[0]["matched_year"] == 2024
        assert results[0]["matched_quarter"] == "Q4"

    def test_nvda_no_spillover_when_q1_is_targeted(self):
        """Direct match to Q1 should win over spillover to Q4."""
        filings = _make_filings([{
            "form": "10-K",
            "filingDate": "2025-02-26",
            "accessionNumber": "nvda-10k-fy25",
            "primaryDocument": "nvda-20250126.htm",
            "reportDate": "2025-01-26",
        }])
        results = filter_filings(filings, [(2024, "Q4"), (2025, "Q1")], {"10-K"})
        assert len(results) == 1
        assert results[0]["matched_year"] == 2025
        assert results[0]["matched_quarter"] == "Q1"
