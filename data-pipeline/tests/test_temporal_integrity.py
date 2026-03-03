"""
Temporal integrity tests — verify no lookahead bias in the pipeline.

These tests validate that:
  1. No filing with filing_date > rebalance_date is included in snapshots
  2. Price data does not extend past quarter end
  3. Financial statement dates are plausible (not from unreleased filings)
  4. Macro data respects quarter-end boundaries
  5. Cross-sectional features match upstream data
  6. Snapshot builder filtering logic is correct with synthetic data

Run:  cd data-pipeline && python -m pytest tests/test_temporal_integrity.py -v
"""

import datetime as dt
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

_PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PIPELINE_DIR / "final_snapshots"))
sys.path.insert(0, str(_PIPELINE_DIR / "quarterly_asset_details"))
sys.path.insert(0, str(_PIPELINE_DIR / "macro"))

from generate_quarterly_json import (
    load_asset_data,
    load_filing_summaries,
    load_sentiment,
    quarter_end_date,
    _add_relative_strength,
    _add_sentiment_z,
)


# ===================================================================
# Helpers
# ===================================================================

def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


# ===================================================================
# 1. FILING DATE FILTERING
# ===================================================================


class TestFilingDateFiltering:
    """Verify load_filing_summaries enforces filing_date <= rebalance_date."""

    def test_filing_after_rebalance_excluded(self, tmp_path):
        """A 10-Q filed AFTER rebalance_date must not appear."""
        ticker_dir = tmp_path / "AAPL" / "2024"
        _write_json(ticker_dir / "Q3.json", {
            "form": "10-Q",
            "filing_date": "2025-01-15",  # filed after Dec 31
            "fiscal_period": "2024-Q3",
        })
        result = load_filing_summaries(tmp_path, "AAPL", dt.date(2024, 12, 31))
        assert result["periodic"] is None
        assert result["event_filings"] == []

    def test_filing_before_rebalance_included(self, tmp_path):
        """A 10-Q filed BEFORE rebalance_date must be included."""
        ticker_dir = tmp_path / "AAPL" / "2024"
        _write_json(ticker_dir / "Q2.json", {
            "form": "10-Q",
            "filing_date": "2024-08-01",
            "fiscal_period": "2024-Q2",
        })
        result = load_filing_summaries(tmp_path, "AAPL", dt.date(2024, 12, 31))
        assert result["periodic"] is not None
        assert result["periodic"]["filing_date"] == "2024-08-01"

    def test_filing_on_rebalance_included(self, tmp_path):
        """A filing on exactly rebalance_date is included."""
        ticker_dir = tmp_path / "AAPL" / "2024"
        _write_json(ticker_dir / "Q3.json", {
            "form": "10-Q",
            "filing_date": "2024-12-31",
            "fiscal_period": "2024-Q3",
        })
        result = load_filing_summaries(tmp_path, "AAPL", dt.date(2024, 12, 31))
        assert result["periodic"] is not None

    def test_most_recent_periodic_selected(self, tmp_path):
        """When multiple periodic filings are eligible, pick most recent by filing_date."""
        ticker_dir = tmp_path / "AAPL"
        _write_json(ticker_dir / "2024" / "Q1.json", {
            "form": "10-Q",
            "filing_date": "2024-05-01",
        })
        _write_json(ticker_dir / "2024" / "Q2.json", {
            "form": "10-Q",
            "filing_date": "2024-08-15",
        })
        _write_json(ticker_dir / "2024" / "Q3.json", {
            "form": "10-Q",
            "filing_date": "2024-11-10",
        })
        result = load_filing_summaries(tmp_path, "AAPL", dt.date(2024, 12, 31))
        assert result["periodic"]["filing_date"] == "2024-11-10"

    def test_8k_within_90_days(self, tmp_path):
        """8-K filings older than 90 days before rebalance are excluded."""
        ticker_dir = tmp_path / "AAPL" / "2024"
        # 100 days before Dec 31 = Sep 22
        _write_json(ticker_dir / "old_8k.json", {
            "form": "8-K",
            "filing_date": "2024-09-20",  # > 90 days before Dec 31
        })
        _write_json(ticker_dir / "recent_8k.json", {
            "form": "8-K",
            "filing_date": "2024-11-15",  # within 90 days
        })
        result = load_filing_summaries(tmp_path, "AAPL", dt.date(2024, 12, 31))
        assert len(result["event_filings"]) == 1
        assert result["event_filings"][0]["filing_date"] == "2024-11-15"

    def test_8k_after_rebalance_excluded(self, tmp_path):
        """8-K filed after rebalance is excluded even if within 90-day window."""
        ticker_dir = tmp_path / "AAPL" / "2025"
        _write_json(ticker_dir / "future_8k.json", {
            "form": "8-K",
            "filing_date": "2025-01-05",
        })
        result = load_filing_summaries(tmp_path, "AAPL", dt.date(2024, 12, 31))
        assert result["event_filings"] == []

    def test_amended_filing_with_later_date(self, tmp_path):
        """10-K/A filed after rebalance is excluded even though original was before."""
        ticker_dir = tmp_path / "AAPL"
        _write_json(ticker_dir / "2024" / "original.json", {
            "form": "10-K",
            "filing_date": "2024-10-30",
        })
        _write_json(ticker_dir / "2024" / "amended.json", {
            "form": "10-K/A",
            "filing_date": "2025-02-15",  # amendment filed after rebalance
        })
        result = load_filing_summaries(tmp_path, "AAPL", dt.date(2024, 12, 31))
        # Only the original should be included
        assert result["periodic"]["form"] == "10-K"
        assert result["periodic"]["filing_date"] == "2024-10-30"

    def test_missing_ticker_returns_empty(self, tmp_path):
        """Non-existent ticker dir returns null periodic and empty events."""
        result = load_filing_summaries(tmp_path, "ZZZZ", dt.date(2024, 12, 31))
        assert result["periodic"] is None
        assert result["event_filings"] == []


# ===================================================================
# 2. QUARTER END DATE CORRECTNESS
# ===================================================================


class TestQuarterEndDates:
    """Verify quarter end dates match calendar conventions."""

    @pytest.mark.parametrize("quarter,expected", [
        ("Q1", dt.date(2025, 3, 31)),
        ("Q2", dt.date(2025, 6, 30)),
        ("Q3", dt.date(2025, 9, 30)),
        ("Q4", dt.date(2025, 12, 31)),
    ])
    def test_quarter_end_dates(self, quarter, expected):
        assert quarter_end_date(2025, quarter) == expected

    def test_leap_year_q1(self):
        """Q1 end is always March 31, not affected by leap year."""
        assert quarter_end_date(2024, "Q1") == dt.date(2024, 3, 31)


# ===================================================================
# 3. PER-TICKER FILE LOADING
# ===================================================================


class TestPerTickerAssetLoading:
    """Verify load_asset_data reads per-ticker files correctly."""

    def test_loads_features(self, tmp_path):
        _write_json(tmp_path / "AAPL" / "2025_Q1.json", {
            "schema_version": "asset_state_v2",
            "ticker": "AAPL",
            "year": 2025,
            "quarter": "Q1",
            "as_of": "2025-03-31",
            "features": {"close": 198.0, "ret_60d": 0.05},
        })
        result = load_asset_data(tmp_path, "AAPL", 2025, "Q1")
        assert result is not None
        assert result["close"] == 198.0
        assert result["ret_60d"] == 0.05

    def test_missing_ticker_returns_none(self, tmp_path):
        result = load_asset_data(tmp_path, "ZZZZ", 2025, "Q1")
        assert result is None

    def test_missing_quarter_returns_none(self, tmp_path):
        (tmp_path / "AAPL").mkdir()
        result = load_asset_data(tmp_path, "AAPL", 2025, "Q1")
        assert result is None


class TestPerTickerSentimentLoading:
    """Verify load_sentiment reads per-ticker files correctly."""

    def test_loads_features(self, tmp_path):
        _write_json(tmp_path / "AAPL" / "2025_Q1.json", {
            "schema_version": "sentiment_v1",
            "ticker": "AAPL",
            "year": 2025,
            "quarter": "Q1",
            "features": {"article_count": 42, "mean_sentiment": 0.12},
        })
        result = load_sentiment(tmp_path, "AAPL", 2025, "Q1")
        assert result is not None
        assert result["article_count"] == 42
        assert result["mean_sentiment"] == 0.12

    def test_missing_returns_none(self, tmp_path):
        result = load_sentiment(tmp_path, "ZZZZ", 2025, "Q1")
        assert result is None


# ===================================================================
# 4. SNAPSHOT-LEVEL LEAKAGE DETECTION ON REAL DATA
# ===================================================================


class TestRealDataLeakage:
    """Run against actual pipeline output if it exists.

    These tests are skipped if the data directories don't exist.
    They verify no temporal violations in generated data.
    """

    SNAPSHOT_DIR = _PIPELINE_DIR / "final_snapshots" / "json_data"
    SUMMARIES_DIR = _PIPELINE_DIR / "EDGAR" / "finished_summaries"
    ASSET_DIR = _PIPELINE_DIR / "quarterly_asset_details" / "data"
    SENTIMENT_DIR = _PIPELINE_DIR / "sentiment" / "data"
    MACRO_DIR = _PIPELINE_DIR / "macro" / "data"

    def _load_snapshots(self):
        if not self.SNAPSHOT_DIR.exists():
            pytest.skip("No snapshot data found")
        snapshots = []
        for f in sorted(self.SNAPSHOT_DIR.glob("snapshot_*.json")):
            with open(f) as fh:
                snapshots.append(json.load(fh))
        if not snapshots:
            pytest.skip("No snapshot files found")
        return snapshots

    def test_no_filing_after_rebalance_in_snapshots(self):
        """Verify every filing_date in every snapshot <= as_of_date."""
        for snap in self._load_snapshots():
            cutoff = snap["as_of_date"]
            for ticker, td in snap.get("ticker_data", {}).items():
                fs = td.get("filing_summary", {})
                if not fs:
                    continue
                periodic = fs.get("periodic")
                if periodic and periodic.get("filing_date"):
                    assert periodic["filing_date"] <= cutoff, (
                        f"{ticker}: periodic filing_date {periodic['filing_date']} "
                        f"> as_of {cutoff}"
                    )
                for ev in fs.get("event_filings", []):
                    if ev.get("filing_date"):
                        assert ev["filing_date"] <= cutoff, (
                            f"{ticker}: 8-K filing_date {ev['filing_date']} "
                            f"> as_of {cutoff}"
                        )

    def test_no_macro_data_after_quarter_end(self):
        """Verify macro data as_of matches quarter end."""
        if not self.MACRO_DIR.exists():
            pytest.skip("No macro data")
        for f in sorted(self.MACRO_DIR.glob("macro_*.json")):
            with open(f) as fh:
                data = json.load(fh)
            year = data["year"]
            quarter = data["quarter"]
            as_of = data["as_of"]
            expected_end = quarter_end_date(year, quarter).isoformat()
            assert as_of <= expected_end, (
                f"macro {f.name}: as_of {as_of} > quarter_end {expected_end}"
            )

    def test_asset_files_as_of_matches_quarter_end(self):
        """Verify per-ticker asset files have as_of <= quarter end."""
        if not self.ASSET_DIR.exists():
            pytest.skip("No asset data")
        for ticker_dir in self.ASSET_DIR.iterdir():
            if not ticker_dir.is_dir():
                continue
            for f in ticker_dir.glob("*.json"):
                with open(f) as fh:
                    data = json.load(fh)
                year = data.get("year")
                quarter = data.get("quarter")
                as_of = data.get("as_of")
                if year and quarter and as_of:
                    expected = quarter_end_date(year, quarter).isoformat()
                    assert as_of <= expected, (
                        f"{f}: as_of {as_of} > quarter_end {expected}"
                    )

    def test_8k_events_within_90_day_window(self):
        """Every 8-K in a snapshot must have filing_date within 90 days of as_of."""
        for snap in self._load_snapshots():
            as_of = dt.date.fromisoformat(snap["as_of_date"])
            cutoff_90 = (as_of - dt.timedelta(days=90)).isoformat()
            for ticker, td in snap.get("ticker_data", {}).items():
                fs = td.get("filing_summary", {})
                for ev in fs.get("event_filings", []):
                    fd = ev.get("filing_date", "")
                    assert fd >= cutoff_90, (
                        f"{ticker}: 8-K filing_date {fd} is older than "
                        f"90-day window (cutoff {cutoff_90})"
                    )


# ===================================================================
# 5. CROSS-SECTIONAL FEATURE CONSISTENCY
# ===================================================================


class TestCrossSectionalConsistency:
    """Verify cross-sectional features are mathematically consistent."""

    def test_relative_strength_sums_near_zero(self):
        """Relative strength (deviation from median) should approximately sum to zero
        for odd N, and exactly for symmetric distributions."""
        ticker_data = {
            "A": {"asset_features": {"ret_60d": 0.05}},
            "B": {"asset_features": {"ret_60d": 0.10}},
            "C": {"asset_features": {"ret_60d": 0.15}},
        }
        _add_relative_strength(ticker_data)
        strengths = [
            td["asset_features"]["relative_strength_60d"]
            for td in ticker_data.values()
        ]
        # For odd N, sum of deviations from median = 0 only if symmetric
        # But median is exact middle value, so it should be close to zero
        assert sum(strengths) == pytest.approx(0.0, abs=0.01)

    def test_sentiment_z_scores_sum_to_zero(self):
        """Z-scores across tickers in a quarter must sum to approximately zero."""
        ticker_data = {
            "A": {"news_sentiment": {"mean_sentiment": 0.10}},
            "B": {"news_sentiment": {"mean_sentiment": 0.20}},
            "C": {"news_sentiment": {"mean_sentiment": 0.30}},
            "D": {"news_sentiment": {"mean_sentiment": 0.15}},
        }
        _add_sentiment_z(ticker_data)
        z_scores = [
            td["news_sentiment"]["cross_sectional_z"]
            for td in ticker_data.values()
        ]
        assert sum(z_scores) == pytest.approx(0.0, abs=1e-5)

    def test_sentiment_z_scores_unit_variance(self):
        """Z-scores should have population std dev = 1."""
        ticker_data = {
            f"T{i}": {"news_sentiment": {"mean_sentiment": v}}
            for i, v in enumerate([0.05, 0.10, 0.15, 0.20, 0.25])
        }
        _add_sentiment_z(ticker_data)
        z_scores = [
            td["news_sentiment"]["cross_sectional_z"]
            for td in ticker_data.values()
        ]
        pop_std = math.sqrt(sum(z ** 2 for z in z_scores) / len(z_scores))
        assert pop_std == pytest.approx(1.0, abs=1e-4)
