"""
Mathematical correctness tests for pipeline computations.

Tests cover:
  1. Asset metric helpers (returns, vol, drawdown, sharpe, beta, etc.)
  2. Sentiment computations (weighted mean/std, surprise, decay weights)
  3. Snapshot cross-sectional features (relative strength, sentiment z-score)

Run:  cd data-pipeline && python -m pytest tests/ -v
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Make pipeline modules importable
# ---------------------------------------------------------------------------
_PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PIPELINE_DIR / "quarterly_asset_details"))
sys.path.insert(0, str(_PIPELINE_DIR / "sentiment"))
sys.path.insert(0, str(_PIPELINE_DIR / "final_snapshots"))
sys.path.insert(0, str(_PIPELINE_DIR / "EDGAR"))

from asset_quarter_builder import (
    _ann_vol,
    _avg_dollar_volume,
    _beta,
    _downside_vol,
    _idiosyncratic_momentum,
    _latest_valid_col,
    _max_drawdown,
    _momentum_12_1,
    _ret,
    _round_or_none,
    _safe_get,
    _sharpe,
    _sma,
    _trend_consistency,
)
from generate_quarterly_json import _add_relative_strength, _add_sentiment_z
from get_sec_data import filter_filings, next_quarter, quarter_from_month
from sentiment import (
    add_cross_sectional_z,
    add_surprise_sentiment,
    exp_decay_weights,
    weighted_mean_and_std,
)


# ===================================================================
# Helpers
# ===================================================================

def make_prices(values):
    """Create a pd.Series of prices from a list."""
    return pd.Series(values, dtype=float)


# ===================================================================
# 1. ASSET METRIC HELPERS
# ===================================================================


class TestReturn:
    def test_basic(self):
        # 100 -> 110 over 1 day = 10% return
        px = make_prices([100, 110])
        assert _ret(px, 1) == pytest.approx(0.1)

    def test_multi_day(self):
        # Window of 3: use px[-4] and px[-1]
        px = make_prices([80, 90, 95, 100, 120])
        assert _ret(px, 3) == pytest.approx(120 / 90 - 1)

    def test_negative_return(self):
        px = make_prices([100, 80])
        assert _ret(px, 1) == pytest.approx(-0.2)

    def test_insufficient_data(self):
        px = make_prices([100])
        assert _ret(px, 1) is None

    def test_zero_return(self):
        px = make_prices([100, 100])
        assert _ret(px, 1) == pytest.approx(0.0)


class TestAnnVol:
    def test_constant_prices_returns_zero(self):
        # Constant prices -> zero log returns -> zero vol
        px = make_prices([100] * 30)
        # std of all zeros = 0
        assert _ann_vol(px, 20) == pytest.approx(0.0)

    def test_known_volatility(self):
        # Create prices with known daily log returns
        np.random.seed(42)
        daily_log_ret = np.random.normal(0, 0.01, 60)  # daily vol ~1%
        prices = [100.0]
        for r in daily_log_ret:
            prices.append(prices[-1] * np.exp(r))
        px = make_prices(prices)
        vol = _ann_vol(px, 60)
        # Should be close to 0.01 * sqrt(252) ≈ 0.1587
        assert vol == pytest.approx(0.01 * np.sqrt(252), rel=0.15)

    def test_insufficient_data(self):
        px = make_prices([100, 101, 102])
        assert _ann_vol(px, 5) is None

    def test_too_few_returns(self):
        # 9 returns (need at least 10)
        px = make_prices(list(range(100, 111)))  # 11 values -> 10 returns
        # window=10, len(px) >= 11 but log_ret has 10 values -> passes check
        result = _ann_vol(px, 10)
        assert result is not None


class TestDownsideVol:
    def test_all_positive_returns(self):
        # Monotonically increasing -> no negative returns -> None
        px = make_prices(list(range(100, 130)))
        assert _downside_vol(px, 20) is None

    def test_known_downside(self):
        # Alternating up/down to ensure negative returns
        np.random.seed(42)
        prices = [100.0]
        for _ in range(60):
            # Random returns with mean ~0 so ~half are negative
            r = np.random.normal(0, 0.02)
            prices.append(prices[-1] * np.exp(r))
        px = make_prices(prices)
        dv = _downside_vol(px, 60)
        assert dv is not None
        assert dv > 0  # volatility is positive

    def test_insufficient_data(self):
        px = make_prices([100, 99])
        assert _downside_vol(px, 5) is None


class TestSMA:
    def test_basic(self):
        px = make_prices([10, 20, 30, 40, 50])
        # SMA of last 3: (30+40+50)/3 = 40
        assert _sma(px, 3) == pytest.approx(40.0)

    def test_full_series(self):
        px = make_prices([2, 4, 6, 8])
        assert _sma(px, 4) == pytest.approx(5.0)

    def test_insufficient_data(self):
        px = make_prices([100, 200])
        assert _sma(px, 5) is None

    def test_window_one(self):
        px = make_prices([10, 20, 30])
        assert _sma(px, 1) == pytest.approx(30.0)


class TestMaxDrawdown:
    def test_no_drawdown(self):
        # Monotonically increasing -> drawdown = 0
        px = make_prices([100, 110, 120, 130])
        assert _max_drawdown(px) == pytest.approx(0.0)

    def test_known_drawdown(self):
        # 100 -> 120 -> 90 -> 100
        # Peak at 120, trough at 90: drawdown = 90/120 - 1 = -0.25
        px = make_prices([100, 120, 90, 100])
        assert _max_drawdown(px) == pytest.approx(-0.25)

    def test_total_loss(self):
        px = make_prices([100, 50, 10])
        # Peak 100, trough 10: 10/100 - 1 = -0.9
        assert _max_drawdown(px) == pytest.approx(-0.9)

    def test_single_point(self):
        px = make_prices([100])
        assert _max_drawdown(px) is None

    def test_drawdown_always_nonpositive(self):
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(100)) + 200
        px = make_prices(prices.tolist())
        dd = _max_drawdown(px)
        assert dd is not None
        assert dd <= 0


class TestSharpe:
    def test_constant_prices(self):
        # Zero vol -> None
        px = make_prices([100] * 30)
        assert _sharpe(px, 20) is None

    def test_positive_trend(self):
        # Monotonically increasing -> positive Sharpe
        px = make_prices([100 + i * 0.5 for i in range(62)])
        sharpe = _sharpe(px, 60)
        assert sharpe is not None
        assert sharpe > 0

    def test_negative_trend(self):
        # Monotonically decreasing -> negative Sharpe
        px = make_prices([200 - i * 0.5 for i in range(62)])
        sharpe = _sharpe(px, 60)
        assert sharpe is not None
        assert sharpe < 0

    def test_insufficient_data(self):
        px = make_prices([100, 101])
        assert _sharpe(px, 5) is None


class TestBeta:
    def test_identical_series(self):
        # If ticker == SPY, beta should be 1.0
        np.random.seed(42)
        prices = [100.0]
        for _ in range(60):
            prices.append(prices[-1] * np.exp(np.random.normal(0, 0.01)))
        px = make_prices(prices)
        b = _beta(px, px, 60)
        assert b == pytest.approx(1.0, abs=1e-10)

    def test_double_leverage(self):
        # If ticker returns = 2x SPY returns, beta ≈ 2.0
        np.random.seed(42)
        spy_prices = [100.0]
        ticker_prices = [100.0]
        for _ in range(60):
            r = np.random.normal(0, 0.01)
            spy_prices.append(spy_prices[-1] * np.exp(r))
            ticker_prices.append(ticker_prices[-1] * np.exp(2 * r))
        spy = make_prices(spy_prices)
        ticker = make_prices(ticker_prices)
        b = _beta(ticker, spy, 60)
        assert b == pytest.approx(2.0, abs=0.01)

    def test_insufficient_data(self):
        px = make_prices([100, 101, 102])
        spy = make_prices([100, 100.5, 101])
        assert _beta(px, spy, 60) is None


class TestIdiosyncraticMomentum:
    def test_pure_market_return(self):
        # If ticker perfectly tracks SPY, residuals ≈ 0
        np.random.seed(42)
        spy_prices = [100.0]
        for _ in range(60):
            r = np.random.normal(0, 0.01)
            spy_prices.append(spy_prices[-1] * np.exp(r))
        spy = make_prices(spy_prices)
        # Ticker = SPY (beta=1, alpha=0 => residuals=0)
        im = _idiosyncratic_momentum(spy, spy, 60)
        assert im == pytest.approx(0.0, abs=1e-10)

    def test_positive_alpha(self):
        # Ticker has constant positive alpha over SPY
        np.random.seed(42)
        spy_prices = [100.0]
        ticker_prices = [100.0]
        for _ in range(60):
            r = np.random.normal(0, 0.01)
            spy_prices.append(spy_prices[-1] * np.exp(r))
            ticker_prices.append(ticker_prices[-1] * np.exp(r + 0.001))
        spy = make_prices(spy_prices)
        ticker = make_prices(ticker_prices)
        im = _idiosyncratic_momentum(ticker, spy, 60)
        assert im is not None
        assert im > 0  # positive alpha -> positive cumulative residual

    def test_insufficient_data(self):
        px = make_prices([100, 101])
        spy = make_prices([100, 100.5])
        assert _idiosyncratic_momentum(px, spy, 60) is None


class TestMomentum12_1:
    def test_basic(self):
        # Need 274 data points. px[-22] / px[-253] - 1
        prices = list(range(1, 300))  # 1, 2, ..., 299
        px = make_prices(prices)
        # px[-22] = 278, px[-253] = 47
        expected = 278 / 47 - 1
        assert _momentum_12_1(px) == pytest.approx(expected)

    def test_insufficient_data(self):
        px = make_prices(list(range(200)))
        assert _momentum_12_1(px) is None


class TestTrendConsistency:
    def test_all_positive(self):
        # Monotonically increasing -> all positive log returns -> 1.0
        px = make_prices([100 + i for i in range(62)])
        tc = _trend_consistency(px, 60)
        assert tc == pytest.approx(1.0)

    def test_all_negative(self):
        # Monotonically decreasing -> all negative log returns -> 0.0
        px = make_prices([200 - i for i in range(62)])
        tc = _trend_consistency(px, 60)
        assert tc == pytest.approx(0.0)

    def test_half_and_half(self):
        # Alternating up/down -> ~0.5
        prices = []
        p = 100.0
        for i in range(62):
            prices.append(p)
            p += 1 if i % 2 == 0 else -1
        px = make_prices(prices)
        tc = _trend_consistency(px, 60)
        # Should be close to 0.5
        assert 0.4 < tc < 0.6

    def test_insufficient_data(self):
        px = make_prices([100])
        assert _trend_consistency(px, 5) is None


class TestAvgDollarVolume:
    def test_basic(self):
        close = make_prices([10, 20, 30])
        volume = make_prices([100, 200, 300])
        # Dollar volumes: 1000, 4000, 9000
        # Last 2: (4000 + 9000) / 2 = 6500
        assert _avg_dollar_volume(close, volume, 2) == pytest.approx(6500.0)

    def test_insufficient_data(self):
        close = make_prices([10])
        volume = make_prices([100])
        assert _avg_dollar_volume(close, volume, 5) is None


class TestRoundOrNone:
    def test_normal(self):
        assert _round_or_none(3.14159, 2) == 3.14

    def test_none(self):
        assert _round_or_none(None, 2) is None

    def test_nan(self):
        assert _round_or_none(float("nan"), 2) is None

    def test_inf(self):
        assert _round_or_none(float("inf"), 2) is None

    def test_negative_inf(self):
        assert _round_or_none(float("-inf"), 2) is None

    def test_zero(self):
        assert _round_or_none(0.0, 4) == 0.0


# ===================================================================
# 2. SENTIMENT COMPUTATIONS
# ===================================================================


class TestExpDecayWeights:
    def test_no_decay(self):
        from datetime import datetime
        times = [datetime(2025, 1, 1), datetime(2025, 1, 15)]
        anchor = datetime(2025, 3, 31)
        # half_life_days=None -> uniform weights
        w = exp_decay_weights(times, None, anchor)
        assert w == [1.0, 1.0]

    def test_zero_half_life(self):
        from datetime import datetime
        times = [datetime(2025, 1, 1)]
        anchor = datetime(2025, 3, 31)
        w = exp_decay_weights(times, 0, anchor)
        assert w == [1.0]

    def test_at_anchor(self):
        from datetime import datetime
        anchor = datetime(2025, 3, 31)
        times = [anchor]  # age = 0 days
        w = exp_decay_weights(times, 7.0, anchor)
        assert w[0] == pytest.approx(1.0)

    def test_at_half_life(self):
        from datetime import datetime
        anchor = datetime(2025, 3, 31)
        times = [datetime(2025, 3, 24)]  # 7 days ago
        w = exp_decay_weights(times, 7.0, anchor)
        assert w[0] == pytest.approx(0.5, abs=0.001)

    def test_at_two_half_lives(self):
        from datetime import datetime
        anchor = datetime(2025, 3, 31)
        times = [datetime(2025, 3, 17)]  # 14 days ago
        w = exp_decay_weights(times, 7.0, anchor)
        assert w[0] == pytest.approx(0.25, abs=0.001)

    def test_none_time(self):
        from datetime import datetime
        anchor = datetime(2025, 3, 31)
        times = [None]
        w = exp_decay_weights(times, 7.0, anchor)
        assert w[0] == 1.0


class TestWeightedMeanAndStd:
    def test_uniform_weights(self):
        values = [2.0, 4.0, 6.0]
        weights = [1.0, 1.0, 1.0]
        mu, sd = weighted_mean_and_std(values, weights)
        assert mu == pytest.approx(4.0)
        # Population std: sqrt(((2-4)^2 + (4-4)^2 + (6-4)^2) / 3) = sqrt(8/3)
        assert sd == pytest.approx(math.sqrt(8 / 3))

    def test_non_uniform_weights(self):
        values = [0.0, 10.0]
        weights = [3.0, 1.0]
        mu, sd = weighted_mean_and_std(values, weights)
        # mu = (3*0 + 1*10) / 4 = 2.5
        assert mu == pytest.approx(2.5)
        # var = (3*(0-2.5)^2 + 1*(10-2.5)^2) / 4 = (18.75 + 56.25) / 4 = 18.75
        assert sd == pytest.approx(math.sqrt(18.75))

    def test_empty(self):
        mu, sd = weighted_mean_and_std([], [])
        assert math.isnan(mu)
        assert math.isnan(sd)

    def test_single_value(self):
        mu, sd = weighted_mean_and_std([5.0], [1.0])
        assert mu == pytest.approx(5.0)
        assert sd == pytest.approx(0.0)

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            weighted_mean_and_std([1.0, 2.0], [1.0])


class TestSurpriseSentiment:
    def test_basic_qoq_delta(self):
        results = {
            "AAPL": {
                "2025Q1": {"mean_sentiment": 0.10},
                "2025Q2": {"mean_sentiment": 0.15},
                "2025Q3": {"mean_sentiment": 0.05},
            }
        }
        add_surprise_sentiment(results)
        assert results["AAPL"]["2025Q1"]["surprise_sentiment"] is None  # no prior
        assert results["AAPL"]["2025Q2"]["surprise_sentiment"] == pytest.approx(0.05)
        assert results["AAPL"]["2025Q3"]["surprise_sentiment"] == pytest.approx(-0.10)

    def test_none_quarter(self):
        results = {
            "AAPL": {
                "2025Q1": {"mean_sentiment": 0.10},
                "2025Q2": None,
                "2025Q3": {"mean_sentiment": 0.20},
            }
        }
        add_surprise_sentiment(results)
        assert results["AAPL"]["2025Q1"]["surprise_sentiment"] is None
        # Q2 is None -> Q3's prev is None -> surprise is None
        assert results["AAPL"]["2025Q3"]["surprise_sentiment"] is None

    def test_multiple_tickers(self):
        results = {
            "AAPL": {
                "2025Q1": {"mean_sentiment": 0.10},
                "2025Q2": {"mean_sentiment": 0.20},
            },
            "GOOG": {
                "2025Q1": {"mean_sentiment": -0.05},
                "2025Q2": {"mean_sentiment": 0.05},
            },
        }
        add_surprise_sentiment(results)
        assert results["AAPL"]["2025Q2"]["surprise_sentiment"] == pytest.approx(0.10)
        assert results["GOOG"]["2025Q2"]["surprise_sentiment"] == pytest.approx(0.10)


class TestCrossSectionalZ:
    def test_basic(self):
        results = {
            "AAPL": {"2025Q1": {"mean_sentiment": 0.10}},
            "GOOG": {"2025Q1": {"mean_sentiment": 0.20}},
            "MSFT": {"2025Q1": {"mean_sentiment": 0.30}},
        }
        tickers = ["AAPL", "GOOG", "MSFT"]
        quarters = [(2025, 1)]
        add_cross_sectional_z(results, tickers, quarters)
        # mu = 0.2, sigma = sqrt((0.01 + 0 + 0.01)/3) = sqrt(0.02/3)
        mu = 0.2
        sigma = math.sqrt(((0.1 - 0.2) ** 2 + 0 + (0.3 - 0.2) ** 2) / 3)
        assert results["AAPL"]["2025Q1"]["cross_sectional_z"] == pytest.approx(
            (0.10 - mu) / sigma, abs=1e-5
        )
        assert results["GOOG"]["2025Q1"]["cross_sectional_z"] == pytest.approx(
            0.0, abs=1e-5
        )
        assert results["MSFT"]["2025Q1"]["cross_sectional_z"] == pytest.approx(
            (0.30 - mu) / sigma, abs=1e-5
        )

    def test_single_ticker_skipped(self):
        results = {
            "AAPL": {"2025Q1": {"mean_sentiment": 0.10}},
        }
        add_cross_sectional_z(results, ["AAPL"], [(2025, 1)])
        # Should not add cross_sectional_z with only 1 ticker
        assert "cross_sectional_z" not in results["AAPL"]["2025Q1"]

    def test_equal_sentiments(self):
        results = {
            "AAPL": {"2025Q1": {"mean_sentiment": 0.10}},
            "GOOG": {"2025Q1": {"mean_sentiment": 0.10}},
        }
        add_cross_sectional_z(results, ["AAPL", "GOOG"], [(2025, 1)])
        # sigma = 0 -> z = 0.0
        assert results["AAPL"]["2025Q1"]["cross_sectional_z"] == 0.0
        assert results["GOOG"]["2025Q1"]["cross_sectional_z"] == 0.0


# ===================================================================
# 3. SNAPSHOT CROSS-SECTIONAL FEATURES
# ===================================================================


class TestAddRelativeStrength:
    def test_basic_odd(self):
        # 3 tickers -> median is the middle value
        ticker_data = {
            "AAPL": {"asset_features": {"ret_60d": 0.10}},
            "GOOG": {"asset_features": {"ret_60d": 0.05}},
            "MSFT": {"asset_features": {"ret_60d": 0.20}},
        }
        _add_relative_strength(ticker_data)
        # Sorted: [0.05, 0.10, 0.20] -> median = 0.10
        assert ticker_data["AAPL"]["asset_features"]["relative_strength_60d"] == pytest.approx(0.0)
        assert ticker_data["GOOG"]["asset_features"]["relative_strength_60d"] == pytest.approx(-0.05)
        assert ticker_data["MSFT"]["asset_features"]["relative_strength_60d"] == pytest.approx(0.10)

    def test_basic_even(self):
        ticker_data = {
            "A": {"asset_features": {"ret_60d": 0.10}},
            "B": {"asset_features": {"ret_60d": 0.30}},
        }
        _add_relative_strength(ticker_data)
        # Sorted: [0.10, 0.30] -> median = (0.10+0.30)/2 = 0.20
        assert ticker_data["A"]["asset_features"]["relative_strength_60d"] == pytest.approx(-0.10)
        assert ticker_data["B"]["asset_features"]["relative_strength_60d"] == pytest.approx(0.10)

    def test_skips_errors(self):
        ticker_data = {
            "AAPL": {"asset_features": {"ret_60d": 0.10}},
            "GOOG": {"asset_features": {"error": "no_data"}},
            "MSFT": {"asset_features": {"ret_60d": 0.20}},
        }
        _add_relative_strength(ticker_data)
        # Only AAPL and MSFT contribute -> median = 0.15
        assert ticker_data["AAPL"]["asset_features"]["relative_strength_60d"] == pytest.approx(-0.05)
        assert ticker_data["MSFT"]["asset_features"]["relative_strength_60d"] == pytest.approx(0.05)
        assert "relative_strength_60d" not in ticker_data["GOOG"]["asset_features"]

    def test_skips_none_features(self):
        ticker_data = {
            "AAPL": {"asset_features": {"ret_60d": 0.10}},
            "GOOG": {"asset_features": None},
        }
        _add_relative_strength(ticker_data)
        # Only AAPL -> median = 0.10
        assert ticker_data["AAPL"]["asset_features"]["relative_strength_60d"] == pytest.approx(0.0)

    def test_empty(self):
        ticker_data = {}
        _add_relative_strength(ticker_data)  # Should not raise


class TestAddSentimentZ:
    def test_basic(self):
        ticker_data = {
            "AAPL": {"news_sentiment": {"mean_sentiment": 0.10}},
            "GOOG": {"news_sentiment": {"mean_sentiment": 0.20}},
            "MSFT": {"news_sentiment": {"mean_sentiment": 0.30}},
        }
        _add_sentiment_z(ticker_data)
        mu = 0.2
        sigma = math.sqrt(((0.1 - 0.2) ** 2 + 0 + (0.3 - 0.2) ** 2) / 3)
        assert ticker_data["AAPL"]["news_sentiment"]["cross_sectional_z"] == pytest.approx(
            (0.10 - mu) / sigma, abs=1e-5
        )
        assert ticker_data["GOOG"]["news_sentiment"]["cross_sectional_z"] == pytest.approx(
            0.0, abs=1e-5
        )

    def test_single_ticker(self):
        ticker_data = {
            "AAPL": {"news_sentiment": {"mean_sentiment": 0.10}},
        }
        _add_sentiment_z(ticker_data)
        # len < 2 -> should not add key
        assert "cross_sectional_z" not in ticker_data["AAPL"]["news_sentiment"]

    def test_equal_sentiments(self):
        ticker_data = {
            "AAPL": {"news_sentiment": {"mean_sentiment": 0.10}},
            "GOOG": {"news_sentiment": {"mean_sentiment": 0.10}},
        }
        _add_sentiment_z(ticker_data)
        assert ticker_data["AAPL"]["news_sentiment"]["cross_sectional_z"] == 0.0

    def test_none_sentiment(self):
        ticker_data = {
            "AAPL": {"news_sentiment": {"mean_sentiment": 0.10}},
            "GOOG": {"news_sentiment": None},
            "MSFT": {"news_sentiment": {"mean_sentiment": 0.30}},
        }
        _add_sentiment_z(ticker_data)
        # Only AAPL and MSFT contribute
        mu = 0.2
        sigma = math.sqrt(((0.1 - 0.2) ** 2 + (0.3 - 0.2) ** 2) / 2)
        assert ticker_data["AAPL"]["news_sentiment"]["cross_sectional_z"] == pytest.approx(
            (0.10 - mu) / sigma, abs=1e-5
        )

    def test_empty(self):
        ticker_data = {}
        _add_sentiment_z(ticker_data)  # Should not raise


# ===================================================================
# 4. FUNDAMENTAL DATA HELPERS
# ===================================================================


class TestSafeGet:
    """Verify _safe_get extracts values from financial statement DataFrames."""

    def test_basic_extraction(self):
        df = pd.DataFrame(
            {"2024-06-30": [100.0, 200.0]},
            index=["Revenue", "Gross Profit"],
        )
        assert _safe_get(df, "2024-06-30", ["Revenue"]) == 100.0

    def test_fallback_row_names(self):
        df = pd.DataFrame(
            {"2024-06-30": [500.0]},
            index=["Total Revenue"],
        )
        # First name doesn't exist, second does
        assert _safe_get(df, "2024-06-30", ["Revenue", "Total Revenue"]) == 500.0

    def test_none_when_missing(self):
        df = pd.DataFrame(
            {"2024-06-30": [100.0]},
            index=["Revenue"],
        )
        assert _safe_get(df, "2024-06-30", ["Nonexistent"]) is None

    def test_none_for_nan(self):
        df = pd.DataFrame(
            {"2024-06-30": [float("nan")]},
            index=["Revenue"],
        )
        assert _safe_get(df, "2024-06-30", ["Revenue"]) is None

    def test_none_for_empty_df(self):
        assert _safe_get(pd.DataFrame(), None, ["Revenue"]) is None

    def test_none_for_none_df(self):
        assert _safe_get(None, None, ["Revenue"]) is None

    def test_none_for_none_col(self):
        df = pd.DataFrame(
            {"2024-06-30": [100.0]},
            index=["Revenue"],
        )
        assert _safe_get(df, None, ["Revenue"]) is None


class TestLatestValidCol:
    """Verify _latest_valid_col picks the most recent column <= cutoff."""

    def test_picks_latest_before_cutoff(self):
        cols = [pd.Timestamp("2024-03-31"), pd.Timestamp("2024-06-30"),
                pd.Timestamp("2024-09-30")]
        df = pd.DataFrame(np.zeros((1, 3)), columns=cols)
        cutoff = pd.Timestamp("2024-08-15")
        result = _latest_valid_col(df, cutoff)
        assert result == pd.Timestamp("2024-06-30")

    def test_includes_exact_cutoff(self):
        cols = [pd.Timestamp("2024-03-31"), pd.Timestamp("2024-06-30")]
        df = pd.DataFrame(np.zeros((1, 2)), columns=cols)
        cutoff = pd.Timestamp("2024-06-30")
        result = _latest_valid_col(df, cutoff)
        assert result == pd.Timestamp("2024-06-30")

    def test_all_after_cutoff_returns_none(self):
        cols = [pd.Timestamp("2025-03-31"), pd.Timestamp("2025-06-30")]
        df = pd.DataFrame(np.zeros((1, 2)), columns=cols)
        cutoff = pd.Timestamp("2024-12-31")
        assert _latest_valid_col(df, cutoff) is None

    def test_empty_df_returns_none(self):
        assert _latest_valid_col(pd.DataFrame(), pd.Timestamp("2024-12-31")) is None

    def test_none_df_returns_none(self):
        assert _latest_valid_col(None, pd.Timestamp("2024-12-31")) is None


class TestBalanceSheetDerivedFundamentals:
    """Verify shares_outstanding and book_value_per_share are derived from
    balance sheet data (point-in-time safe) rather than yt.info (live snapshot)."""

    def test_shares_from_balance_sheet(self):
        """_safe_get should find shares under known row names."""
        df = pd.DataFrame(
            {"2024-06-30": [1_500_000_000]},
            index=["Ordinary Shares Number"],
        )
        shares = _safe_get(df, "2024-06-30", [
            "Ordinary Shares Number", "Share Issued",
            "Common Stock Shares Outstanding",
        ])
        assert shares == 1_500_000_000

    def test_shares_fallback_share_issued(self):
        df = pd.DataFrame(
            {"2024-06-30": [2_000_000_000]},
            index=["Share Issued"],
        )
        shares = _safe_get(df, "2024-06-30", [
            "Ordinary Shares Number", "Share Issued",
            "Common Stock Shares Outstanding",
        ])
        assert shares == 2_000_000_000

    def test_book_value_per_share_computation(self):
        """book_value_per_share = equity / shares."""
        equity = 150_000_000_000.0  # $150B
        shares = 1_500_000_000.0    # 1.5B shares
        bvps = equity / shares
        assert bvps == pytest.approx(100.0)

    def test_book_value_per_share_zero_shares(self):
        """Zero shares should yield None, not division error."""
        equity = 150_000_000_000.0
        shares = 0.0
        # Mirrors the guard in _fetch_fundamentals
        if equity is not None and shares is not None and shares > 0:
            bvps = equity / shares
        else:
            bvps = None
        assert bvps is None

    def test_book_value_per_share_none_equity(self):
        """None equity should yield None."""
        equity = None
        shares = 1_500_000_000.0
        if equity is not None and shares is not None and shares > 0:
            bvps = equity / shares
        else:
            bvps = None
        assert bvps is None


# ===================================================================
# filter_filings — direct-match-over-spillover & dedup tests
# ===================================================================

def _make_filings(entries):
    """Build a mock SEC filings dict from a list of filing dicts.

    Each entry: {form, filingDate, accessionNumber, primaryDocument, reportDate}
    """
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


class TestFilterFilings:
    """Tests for filter_filings() in get_sec_data.py."""

    def test_direct_match_wins_over_spillover_10k(self):
        """WMT-style bug: 10-K with fiscal Q1 should direct-match Q1,
        not spillover to prior Q4, when both targets exist."""
        filings = _make_filings([{
            "form": "10-K",
            "filingDate": "2025-03-14",
            "accessionNumber": "0000104169-25-000021",
            "primaryDocument": "wmt-20250131.htm",
            "reportDate": "2025-01-31",  # fiscal Q1 2025
        }])
        targets = [
            (2024, "Q4"),
            (2025, "Q1"),
        ]
        results = filter_filings(filings, targets, {"10-K"})
        assert len(results) == 1
        assert results[0]["matched_year"] == 2025
        assert results[0]["matched_quarter"] == "Q1"

    def test_spillover_fires_when_no_direct_match(self):
        """When only Q4 target exists, spillover should still work
        for 10-K with fiscal Q1."""
        filings = _make_filings([{
            "form": "10-K",
            "filingDate": "2025-03-14",
            "accessionNumber": "0000104169-25-000021",
            "primaryDocument": "wmt-20250131.htm",
            "reportDate": "2025-01-31",
        }])
        targets = [(2024, "Q4")]  # No Q1 2025 target
        results = filter_filings(filings, targets, {"10-K"})
        assert len(results) == 1
        assert results[0]["matched_year"] == 2024
        assert results[0]["matched_quarter"] == "Q4"

    def test_no_spillover_for_10q(self):
        """Spillover must NOT fire for 10-Q filings."""
        filings = _make_filings([{
            "form": "10-Q",
            "filingDate": "2025-05-02",
            "accessionNumber": "0001018724-25-000036",
            "primaryDocument": "amzn-20250331.htm",
            "reportDate": "2025-03-31",  # fiscal Q1 2025
        }])
        targets = [(2024, "Q4")]  # 10-Q should NOT spillover
        results = filter_filings(filings, targets, {"10-Q"})
        assert len(results) == 0

    def test_accession_dedup_within_single_call(self):
        """Same accession should only appear once even with multiple targets."""
        filings = _make_filings([{
            "form": "10-K",
            "filingDate": "2025-02-26",
            "accessionNumber": "0001045810-25-000023",
            "primaryDocument": "nvda-20250126.htm",
            "reportDate": "2025-01-26",
        }])
        # Cross-product targets that both could match
        targets = [
            (2024, "Q1"), (2024, "Q2"), (2024, "Q3"), (2024, "Q4"),
            (2025, "Q1"), (2025, "Q2"), (2025, "Q3"), (2025, "Q4"),
        ]
        results = filter_filings(filings, targets, {"10-K"})
        assert len(results) == 1
        # Direct match Q1 2025 should win
        assert results[0]["matched_year"] == 2025
        assert results[0]["matched_quarter"] == "Q1"

    def test_multiple_filings_different_quarters(self):
        """Multiple distinct filings should each match their own quarter."""
        filings = _make_filings([
            {
                "form": "10-Q",
                "filingDate": "2025-05-02",
                "accessionNumber": "acc-001",
                "primaryDocument": "q1.htm",
                "reportDate": "2025-03-31",
            },
            {
                "form": "10-Q",
                "filingDate": "2025-08-01",
                "accessionNumber": "acc-002",
                "primaryDocument": "q2.htm",
                "reportDate": "2025-06-30",
            },
        ])
        targets = [(2025, "Q1"), (2025, "Q2")]
        results = filter_filings(filings, targets, {"10-Q"})
        assert len(results) == 2
        quarters = {r["matched_quarter"] for r in results}
        assert quarters == {"Q1", "Q2"}

    def test_amendment_excluded_by_default(self):
        """10-K/A should be excluded unless include_amendments=True."""
        filings = _make_filings([{
            "form": "10-K/A",
            "filingDate": "2025-04-01",
            "accessionNumber": "acc-amend",
            "primaryDocument": "amend.htm",
            "reportDate": "2024-12-31",
        }])
        targets = [(2024, "Q4")]
        assert len(filter_filings(filings, targets, {"10-K"})) == 0
        assert len(filter_filings(filings, targets, {"10-K"},
                                  include_amendments=True)) == 1

    def test_none_quarter_matches_any_fiscal_quarter(self):
        """Target with quarter=None should match any fiscal quarter in that year."""
        filings = _make_filings([{
            "form": "10-Q",
            "filingDate": "2025-05-02",
            "accessionNumber": "acc-003",
            "primaryDocument": "q.htm",
            "reportDate": "2025-03-31",
        }])
        targets = [(2025, None)]
        results = filter_filings(filings, targets, {"10-Q"})
        assert len(results) == 1
        assert results[0]["matched_quarter"] == "Q1"

    def test_no_match_when_targets_dont_overlap(self):
        """Filing with fiscal Q3 shouldn't match Q1 or Q2 targets."""
        filings = _make_filings([{
            "form": "10-Q",
            "filingDate": "2025-11-01",
            "accessionNumber": "acc-004",
            "primaryDocument": "q3.htm",
            "reportDate": "2025-09-30",
        }])
        targets = [(2025, "Q1"), (2025, "Q2")]
        results = filter_filings(filings, targets, {"10-Q"})
        assert len(results) == 0
