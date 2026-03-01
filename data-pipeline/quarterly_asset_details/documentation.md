
# Asset Feature Builder — Data Flow & Methodology

This document describes the asset quarter builder (`asset_quarter_builder.py`), which produces per-ticker per-quarter JSON files containing ~28 price, volatility, momentum, and fundamental metrics.

The goal is:

* Reproducibility
* Strict point-in-time safety (no future leakage)
* Per-ticker output for incremental processing
* Clear separation of per-ticker features from cross-sectional features

---

# 1. Pipeline Overview

```
Yahoo Finance (yfinance)
  -> asset_quarter_builder.py
  -> data/{TICKER}/{YEAR}_Q{#}.json   (one file per ticker per quarter)
```

The script fetches price history and quarterly financial statements from yfinance, computes ~28 metrics per ticker, and writes one JSON file per ticker per quarter.

Cross-sectional features (relative_strength_60d) are computed at snapshot build time (Stage 6), not here. This keeps each file independent and allows adding a single ticker without reprocessing all others.

---

# 2. Output

## 2.1 Location

```
data-pipeline/quarterly_asset_details/data/{TICKER}/{YEAR}_Q{#}.json
```

Example: `data/AAPL/2025_Q1.json`

## 2.2 Schema

```json
{
  "schema_version": "asset_state_v2",
  "ticker": "AAPL",
  "year": 2025,
  "quarter": "Q1",
  "as_of": "2025-03-31",
  "features": {
    "close": 216.95,
    "ret_20d": -0.099,
    "ret_60d": -0.1351,
    "ret_120d": -0.0234,
    "ret_252d": 0.1823,
    "vol_20d": 0.2878,
    "vol_60d": 0.2747,
    "vol_120d": 0.2345,
    "downside_vol_60d": 0.1895,
    "drawdown_60d": -0.1984,
    "max_drawdown_1y": -0.2133,
    "sma_20d": 220.5,
    "sma_50d": 225.3,
    "sma_200d": 218.7,
    "momentum_200d": 0.9919,
    "momentum_12_1": 0.0812,
    "idiosyncratic_momentum": 0.023,
    "trend_consistency": 0.5667,
    "sharpe_60d": -1.4159,
    "beta_1y": 0.9742,
    "size_log_mcap": 26.12,
    "value_book_to_market": 0.0543,
    "avg_dollar_volume_20d": 12345678.90,
    "gross_margin": 0.4705,
    "roe": 0.3821,
    "free_cash_flow_yield": 0.0312,
    "debt_to_equity": 1.8721,
    "earnings_surprise_pct": 0.0219
  }
}
```

---

# 3. Feature Reference

## 3.1 Price & Returns

| Feature | Definition | Units |
|---|---|---|
| `close` | Last closing price on or before quarter end | USD |
| `ret_20d` | 20-trading-day total return | decimal |
| `ret_60d` | 60-trading-day total return | decimal |
| `ret_120d` | 120-trading-day total return | decimal |
| `ret_252d` | 252-trading-day total return (1 year) | decimal |

## 3.2 Volatility & Risk

| Feature | Definition | Units |
|---|---|---|
| `vol_20d` | 20-day annualized volatility (daily log returns * sqrt(252)) | decimal |
| `vol_60d` | 60-day annualized volatility | decimal |
| `vol_120d` | 120-day annualized volatility | decimal |
| `downside_vol_60d` | 60-day downside volatility (negative returns only) | decimal |
| `drawdown_60d` | Max drawdown over last 60 trading days | decimal (negative) |
| `max_drawdown_1y` | Max drawdown over last 252 trading days | decimal (negative) |
| `sharpe_60d` | 60-day Sharpe ratio (annualized return / annualized vol) | ratio |

## 3.3 Trend & Momentum

| Feature | Definition | Units |
|---|---|---|
| `sma_20d` | 20-day simple moving average | USD |
| `sma_50d` | 50-day simple moving average | USD |
| `sma_200d` | 200-day simple moving average | USD |
| `momentum_200d` | close / sma_200d (price relative to long-term trend) | ratio |
| `momentum_12_1` | 12-month return excluding last month (skip-month momentum) | decimal |
| `idiosyncratic_momentum` | Residual momentum after removing market beta (OLS residual cumulative return) | decimal |
| `trend_consistency` | Fraction of trailing monthly returns that are positive | decimal [0, 1] |

## 3.4 Market Structure

| Feature | Definition | Units |
|---|---|---|
| `beta_1y` | 1-year beta vs SPY (covariance / variance of SPY returns) | ratio |
| `size_log_mcap` | log(market cap) = log(close * shares_outstanding) | log USD |
| `value_book_to_market` | book_value_per_share / close | ratio |
| `avg_dollar_volume_20d` | Mean of (close * volume) over last 20 days | USD |

## 3.5 Fundamentals

All fundamental data is derived from quarterly financial statements via yfinance, filtered to fiscal periods <= quarter end for point-in-time safety.

| Feature | Source | Definition |
|---|---|---|
| `gross_margin` | Income statement | Gross Profit / Total Revenue |
| `roe` | Income + Balance sheet | Net Income / Stockholders Equity |
| `free_cash_flow_yield` | Cashflow + Balance sheet | FCF / Market Cap |
| `debt_to_equity` | Balance sheet | Total Debt / Stockholders Equity |
| `earnings_surprise_pct` | Earnings dates | (Actual EPS - Estimated EPS) / \|Estimated EPS\| |

**Point-in-time safety for fundamentals:**

- `shares_outstanding` and `book_value_per_share` are derived from the quarterly balance sheet (not from `yt.info`, which returns live data)
- `_latest_valid_col(df, cutoff)` selects the most recent financial statement column with date <= quarter end
- `_safe_get(df, col, row_names)` extracts values with multiple fallback row names for cross-company compatibility

## 3.6 Cross-Sectional (computed at snapshot build time)

| Feature | Definition | Computed In |
|---|---|---|
| `relative_strength_60d` | ret_60d - median(all ret_60d in quarter) | `generate_quarterly_json.py` |

This feature requires all tickers to be present, so it is computed in the snapshot builder, not here.

---

# 4. Computation Details

## 4.1 Price History

A single yfinance download covers a 600-calendar-day lookback from the quarter end date. This ensures sufficient history for 252-trading-day windows plus the 21-day skip for `momentum_12_1`.

## 4.2 Beta Computation

```python
# Covariance matrix of asset returns vs SPY returns
cov = np.cov(asset_returns, spy_returns)
beta = cov[0, 1] / cov[1, 1]
```

Requires >= 60 overlapping daily returns. SPY is downloaded once per quarter and reused for all tickers.

## 4.3 Idiosyncratic Momentum

OLS regression of asset returns on SPY returns over 252 trading days, excluding the most recent 21 days:

```python
# alpha + beta * SPY_ret = asset_ret
# residuals = asset_ret - (alpha + beta * SPY_ret)
# idiosyncratic_momentum = cumulative product of (1 + residuals) - 1
```

## 4.4 Rounding

All output values are rounded at serialization time via `_round_or_none(val, decimals)`, which also maps NaN and Inf to None.

---

# 5. Usage

```bash
# All supported tickers, quarter range
python asset_quarter_builder.py --start 2024Q4 --end 2025Q3 --supported

# Specific tickers (e.g. adding a new one)
python asset_quarter_builder.py --start 2024Q4 --end 2025Q3 --tickers NFLX

# Single quarter
python asset_quarter_builder.py --year 2025 --quarter Q1 --supported
```

## CLI Arguments

| Argument | Description |
|---|---|
| `--start` | Start quarter, e.g. `2024Q4` |
| `--end` | End quarter, e.g. `2025Q3` |
| `--year` | Year (single quarter mode) |
| `--quarter` | Quarter: Q1, Q2, Q3, Q4 (single quarter mode) |
| `--tickers` | Comma-separated tickers |
| `--supported` | Use all tickers from `supported_tickers.yaml` |

---

# 6. Point-in-Time Safety

* **Price data**: yfinance download uses `end = quarter_end + 1 day` (exclusive), so no data after quarter end is included
* **Financial statements**: `_latest_valid_col()` filters to columns with dates <= quarter end. This means only reported (not future) fiscal periods are used
* **Shares outstanding**: Derived from quarterly balance sheet (`Ordinary Shares Number` / `Share Issued`), not from the live `yt.info` snapshot
* **Book value per share**: Computed as `equity / shares` from the same balance sheet column — point-in-time safe
* **Earnings surprise**: `earnings_dates` filtered to dates <= quarter end cutoff

---

# 7. Limitations

* **yfinance reliability**: Rate limits and MultiIndex errors can cause intermittent failures. Retry typically resolves.
* **Financial statement row names**: Different companies use different row names (e.g., "Total Revenue" vs "Revenue"). The `_safe_get()` function handles this with fallback lists, but novel names may return None.
* **Earnings dates**: yfinance `earnings_dates` may not include estimates for all companies, resulting in null `earnings_surprise_pct`.
* **No parallel mode**: The script processes tickers sequentially within each quarter.
