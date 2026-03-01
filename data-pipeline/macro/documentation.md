
# Macro Market State — Data Flow & Methodology

This document describes the augmented market state builder (`metrics_gather.py`), which produces a multi-layer JSON snapshot of macro, volatility, market internals, and price data for each quarter.

The goal is:

* Reproducibility
* Transparent data provenance
* Quarterly granularity with QoQ/YoY deltas
* Incremental caching for efficient re-runs

---

# 1. Pipeline Overview

```
FRED API  +  Yahoo Finance  +  Wikipedia (S&P 500 list)
  -> metrics_gather.py
  -> augmented_market_state_v3.json
```

The script produces a single JSON file containing four layers of market data for a given year:

| Layer | Name | Source | Scope |
|-------|------|--------|-------|
| L1 | Macro Regime | FRED | Rates, curve, inflation, growth, credit, liquidity, labor, FX, commodities |
| L4 | Vol & Risk Pricing | FRED + Yahoo Finance | VIX, MOVE, SKEW, equity-bond correlation |
| L5 | Market Internals | Wikipedia + Yahoo Finance | Breadth (% > 200DMA), concentration (top-10 cap share), equal-weight vs cap-weight |
| L6 | Price Summary | Yahoo Finance | Per-ticker: current price, 60D return/vol/maxDD, SMA20/SMA50, daily bars |

---

# 2. Layer 1: Macro Regime

## 2.1 FRED Series Map

| Metric Key | FRED Series | Description | Units |
|-----------|-------------|-------------|-------|
| L1-MON-FF | DFF | Effective Federal Funds Rate | % |
| L1-MON-2Y | DGS2 | 2-Year Treasury Yield | % |
| L1-MON-10Y | DGS10 | 10-Year Treasury Yield | % |
| L1-MON-30Y | DGS30 | 30-Year Treasury Yield | % |
| L1-MON-REAL | DFII10 | 10-Year TIPS (Real Yield) | % |
| L1-MON-CURVE | computed | 10Y - 2Y spread | percentage points |
| L1-LIQ-NET | computed | WALCL - WDTGAL - RRPONTSYD | USD |
| L1-INF-CPI | CPIAUCSL | CPI YoY (computed from index) | % YoY |
| L1-INF-CORE | CPILFESL | Core CPI YoY (computed from index) | % YoY |
| L1-GRW-ISM | NAPM | ISM Manufacturing PMI | index |
| L1-GRW-SRV | NAPMNOI | ISM Services Index | index |
| L1-CRED-HY | BAMLH0A0HYM2 | HY OAS | % |
| L1-CRED-DEF_PROXY | BAMLH0A3HYC | HY CCC OAS (proxy) | % |
| L1-LAB-UNRATE | UNRATE | Unemployment Rate | % |
| L1-LAB-ICSA | ICSA | Initial Jobless Claims | count |
| L1-FX-DXY | DTWEXBGS | Broad USD Index | index |
| L1-COM-WTI | DCOILWTICO | WTI Crude Oil | USD/bbl |

## 2.2 Net Liquidity Computation

```
Net Liquidity = WALCL (Fed Total Assets)
              - WDTGAL (Treasury General Account)
              - RRPONTSYD (Overnight Reverse Repo)
```

All three components are fetched from FRED and aligned on a daily index. Deltas are computed in USD (not basis points).

## 2.3 CPI YoY Computation

CPI and Core CPI are reported as indices, not rates. YoY inflation is computed as:

```
cpi_yoy = (CPI_index_now / CPI_index_1yr_ago - 1) * 100
```

The comparison date is the same calendar date one year prior to the quarter end.

## 2.4 Deltas

For rate-type metrics (yields, spreads, OAS), deltas are computed in basis points:

* **QoQ delta**: `(value_now - value_prev_quarter_end) * 100 bps`
* **YoY delta**: `(value_now - value_same_quarter_last_year) * 100 bps`

---

# 3. Layer 4: Volatility & Risk Pricing

## 3.1 FRED Series

| Metric Key | FRED Series | Description |
|-----------|-------------|-------------|
| L4-VIX | VIXCLS | CBOE VIX Index |
| L4-MOVE | MOVE | Merrill Lynch MOVE Index |
| L4-SKEW | SKEW | CBOE SKEW Index |

## 3.2 Equity-Bond Correlation

Computed from daily log returns of SPY and TLT over the last ~60 trading days of each quarter:

1. Fetch SPY and TLT daily prices (120 calendar days lookback to ensure ~60 trading days)
2. Compute log returns: `ln(close_t / close_{t-1})`
3. Compute Pearson correlation of SPY and TLT log returns
4. Requires at least 30 observations; null otherwise

---

# 4. Layer 5: Market Internals

## 4.1 Breadth: % Above 200DMA

1. Fetch S&P 500 constituent tickers from Wikipedia
2. Deterministic sampling: every k-th ticker (configurable via `--breadth-sample`, `--breadth-max`)
3. For each sampled ticker at each quarter end:
   - Download 420 calendar days of price history (ensures 200+ trading days)
   - Compute 200-day SMA from Adjusted Close
   - Check if last close > 200DMA
4. Report: `% above = (above_count / valid_count) * 100`

## 4.2 Concentration: Top-10 Market Cap Share

1. For sampled tickers (up to 150), fetch current market cap via `yfinance`
2. Sort descending by market cap
3. Compute: `top10_share = sum(top_10_mcap) / sum(all_mcap) * 100`

This is approximate: uses current market cap snapshot, not historical quarter-end values.

## 4.3 Equal-Weight vs Cap-Weight

Proxy: RSP (S&P 500 Equal Weight ETF) vs SPY quarterly total return.

```
diff = RSP_quarterly_return - SPY_quarterly_return (in percentage points)
```

Negative diff indicates cap-weighted outperformance (narrow market).

---

# 5. Layer 6: Price Summary

## 5.1 Single-Download Optimization

For each ticker, price history is downloaded **once** for the entire `[start_date, end_date]` window, then sliced per-quarter from the preloaded DataFrame. This eliminates redundant API calls.

## 5.2 Per-Quarter Metrics

| Metric Key | Description | Units |
|-----------|-------------|-------|
| CURRENT_PRICE | Last close on or before quarter end | USD |
| PX-RET60 | 60-trading-day return | % |
| PX-VOL60 | 60-day annualized volatility | fraction |
| PX-DD60 | Max drawdown over ~60 trading days | fraction |
| PX-SMA20 | 20-day SMA at quarter end | USD |
| PX-SMA50 | 50-day SMA at quarter end | USD |

## 5.3 Daily Bars

Each quarter includes a `daily_bars` array with `{timestamp, close}` entries for the quarter period only.

---

# 6. FRED Incremental CSV Cache

## 6.1 Mechanism

Each FRED series is cached as a flat CSV file next to the script:

```
metrics_gather.py
DFF_fred_cache.csv
DGS10_fred_cache.csv
VIXCLS_fred_cache.csv
```

Naming: `{SERIES_ID}_fred_cache.csv` or `{SERIES_ID}_{frequency}_fred_cache.csv`.

## 6.2 Incremental Fetch Logic

1. Load existing CSV (columns: `date`, `value`)
2. Determine cached date range `[cached_start, cached_end]`
3. If requested range extends before `cached_start`: fetch `[request_start, cached_start - 1 day]`
4. If requested range extends after `cached_end`: fetch `[cached_end + 1 day, request_end]`
5. Merge cached + new, deduplicate (keep latest), sort by date
6. Save updated CSV
7. Return only the requested `[start, end]` slice

This ensures:
* First run fetches everything from API
* Subsequent runs only fetch missing date ranges
* Cache grows monotonically over time
* No stale data: new ranges always fetched fresh

## 6.3 Historical Lookback Window

The `--back-years` argument (default: 2) controls how far back data is fetched:

```
start_date = date(year - back_years, 1, 1)
end_date = date(year, 12, 31)
```

This ensures the cache accumulates enough history for YoY deltas and long moving averages.

---

# 7. Output Schema

```json
{
  "schema_version": "augmented_market_state_v3",
  "generated_at_utc": "2025-01-15T12:00:00Z",
  "year": 2025,
  "tickers": ["AAPL", "NVDA"],
  "layers": {
    "L1": {
      "layer": "L1",
      "year": 2025,
      "quarters": {
        "Q1": {
          "asof": "2025-03-31",
          "metrics": {
            "L1-MON-FF": {"value": 4.33, "units": "%"},
            "L1-MON-10Y": {"value": 4.21, "units": "%", "delta_bps_qoq": -12.0, "delta_bps_yoy": -5.0}
          }
        }
      },
      "errors": {},
      "notes": {}
    },
    "L4": { "..." : "..." },
    "L5": { "..." : "..." },
    "L6": {
      "layer": "L6",
      "tickers": {
        "AAPL": {
          "quarters": {
            "Q1": {
              "asof": "2025-03-31",
              "metrics": {
                "CURRENT_PRICE": {"value": 178.5, "units": "USD"},
                "PX-RET60": {"value": 5.2, "units": "%"}
              },
              "daily_bars": [{"timestamp": "2025-01-02", "close": 170.1}]
            }
          }
        }
      }
    }
  },
  "coverage": {
    "L1": {"metric_slots": 72, "nonnull": 68},
    "L4": {"metric_slots": 16, "nonnull": 14}
  }
}
```

The `coverage` section counts total metric slots and non-null values for quick quality assessment.

---

# 8. Example Commands

```bash
# Basic run for 2025 with default settings
python macro_quarter_builder.py --year 2025 --tickers AAPL,NVDA,MSFT

# With FRED API key and custom output
FRED_API_KEY=your_key python macro_quarter_builder.py \
    --year 2025 \
    --tickers AAPL,NVDA,MSFT,GOOG,AMZN,META,JPM,GS \
    --out data/augmented_market_state_v3.json

# Fast run: skip breadth computation (fewer S&P 500 tickers)
python macro_quarter_builder.py --year 2025 --tickers AAPL --breadth-sample 50 --breadth-max 60

# Extended lookback for deeper history
python macro_quarter_builder.py --year 2025 --tickers AAPL --back-years 5
```

---

# 9. Integration with Quarterly Snapshot Builder

The snapshot builder reads from `augmented_market_state_v3.json` via `load_macro()`:

* Extracts Layer 1 (macro_metrics), Layer 4 (vol_metrics), and Layer 5 (internals_metrics) for each quarter
* Matched by year and quarter label (Q1-Q4)
* Layer 6 (prices) is separately handled by the snapshot builder's price feature stubs

The macro data is market-wide (not per-ticker), providing regime context for all tickers in the universe.

---

# 10. Limitations

* **FRED availability**: Some series may be discontinued or have reporting lags. Errors are captured in the `errors` dict per layer.
* **Breadth approximation**: Uses sampled S&P 500 constituents, not the full index. Sampling is deterministic but approximate.
* **Concentration snapshot**: Uses current market caps, not historical quarter-end values.
* **Yahoo Finance reliability**: Rate limits and data gaps can cause missing quarters. Errors are logged per-ticker per-quarter.
* **No real-time data**: All values are as-of quarter end dates. Intra-quarter data is not captured.
