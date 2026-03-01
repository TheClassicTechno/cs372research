
# Macro Market State — Data Flow & Methodology

This document describes the macro quarter builder (`macro_quarter_builder.py`), which produces a per-quarter JSON snapshot of macro regime and volatility data.

The goal is:

* Reproducibility
* Transparent data provenance
* Quarterly granularity with QoQ deltas
* Incremental caching for efficient re-runs

---

# 1. Pipeline Overview

```
FRED API  +  Yahoo Finance
  -> macro_quarter_builder.py
  -> macro/data/macro_{YEAR}_{QUARTER}.json   (one file per quarter)
```

The script produces one JSON file per quarter containing two layers of market data:

| Layer | Name | Source | Scope |
|-------|------|--------|-------|
| L1 | Macro Regime | FRED | Rates, curve, inflation, growth, credit, labor, FX, commodities |
| L4 | Vol & Risk Pricing | FRED + Yahoo Finance | VIX, equity-bond correlation |

---

# 2. Layer 1: Macro Regime

## 2.1 FRED Series Map

| Metric Key | FRED Series | Description | Units |
|-----------|-------------|-------------|-------|
| FF | DFF | Effective Federal Funds Rate | % |
| 2Y | DGS2 | 2-Year Treasury Yield | % |
| 10Y | DGS10 | 10-Year Treasury Yield | % |
| 30Y | DGS30 | 30-Year Treasury Yield | % |
| REAL10 | DFII10 | 10-Year TIPS (Real Yield) | % |
| CURVE | computed | 10Y - 2Y spread | percentage points |
| CPI_YOY | CPIAUCSL | CPI YoY (computed from index) | % YoY |
| CORE_YOY | CPILFESL | Core CPI YoY (computed from index) | % YoY |
| INDPRO | INDPRO | Industrial Production Index | index |
| HY_OAS | BAMLH0A0HYM2 | HY OAS | % |
| UNRATE | UNRATE | Unemployment Rate | % |
| DXY | DTWEXBGS | Broad USD Index | index |
| WTI | DCOILWTICO | WTI Crude Oil | USD/bbl |

## 2.2 CPI YoY Computation

CPI and Core CPI are reported as indices, not rates. YoY inflation is computed as:

```
cpi_yoy = (CPI_index_now / CPI_index_1yr_ago - 1) * 100
```

The comparison date is the same calendar date one year prior to the quarter end.

## 2.3 Deltas

For rate-type metrics (yields, spreads, OAS), deltas are computed in basis points:

* **QoQ delta**: `(value_now - value_prev_quarter_end) * 100 bps`

---

# 3. Layer 4: Volatility & Risk Pricing

## 3.1 FRED Series

| Metric Key | FRED Series | Description |
|-----------|-------------|-------------|
| VIX | VIXCLS | CBOE VIX Index |

## 3.2 Equity-Bond Correlation

Computed from daily log returns of SPY and TLT over the last ~60 trading days of each quarter:

1. Fetch SPY and TLT daily prices (120 calendar days lookback to ensure ~60 trading days)
2. Filter to rows with index <= quarter end date (guards against yfinance returning extra data)
3. Compute log returns: `ln(close_t / close_{t-1})`
4. Compute Pearson correlation of SPY and TLT log returns
5. Requires at least 30 observations; null otherwise

---

# 4. FRED Incremental CSV Cache

## 4.1 Mechanism

Each FRED series is cached as a flat CSV file next to the script:

```
macro_quarter_builder.py
DFF_fred_cache.csv
DGS10_fred_cache.csv
VIXCLS_fred_cache.csv
```

Naming: `{SERIES_ID}_fred_cache.csv`.

## 4.2 Incremental Fetch Logic

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

## 4.3 Historical Lookback Window

The `--back-years` argument (default: 2) controls how far back data is fetched:

```
start_date = date(year - back_years, 1, 1)
end_date = quarter_end
```

This ensures the cache accumulates enough history for YoY deltas.

---

# 5. Output Schema

```json
{
  "schema_version": "macro_state_v4_quarter_aware",
  "generated_at_utc": "2026-03-01T10:41:42+00:00",
  "year": 2025,
  "quarter": "Q1",
  "as_of": "2025-03-31",
  "layers": {
    "L1": {
      "layer": "L1",
      "year": 2025,
      "quarter": "Q1",
      "asof": "2025-03-31",
      "metrics": {
        "FF":       { "value": 4.33 },
        "2Y":       { "value": 3.89, "delta_bps_qoq": -36.0 },
        "10Y":      { "value": 4.23, "delta_bps_qoq": -35.0 },
        "30Y":      { "value": 4.54 },
        "CURVE":    { "value": 0.34 },
        "REAL10":   { "value": 1.82 },
        "CPI_YOY":  { "value": 2.38 },
        "CORE_YOY": { "value": 3.12 },
        "INDPRO":   { "value": 103.5 },
        "HY_OAS":   { "value": 3.21 },
        "UNRATE":   { "value": 4.2 },
        "DXY":      { "value": 104.1 },
        "WTI":      { "value": 71.5 }
      }
    },
    "L4": {
      "layer": "L4",
      "year": 2025,
      "quarter": "Q1",
      "asof": "2025-03-31",
      "metrics": {
        "VIX":          { "value": 22.28 },
        "EQ_BOND_CORR": { "value": 0.103 }
      }
    }
  }
}
```

---

# 6. Example Commands

```bash
# Quarter range (recommended)
python macro_quarter_builder.py --start 2024Q4 --end 2025Q3

# Single quarter
python macro_quarter_builder.py --year 2025 --quarter Q1

# With explicit FRED key
python macro_quarter_builder.py --start 2024Q4 --end 2025Q3 --fred-key $FRED_API_KEY
```

---

# 7. Integration with Snapshot Builder

The snapshot builder (`generate_quarterly_json.py`) reads from `macro/data/macro_{YEAR}_{QUARTER}.json` via `load_macro()`:

* Extracts Layer 1 (macro_metrics) and Layer 4 (vol_metrics) for each quarter
* Matched by year and quarter from the filename
* The macro data is market-wide (not per-ticker), providing regime context for all tickers in the universe

---

# 8. Point-in-Time Safety

* All FRED values use `value_on_or_before(series, q_end)` — only data published on or before the quarter end date is included
* SPY/TLT price data for equity-bond correlation is filtered to `index <= q_end` after download, guarding against yfinance returning extra rows
* The `as_of` field in each output file records the quarter end date for audit

---

# 9. Limitations

* **FRED availability**: Some series may be discontinued or have reporting lags. Errors are logged per-metric, not fatal.
* **Yahoo Finance reliability**: Rate limits and data gaps can cause missing VIX or correlation values.
* **No real-time data**: All values are as-of quarter end dates. Intra-quarter data is not captured.
