
# Quarterly Snapshot Builder — Data Flow & Methodology

This document describes the quarterly snapshot builder (`quarterly_snapshot_builder.py`), which merges all upstream pipeline outputs into a single canonical per-ticker per-quarter JSON snapshot for trading agent consumption.

The goal is:

* Single source of truth per ticker per quarter
* Strict point-in-time safety (no future leakage)
* Clear data provenance
* Deterministic, auditable output

---

# 1. Pipeline Overview

```
EDGAR/finished_summaries/     (filing summaries)
sentiment/data/               (sentiment_output.json)
macro/data/                   (augmented_market_state_v3.json)
Fundamentals API              (stub)
Price History                 (stub)
Ownership / Exposure          (stub)
  |
  v
quarterly_snapshot_builder.py
  |
  v
quarterly_snapshot/data/{YEAR}{QUARTER}/{TICKER}.json
```

The builder is a **read-only consumer** of all upstream pipelines. It does not modify or re-fetch any source data.

---

# 2. Snapshot Schema

Each snapshot JSON has this top-level structure:

```json
{
  "ticker": "AAPL",
  "as_of_date": "2025-03-31",
  "year": 2025,
  "quarter": "Q1",
  "fundamentals": { ... },
  "filing_summary": {
    "periodic": { ... },
    "event_filings": [ ... ]
  },
  "news_sentiment": { ... },
  "macro_regime": { ... },
  "price_features": { ... },
  "ownership_signals": { ... },
  "exposure_profile": { ... }
}
```

### Required Top-Level Keys

All seven data sections are required. Missing data is represented as `null` values within the section, never as missing keys.

| Key | Source | Description |
|-----|--------|-------------|
| `ticker` | CLI input | Uppercase ticker symbol |
| `as_of_date` | CLI input | Rebalance date (ISO 8601) |
| `year` | Derived | Calendar year from rebalance date |
| `quarter` | Derived | Quarter label (Q1-Q4) from rebalance date |
| `fundamentals` | Structured API (stub) | Revenue, EPS, debt, ratios |
| `filing_summary` | EDGAR summarization pipeline | Periodic filing + recent 8-Ks |
| `news_sentiment` | Sentiment pipeline | Article count, mean, volatility, surprise, z-score |
| `macro_regime` | Macro pipeline | L1 macro, L4 vol, L5 internals |
| `price_features` | Price history (stub) | Returns, vol, drawdown, SMAs |
| `ownership_signals` | SEC Form 4/13F (stub) | Insider activity, institutional % |
| `exposure_profile` | Factor model (stub) | Sector, industry, beta |

---

# 3. Data Loading Details

## 3.1 Filing Summaries (`load_filing_summaries`)

Reads from `EDGAR/finished_summaries/{TICKER}/` directory.

**Periodic filing** (most recent 10-Q or 10-K):
* Scans all `*_summary.json` files under the ticker directory
* Filters by `filing_date <= rebalance_date`
* Returns the most recent by filing date
* Includes 10-Q/A and 10-K/A amendments

**Event filings** (recent 8-Ks):
* Same scan, filtered to 8-K and 8-K/A forms
* Additional filter: `filing_date >= rebalance_date - 90 days`
* Sorted by filing date, most recent first

Returns:
```json
{
  "periodic": { "form": "10-Q", "filing_date": "2025-02-15", "mda": {...}, ... },
  "event_filings": [
    { "form": "8-K", "filing_date": "2025-03-10", "events": [...] }
  ]
}
```

## 3.2 Sentiment (`load_sentiment`)

Reads from `sentiment/data/sentiment_output.json`.

* Looks up `results.{TICKER}.{YEAR}{QUARTER}` (e.g., `results.AAPL.2025Q1`)
* Returns the per-quarter sentiment dict directly:
  ```json
  {
    "article_count": 45,
    "mean_sentiment": 0.23,
    "sentiment_volatility": 0.15,
    "surprise_sentiment": 0.05,
    "cross_sectional_z": 1.2
  }
  ```
* Returns `null` if ticker or quarter not found

## 3.3 Macro Regime (`load_macro`)

Reads from `macro/data/augmented_market_state_v3.json`.

* Extracts Layer 1 (macro_metrics), Layer 4 (vol_metrics), and Layer 5 (internals_metrics) for the given quarter
* Returns:
  ```json
  {
    "macro_metrics": { "L1-MON-FF": {"value": 4.33, "units": "%"}, ... },
    "vol_metrics": { "L4-VIX": {"value": 18.5, "units": "index"}, ... },
    "internals_metrics": { "L5-200DMA": {"value": 65.0, "units": "%"}, ... }
  }
  ```
* Returns `null` if macro file not found or quarter not present

## 3.4 Stub Data Sources

The following are currently stubbed with placeholder values. Each returns a dict with `"source": "STUB"` to indicate non-live data.

| Function | Future Integration |
|----------|--------------------|
| `fetch_fundamentals_api(ticker, as_of)` | SEC XBRL company-facts endpoint or financial data vendor |
| `fetch_price_history(ticker, end_date)` | Yahoo Finance or vendor API |
| `compute_price_features(history, as_of)` | Internal computation from price history |
| `fetch_ownership_signals(ticker, as_of)` | SEC Form 4 / 13F data |
| `fetch_exposure_profile(ticker, as_of)` | Factor model or sector classification |

When stubs are replaced with real integrations, each must enforce `as_of <= rebalance_date` filtering.

---

# 4. Point-in-Time Safety

This is the most critical property of the snapshot builder. Every data source is filtered so that **only information publicly available on or before the rebalance date** is included.

## 4.1 Rules by Data Source

| Source | Point-in-Time Rule |
|--------|-------------------|
| Filing summaries | `filing_date <= rebalance_date` |
| 8-K events | `filing_date` within 90 days before `rebalance_date` |
| Sentiment | Matched by year/quarter (articles within quarter boundaries) |
| Macro | Matched by year/quarter (quarter-end values only) |
| Fundamentals (future) | API must return data from filings with `filing_date <= as_of` |
| Price features (future) | All prices `<= rebalance_date` only |
| Ownership (future) | Filing or reporting date `<= as_of` |

## 4.2 Leakage Detection

`ensure_no_future_leakage(snapshot, rebalance_date)` checks:

1. **Filing dates**: Every `filing_date` in `filing_summary.periodic` and `filing_summary.event_filings` must be `<= rebalance_date`
2. **Price feature end date**: `price_features.as_of_date` must be `<= rebalance_date`

If violations are detected, they are logged as warnings in the pipeline report. Snapshots with leakage are still saved (for debugging) but flagged.

## 4.3 Why This Matters

In backtesting and multi-agent research:
* Using data that wasn't yet available on the decision date creates **lookahead bias**
* This inflates backtested returns and invalidates research conclusions
* Point-in-time snapshots ensure every agent decision is based only on information available at that moment

---

# 5. Quarter Resolution

## 5.1 Rebalance Date to Quarter Mapping

`quarter_from_date(d)` assigns a date to the quarter whose end date it falls on or before:

| Date Range | Quarter |
|-----------|---------|
| Jan 1 - Mar 31 | Q1 |
| Apr 1 - Jun 30 | Q2 |
| Jul 1 - Sep 30 | Q3 |
| Oct 1 - Dec 31 | Q4 |

Typical rebalance dates are quarter-end dates (Mar 31, Jun 30, Sep 30, Dec 31), but any date within the quarter is valid.

## 5.2 Quarter End Dates

Used for boundary calculations:

| Quarter | End Date |
|---------|----------|
| Q1 | March 31 |
| Q2 | June 30 |
| Q3 | September 30 |
| Q4 | December 31 |

---

# 6. Output Structure

```
quarterly_snapshot/data/
  2025Q1/
    AAPL.json
    NVDA.json
    MSFT.json
  2025Q2/
    AAPL.json
    NVDA.json
    MSFT.json
```

Each file is one complete snapshot. All writes are atomic (tmp file -> rename).

---

# 7. Validation

## 7.1 Schema Validation

`validate_snapshot_schema()` checks:
* All required top-level keys present (`ticker`, `as_of_date`, `year`, `quarter`, `fundamentals`, `filing_summary`, `news_sentiment`, `macro_regime`, `price_features`, `ownership_signals`, `exposure_profile`)
* `ticker` is a string
* `as_of_date` is an ISO date string

## 7.2 Pipeline Report

`run_pipeline()` returns a summary:
```json
{
  "total_planned": 16,
  "built": 16,
  "errors": [],
  "leakage_warnings": []
}
```

---

# 8. Example Commands

```bash
# Build final_snapshots for 3 tickers at 2 rebalance dates
python quarterly_quarterly_json.py \
    --tickers AAPL,NVDA,MSFT \
    --rebalance-dates 2025-03-31,2025-06-30

# Build with custom data directories
python quarterly_quarterly_json.py \
    --tickers AAPL,NVDA \
    --rebalance-dates 2025-03-31,2025-06-30,2025-09-30,2025-12-31 \
    --summaries-dir ./EDGAR/finished_summaries \
    --sentiment-dir ./sentiment/data \
    --macro-dir ./macro/data \
    --output-dir ./quarterly_asset_details/data

# Single ticker, single date (debugging)
python quarterly_quarterly_json.py \
    --tickers AAPL \
    --rebalance-dates 2025-03-31
```

---

# 9. Future API Integration Points

When replacing stubs with real data sources:

### Fundamentals (`fetch_fundamentals_api`)
* **Recommended**: SEC XBRL company-facts endpoint (`data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json`)
* Returns structured financial data with filing dates for point-in-time filtering
* Expected fields: revenue_ttm, net_income_ttm, eps_diluted_ttm, total_debt, cash, book_value, shares_outstanding, pe_ratio, price_to_book, debt_to_equity, current_ratio, roe

### Price History (`fetch_price_history`)
* **Recommended**: Yahoo Finance (`yfinance`) or financial data vendor
* Must return daily OHLCV data with dates
* `compute_price_features()` already contains full implementation logic; it just needs non-empty input data

### Ownership Signals (`fetch_ownership_signals`)
* **Recommended**: SEC Form 4 (insider transactions) and 13F (institutional holdings)
* Expected fields: insider_net_shares_90d, institutional_ownership_pct

### Exposure Profile (`fetch_exposure_profile`)
* **Recommended**: Factor model or sector classification provider
* Expected fields: sector, industry, market_cap_bucket, beta

Each integration must:
1. Accept an `as_of` date parameter
2. Return only data available on or before that date
3. Include a `"source"` field identifying the data provider
4. Include an `"as_of_date"` field for audit trail

---

# 10. Component Integration Diagram

```
                    +-----------------------+
                    |   Trading Agents      |
                    |  (read snapshots)     |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    | quarterly_snapshot/    |
                    | data/2025Q1/AAPL.json |
                    +-----------+-----------+
                                |
              +-----------------+------------------+
              |                 |                   |
   +----------v------+  +------v--------+  +-------v-------+
   | EDGAR/           |  | sentiment/    |  | macro/        |
   | finished_        |  | data/         |  | data/         |
   | summaries/       |  | sentiment_    |  | augmented_    |
   |                  |  | output.json   |  | market_state_ |
   +--------+---------+  +------+--------+  | v3.json       |
            |                    |           +-------+-------+
   +--------v---------+         |                   |
   | EDGAR/            |  Sentiment         Macro Pipeline
   | raw_filings/      |  Pipeline          (metrics_gather.py)
   | (Stage 1 .txt)    |                       |
   +--------+----------+               FRED + Yahoo Finance
            |                            + Wikipedia
   SEC EDGAR API
   (get_sec_data.py)
```

---

# 11. Limitations

* **Stub data**: Fundamentals, price features, ownership, and exposure are currently placeholder values. Snapshots are structurally complete but lack these live data sources.
* **Single-file sentiment**: Sentiment is loaded from a single `sentiment_output.json` file. If the sentiment pipeline produces per-quarter files, the loader will need updating.
* **Macro granularity**: Macro data is matched by quarter label, not by exact date. Intra-quarter rebalance dates use the same quarter's macro data.
* **No retry logic**: If an upstream file is missing or corrupt, the corresponding section is null. The pipeline does not retry or fall back to alternative sources.
* **Sequential processing**: Snapshots are built sequentially (no parallelism). For large universes, this may be slow but ensures deterministic output.
