# Final Snapshots — Quarterly Snapshot Builder & Memo Generator

## Purpose

`generate_quarterly_json.py` is the merge stage of the data pipeline. It combines all upstream data sources into a single, self-contained JSON file per quarter. Each snapshot file is the **full data payload** that gets injected into an agent prompt for all tickers at a specific rebalance date.

An agent receiving this file has everything it needs to reason about portfolio allocation for that quarter: macro regime, per-ticker price/fundamental metrics, SEC filing summaries, and news sentiment — all in one document, with no future data leakage.

## Usage

```bash
# All supported tickers, Q4 2024 through Q3 2025
python generate_quarterly_json.py --start 2024Q4 --end 2025Q3 --supported

# Specific tickers, single quarter
python generate_quarterly_json.py --year 2025 --quarter Q1 --tickers AAPL,NVDA,MSFT

# Custom output directory
python generate_quarterly_json.py --start 2025Q1 --end 2025Q3 --supported --output-dir /tmp/snapshots
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
| `--summaries-dir` | Override EDGAR summaries path |
| `--sentiment-dir` | Override sentiment data path |
| `--macro-dir` | Override macro data path |
| `--assets-dir` | Override asset data path |
| `--output-dir` | Override output directory |

## Output

### Location

```
data-pipeline/final_snapshots/json_data/snapshot_{YEAR}_{QUARTER}.json
```

### Schema

Each file contains all requested tickers for that quarter in a single JSON document:

```json
{
  "year": 2025,
  "quarter": "Q1",
  "as_of_date": "2025-03-31",
  "tickers": ["CAT", "DAL", "JPM", "..."],

  "macro_regime": {
    "macro_metrics": {
      "FF":      { "value": 4.33 },
      "2Y":      { "value": 3.89, "delta_bps_qoq": -36.0 },
      "10Y":     { "value": 4.23, "delta_bps_qoq": -35.0 },
      "CPI_YOY": { "value": 2.38 },
      "UNRATE":  { "value": 4.2 },
      "..."
    },
    "vol_metrics": {
      "VIX":          { "value": 22.28 },
      "EQ_BOND_CORR": { "value": 0.103 }
    }
  },

  "ticker_data": {
    "AAPL": {
      "filing_summary": {
        "periodic": {
          "form": "10-Q",
          "filing_date": "2025-01-31",
          "operating_state": "...",
          "cost_structure": "...",
          "material_events": "...",
          "macro_exposures": "...",
          "forward_outlook": "...",
          "uncertainty_profile": "..."
        },
        "event_filings": []
      },
      "news_sentiment": {
        "article_count": 227,
        "mean_sentiment": 0.2289,
        "sentiment_volatility": 0.5307,
        "surprise_sentiment": null,
        "cross_sectional_z": 0.7776
      },
      "asset_features": {
        "close": 216.95,
        "ret_20d": -0.099,
        "ret_60d": -0.1351,
        "vol_60d": 0.2878,
        "beta_1y": 0.9742,
        "gross_margin": 0.4705,
        "relative_strength_60d": -0.0369,
        "..."
      }
    },
    "NVDA": { "..." },
    "...": "..."
  }
}
```

### Top-Level Structure

| Key | Description |
|---|---|
| `year` | Calendar year |
| `quarter` | Quarter label (Q1-Q4) |
| `as_of_date` | Quarter end date (rebalance date) |
| `tickers` | List of tickers included in this snapshot |
| `macro_regime` | Macro economic data — shared across all tickers |
| `ticker_data` | Per-ticker data keyed by symbol |

### Per-Ticker Data (`ticker_data.{TICKER}`)

| Section | Source | Description |
|---|---|---|
| `filing_summary` | EDGAR summarization pipeline | Most recent 10-Q/10-K summary + 8-K events from last 90 days |
| `news_sentiment` | Finnhub + FinBERT pipeline | 5 sentiment features: article_count, mean_sentiment, sentiment_volatility, surprise_sentiment, cross_sectional_z |
| `asset_features` | Yahoo Finance + yfinance | ~28 price, volatility, momentum, fundamental, and cross-sectional metrics |

Any section is `null` if its upstream data is unavailable for that ticker/quarter.

## Data Sources

The script reads from four upstream pipelines, all using per-ticker file layouts:

```
data-pipeline/
  EDGAR/finished_summaries/{TICKER}/{YEAR}/Q{#}.json    <- filing summaries
  sentiment/data/{TICKER}/{YEAR}_Q{#}.json              <- sentiment features
  macro/data/macro_{YEAR}_Q{#}.json                     <- macro regime
  quarterly_asset_details/data/{TICKER}/{YEAR}_Q{#}.json <- asset features
```

## Cross-Sectional Features

Two cross-sectional features are computed at this stage (not in upstream pipelines) because they require all tickers to be present:

### `relative_strength_60d`

```
relative_strength_60d = ret_60d - median(all ret_60d in quarter)
```

Added to `asset_features` for each ticker. The sum of deviations from the median is approximately zero.

### `cross_sectional_z`

```
mu    = mean(mean_sentiment across tickers)
sigma = std(mean_sentiment across tickers)  # population std
z     = (mean_sentiment - mu) / sigma
```

Added to `news_sentiment` for each ticker. Z-scores sum to zero and have population standard deviation of 1.

## Point-in-Time Safety

Every data source is filtered so that only information publicly available on or before the quarter end date is included:

- **Filing summaries**: `filing_date <= rebalance_date`
- **8-K events**: `filing_date` within 90 days before rebalance_date
- **Sentiment**: matched by year/quarter (articles within quarter boundaries)
- **Macro**: matched by year/quarter (FRED/yfinance data as-of quarter end)
- **Asset features**: matched by year/quarter (all rolling windows end on quarter end)

The script runs a leakage check on every ticker via `check_leakage()` and warns if any filing dates exceed the rebalance date.

## Design Decisions

**Single file per quarter**: All tickers are in one JSON rather than separate files. This is the format that gets passed directly to agent prompts — one file = one complete information set for a rebalance decision.

**Macro at top level**: Macro regime is the same for all tickers in a quarter, so it lives at the document root rather than being duplicated inside each ticker.

**Null propagation**: If an upstream pipeline hasn't been run for a given quarter/ticker, the corresponding section is `null` rather than being omitted. This makes it explicit what data is missing.

**Cross-sectional features at merge time**: `relative_strength_60d` and `cross_sectional_z` require data from all tickers simultaneously. Computing them here (not in upstream scripts) keeps upstream files independent and allows adding a single ticker without reprocessing all others.

---

# Memo Generator — `generate_quarterly_memo.py`

## Purpose

Converts snapshot JSON files into structured plain-text memos designed for direct injection into agent prompts. Each memo is a human/agent-readable version of the snapshot with tagged evidence IDs that agents can cite in their reasoning.

## Usage

```bash
# All tickers in snapshot, single quarter
python generate_quarterly_memo.py --year 2025 --quarter Q1

# Quarter range, all supported tickers
python generate_quarterly_memo.py --start 2024Q4 --end 2025Q3 --supported

# Specific tickers only
python generate_quarterly_memo.py --start 2025Q1 --end 2025Q3 --tickers AAPL,NVDA,MSFT

# Custom directories
python generate_quarterly_memo.py --year 2025 --quarter Q1 --input-dir /tmp/json --output-dir /tmp/memos
```

## CLI Arguments

| Argument | Description |
|---|---|
| `--start` | Start quarter, e.g. `2024Q4` |
| `--end` | End quarter, e.g. `2025Q3` |
| `--year` | Year (single quarter mode) |
| `--quarter` | Quarter: Q1, Q2, Q3, Q4 (single quarter mode) |
| `--tickers` | Comma-separated tickers to include (default: all in snapshot) |
| `--supported` | Use all tickers from `supported_tickers.yaml` |
| `--input-dir` | Override snapshot JSON input path (default: `json_data/`) |
| `--output-dir` | Override memo output path (default: `memo_data/`) |

## Output

### Location

```
data-pipeline/final_snapshots/memo_data/memo_{YEAR}_{QUARTER}.txt
```

### Structure

Each memo contains the following sections in order:

1. **Header** — Quarter, as-of date, ticker count, agent instructions
2. **Layer 1 Macro Regime** — Rates, inflation, growth, credit, volatility with tagged IDs (`[L1-10Y]`, `[L1-VIX]`, etc.)
3. **Cross-Sectional Summary Table** — Compact comparison: close, 60D return, 60D vol, beta, Sharpe, sentiment Z, gross margin
4. **Per-Ticker Sections** — Full detail for each ticker:
   - Price & Returns (`[AAPL-PX]`, `[AAPL-RET60]`, ...)
   - Volatility & Risk (`[AAPL-VOL60]`, `[AAPL-BETA]`, ...)
   - Trend & Momentum (`[AAPL-SMA200]`, `[AAPL-MOM12_1]`, ...)
   - Fundamentals (`[AAPL-GM]`, `[AAPL-ROE]`, `[AAPL-FCFY]`, ...)
   - Sentiment (`[AAPL-SENT]`, `[AAPL-CSZ]`, ...)
   - Filing Summary (`[AAPL-F1]`, `[AAPL-F2]`, ...) + 8-K events (`[AAPL-EV1]`, ...)

### Evidence ID Convention

All data points are tagged with bracketed IDs for agent citation:

| Pattern | Example | Description |
|---|---|---|
| `[L1-*]` | `[L1-10Y]` | Macro regime metrics |
| `[TICKER-PX]` | `[AAPL-PX]` | Closing price |
| `[TICKER-RET*]` | `[AAPL-RET60]` | Return metrics |
| `[TICKER-VOL*]` | `[AAPL-VOL60]` | Volatility metrics |
| `[TICKER-BETA]` | `[NVDA-BETA]` | 1Y beta |
| `[TICKER-SENT]` | `[MSFT-SENT]` | Mean sentiment |
| `[TICKER-CSZ]` | `[GOOG-CSZ]` | Cross-sectional sentiment Z |
| `[TICKER-F*]` | `[JPM-F1]` | Filing evidence sentences (one fact per ID) |
| `[TICKER-EC*]` | `[GS-EC1]` | Earnings call evidence sentences (one fact per ID) |
| `[TICKER-EV*]` | `[DAL-EV1]` | 8-K event filings |

### Ticker Filtering

When `--tickers` or `--supported` is used, the memo only includes the specified tickers. Tickers not found in the snapshot data are listed in a warning at the top. The cross-sectional summary table and per-ticker sections reflect only the filtered set.
