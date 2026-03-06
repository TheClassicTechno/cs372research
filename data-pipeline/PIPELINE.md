# Data Pipeline â€” End-to-End Guide

This document describes how to run the full data pipeline from scratch, producing quarterly snapshot JSON and agent-readable memos in `final_snapshots/`.

---

## Adding a New Ticker

To add a new ticker to the universe, just add an entry to `supported_tickers.yaml`:

```yaml
  - symbol: NFLX
    sector: Communication Services
    description: Netflix â€” Streaming media and entertainment
```

That's it. Every script in the pipeline reads from `supported_tickers.yaml` via the `--supported` flag, so the new ticker will automatically be picked up by all stages â€” EDGAR downloads, filing summarization, asset features, sentiment scoring, snapshot building, and memo generation. No other code changes needed.

---

## One-Command Pipeline â€” `run_pipeline.py`

Instead of running the data-gathering scripts manually, use `run_pipeline.py` to orchestrate them:

```bash
# Add a new ticker and run full pipeline
python run_pipeline.py --ticker NFLX \
  --sector "Communication Services" \
  --description "Netflix -- Streaming" \
  --start 2024Q4 --end 2025Q3

# Re-run pipeline for all existing supported tickers
python run_pipeline.py --start 2024Q4 --end 2025Q3

# Force re-run (ignore cached output)
python run_pipeline.py --start 2024Q4 --end 2025Q3 --force

# Preview commands without executing
python run_pipeline.py --ticker NFLX --sector Tech \
  --description "..." --start 2024Q4 --end 2025Q3 --dry-run
```

| Arg | Required | Description |
|-----|----------|-------------|
| `--ticker` | No | New ticker to add (omit to run for all supported) |
| `--sector` | With `--ticker` | Sector for new ticker |
| `--description` | With `--ticker` | Description for new ticker |
| `--start` | Yes | Start quarter, e.g. `2024Q4` |
| `--end` | Yes | End quarter, e.g. `2025Q3` |
| `--force` | No | Re-run all stages, ignoring cached output |
| `--dry-run` | No | Print commands without executing |

**What it does:**

1. Pre-flight checks (API keys, script existence)
2. Adds the new ticker to `supported_tickers.yaml` (if `--ticker` given and not already present)
3. Runs 6 data-gathering stages sequentially, stopping on first failure
4. Prints a summary table with pass/fail status and elapsed time per stage

When `--ticker` is provided, stages 1-3 (EDGAR + earnings calls) and 5-6 (assets, sentiment) target only the new ticker - no need to reprocess all existing tickers. Stage 4 (macro) is skipped if all output files already exist (unless `--force` is set).

When `--force` is set, it passes `--force` to stages 2, 3, and 6 (which have skip-if-exists behavior) and forces stage 4 to rebuild even if output files exist. Stages 1 (EDGAR download) and 5 (assets) always run regardless.

Snapshot building (Stage 7) and memo generation (Stage 8) are not included - those are experiment-specific and should be run separately.

**Required env vars:** `ANTHROPIC_API_KEY`, `FINNHUB_KEY`, `HF_TOKEN`. Optional: `FRED_API_KEY`.

---

## API Keys

| Key | Required By | How to Set |
|-----|-------------|------------|
| Finnhub API key | `sentiment.py` | `--api-key $FINNHUB_KEY` |
| Anthropic API key | `filing_summarization_pipeline.py` | `export ANTHROPIC_API_KEY=sk-ant-...` or `--api-key` |
| Hugging Face token | `earnings_calls/get_earring_calls_summary.py` | `export HF_TOKEN=hf_...` or `--hf-token` |
| FRED API key | `macro_quarter_builder.py` | `export FRED_API_KEY=your_key` or `--fred-key` |

- **Finnhub** â€” Free tier: 60 req/min. Register at https://finnhub.io. Only ~1 year of news history available on the free plan.
- **Anthropic** â€” Required for Claude-based filing summarization. Each filing = 1 API call.
- **FRED** â€” Free. Register at https://fred.stlouisfed.org/docs/api/api_key.html. Script warns but still runs if missing (some series will be null).

**No API key needed:** `get_sec_data.py` (SEC public API), `asset_quarter_builder.py` (yfinance public API), `generate_quarterly_json.py`, `generate_quarterly_memo.py`.

**SEC compliance:** `get_sec_data.py` requires a valid User-Agent header. Edit `SEC_HEADERS` at the top of the file with your name and email before running.

---

## Pipeline Stages

```
Stage 1: EDGAR/get_sec_data.py                            SEC EDGAR -> raw HTML + clean text
Stage 2: EDGAR/filing_summarization_pipeline.py            clean text -> summary JSON (Claude)
Stage 3: earnings_calls/get_earring_calls_summary.py       HF transcripts -> Claude summary + FinBERT sentiment
Stage 4: macro/macro_quarter_builder.py                    FRED + yfinance -> macro JSON
Stage 5: quarterly_asset_details/asset_quarter_builder.py  yfinance -> per-ticker asset JSON
Stage 6: sentiment/sentiment.py                            Finnhub + FinBERT -> per-ticker sentiment JSON
Stage 7: final_snapshots/generate_quarterly_json.py        merge all -> snapshot JSON
Stage 8: final_snapshots/generate_quarterly_memo.py        snapshot JSON -> agent memo
```

**Dependency graph:**

```
Stages 1->2  (EDGAR download then summarize)
Stage 3      (earnings calls - independent)
Stage 4      (macro - independent)
Stage 5      (assets - independent)
Stage 6      (sentiment - independent)
Stage 7      (merge - requires 2, 3, 4, 5, 6 complete)
Stage 8      (memo - requires 7 complete)
```

Stages 1->2, 3, 4, 5, and 6 are **independent** - run them in parallel.

---

## Stage 1: Download Filing Text from SEC EDGAR

```bash
cd data-pipeline

# All supported tickers, last 4 quarters
python EDGAR/get_sec_data.py --supported --last-n 4

# Specific tickers, specific quarters
python EDGAR/get_sec_data.py --tickers AAPL,NVDA,MSFT --years 2024,2025 --quarters Q1,Q2,Q3,Q4

# With parallel downloads
python EDGAR/get_sec_data.py --supported --last-n 4 --parallel --workers 3
```

**What it does:** Downloads HTML filings from SEC EDGAR, saves raw HTML, converts to clean plain text.

**Output:**
```
EDGAR/raw_html/{TICKER}/{TICKER}_{YEAR}_{QUARTER}_{FORM}_{FILINGDATE}.html
EDGAR/clean_text/{TICKER}/{TICKER}_{YEAR}_{QUARTER}_{FORM}_{FILINGDATE}.txt
```

**Re-runs:** Skips already-downloaded filings by default.

---

## Stage 2: Summarize Filings via Claude

```bash
# All supported tickers
python EDGAR/filing_summarization_pipeline.py --supported

# Specific tickers
python EDGAR/filing_summarization_pipeline.py --ticker AAPL,NVDA,MSFT

# Re-summarize everything
python EDGAR/filing_summarization_pipeline.py --supported --force
```

**What it does:** Sends each filing's full text to Claude for structured extraction. Produces JSON summaries with: operating_state, cost_structure, material_events, macro_exposures, forward_outlook, uncertainty_profile.

**Output:** `EDGAR/finished_summaries/{TICKER}/{YEAR}/Q{#}.json`

**Re-runs:** Skips existing summaries by default. Use `--force` to re-summarize.

**Cost:** Each filing = 1 Claude API call. Model default: `claude-sonnet-4-20250514`.

---

## Stage 3: Build Earnings Call Summaries (Hugging Face + Claude + FinBERT)

```bash
# All supported tickers, quarter range
python earnings_calls/get_earring_calls_summary.py --start 2024Q4 --end 2025Q3 --supported --hf-token $HF_TOKEN

# Specific tickers only
python earnings_calls/get_earring_calls_summary.py --start 2025Q1 --end 2025Q3 --tickers AAPL,NVDA --hf-token $HF_TOKEN

# Rebuild existing files
python earnings_calls/get_earring_calls_summary.py --start 2025Q1 --end 2025Q3 --supported --hf-token $HF_TOKEN --force
```

**What it does:** Pulls earnings-call transcripts from `kurry/sp500_earnings_transcripts`, summarizes each ticker-quarter transcript with Claude, and computes FinBERT transcript sentiment features.

**Output:** `earnings_calls/data/{TICKER}/{YEAR}_Q{#}.json`

**Re-runs:** Skips existing files by default. Use `--force` to rebuild.

**Missing ticker behavior:** If a ticker has no matching transcript rows in the dataset, it is skipped for that quarter.

---

## Stage 4: Build Macro Data

```bash
# Quarter range (recommended)
python macro/macro_quarter_builder.py --start 2024Q4 --end 2025Q3

# Single quarter
python macro/macro_quarter_builder.py --year 2025 --quarter Q1

# With explicit FRED key
python macro/macro_quarter_builder.py --start 2024Q4 --end 2025Q3 --fred-key $FRED_API_KEY
```

**What it does:** Fetches macro data from FRED (rates, inflation, credit, employment, industrial production) and yfinance (VIX, equity-bond correlation). Computes QoQ deltas. No ticker-specific data.

**Output:** `macro/data/macro_{YEAR}_{QUARTER}.json`

**Re-runs:** Always rebuilds. FRED data is cached in CSV files next to the script; only missing date ranges are fetched from the API.

---

## Stage 5: Build Asset Features

```bash
# All supported tickers, quarter range
python quarterly_asset_details/asset_quarter_builder.py --start 2024Q4 --end 2025Q3 --supported

# Specific tickers (e.g. just the new one)
python quarterly_asset_details/asset_quarter_builder.py --start 2025Q1 --end 2025Q3 --tickers AAPL,NVDA,MSFT
```

**What it does:** Fetches price history from yfinance and computes ~28 metrics per ticker: returns (20d/60d/120d/252d), volatility, beta, Sharpe, drawdowns, SMA crossovers, momentum, fundamentals (gross margin, ROE, FCF yield, debt/equity, book-to-market, earnings surprise). All fundamental data (shares outstanding, book value) is derived from quarterly balance sheets for point-in-time safety.

**Output:** `quarterly_asset_details/data/{TICKER}/{YEAR}_Q{#}.json` (one file per ticker per quarter)

**Re-runs:** Always overwrites existing output for the ticker/quarter.

---

## Stage 6: Build Sentiment Features

```bash
# All supported tickers, quarter range
python sentiment/sentiment.py --start 2025Q1 --end 2025Q3 --supported --api-key $FINNHUB_KEY

# With parallel news fetching
python sentiment/sentiment.py --start 2025Q1 --end 2025Q3 --supported --api-key $FINNHUB_KEY --workers 4

# Re-score everything (ignore cache)
python sentiment/sentiment.py --start 2025Q1 --end 2025Q3 --supported --api-key $FINNHUB_KEY --force
```

**What it does:** Fetches news articles from Finnhub, scores with FinBERT (local model), computes 4 features per ticker: article_count, mean_sentiment, sentiment_volatility, surprise_sentiment. Cross-sectional z-score is computed at snapshot build time (Stage 6), not here.

**Output:** `sentiment/data/{TICKER}/{YEAR}_Q{#}.json` (one file per ticker per quarter)

**Re-runs:** Skips tickers with existing output files. Use `--force` to re-score. News responses are cached in `sentiment/news_cache/`.

**Note:** Finnhub free tier only returns ~1 year of history. Older quarters will have null sentiment.

---

## Stage 7: Build Quarterly Snapshots

```bash
# All supported tickers, full range
python final_snapshots/generate_quarterly_json.py --start 2024Q4 --end 2025Q3 --supported

# Specific tickers, single quarter
python final_snapshots/generate_quarterly_json.py --year 2025 --quarter Q1 --tickers AAPL,NVDA,MSFT
```

**What it does:** Merges all upstream data (filings, macro, assets, sentiment) into a single JSON per quarter. Computes cross-sectional features (relative_strength_60d, cross_sectional_z) across all tickers. Enforces point-in-time safety â€” no data after the quarter end date. Runs leakage checks.

**Output:** `final_snapshots/json_data/snapshot_{YEAR}_{QUARTER}.json`

**Inputs required (all must exist):**

| Source | Path |
|--------|------|
| Filing summaries | `EDGAR/finished_summaries/{TICKER}/{YEAR}/Q{#}.json` |
| Earnings calls | `earnings_calls/data/{TICKER}/{YEAR}_Q{#}.json` |
| Macro data | `macro/data/macro_{YEAR}_{QUARTER}.json` |
| Asset features | `quarterly_asset_details/data/{TICKER}/{YEAR}_Q{#}.json` |
| Sentiment | `sentiment/data/{TICKER}/{YEAR}_Q{#}.json` |

Any missing source -> that section is `null` in the snapshot (not fatal).

---

## Stage 8: Generate Agent Memos

```bash
# All tickers in snapshot
python final_snapshots/generate_quarterly_memo.py --start 2024Q4 --end 2025Q3

# Specific tickers only
python final_snapshots/generate_quarterly_memo.py --start 2025Q1 --end 2025Q3 --tickers AAPL,NVDA,MSFT

# All supported tickers
python final_snapshots/generate_quarterly_memo.py --start 2024Q4 --end 2025Q3 --supported
```

**What it does:** Converts snapshot JSON into structured plain-text memos with tagged evidence IDs (`[L1-10Y]`, `[AAPL-RET60]`, etc.) that agents can cite in their reasoning.

**Output:** `final_snapshots/memo_data/memo_{YEAR}_{QUARTER}.txt`

**Input:** `final_snapshots/json_data/snapshot_{YEAR}_{QUARTER}.json`

---

## Full Run â€” Copy-Paste Script

```bash
cd data-pipeline

# --- Independent stages (run in parallel or sequence) ---

# 1. Download filings from SEC EDGAR
python EDGAR/get_sec_data.py --supported --last-n 4 --parallel --workers 3

# 2. Summarize filings via Claude
python EDGAR/filing_summarization_pipeline.py --supported --parallel

# 3. Build earnings call summaries
python earnings_calls/get_earring_calls_summary.py --start 2024Q4 --end 2025Q3 --supported --hf-token $HF_TOKEN

# 4. Build macro data
python macro/macro_quarter_builder.py --start 2024Q4 --end 2025Q3

# 5. Build asset features
python quarterly_asset_details/asset_quarter_builder.py --start 2024Q4 --end 2025Q3 --supported

# 6. Build sentiment features
python sentiment/sentiment.py --start 2025Q1 --end 2025Q3 --supported --api-key $FINNHUB_KEY --workers 4

# --- Merge stage (requires 2-6 complete) ---

# 7. Build quarterly snapshots
python final_snapshots/generate_quarterly_json.py --start 2024Q4 --end 2025Q3 --supported

# 8. Generate agent memos
python final_snapshots/generate_quarterly_memo.py --start 2024Q4 --end 2025Q3 --supported
```

---

## Final Output

After a full run, the agent-consumable files live in:

```
data-pipeline/final_snapshots/
  json_data/
    snapshot_2024_Q4.json       <- full data payload (JSON)
    snapshot_2025_Q1.json
    snapshot_2025_Q2.json
    snapshot_2025_Q3.json
  memo_data/
    memo_2024_Q4.txt            <- agent-readable memo (plain text)
    memo_2025_Q1.txt
    memo_2025_Q2.txt
    memo_2025_Q3.txt
```

Each snapshot JSON contains: macro regime, per-ticker asset features (~28 metrics), sentiment (5 features), and filing summaries â€” all point-in-time safe. See `final_snapshots/documentation.md` for the full schema.

---

## Output Directory Structure

```
data-pipeline/
  supported_tickers.yaml                <- 20-ticker universe (single source of truth)
  run_pipeline.py                       <- orchestrator (stages 1-6)
  earnings_calls/
    get_earring_calls_summary.py
    data/{TICKER}/{YEAR}_Q{#}.json      <- per-ticker earnings-call summary + sentiment
  EDGAR/
    get_sec_data.py
    filing_summarization_pipeline.py
    raw_html/{TICKER}/*.html            <- raw SEC HTML
    clean_text/{TICKER}/*.txt           <- full plain text
    finished_summaries/{TICKER}/{YEAR}/Q{#}.json
  macro/
    macro_quarter_builder.py
    data/macro_{YEAR}_{QUARTER}.json
  quarterly_asset_details/
    asset_quarter_builder.py
    data/{TICKER}/{YEAR}_Q{#}.json      <- per-ticker asset features
  sentiment/
    sentiment.py
    data/{TICKER}/{YEAR}_Q{#}.json      <- per-ticker sentiment features
    news_cache/                         <- Finnhub response cache
  final_snapshots/
    generate_quarterly_json.py          <- snapshot builder
    generate_quarterly_memo.py          <- memo generator
    documentation.md
    json_data/snapshot_{YEAR}_{QUARTER}.json
    memo_data/memo_{YEAR}_{QUARTER}.txt
  tests/
    test_computations.py                <- mathematical correctness tests
    test_temporal_integrity.py          <- temporal leakage tests
```

---

## Testing

```bash
# Run all tests
cd data-pipeline && python -m pytest tests/ -v

# Mathematical correctness only
python -m pytest tests/test_computations.py -v

# Temporal integrity / leakage detection only
python -m pytest tests/test_temporal_integrity.py -v
```

---

## Troubleshooting

**"Ticker not found"** â€” The ticker isn't in SEC's company_tickers.json mapping. Check spelling and that the company is SEC-registered.

**FRED fetch errors** â€” Verify your FRED API key. Some series may return null without a key. Errors are logged per-metric, not fatal.

**Finnhub null sentiment** â€” Free tier only has ~1 year of news history. Older quarters will have null sentiment. Not a bug.

**Claude API rate limits** â€” The summarization pipeline sleeps 1 second between calls by default. Adjust `RATE_LIMIT_SECONDS` for higher-tier plans.

**yfinance MultiIndex errors** â€” If yfinance returns garbled data, retry. This is a known intermittent issue with the yfinance library.

**Missing sections in snapshot** â€” Any upstream data not found -> that section is `null` in the snapshot. Run the relevant upstream stage first.

---

## Per-Component Documentation

Each subdirectory has its own `documentation.md`:

- `EDGAR/documentation.md` â€” Filing download, text extraction, summarization schema
- `macro/documentation.md` â€” FRED series map, layer definitions, caching
- `quarterly_asset_details/documentation.md` â€” Asset feature computation, per-ticker output schema
- `sentiment/documentation.md` â€” Sentiment features, FinBERT pipeline, rate limiting
- `final_snapshots/documentation.md` â€” Snapshot schema, memo format, evidence IDs, point-in-time safety
