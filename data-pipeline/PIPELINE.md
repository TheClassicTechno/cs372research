# Data Pipeline — End-to-End Guide

This document describes how to run the full data pipeline from scratch, producing quarterly snapshots that trading agents consume.

---

## Prerequisites

**Python packages:**

```bash
pip install requests pandas numpy yfinance
```

**API keys:**

| Key | Required By | How to Set |
|-----|-------------|------------|
| FRED API key | `metrics_gather.py` | `export FRED_API_KEY=your_key` or `--fred-key` |
| Anthropic API key | `filing_summarization_pipeline.py` | `export ANTHROPIC_API_KEY=sk-ant-...` or `--api-key` |

The FRED key is free — register at https://fred.stlouisfed.org/docs/api/api_key.html.

**SEC compliance:** `get_sec_data.py` requires a valid User-Agent header. Edit the `SEC_HEADERS` dict at the top of the file with your name and email before running.

---

## Pipeline Stages

```
Stage 1: get_sec_data.py                 (SEC EDGAR -> raw .txt)
Stage 2: filing_summarization_pipeline.py (raw .txt -> summary .json)
Stage 3: metrics_gather.py               (FRED + Yahoo -> macro .json)
Stage 4: quarterly_snapshot_builder.py    (merge all -> per-ticker snapshots)
```

Stages 1+2 (EDGAR) and Stage 3 (macro) are **independent** — run them in parallel if you want. Stage 4 depends on all prior stages being complete.

The sentiment pipeline is assumed to have already run and produced `sentiment/data/sentiment_output.json`. See `sentiment/documentation.md` for details.

---

## Stage 1: Download & Extract Filing Text

```bash
cd data-pipeline

# Option A: Last N completed quarters (recommended for first run)
python EDGAR/get_sec_data.py \
    --tickers AAPL,NVDA,MSFT,GOOG,AMZN,META,JPM,GS \
    --last-n 4 \
    --output EDGAR/raw_filings

# Option B: Specific year + quarters
python EDGAR/get_sec_data.py \
    --tickers AAPL,NVDA,MSFT \
    --years 2024,2025 \
    --quarters Q1,Q2,Q3,Q4 \
    --output EDGAR/raw_filings

# Option C: Annual filings only (10-K)
python EDGAR/get_sec_data.py \
    --tickers AAPL,NVDA,MSFT \
    --years 2024 \
    --quarters ANNUAL \
    --output EDGAR/raw_filings

# Option D: Full institutional bundle (10-K, 10-Q, 8-K, Form 4, etc.)
python EDGAR/get_sec_data.py \
    --tickers AAPL \
    --last-n 4 \
    --bundle core \
    --include-amendments \
    --parallel --workers 3 \
    --output EDGAR/raw_filings
```

**What it does:** Downloads HTML filings from SEC EDGAR, strips tags, extracts narrative sections (MD&A, Risk Factors), saves structured `.txt` files.

**Output:** `EDGAR/raw_filings/{TICKER}/{YEAR}/{QUARTER}/{FORM}_{DATE}.txt`

**Re-runs:** Skips already-extracted filings by default. Use `--force-refresh` to re-extract everything.

---

## Stage 2: Summarize Filings via Claude

```bash
python EDGAR/filing_summarization_pipeline.py \
    --raw-dir EDGAR/raw_filings \
    --out-dir EDGAR/finished_summaries \
    --tickers AAPL,NVDA,MSFT,GOOG,AMZN,META,JPM,GS
```

**What it does:** Sends each filing's narrative text to Claude for structured extraction. Produces JSON summaries with qualitative signals (demand outlook, margin trends, risk factor changes, material events).

**Output:** `EDGAR/finished_summaries/{TICKER}/{YEAR}/{QUARTER}/{FORM}_summary.json`

**Re-runs:** Skips existing summaries by default. Use `--force` to re-summarize.

**Cost note:** Each filing = 1-3 Claude API calls depending on length. Budget accordingly for large universes.

---

## Stage 3: Build Macro Snapshot

```bash
python macro/macro_quarter_builder.py \
    --year 2025 \
    --tickers AAPL,NVDA,MSFT,GOOG,AMZN,META,JPM,GS \
    --out macro/data/augmented_market_state_v3.json
```

**What it does:** Fetches macro data (rates, inflation, credit, liquidity) from FRED, volatility/risk pricing from FRED + Yahoo Finance, market internals (breadth, concentration) from S&P 500 constituents, and per-ticker price summaries from Yahoo Finance.

**Output:** `macro/data/augmented_market_state_v3.json`

**Re-runs:** FRED data is incrementally cached in CSV files next to the script. Only missing date ranges are fetched on subsequent runs.

**Options:**

```bash
# Faster run (fewer S&P 500 tickers for breadth calc)
python macro/macro_quarter_builder.py --year 2025 --tickers AAPL --breadth-sample 50

# Deeper history for YoY comparisons
python macro/macro_quarter_builder.py --year 2025 --tickers AAPL --back-years 5
```

---

## Stage 4: Build Quarterly Snapshots

```bash
python quarterly_snapshot/quarterly_snapshot_builder.py \
    --tickers AAPL,NVDA,MSFT,GOOG,AMZN,META,JPM,GS \
    --rebalance-dates 2025-03-31,2025-06-30,2025-09-30,2025-12-31
```

**What it does:** Merges all upstream data into one JSON per ticker per quarter. Enforces point-in-time safety (no data after the rebalance date). Validates schema and checks for future leakage.

**Output:** `quarterly_snapshot/data/{YEAR}{QUARTER}/{TICKER}.json`

**Data sources read (all must exist before running):**

| Source | Default Path |
|--------|-------------|
| Filing summaries | `EDGAR/finished_summaries/` |
| Sentiment | `sentiment/data/sentiment_output.json` |
| Macro | `macro/data/augmented_market_state_v3.json` |

Custom paths can be passed via `--summaries-dir`, `--sentiment-dir`, `--macro-dir`.

---

## Full Run — Copy-Paste Script

```bash
cd data-pipeline

# 1. Download and extract filing text
python EDGAR/get_sec_data.py \
    --tickers AAPL,NVDA,MSFT,GOOG,AMZN,META,JPM,GS \
    --last-n 4 \
    --output EDGAR/raw_filings

# 2. Summarize filings via Claude
python EDGAR/filing_summarization_pipeline.py \
    --raw-dir EDGAR/raw_filings \
    --out-dir EDGAR/finished_summaries \
    --tickers AAPL,NVDA,MSFT,GOOG,AMZN,META,JPM,GS

# 3. Build macro snapshot (independent of steps 1-2)
python macro/macro_quarter_builder.py \
    --year 2025 \
    --tickers AAPL,NVDA,MSFT,GOOG,AMZN,META,JPM,GS \
    --out macro/data/augmented_market_state_v3.json

# 4. Build quarterly snapshots (requires 2 + 3 + sentiment to be done)
python quarterly_snapshot/quarterly_snapshot_builder.py \
    --tickers AAPL,NVDA,MSFT,GOOG,AMZN,META,JPM,GS \
    --rebalance-dates 2025-03-31,2025-06-30,2025-09-30,2025-12-31
```

---

## Output Directory Structure

After a full run, `data-pipeline/` looks like:

```
data-pipeline/
  EDGAR/
    get_sec_data.py
    filing_summarization_pipeline.py
    documentation.md
    raw_filings/
      AAPL/
        2025/Q1/10-Q_2025-05-02.txt
        2025/Q4/10-K_2025-10-31.txt
    finished_summaries/
      AAPL/
        2025/Q1/10-Q_summary.json
        2025/Q4/10-K_summary.json
  macro/
    metrics_gather.py
    documentation.md
    data/
      augmented_market_state_v3.json
    DFF_fred_cache.csv          (auto-generated FRED cache)
    DGS10_fred_cache.csv
    ...
  sentiment/
    documentation.md
    data/
      sentiment_output.json
  quarterly_snapshot/
    quarterly_snapshot_builder.py
    documentation.md
    data/
      2025Q1/
        AAPL.json
        NVDA.json
      2025Q2/
        AAPL.json
        NVDA.json
```

---

## Troubleshooting

**"Ticker not found"** — The ticker isn't in SEC's company_tickers.json mapping. Check spelling and that the company is SEC-registered.

**FRED fetch errors** — Verify your FRED API key. Some series (MOVE, SKEW) may not be available without a key. Errors are logged per-metric in the output JSON, not fatal.

**Empty filing sections** — Some filings don't have standard Item 7/Item 2 headings (especially older filings or non-standard formats). The text extractor logs these as `[SECTION NOT FOUND]`.

**Claude API rate limits** — The summarization pipeline sleeps 1 second between calls by default. For large batches, this is conservative. Adjust if you have a higher rate limit.

**Missing sentiment data** — The snapshot builder sets `news_sentiment: null` for any ticker/quarter not found in `sentiment_output.json`. This is expected for tickers or quarters not covered by the sentiment pipeline.

---

## Per-Component Documentation

Each subdirectory has its own detailed `documentation.md`:

* `EDGAR/documentation.md` — Filing download, text extraction, summarization schema
* `macro/documentation.md` — FRED series map, layer definitions, caching logic
* `sentiment/documentation.md` — Sentiment feature derivation methodology
* `quarterly_snapshot/documentation.md` — Snapshot schema, point-in-time rules, integration diagram
