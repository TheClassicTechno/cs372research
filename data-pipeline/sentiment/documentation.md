# Sentiment Pipeline Documentation

## Overview

`sentiment.py` builds quarterly sentiment features for a configurable ticker universe using Finnhub news data scored by FinBERT. Output is one JSON file per quarter.

## Pipeline Flow

```
Finnhub company-news API
        ↓
Deduplication + text construction (headline + summary)
        ↓
FinBERT sentiment scoring (batched, MPS/GPU or CPU)
        ↓
Optional exponential time-decay weighting
        ↓
Quarterly aggregation (mean, std)
        ↓
Surprise sentiment (QoQ delta)
        ↓
Cross-sectional z-score (per quarter)
        ↓
Per-quarter JSON output
```

## Usage

```bash
# Quarter range with all supported tickers
python sentiment.py --start 2024Q4 --end 2025Q3 --supported --api-key $FINNHUB_KEY

# Single quarter, custom tickers
python sentiment.py --year 2025 --quarter 2 --tickers AAPL,NVDA --api-key $FINNHUB_KEY

# Parallel news fetching (recommended)
python sentiment.py --start 2025Q1 --end 2025Q3 --supported --api-key $FINNHUB_KEY --workers 4

# With exponential decay weighting
python sentiment.py --start 2025Q1 --end 2025Q3 --supported --api-key $FINNHUB_KEY --half-life-days 7

# Force re-score (ignore cached output files)
python sentiment.py --start 2025Q1 --end 2025Q3 --supported --api-key $FINNHUB_KEY --force
```

## CLI Arguments

| Argument | Description |
|---|---|
| `--start` | Start quarter, e.g. `2024Q4` |
| `--end` | End quarter, e.g. `2025Q3` |
| `--year` | Year (single quarter mode) |
| `--quarter` | Quarter number 1-4 (single quarter mode) |
| `--tickers` | Comma-separated tickers |
| `--supported` | Use all 19 tickers from `supported_tickers.yaml` |
| `--api-key` | Finnhub API key (required) |
| `--batch-size` | FinBERT batch size (default: 16) |
| `--half-life-days` | Exponential decay half-life in days; omit for uniform weights |
| `--workers` | Parallel news fetch workers; 0 = sequential (default: 0) |
| `--force` | Re-score even if output file already exists |
| `--out` | Custom output path (single quarter only) |

## Output

### Location

```
data-pipeline/sentiment/data/sentiment_{YEAR}_Q{#}.json
```

### Schema

```json
{
  "meta": {
    "year": 2025,
    "quarter": "Q1",
    "quarter_key": "2025Q1",
    "tickers": ["CAT", "DAL", "..."],
    "half_life_days": null
  },
  "results": {
    "AAPL": {
      "article_count": 227,
      "mean_sentiment": 0.228929,
      "sentiment_volatility": 0.530728,
      "surprise_sentiment": null,
      "cross_sectional_z": 0.777615
    },
    "NVDA": { ... },
    "...": null
  }
}
```

A ticker's value is `null` if no articles were found for that quarter.

## Output Features

Five features per ticker per quarter:

| Feature | Definition | Interpretation |
|---|---|---|
| `article_count` | Number of deduplicated articles scored | Attention proxy |
| `mean_sentiment` | Mean of FinBERT scores across articles | Aggregate tone, bounded ~[-1, 1] |
| `sentiment_volatility` | Std dev of article-level scores | Disagreement / narrative uncertainty |
| `surprise_sentiment` | `mean_t - mean_{t-1}` (QoQ delta) | Narrative momentum; `null` for first quarter |
| `cross_sectional_z` | `(mean - mu) / sigma` across tickers in same quarter | Relative strength vs peers |

## Feature Derivation

### Article-Level Scoring

Each article text is constructed as `headline + ". " + summary` and scored by ProsusAI/finbert:

```
sentiment = P(positive) - P(negative)
```

This yields a continuous score in approximately [-1, 1].

### Exponential Time-Decay (Optional)

If `--half-life-days` is provided, article weights are:

```
weight = exp(-ln(2) * age_days / half_life_days)
```

Where `age_days = (quarter_end - publish_time)`. More recent articles receive higher weight. If omitted, all articles are weighted equally.

Weighted mean and std replace simple mean/std when decay is active.

### Surprise Sentiment

```
surprise_t = mean_t - mean_{t-1}
```

First quarter in the range has `surprise = null`. Captures narrative shift between consecutive quarters.

### Cross-Sectional Z-Score

For each quarter, computed across all tickers with data:

```
mu_t    = mean(mean_sentiment across tickers)
sigma_t = std(mean_sentiment across tickers)
z_i,t   = (mean_sentiment_i,t - mu_t) / sigma_t
```

Removes time-level drift. Enables relative ranking across tickers.

## Architecture

### Two-Stage Processing

**Stage 1 — News fetch + text preparation** (optionally parallel via `--workers`):
- Queries Finnhub `/company-news` per ticker per quarter
- Deduplicates headlines, drops missing summaries
- Caches raw API responses to `news_cache/`

**Stage 2 — FinBERT scoring** (single process, MPS/GPU):
- Loads model once, scores all articles in batches
- Batched inference with configurable `--batch-size`

### Rate Limiting

Finnhub free tier allows 60 requests/minute. The script enforces 1.1 seconds between API calls via a thread-safe lock, keeping throughput at ~55 req/min.

### Caching

Two cache layers:
1. **News cache** (`news_cache/`): raw Finnhub responses cached by `{TICKER}_{START}_{END}.json`. Persists across runs — delete manually to re-fetch.
2. **Output cache**: if a quarter's output file exists and contains all requested tickers, it is loaded instead of re-scoring. Use `--force` to bypass.

FinBERT is only loaded if at least one quarter needs scoring.

### Device Selection

- Apple MPS if available (fast on M-series Macs)
- Otherwise CPU
- Controlled automatically; no flag needed

## Ticker Universe

When using `--supported`, tickers are loaded from `data-pipeline/supported_tickers.yaml` (19 tickers). The SECTOR_MAP in the script covers all 19:

| Sector | Tickers |
|---|---|
| Technology | AAPL, NVDA, MSFT |
| Financials | JPM, GS, BAC |
| Healthcare | UNH, LLY, JNJ |
| Industrials | CAT, DAL |
| Consumer Defensive | COST, WMT |
| Consumer Discretionary | AMZN, TSLA |
| Communication Services | GOOG, META |
| Energy | XOM |
| Real Estate | AMT |

## Limitations

- Finnhub free tier only returns ~1 year of news history. Older quarters will return null.
- FinBERT max token length is 512; longer articles are truncated.
- Cross-sectional z-score requires >= 2 tickers with data in a quarter.
- Sentiment may already be priced in by the time of the rebalance date.

## How Agents Should Use These Features

Agents should treat these as probabilistic signals, not deterministic truths.

**`cross_sectional_z`** — Primary ranking signal. Higher z = stronger sentiment relative to peers. Supports long-short and ranking-based allocation.

**`surprise_sentiment`** — Momentum signal. Positive surprise strengthens conviction; negative surprise weakens it. Strong negative surprise may override positive level.

**`mean_sentiment`** — Absolute tone. Level matters less than relative strength. Level + surprise together form a regime signal.

**`sentiment_volatility`** — Risk/uncertainty signal. High volatility = disagreement. Strong signal + low volatility = high conviction. Strong signal + high volatility = unstable thesis.

**`article_count`** — Attention proxy. High count = more credible signals. Low count = possible underreaction. Extremely high = possible crowded trade.

### Signal Combinations

Strong long: high `cross_sectional_z` + positive `surprise_sentiment` + low `sentiment_volatility` + sufficient `article_count`.

Weak / avoid: negative `cross_sectional_z` + negative `surprise_sentiment` + high `sentiment_volatility`.

Conflict: high `cross_sectional_z` + strong negative `surprise_sentiment` — agents should debate whether sentiment is peaking, reversing, or noisy.
