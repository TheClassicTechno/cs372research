#!/usr/bin/env python3
"""
================================================================================
FINNHUB → FINBERT → QUARTERLY SENTIMENT FEATURE PIPELINE
================================================================================

OVERVIEW
--------
This script builds a complete quarterly sentiment feature dataset for a list
of tickers using:

    1) Finnhub news API
    2) FinBERT financial sentiment model
    3) Optional exponential time-decay weighting
    4) Cross-sectional and sector normalization
    5) Rolling 12-month normalization
    6) Optional Sharpe ratio regression against sentiment features
    7) Optional multiprocessing (IO or full scoring parallelism)

The output is a JSON file containing:

    - Per-ticker, per-quarter sentiment features
    - Cross-sectional standardizations
    - Sector-relative adjustments
    - Rolling z-scores
    - Optional regression diagnostics

This pipeline is designed for:
    - Multi-agent trading experiments
    - Sentiment factor research
    - Economic backtesting
    - Research reproducibility


================================================================================
PIPELINE FLOW
================================================================================

RAW NEWS (Finnhub)
        ↓
Deduplication + Text Construction
        ↓
FinBERT Sentiment Scoring (batched)
        ↓
Optional Exponential Time-Decay Weighting
        ↓
Quarterly Aggregation (mean, std)
        ↓
Surprise (QoQ delta)
        ↓
Volume Z-score (per ticker)
        ↓
Cross-Sectional Z-score (per quarter)
        ↓
Sector Baselines (per quarter)
        ↓
Rolling 12-Month Normalization
        ↓
(Optional) Sharpe Regression
        ↓
JSON Output


================================================================================
1) NEWS INGESTION
================================================================================

For each ticker and quarter:

- We compute quarter boundaries via get_quarter_range().
- We query Finnhub's /company-news endpoint.
- Results are cached locally to avoid redundant API calls.

Deduplication steps:
    - Remove duplicate headlines.
    - Drop rows with missing finished_summaries.
    - Construct article text as:
          headline + ". " + summary

We retain article publication timestamps (UTC) for decay weighting.


================================================================================
2) SENTIMENT SCORING (FinBERT)
================================================================================

Each article is scored using:

    ProsusAI/finbert

FinBERT outputs logits for:
    [negative, neutral, positive]

We convert to probabilities via softmax, then compute a scalar sentiment:

    sentiment = P(positive) − P(negative)

This yields a continuous score in approximately [-1, 1].

Scoring is batched for efficiency.

Device selection:
    - Uses Apple MPS if available
    - Otherwise CPU
    - Full multiprocessing forces CPU per worker


================================================================================
3) EXPONENTIAL TIME DECAY (OPTIONAL)
================================================================================

If --half-life-days is provided, article weights are computed as:

    weight = exp( -ln(2) * age_days / half_life_days )

Where:
    age_days = (quarter_end - publish_time)

Interpretation:
    - More recent articles receive higher weight.
    - Half-life defines when weight halves.
    - If omitted → uniform weighting.

Weighted statistics are then computed:

    weighted_mean
    weighted_std

This makes the quarter’s sentiment more sensitive to recent narrative shifts.


================================================================================
4) QUARTERLY AGGREGATION
================================================================================

For each (ticker, quarter), we compute:

    article_count
    mean_sentiment
    sentiment_volatility
    decay_half_life_days

If no articles exist → quarter entry is None.


================================================================================
5) SURPRISE SENTIMENT (QoQ)
================================================================================

Computed as:

    surprise_t = mean_t − mean_{t−1}

First available quarter has:
    surprise = None

This captures narrative momentum.


================================================================================
6) VOLUME Z-SCORE (PER TICKER)
================================================================================

Within each ticker across quarters:

    volume_z = (article_count − mean_count) / std_count

This captures abnormal attention spikes relative to that ticker’s own history.


================================================================================
7) CROSS-SECTIONAL NORMALIZATION (PER QUARTER)
================================================================================

For each quarter:

    mu = mean of mean_sentiment across tickers
    sigma = std of mean_sentiment across tickers

Then:

    cross_sectional_z = (mean_sentiment − mu) / sigma

This removes time-level drift and enables ranking-based trading.

Additionally, per sector:

    sector_baseline = avg(mean_sentiment within sector)
    relative_sentiment = mean_sentiment − sector_baseline
    sector_volume_baseline = avg(article_count within sector)
    relative_volume = article_count − sector_volume_baseline


================================================================================
8) ROLLING 12-MONTH NORMALIZATION
================================================================================

For each ticker:

- Rolling window of last 4 quarters (≈ 12 months)
- Compute rolling mean and std of mean_sentiment
- Add:

    rolling_12m_mean_sentiment
    rolling_12m_std_sentiment
    rolling_12m_z_sentiment

If < 2 quarters available:
    std = None
    z = None

This captures medium-term regime positioning.


================================================================================
9) MULTIPROCESSING MODES
================================================================================

Two modes:

------------------------------------------------
mp-mode = "io"   (RECOMMENDED)
------------------------------------------------

Stage 1:
    Parallel fetch + prepare texts

Stage 2:
    Single-process FinBERT scoring
    (Fast on GPU/MPS)

Best for:
    Apple MPS
    GPU acceleration
    Memory efficiency

------------------------------------------------
mp-mode = "full"
------------------------------------------------

Each worker:
    - Loads its own FinBERT model
    - Scores one ticker fully

True parallel scoring across tickers,
but memory-heavy and CPU-bound.

Best for:
    Large CPU servers
    No GPU/MPS


================================================================================
10) OPTIONAL SHARPE REGRESSION
================================================================================

If --enable-sharpe:

- Fetch price data via yfinance
- Compute quarterly Sharpe:

      sharpe = mean(daily_returns) / std(daily_returns) * sqrt(252)

Then perform pooled OLS:

    sharpe ~ rolling_12m_z_sentiment
             + surprise_sentiment
             + sentiment_volatility

This provides exploratory economic validation.


================================================================================
11) OUTPUT STRUCTURE
================================================================================

JSON structure:

{
    "meta": {...},
    "results": {
        "TICKER": {
            "YEARQ#": {
                sentiment features...
            }
        }
    },
    "regressions": {...}
}

All transformations are deterministic given:
    - API data
    - Sentiment model
    - Quarter boundaries


================================================================================
12) DESIGN PHILOSOPHY
================================================================================

This pipeline prioritizes:

    - Statistical transparency
    - Deterministic transformations
    - Modular structure
    - Economic interpretability
    - Multi-agent trading compatibility

It separates:

    Data ingestion
    Feature construction
    Normalization
    Optional economic validation

This makes it suitable for:

    Research
    Backtesting
    Agent simulation
    Academic publication


================================================================================
END OF DOCUMENTATION BLOCK
================================================================================
"""


"""
Finnhub news sentiment pipeline with:
- multiprocessing across tickers (io-prep parallel or full scoring parallel)
- exponential time-decay weighting (article-level)
- rolling 12-month normalization (rolling 4 quarters) for sentiment
- optional Sharpe regression vs sentiment (requires yfinance)

Example:
  python sentiment.py --tickers NVDA,AAPL,MSFT --year 2024 --api-key $FINNHUB_KEY --output out.json

With multiprocessing (IO-only parallelism; recommended on MPS/GPU):
  python sentiment.py --tickers NVDA,AAPL,MSFT --year 2024 --api-key ... --workers 4 --mp-mode io

With full multiprocessing (each worker loads FinBERT; CPU-friendly but heavy):
  python sentiment.py --tickers NVDA,AAPL,MSFT --year 2024 --api-key ... --workers 3 --mp-mode full

With exponential decay:
  python sentiment.py ... --half-life-days 7

With Sharpe regression (requires yfinance):
  python sentiment.py ... --enable-sharpe
"""

import argparse
import concurrent.futures as cf
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import torch
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# =====================================================
# CONFIGURATION
# =====================================================

CACHE_DIR = "news_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

SECTOR_MAP = {
    "NVDA": "Tech",
    "AAPL": "Tech",
    "MSFT": "Tech",
    "BAC": "Financials",
    "AXP": "Financials",
    "CVX": "Energy",
    "KO": "Consumer",
    "UBER": "Consumer",
}

FINBERT_MODEL_NAME = "ProsusAI/finbert"


# =====================================================
# TIME / QUARTER UTILS
# =====================================================

def get_quarter_range(year: int, quarter: int) -> Tuple[str, str]:
    start = datetime(year, 1, 1) + relativedelta(months=3 * (quarter - 1))
    end = start + relativedelta(months=3) - relativedelta(days=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def quarter_key(year: int, quarter: int) -> str:
    return f"{year}Q{quarter}"


def parse_finnhub_datetime(article: dict) -> Optional[datetime]:
    """
    Finnhub company-news items include 'datetime' (unix seconds).
    Returns UTC datetime or None.
    """
    ts = article.get("datetime")
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc)
    except Exception:
        return None


# =====================================================
# FINBERT (batched) - main process path
# =====================================================

def init_finbert(device_str: Optional[str] = None):
    print("Loading FinBERT...")
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)

    if device_str is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    model.to(device)
    model.eval()
    print(f"Using device: {device}")
    return tokenizer, model, device


def finbert_score_batch(
    texts: List[str],
    tokenizer,
    model,
    device,
    batch_size: int = 16,
) -> List[float]:
    scores: List[float] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # FinBERT logits order: [negative, neutral, positive]
        batch_scores = (probs[:, 2] - probs[:, 0]).detach().cpu().tolist()
        scores.extend([float(x) for x in batch_scores])
    return scores


# =====================================================
# NEWS FETCH + PREP
# =====================================================

def fetch_news(ticker: str, start: str, end: str, api_key: str) -> List[dict]:
    cache_path = os.path.join(CACHE_DIR, f"{ticker}_{start}_{end}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    url = "https://finnhub.io/api/v1/company-news"
    params = {"symbol": ticker, "from": start, "to": end, "token": api_key}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    with open(cache_path, "w") as f:
        json.dump(data, f)

    return data


@dataclass
class PreparedQuarter:
    ticker: str
    qname: str
    texts: List[str]
    published_utc: List[Optional[datetime]]  # aligns to texts


def prepare_texts_for_ticker_quarter(
    ticker: str,
    year: int,
    quarter: int,
    api_key: str,
) -> PreparedQuarter:
    start, end = get_quarter_range(year, quarter)
    raw_news = fetch_news(ticker, start, end, api_key)
    df = pd.DataFrame(raw_news)

    qname = quarter_key(year, quarter)
    if df.empty:
        return PreparedQuarter(ticker=ticker, qname=qname, texts=[], published_utc=[])

    # Dedup and basic hygiene
    if "headline" in df.columns:
        df = df.drop_duplicates(subset=["headline"])
    df = df[df.get("summary").notnull()]

    # Build text
    df["text"] = df["headline"].fillna("").astype(str) + ". " + df["summary"].fillna("").astype(str)

    # Parse publish times
    published = []
    for a in raw_news:
        published.append(parse_finnhub_datetime(a))

    # Align publish times to df rows as best-effort:
    # If dataframe row count differs from raw_news after filtering, we re-derive timestamps from df if present.
    if len(published) != len(df):
        # Finnhub sometimes includes "datetime" column in df
        if "datetime" in df.columns:
            published = []
            for ts in df["datetime"].tolist():
                try:
                    published.append(datetime.fromtimestamp(int(ts), tz=timezone.utc))
                except Exception:
                    published.append(None)
        else:
            published = [None] * len(df)

    texts = df["text"].tolist()
    if len(published) != len(texts):
        published = (published[: len(texts)] + [None] * len(texts))[: len(texts)]

    return PreparedQuarter(ticker=ticker, qname=qname, texts=texts, published_utc=published)


# =====================================================
# EXPONENTIAL DECAY WEIGHTING (within quarter)
# =====================================================

def exp_decay_weights(
    published_times: List[Optional[datetime]],
    half_life_days: Optional[float],
    anchor_time: datetime,
) -> List[float]:
    """
    weight = exp(-ln(2) * age_days / half_life_days)
    If half_life_days is None: uniform weights.
    """
    if not half_life_days or half_life_days <= 0:
        return [1.0] * len(published_times)

    ln2 = math.log(2.0)
    weights: List[float] = []
    for dt in published_times:
        if dt is None:
            weights.append(1.0)
            continue
        age_days = max(0.0, (anchor_time - dt).total_seconds() / (3600 * 24))
        w = math.exp(-ln2 * age_days / half_life_days)
        weights.append(float(w))
    return weights


def weighted_mean_and_std(values: List[float], weights: List[float]) -> Tuple[float, float]:
    """
    Returns (weighted_mean, weighted_std) using population-style weighted variance.
    """
    if not values:
        return (float("nan"), float("nan"))
    if len(values) != len(weights):
        raise ValueError("values and weights must have same length")

    wsum = sum(weights)
    if wsum <= 0:
        return (float("nan"), float("nan"))

    mu = sum(w * x for w, x in zip(weights, values)) / wsum
    var = sum(w * (x - mu) ** 2 for w, x in zip(weights, values)) / wsum
    return float(mu), float(math.sqrt(var))


# =====================================================
# SHARPE (optional) + REGRESSION
# =====================================================

def fetch_quarter_prices_yfinance(ticker: str, start: str, end: str) -> pd.Series:
    """
    Returns a Series of adjusted close prices indexed by date.
    Requires yfinance installed.
    """
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return pd.Series(dtype=float)
    col = "Close" if "Close" in df.columns else df.columns[0]
    s = df[col].dropna()
    s.index = pd.to_datetime(s.index)
    return s


def compute_quarter_sharpe_from_prices(prices: pd.Series) -> Optional[float]:
    """
    Simple annualized Sharpe using daily returns and rf=0:
      sharpe = mean(daily_ret)/std(daily_ret) * sqrt(252)
    """
    if prices is None or prices.empty or len(prices) < 10:
        return None
    rets = prices.pct_change().dropna()
    if rets.empty:
        return None
    mu = float(rets.mean())
    sigma = float(rets.std())
    if sigma <= 0:
        return None
    return float(mu / sigma * math.sqrt(252.0))


def ols_fit(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Basic OLS with intercept using numpy.linalg.lstsq.
    Returns coefficients, R^2, and n.
    """
    import numpy as np

    X_ = X.copy()
    X_["intercept"] = 1.0
    cols = ["intercept"] + [c for c in X.columns]
    A = X_[cols].to_numpy(dtype=float)
    b = y.to_numpy(dtype=float)

    coef, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    yhat = A @ coef
    ss_res = float(((b - yhat) ** 2).sum())
    ss_tot = float(((b - b.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

    out = {
        "n": int(len(y)),
        "rank": int(rank),
        "r2": r2,
        "coefficients": {cols[i]: float(coef[i]) for i in range(len(cols))},
    }
    return out


# =====================================================
# ROLLING 12-MONTH NORMALIZATION (rolling 4 quarters)
# =====================================================

def add_rolling_12m_zscores(results: Dict[str, Dict[str, Any]]) -> None:
    """
    For each ticker, compute rolling mean/std over last 4 observed quarters of mean_sentiment,
    then add:
      rolling_12m_mean, rolling_12m_std, rolling_12m_z
    """
    for ticker, tmap in results.items():
        # sort quarter keys like "2024Q1"
        keys = sorted(tmap.keys())
        series = []
        for k in keys:
            rec = tmap[k]
            if rec and rec.get("mean_sentiment") is not None:
                series.append((k, float(rec["mean_sentiment"])))

        # rolling over series order
        for i, (k, _) in enumerate(series):
            window = series[max(0, i - 3) : i + 1]  # up to 4 quarters
            vals = [v for _, v in window]
            if len(vals) < 2:
                mu = float(sum(vals) / len(vals))
                sd = None
                z = None
            else:
                mu = float(sum(vals) / len(vals))
                var = float(sum((x - mu) ** 2 for x in vals) / len(vals))
                sd = float(math.sqrt(var))
                z = float((series[i][1] - mu) / sd) if sd and sd > 0 else 0.0

            rec = results[ticker].get(k)
            if rec:
                rec["rolling_12m_mean_sentiment"] = mu
                rec["rolling_12m_std_sentiment"] = sd
                rec["rolling_12m_z_sentiment"] = z


# =====================================================
# MULTIPROCESSING WORKERS
# =====================================================

def worker_full_compute_one_ticker(args: Tuple[str, int, List[int], str, int, Optional[float]]) -> Tuple[str, Dict[str, Any]]:
    """
    FULL multiprocessing: each worker loads FinBERT and processes one ticker across quarters.
    Heavy but true "across tickers" parallelism.
    """
    ticker, year, quarters, api_key, batch_size, half_life_days = args

    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    t_res: Dict[str, Any] = {}

    for q in quarters:
        qname = quarter_key(year, q)
        start, end = get_quarter_range(year, q)
        raw_news = fetch_news(ticker, start, end, api_key)
        df = pd.DataFrame(raw_news)

        if df.empty:
            t_res[qname] = None
            continue

        if "headline" in df.columns:
            df = df.drop_duplicates(subset=["headline"])
        df = df[df.get("summary").notnull()]
        df["text"] = df["headline"].fillna("").astype(str) + ". " + df["summary"].fillna("").astype(str)

        texts = df["text"].tolist()
        if not texts:
            t_res[qname] = None
            continue

        # times + weights
        published = []
        if "datetime" in df.columns:
            for ts in df["datetime"].tolist():
                try:
                    published.append(datetime.fromtimestamp(int(ts), tz=timezone.utc))
                except Exception:
                    published.append(None)
        else:
            published = [None] * len(texts)

        anchor = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        weights = exp_decay_weights(published, half_life_days, anchor)

        # Score batches on CPU
        sentiments: List[float] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            batch_scores = (probs[:, 2] - probs[:, 0]).cpu().tolist()
            sentiments.extend([float(x) for x in batch_scores])

        mu, sd = weighted_mean_and_std(sentiments, weights[: len(sentiments)])
        t_res[qname] = {
            "article_count": int(len(sentiments)),
            "mean_sentiment": float(mu),
            "sentiment_volatility": float(sd),
            "decay_half_life_days": half_life_days,
        }

    return ticker, t_res


# =====================================================
# MAIN
# =====================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", required=True, help="Comma-separated tickers")
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--quarter", type=int, choices=[1, 2, 3, 4], help="If omitted, computes all 4 quarters")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--output", default="sentiment_output.json")

    # batching / decay
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--half-life-days", type=float, default=None, help="Exponential decay half-life in days (article-level). Omit for uniform weights.")

    # multiprocessing
    parser.add_argument("--workers", type=int, default=0, help="Number of worker processes. 0 disables multiprocessing.")
    parser.add_argument(
        "--mp-mode",
        choices=["io", "full"],
        default="io",
        help="io = parallel fetch/prep only (recommended). full = each worker loads FinBERT and scores its ticker.",
    )

    # sharpe regression
    parser.add_argument("--enable-sharpe", action="store_true", help="Compute quarterly Sharpe via yfinance and regress Sharpe vs sentiment features.")

    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    year = args.year
    quarters = [args.quarter] if args.quarter else [1, 2, 3, 4]
    qnames = [quarter_key(year, q) for q in quarters]

    results: Dict[str, Dict[str, Any]] = {t: {} for t in tickers}
    regressions: Dict[str, Any] = {}

    # -------------------------------------------------
    # OPTION A: FULL multiprocessing (true across tickers)
    # -------------------------------------------------
    if args.workers and args.workers > 0 and args.mp_mode == "full":
        work = [(t, year, quarters, args.api_key, args.batch_size, args.half_life_days) for t in tickers]
        with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(worker_full_compute_one_ticker, w) for w in work]
            for fut in tqdm(cf.as_completed(futures), total=len(futures), desc="Tickers (full mp)"):
                ticker, t_res = fut.result()
                results[ticker] = t_res

    # -------------------------------------------------
    # OPTION B: IO multiprocessing (recommended):
    #   1) parallel fetch + prepare texts
    #   2) single-process FinBERT scoring (fast on MPS/GPU)
    # -------------------------------------------------
    else:
        tokenizer, model, device = init_finbert()

        prepared: Dict[str, Dict[str, PreparedQuarter]] = {t: {} for t in tickers}

        # Stage 1: parallel prep
        prep_jobs = [(t, year, q, args.api_key) for t in tickers for q in quarters]
        if args.workers and args.workers > 0:
            with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
                futures = [
                    ex.submit(prepare_texts_for_ticker_quarter, t, year, q, args.api_key)
                    for (t, year, q, _) in prep_jobs
                ]
                for fut in tqdm(cf.as_completed(futures), total=len(futures), desc="Prep (io mp)"):
                    pq = fut.result()
                    prepared[pq.ticker][pq.qname] = pq
        else:
            for t in tickers:
                for q in quarters:
                    pq = prepare_texts_for_ticker_quarter(t, year, q, args.api_key)
                    prepared[pq.ticker][pq.qname] = pq

        # Stage 2: FinBERT scoring (single process)
        for ticker in tickers:
            for q in quarters:
                qname = quarter_key(year, q)
                pq = prepared[ticker].get(qname)
                if pq is None or not pq.texts:
                    results[ticker][qname] = None
                    continue

                start, end = get_quarter_range(year, q)
                anchor = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                weights = exp_decay_weights(pq.published_utc, args.half_life_days, anchor)

                sentiments = finbert_score_batch(
                    pq.texts,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    batch_size=args.batch_size,
                )
                if not sentiments:
                    results[ticker][qname] = None
                    continue

                # Weighted stats
                w = weights[: len(sentiments)]
                mu, sd = weighted_mean_and_std(sentiments, w)

                results[ticker][qname] = {
                    "article_count": int(len(sentiments)),
                    "mean_sentiment": float(mu),
                    "sentiment_volatility": float(sd),
                    "decay_half_life_days": args.half_life_days,
                }

    # -------------------------------------------------
    # Surprise sentiment (QoQ delta)
    # -------------------------------------------------
    for ticker in tickers:
        prev = None
        for q in sorted(results[ticker].keys()):
            rec = results[ticker][q]
            cur = rec["mean_sentiment"] if rec else None
            surprise = None if prev is None or cur is None else float(cur - prev)
            if rec:
                rec["surprise_sentiment"] = surprise
            prev = cur

    # -------------------------------------------------
    # Volume z-score per ticker
    # -------------------------------------------------
    for ticker in tickers:
        counts = [results[ticker][q]["article_count"] for q in results[ticker] if results[ticker][q]]
        if len(counts) < 2:
            continue
        mu = sum(counts) / len(counts)
        sigma = math.sqrt(sum((x - mu) ** 2 for x in counts) / len(counts))
        for q in results[ticker]:
            if results[ticker][q]:
                x = results[ticker][q]["article_count"]
                results[ticker][q]["volume_zscore"] = float((x - mu) / sigma) if sigma > 0 else 0.0

    # -------------------------------------------------
    # Cross-sectional + sector baselines (per quarter)
    # -------------------------------------------------
    for q in quarters:
        qname = quarter_key(year, q)
        sents = [results[t][qname]["mean_sentiment"] for t in tickers if results[t].get(qname)]
        if len(sents) < 2:
            continue
        mu = sum(sents) / len(sents)
        sigma = math.sqrt(sum((x - mu) ** 2 for x in sents) / len(sents))

        sector_sents: Dict[str, List[float]] = {}
        sector_vols: Dict[str, List[int]] = {}

        for ticker in tickers:
            rec = results[ticker].get(qname)
            if not rec:
                continue
            sector = SECTOR_MAP.get(ticker)
            if not sector:
                continue
            sector_sents.setdefault(sector, []).append(float(rec["mean_sentiment"]))
            sector_vols.setdefault(sector, []).append(int(rec["article_count"]))

        sector_sent_mean = {s: sum(v) / len(v) for s, v in sector_sents.items()}
        sector_vol_mean = {s: sum(v) / len(v) for s, v in sector_vols.items()}

        for ticker in tickers:
            rec = results[ticker].get(qname)
            if not rec:
                continue

            s = float(rec["mean_sentiment"])
            sector = SECTOR_MAP.get(ticker)

            rec["cross_sectional_z"] = float((s - mu) / sigma) if sigma > 0 else 0.0

            base_sent = sector_sent_mean.get(sector) if sector else None
            base_vol = sector_vol_mean.get(sector) if sector else None

            rec["sector_baseline"] = base_sent
            rec["relative_sentiment"] = float(s - base_sent) if base_sent is not None else None

            rec["sector_volume_baseline"] = base_vol
            rec["relative_volume"] = float(rec["article_count"] - base_vol) if base_vol is not None else None

    # -------------------------------------------------
    # Rolling 12-month normalization (rolling 4 quarters)
    # -------------------------------------------------
    add_rolling_12m_zscores(results)

    # -------------------------------------------------
    # Optional: Sharpe regression vs sentiment
    # -------------------------------------------------
    if args.enable_sharpe:
        rows = []
        for ticker in tickers:
            for q in quarters:
                qname = quarter_key(year, q)
                rec = results[ticker].get(qname)
                if not rec:
                    continue
                start, end = get_quarter_range(year, q)
                try:
                    prices = fetch_quarter_prices_yfinance(ticker, start, end)
                    sharpe = compute_quarter_sharpe_from_prices(prices)
                except Exception:
                    sharpe = None

                rec["quarterly_sharpe"] = sharpe

                if sharpe is None:
                    continue

                rows.append(
                    {
                        "ticker": ticker,
                        "quarter": qname,
                        "mean_sentiment": rec.get("mean_sentiment"),
                        "sentiment_volatility": rec.get("sentiment_volatility"),
                        "surprise_sentiment": rec.get("surprise_sentiment"),
                        "rolling_12m_z_sentiment": rec.get("rolling_12m_z_sentiment"),
                        "cross_sectional_z": rec.get("cross_sectional_z"),
                        "sharpe": sharpe,
                    }
                )

        df_reg = pd.DataFrame(rows).dropna(subset=["sharpe"])
        if len(df_reg) >= 8:
            # Choose a small feature set to avoid overfitting in tiny samples
            feature_cols = [c for c in ["rolling_12m_z_sentiment", "surprise_sentiment", "sentiment_volatility"] if c in df_reg.columns]
            X = df_reg[feature_cols].fillna(0.0)
            y = df_reg["sharpe"].astype(float)
            regressions["pooled_sharpe_on_sentiment"] = {
                "features": feature_cols,
                "ols": ols_fit(X, y),
            }
        else:
            regressions["pooled_sharpe_on_sentiment"] = {
                "warning": "Not enough (ticker,quarter) samples with Sharpe to run regression (need ~8+).",
                "n": int(len(df_reg)),
            }

    # -------------------------------------------------
    # Save
    # -------------------------------------------------
    out = {
        "meta": {
            "year": year,
            "quarters": qnames,
            "tickers": tickers,
            "half_life_days": args.half_life_days,
            "mp_mode": args.mp_mode,
            "workers": args.workers,
            "enable_sharpe": bool(args.enable_sharpe),
        },
        "results": results,
        "regressions": regressions,
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print("\nSaved to", args.output)


if __name__ == "__main__":
    main()