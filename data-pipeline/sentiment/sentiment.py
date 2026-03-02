#!/usr/bin/env python3
"""
Finnhub -> FinBERT -> Quarterly Sentiment Feature Pipeline.

Builds per-ticker quarterly sentiment features using:
  1) Finnhub news API
  2) FinBERT financial sentiment model (ProsusAI/finbert)
  3) Optional exponential time-decay weighting
  4) Cross-sectional z-score normalization

Output:
  data-pipeline/sentiment/data/sentiment_{YEAR}_Q{#}.json

Per-ticker features: article_count, mean_sentiment, sentiment_volatility,
surprise_sentiment, cross_sectional_z.

Examples:

  # Quarter range with supported tickers
  python sentiment.py --start 2025Q1 --end 2025Q3 --supported --api-key $FINNHUB_KEY

  # Single quarter, custom tickers
  python sentiment.py --year 2025 --quarter 2 --tickers AAPL,NVDA --api-key $FINNHUB_KEY

  # Parallel news fetching (recommended)
  python sentiment.py --start 2025Q1 --end 2025Q3 --supported --api-key $FINNHUB_KEY --workers 4

  # With exponential decay
  python sentiment.py --start 2025Q1 --end 2025Q3 --supported --api-key $FINNHUB_KEY --half-life-days 7
"""

import argparse
import concurrent.futures as cf
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import torch
import yaml
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# =====================================================
# PATHS
# =====================================================

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA_DIR = _SCRIPT_DIR / "data"
_SUPPORTED_TICKERS_PATH = _SCRIPT_DIR.parent / "supported_tickers.yaml"
_CACHE_DIR = _SCRIPT_DIR / "news_cache"

# =====================================================
# CONFIGURATION
# =====================================================

FINNHUB_RATE_LIMIT = 1.1  # seconds between API calls (~55 req/min, under 60/min free tier)
_finnhub_lock = threading.Lock()

SECTOR_MAP = {
    "CAT": "Industrials",
    "DAL": "Industrials",
    "JPM": "Financials",
    "GS": "Financials",
    "BAC": "Financials",
    "UNH": "Healthcare",
    "LLY": "Healthcare",
    "JNJ": "Healthcare",
    "AMT": "Real Estate",
    "XOM": "Energy",
    "COST": "Consumer Defensive",
    "WMT": "Consumer Defensive",
    "AAPL": "Technology",
    "NVDA": "Technology",
    "MSFT": "Technology",
    "GOOG": "Communication Services",
    "AMZN": "Consumer Discretionary",
    "META": "Communication Services",
    "TSLA": "Consumer Discretionary",
}

FINBERT_MODEL_NAME = "ProsusAI/finbert"


# =====================================================
# TICKER LOADING
# =====================================================

def load_supported_tickers(yaml_path: Path = _SUPPORTED_TICKERS_PATH) -> list:
    """Load ticker symbols from supported_tickers.yaml."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return [entry["symbol"] for entry in data["supported_tickers"]]


# =====================================================
# TIME / QUARTER UTILS
# =====================================================

def get_quarter_range(year: int, quarter: int) -> Tuple[str, str]:
    """Return (start_date, end_date) strings for a quarter."""
    start = datetime(year, 1, 1) + relativedelta(months=3 * (quarter - 1))
    end = start + relativedelta(months=3) - relativedelta(days=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def quarter_key(year: int, quarter: int) -> str:
    return f"{year}Q{quarter}"


def parse_quarter_string(qstr: str) -> Tuple[int, int]:
    """'2025Q2' -> (2025, 2)"""
    year = int(qstr[:4])
    q = int(qstr[5])
    if q not in (1, 2, 3, 4):
        raise ValueError(f"Invalid quarter: {qstr}")
    return year, q


def next_quarter(year: int, quarter: int) -> Tuple[int, int]:
    if quarter < 4:
        return year, quarter + 1
    return year + 1, 1


def quarter_range_list(start: str, end: str) -> List[Tuple[int, int]]:
    """Generate list of (year, quarter_int) from start to end inclusive."""
    y, q = parse_quarter_string(start)
    end_y, end_q = parse_quarter_string(end)
    result = []
    while True:
        result.append((y, q))
        if y == end_y and q == end_q:
            break
        y, q = next_quarter(y, q)
    return result


def parse_finnhub_datetime(article: dict) -> Optional[datetime]:
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
    print("Loading FinBERT...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)

    if device_str is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    model.to(device)
    model.eval()
    print(f"Using device: {device}", flush=True)
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
    cache_dir = _CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{ticker}_{start}_{end}.json"
    if cache_path.exists():
        with open(cache_path, "r") as f:
            return json.load(f)

    with _finnhub_lock:
        time.sleep(FINNHUB_RATE_LIMIT)

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
    year: int
    quarter: int
    qname: str
    texts: List[str]
    published_utc: List[Optional[datetime]]


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
        return PreparedQuarter(ticker=ticker, year=year, quarter=quarter,
                               qname=qname, texts=[], published_utc=[])

    # Dedup and basic hygiene
    if "headline" in df.columns:
        df = df.drop_duplicates(subset=["headline"])
    df = df[df.get("summary").notnull()]

    # Build text
    df["text"] = df["headline"].fillna("").astype(str) + ". " + df["summary"].fillna("").astype(str)

    # Parse publish times — derive from df rows after filtering
    published = []
    if "datetime" in df.columns:
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

    return PreparedQuarter(ticker=ticker, year=year, quarter=quarter,
                           qname=qname, texts=texts, published_utc=published)


# =====================================================
# EXPONENTIAL DECAY WEIGHTING (within quarter)
# =====================================================

def exp_decay_weights(
    published_times: List[Optional[datetime]],
    half_life_days: Optional[float],
    anchor_time: datetime,
) -> List[float]:
    if not half_life_days or half_life_days <= 0:
        return [1.0] * len(published_times)

    ln2 = math.log(2.0)
    weights: List[float] = []
    for dt_val in published_times:
        if dt_val is None:
            weights.append(1.0)
            continue
        age_days = max(0.0, (anchor_time - dt_val).total_seconds() / (3600 * 24))
        w = math.exp(-ln2 * age_days / half_life_days)
        weights.append(float(w))
    return weights


def weighted_mean_and_std(values: List[float], weights: List[float]) -> Tuple[float, float]:
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
# SCORE ONE QUARTER  (all tickers for a single quarter)
# =====================================================

def score_quarter(
    tickers: List[str],
    year: int,
    quarter: int,
    api_key: str,
    half_life_days: Optional[float],
    batch_size: int,
    workers: int,
    tokenizer,
    model,
    device,
) -> Dict[str, Any]:
    """
    Fetch news + score sentiment for all tickers in a single quarter.
    Returns {ticker: {features...} or None}.
    """
    qname = quarter_key(year, quarter)
    results: Dict[str, Any] = {}

    # Stage 1: prepare texts (optionally parallel IO)
    prepared: Dict[str, PreparedQuarter] = {}
    if workers > 0:
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(prepare_texts_for_ticker_quarter, t, year, quarter, api_key): t
                for t in tickers
            }
            for fut in tqdm(cf.as_completed(futures), total=len(futures),
                            desc=f"  {qname} fetch", leave=False):
                pq = fut.result()
                prepared[pq.ticker] = pq
    else:
        for t in tqdm(tickers, desc=f"  {qname} fetch", leave=False):
            pq = prepare_texts_for_ticker_quarter(t, year, quarter, api_key)
            prepared[t] = pq

    # Stage 2: FinBERT scoring (single process — fast on MPS/GPU)
    start_str, end_str = get_quarter_range(year, quarter)
    anchor = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    for t in tqdm(tickers, desc=f"  {qname} score", leave=False):
        pq = prepared.get(t)
        if pq is None or not pq.texts:
            results[t] = None
            continue

        weights = exp_decay_weights(pq.published_utc, half_life_days, anchor)
        sentiments = finbert_score_batch(
            pq.texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=batch_size,
        )
        if not sentiments:
            results[t] = None
            continue

        w = weights[: len(sentiments)]
        mu, sd = weighted_mean_and_std(sentiments, w)

        results[t] = {
            "article_count": int(len(sentiments)),
            "mean_sentiment": round(float(mu), 6),
            "sentiment_volatility": round(float(sd), 6),
        }

    return results


# =====================================================
# POST-PROCESSING: surprise, volume z, cross-sectional
# =====================================================

def add_surprise_sentiment(all_results: Dict[str, Dict[str, Any]]) -> None:
    """QoQ delta of mean_sentiment."""
    for ticker, tmap in all_results.items():
        prev = None
        for q in sorted(tmap.keys()):
            rec = tmap[q]
            cur = rec["mean_sentiment"] if rec else None
            surprise = None if prev is None or cur is None else round(float(cur - prev), 6)
            if rec:
                rec["surprise_sentiment"] = surprise
            prev = cur


def add_cross_sectional_z(
    all_results: Dict[str, Dict[str, Any]],
    tickers: List[str],
    quarters: List[Tuple[int, int]],
) -> None:
    """Cross-sectional z-score of mean_sentiment per quarter."""
    for year, q in quarters:
        qname = quarter_key(year, q)
        sents = [all_results[t][qname]["mean_sentiment"]
                 for t in tickers if all_results[t].get(qname)]
        if len(sents) < 2:
            continue

        mu = sum(sents) / len(sents)
        sigma = math.sqrt(sum((x - mu) ** 2 for x in sents) / len(sents))

        for ticker in tickers:
            rec = all_results[ticker].get(qname)
            if not rec:
                continue
            s = float(rec["mean_sentiment"])
            rec["cross_sectional_z"] = round(float((s - mu) / sigma), 6) if sigma > 0 else 0.0


# =====================================================
# PER-QUARTER OUTPUT BUILDER
# =====================================================

def build_quarter_doc(
    year: int,
    quarter: int,
    tickers: List[str],
    all_results: Dict[str, Dict[str, Any]],
    half_life_days: Optional[float],
) -> dict:
    """Build the JSON document for a single quarter."""
    qname = quarter_key(year, quarter)
    ticker_data = {}
    for t in tickers:
        rec = all_results[t].get(qname)
        ticker_data[t] = rec  # may be None

    return {
        "meta": {
            "year": year,
            "quarter": f"Q{quarter}",
            "quarter_key": qname,
            "tickers": tickers,
            "half_life_days": half_life_days,
        },
        "results": ticker_data,
    }


# =====================================================
# MAIN
# =====================================================

def main():
    p = argparse.ArgumentParser(
        description="Build quarterly sentiment features from Finnhub + FinBERT",
    )
    # Single quarter mode
    p.add_argument("--year", type=int, default=None,
                   help="Year (single quarter mode)")
    p.add_argument("--quarter", type=int, default=None, choices=[1, 2, 3, 4],
                   help="Quarter number (single quarter mode)")
    # Range mode
    p.add_argument("--start", type=str, default=None,
                   help="Start quarter, e.g. 2024Q4")
    p.add_argument("--end", type=str, default=None,
                   help="End quarter, e.g. 2025Q3")
    # Tickers
    p.add_argument("--tickers", default=None,
                   help="Comma-separated tickers")
    p.add_argument("--supported", action="store_true", default=False,
                   help="Use all tickers from supported_tickers.yaml")
    # API
    p.add_argument("--api-key", required=True,
                   help="Finnhub API key")
    # Options
    p.add_argument("--batch-size", type=int, default=16,
                   help="FinBERT batch size (default: 16)")
    p.add_argument("--half-life-days", type=float, default=None,
                   help="Exponential decay half-life in days. Omit for uniform weights.")
    p.add_argument("--workers", type=int, default=0,
                   help="Parallel news fetch workers. 0 = sequential. (default: 0)")
    p.add_argument("--force", action="store_true", default=False,
                   help="Re-score even if output file already exists")
    args = p.parse_args()

    # -- Resolve quarters --
    if args.start and args.end:
        quarters = quarter_range_list(args.start, args.end)
    elif args.year and args.quarter:
        quarters = [(args.year, args.quarter)]
    else:
        p.error("specify --year/--quarter or --start/--end")

    # -- Resolve tickers --
    if args.supported:
        tickers = load_supported_tickers()
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        p.error("either --tickers or --supported is required")

    print(f"Sentiment pipeline: {len(quarters)} quarter(s), "
          f"{len(tickers)} ticker(s)", flush=True)

    # -- Determine which quarters need scoring --
    all_results: Dict[str, Dict[str, Any]] = {t: {} for t in tickers}
    quarters_to_score: List[Tuple[int, int]] = []

    for year, q in quarters:
        qname = quarter_key(year, q)

        if not args.force:
            # Check per-ticker cache files
            missing = []
            cached = []
            for t in tickers:
                ticker_file = _DEFAULT_DATA_DIR / t / f"{year}_Q{q}.json"
                if ticker_file.exists():
                    try:
                        with open(ticker_file, "r") as f:
                            data = json.load(f)
                        all_results[t][qname] = data.get("features")
                        cached.append(t)
                    except (json.JSONDecodeError, OSError):
                        missing.append(t)
                else:
                    missing.append(t)
            if not missing:
                tqdm.write(f"  {qname}: loaded from cache ({len(cached)} tickers)")
                continue
            elif cached:
                tqdm.write(f"  {qname}: {len(missing)} ticker(s) missing, will re-score all")

        quarters_to_score.append((year, q))

    if not quarters_to_score:
        print("All quarters cached. Skipping FinBERT. "
              "Use --force to re-score.", flush=True)
    else:
        # -- Load FinBERT only if there's work to do --
        tokenizer, model, device = init_finbert()

        for year, q in tqdm(quarters_to_score, desc="Scoring", unit="qtr"):
            qname = quarter_key(year, q)
            q_results = score_quarter(
                tickers=tickers,
                year=year,
                quarter=q,
                api_key=args.api_key,
                half_life_days=args.half_life_days,
                batch_size=args.batch_size,
                workers=args.workers,
                tokenizer=tokenizer,
                model=model,
                device=device,
            )
            for t in tickers:
                all_results[t][qname] = q_results.get(t)

    # -- Post-processing (needs all quarters) --
    print("Post-processing...", flush=True)
    add_surprise_sentiment(all_results)

    # -- Save per-ticker files --
    count = 0
    for t in tickers:
        ticker_dir = _DEFAULT_DATA_DIR / t
        ticker_dir.mkdir(parents=True, exist_ok=True)
        for year, q in quarters:
            qname = quarter_key(year, q)
            rec = all_results[t].get(qname)
            wrapped = {
                "schema_version": "sentiment_v1",
                "ticker": t,
                "year": year,
                "quarter": f"Q{q}",
                "features": rec,
            }
            try:
                from provenance import inline_provenance
                wrapped.update(inline_provenance())
            except ImportError:
                pass
            out_path = ticker_dir / f"{year}_Q{q}.json"
            with open(out_path, "w") as f:
                json.dump(wrapped, f, indent=2)
            count += 1

    print(f"Done: wrote {count} file(s) for {len(tickers)} ticker(s), "
          f"{len(quarters)} quarter(s).", flush=True)


if __name__ == "__main__":
    main()
