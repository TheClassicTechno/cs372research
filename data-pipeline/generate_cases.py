#!/usr/bin/env python3
"""Generate Case-compliant JSON for the trading simulation.

Fetches stock prices (yfinance), news (Finnhub), and earnings (HuggingFace),
enriches with sentiment and impact scoring, and outputs cases in the format
expected by simulation/case_loader.py.

API keys are read from .env (FINNHUB_API_KEY, HF_TOKEN for HuggingFace).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Optional imports for fetch logic
def _ensure_deps():
    """Ensure pipeline dependencies are available."""
    try:
        import yfinance  # noqa: F401
        import finnhub  # noqa: F401
        from datasets import load_dataset  # noqa: F401
        from nltk.sentiment import SentimentIntensityAnalyzer  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Pipeline dependencies missing. Run: uv add yfinance finnhub-python datasets nltk"
        ) from e


def _ts(x) -> pd.Timestamp:
    """Parse timestamp to UTC-aware pandas Timestamp."""
    try:
        ts = pd.Timestamp(x)
        if ts.tzinfo:
            return ts.tz_convert("UTC")
        return ts.tz_localize("UTC")
    except Exception:
        try:
            from dateutil import parser as dateutil_parser
            ts = pd.Timestamp(dateutil_parser.parse(str(x))).tz_localize("UTC")
            if ts.year < 1970:
                return pd.NaT
            return ts
        except Exception:
            return pd.NaT


EVENT_PATTERNS = {
    "earnings": r"earnings|EPS|quarterly results|Q[1-4]|beat|miss|guidance",
    "guidance": r"guidance|outlook|forecast|revenue (guidance|outlook)|raised guidance|lowered guidance",
    "mna": r"acquire|acquisition|merger|takeover|buyout|deal worth|definitive agreement",
    "lawsuit": r"lawsuit|class action|litigation|sued|antitrust|settlement",
    "regulatory": r"SEC|FTC|DoJ|CMA|EU Commission|approval|clearance|investigation|probe",
    "product": r"launch|unveil|product|chip|AI model|platform|feature|partnership",
}


def _map_kind(news_type: list[str] | str, news_source: str) -> str:
    """Map pipeline tags to CaseDataItem kind: earnings | news | other."""
    tags = news_type if isinstance(news_type, list) else []
    if "earnings" in tags or news_source == "earnings":
        return "earnings"
    if "news" in tags or news_source == "finnhub":
        return "news"
    return "other"


def fetch_stock_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily close prices from yfinance."""
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "close_price", "daily_pct_change"])

    # Normalize: yfinance uses MultiIndex (Open, TICKER) for single ticker
    if isinstance(df.columns, pd.MultiIndex):
        close_cols = [c for c in df.columns if c[0] == "Close"]
        close_col = close_cols[0] if close_cols else df.columns[0]
    else:
        close_col = "Close" if "Close" in df.columns else df.columns[0]
    close_series = df[close_col].copy()
    df_out = pd.DataFrame({"close_price": close_series})
    df_out = df_out.reset_index()
    df_out = df_out.rename(columns={df_out.columns[0]: "timestamp"})
    df_out["timestamp"] = pd.to_datetime(df_out["timestamp"])
    if df_out["timestamp"].dt.tz is None:
        df_out["timestamp"] = df_out["timestamp"].dt.tz_localize("UTC")
    df_out["daily_pct_change"] = df_out["close_price"].pct_change() * 100
    return df_out.dropna(subset=["close_price"])


def fetch_finnhub_news(ticker: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    """Fetch company news from Finnhub."""
    import finnhub

    client = finnhub.Client(api_key=api_key)
    rows = []
    current = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end).date()

    while current <= end_date:
        month_end = (current + pd.DateOffset(months=1) - pd.DateOffset(days=1)).date()
        fetch_end = min(month_end, end_date)
        try:
            data = client.company_news(
                ticker, _from=current.isoformat(), to=fetch_end.isoformat()
            )
            for a in data:
                ts = _ts(pd.Timestamp.utcfromtimestamp(a.get("datetime", 0)))
                rows.append({
                    "provider": "finnhub",
                    "ticker": ticker,
                    "title": a.get("headline"),
                    "description": a.get("summary"),
                    "published_at": ts,
                })
        except Exception as e:
            logger.warning("Finnhub error %s to %s: %s", current, fetch_end, e)
        current = (fetch_end + pd.DateOffset(days=1)).date()

    return pd.DataFrame(rows)


def fetch_earnings(ticker: str, year: int) -> pd.DataFrame:
    """Fetch earnings transcripts from HuggingFace."""
    from datasets import load_dataset

    ds = load_dataset("kurry/sp500_earnings_transcripts", split="train")
    filtered = ds.filter(
        lambda x: x.get("symbol") == ticker and x.get("year") == year
    )
    df = filtered.to_pandas()
    if df.empty:
        return pd.DataFrame(
            columns=["provider", "ticker", "published_at", "description"]
        )
    df = df.rename(columns={"symbol": "ticker", "date": "published_at", "content": "description"})
    df["provider"] = "earnings_transcript"
    df["published_at"] = df["published_at"].apply(_ts)
    return df


def combine_and_enrich(
    news_df: pd.DataFrame,
    earnings_df: pd.DataFrame,
    close_prices_df: pd.DataFrame,
    market_close_hour: int = 16,
) -> pd.DataFrame:
    """Merge news and earnings, add sentiment, tags, and impact score."""
    from nltk.sentiment import SentimentIntensityAnalyzer
    from pytz import timezone

    try:
        import nltk
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)

    sia = SentimentIntensityAnalyzer()
    ny_tz = timezone("America/New_York")

    def sentiment(text: str) -> float:
        return sia.polarity_scores(text or "").get("compound", 0.0)

    def tag_events(text: str) -> list[str]:
        t = (text or "").lower()
        return [k for k, pat in EVENT_PATTERNS.items() if re.search(pat, t)]

    trading_dates = pd.Series(close_prices_df["timestamp"].dt.normalize().unique()).sort_values()

    def get_trading_date(ts) -> pd.Timestamp | pd.NaT:
        if pd.isna(ts):
            return pd.NaT
        ts = _ts(ts)
        ny = ts.astimezone(ny_tz)
        effective = ny.date()
        if ny.time().hour >= market_close_hour:
            from datetime import timedelta
            effective = effective + timedelta(days=1)
        effective_utc = pd.Timestamp(effective).tz_localize("UTC")
        idx = trading_dates.searchsorted(effective_utc)
        if idx < len(trading_dates):
            return trading_dates.iloc[idx]
        return pd.NaT

    def get_pct_change(td) -> float:
        if pd.isna(td):
            return 0.0
        m = close_prices_df["timestamp"].dt.normalize() == td
        if m.any():
            v = close_prices_df.loc[m, "daily_pct_change"].iloc[0]
            return float(v) if pd.notna(v) else 0.0
        return 0.0

    def impact_score(sent: float, pub_ts, trading_date, half_life_hours: float = 24) -> float:
        if pd.isna(pub_ts):
            decay = 0.5
        else:
            age_h = max(0, (pd.Timestamp.utcnow() - _ts(pub_ts)).total_seconds() / 3600)
            decay = 0.5 ** (age_h / half_life_hours)
        pct = get_pct_change(trading_date)
        return float(sent * decay * (1 + pct / 100.0))

    def process_df(df: pd.DataFrame, news_source: str) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df["combined_text"] = (
            df.get("title", pd.Series([""] * len(df))).fillna("")
            + " "
            + df.get("description", pd.Series([""] * len(df))).fillna("")
        )
        df["sentiment_score"] = df["combined_text"].apply(sentiment)
        df["news_type"] = df["combined_text"].apply(tag_events)
        df["trading_date"] = df["published_at"].apply(get_trading_date)
        df["impact_score"] = df.apply(
            lambda r: impact_score(r["sentiment_score"], r["published_at"], r["trading_date"]),
            axis=1,
        )
        df["news_source"] = news_source
        df["content"] = df["combined_text"]
        return df

    news_processed = process_df(news_df, "finnhub")
    earn_processed = process_df(earnings_df, "earnings")
    combined = pd.concat([news_processed, earn_processed], ignore_index=True)
    combined = combined.sort_values("published_at").reset_index(drop=True)
    return combined


def build_case_json(
    ticker: str,
    combined_df: pd.DataFrame,
    stock_prices_df: pd.DataFrame,
    quarter: dict,
) -> dict:
    """Build Case-valid JSON for one quarter."""
    start = quarter["start"]
    end = quarter["end"]
    q_name = quarter["name"]

    news_q = combined_df[
        (combined_df["published_at"] >= start) & (combined_df["published_at"] <= end)
    ]
    prices_q = stock_prices_df[
        (stock_prices_df["timestamp"] >= start) & (stock_prices_df["timestamp"] <= end)
    ]

    items = []
    for _, row in news_q.iterrows():
        kind = _map_kind(row.get("news_type", []), row.get("news_source", "other"))
        content = row.get("content", "")
        impact = row.get("impact_score")
        impact_val = float(impact) if pd.notna(impact) else None
        items.append({
            "kind": kind,
            "content": content,
            "impact_score": impact_val,
        })

    daily_bars = []
    current_price = None
    if not prices_q.empty:
        for _, r in prices_q.iterrows():
            ts = r["timestamp"]
            ts_str = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
            daily_bars.append({"timestamp": ts_str, "close": float(r["close_price"])})
        current_price = float(prices_q["close_price"].iloc[-1])

    stock_data_entry = {
        "ticker": ticker,
        "current_price": current_price if current_price is not None else 0.0,
        "daily_bars": daily_bars,
    }

    return {
        "case_data": {"items": items},
        "stock_data": {ticker: stock_data_entry},
    }


def run(
    ticker: str,
    year: int,
    output_dir: str = "data/cases",
    finnhub_key: str | None = None,
    use_cache: bool = False,
    cache_dir: str | None = None,
) -> list[Path]:
    """Fetch data, build cases, and write JSON files. Returns paths of written files."""
    load_dotenv()
    api_key = finnhub_key or os.environ.get("FINNHUB_API_KEY", "")
    if not api_key:
        raise ValueError("FINNHUB_API_KEY not set. Add it to .env or pass --finnhub-key.")

    _ensure_deps()

    start = f"{year}-01-01"
    end = f"{year}-12-31"

    if use_cache and cache_dir:
        cache_path = Path(cache_dir)
        cp_file = cache_path / f"close_prices_{ticker}_{year}.parquet"
        ne_file = cache_path / f"news_earnings_{ticker}_{year}.parquet"
        if cp_file.exists() and ne_file.exists():
            close_prices_df = pd.read_parquet(cp_file)
            combined_df = pd.read_parquet(ne_file)
            combined_df["published_at"] = pd.to_datetime(combined_df["published_at"])
            stock_prices_df = close_prices_df
        else:
            close_prices_df = fetch_stock_prices(ticker, start, end)
            stock_prices_df = close_prices_df
            news_df = fetch_finnhub_news(ticker, start, end, api_key)
            earnings_df = fetch_earnings(ticker, year)
            combined_df = combine_and_enrich(news_df, earnings_df, close_prices_df)
            cache_path.mkdir(parents=True, exist_ok=True)
            close_prices_df.to_parquet(cp_file, index=False)
            combined_df.to_parquet(ne_file, index=False)
    else:
        close_prices_df = fetch_stock_prices(ticker, start, end)
        stock_prices_df = close_prices_df
        news_df = fetch_finnhub_news(ticker, start, end, api_key)
        earnings_df = fetch_earnings(ticker, year)
        combined_df = combine_and_enrich(news_df, earnings_df, close_prices_df)

    quarters = [
        {
            "name": f"{year}_Q1",
            "start": pd.Timestamp(f"{year}-01-01").tz_localize("UTC"),
            "end": pd.Timestamp(f"{year}-03-31").tz_localize("UTC"),
        },
        {
            "name": f"{year}_Q2",
            "start": pd.Timestamp(f"{year}-04-01").tz_localize("UTC"),
            "end": pd.Timestamp(f"{year}-06-30").tz_localize("UTC"),
        },
        {
            "name": f"{year}_Q3",
            "start": pd.Timestamp(f"{year}-07-01").tz_localize("UTC"),
            "end": pd.Timestamp(f"{year}-09-30").tz_localize("UTC"),
        },
        {
            "name": f"{year}_Q4",
            "start": pd.Timestamp(f"{year}-10-01").tz_localize("UTC"),
            "end": pd.Timestamp(f"{year}-12-31").tz_localize("UTC"),
        },
    ]

    out = Path(output_dir) / ticker
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for q in quarters:
        case_json = build_case_json(ticker, combined_df, stock_prices_df, q)
        path = out / f"{q['name']}.json"
        path.write_text(json.dumps(case_json, indent=2), encoding="utf-8")
        written.append(path)
        logger.info("Wrote %s", path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Case-compliant JSON for the trading simulation."
    )
    parser.add_argument("--ticker", required=True, help="Stock ticker (e.g. NVDA)")
    parser.add_argument("--year", type=int, required=True, help="Year (e.g. 2025)")
    parser.add_argument(
        "--output-dir",
        default="data/cases",
        help="Output directory (default: data/cases). Files go to {output-dir}/{ticker}/{year}_Qn.json",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached parquet files if present",
    )
    parser.add_argument(
        "--cache-dir",
        default="data-pipeline/cache",
        help="Directory for cache when --use-cache (default: data-pipeline/cache)",
    )
    parser.add_argument(
        "--finnhub-key",
        default=None,
        help="Finnhub API key (overrides FINNHUB_API_KEY from .env)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    paths = run(
        ticker=args.ticker,
        year=args.year,
        output_dir=args.output_dir,
        finnhub_key=args.finnhub_key,
        use_cache=args.use_cache,
        cache_dir=args.cache_dir if args.use_cache else None,
    )
    print(f"Generated {len(paths)} case(s):")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
