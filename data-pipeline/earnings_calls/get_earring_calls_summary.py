#!/usr/bin/env python3
"""
Earnings Calls Pipeline: Hugging Face transcripts -> Claude summary + FinBERT sentiment.

Source dataset:
  kurry/sp500_earnings_transcripts

Output:
  data-pipeline/earnings_calls/data/{TICKER}/{YEAR}_Q{#}.json

Behavior:
  - Supports either --tickers or --supported.
  - Supports either --year/--quarter or --start/--end.
  - If no transcript is found for a ticker-quarter, that ticker-quarter is skipped.
  - Uses HF token for gated/private dataset access.
"""

import argparse
import datetime as dt
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
import yaml
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DATASET_NAME = "kurry/sp500_earnings_transcripts"
FINBERT_MODEL_NAME = "ProsusAI/finbert"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
SUPPORTED_TICKERS_PATH = PIPELINE_DIR / "supported_tickers.yaml"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data"


def load_supported_tickers(yaml_path: Path = SUPPORTED_TICKERS_PATH) -> List[str]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [entry["symbol"] for entry in data["supported_tickers"]]


def parse_quarter_string(qstr: str) -> Tuple[int, int]:
    qstr = qstr.strip().upper()
    year = int(qstr[:4])
    quarter = int(qstr[5])
    if qstr[4] != "Q" or quarter not in (1, 2, 3, 4):
        raise ValueError(f"Invalid quarter: {qstr}")
    return year, quarter


def next_quarter(year: int, quarter: int) -> Tuple[int, int]:
    if quarter < 4:
        return year, quarter + 1
    return year + 1, 1


def quarter_range_list(start: str, end: str) -> List[Tuple[int, int]]:
    y, q = parse_quarter_string(start)
    ey, eq = parse_quarter_string(end)
    out: List[Tuple[int, int]] = []
    while True:
        out.append((y, q))
        if y == ey and q == eq:
            break
        y, q = next_quarter(y, q)
    return out


def _quarter_from_date_string(value: str) -> Optional[int]:
    text = str(value).strip()
    if not text:
        return None
    # Accept YYYY-MM-DD and common variants.
    m = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text)
    if not m:
        return None
    month = int(m.group(2))
    if month < 1 or month > 12:
        return None
    return ((month - 1) // 3) + 1


def _quarter_from_any(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value if value in (1, 2, 3, 4) else None
    text = str(value).strip().upper()
    if text in ("Q1", "Q2", "Q3", "Q4"):
        return int(text[1])
    if text in ("1", "2", "3", "4"):
        return int(text)
    return None


def _pick_first_existing(cols: List[str], candidates: List[str]) -> Optional[str]:
    colset = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in colset:
            return colset[cand.lower()]
    return None


def detect_columns(ds: Dataset) -> Dict[str, Optional[str]]:
    cols = list(ds.column_names)
    symbol_col = _pick_first_existing(cols, ["symbol", "ticker"])
    year_col = _pick_first_existing(cols, ["year", "fiscal_year"])
    quarter_col = _pick_first_existing(cols, ["quarter", "fiscal_quarter"])
    date_col = _pick_first_existing(cols, ["date", "call_date", "earnings_date", "event_date"])
    text_col = _pick_first_existing(
        cols,
        ["transcript", "text", "content", "call_transcript", "earnings_call"],
    )
    return {
        "symbol": symbol_col,
        "year": year_col,
        "quarter": quarter_col,
        "date": date_col,
        "text": text_col,
    }


def build_transcript_index(
    ds: Dataset,
    cols: Dict[str, Optional[str]],
) -> Dict[Tuple[str, int, Optional[int]], List[Dict[str, Any]]]:
    """Scan dataset once and return a dict keyed by (TICKER, year, quarter).

    Records whose quarter cannot be inferred are stored under quarter=None
    so the lookup function can apply the single-record-year fallback.
    """
    from collections import defaultdict

    symbol_col = cols["symbol"]
    year_col = cols["year"]
    quarter_col = cols["quarter"]
    date_col = cols["date"]

    if not symbol_col or not year_col:
        return {}

    index: Dict[Tuple[str, int, Optional[int]], List[Dict[str, Any]]] = defaultdict(list)
    for row in tqdm(ds, desc="Indexing transcripts", unit="row"):
        ticker = str(row.get(symbol_col, "")).upper()
        try:
            year = int(row.get(year_col, -1))
        except (ValueError, TypeError):
            continue
        if year < 0:
            continue

        q_val: Optional[int] = None
        if quarter_col:
            q_val = _quarter_from_any(row.get(quarter_col))
        if q_val is None and date_col:
            q_val = _quarter_from_date_string(str(row.get(date_col, "")))

        index[(ticker, year, q_val)].append(dict(row))

    return dict(index)


def lookup_records(
    index: Dict[Tuple[str, int, Optional[int]], List[Dict[str, Any]]],
    ticker: str,
    year: int,
    quarter: int,
) -> List[Dict[str, Any]]:
    """O(1) lookup from pre-built index, with single-record-year fallback."""
    records = index.get((ticker, year, quarter), [])
    if records:
        return records

    # Fallback: if there's exactly one record for this ticker-year with unknown
    # quarter, assume it matches (preserves original behavior).
    unknown_q = index.get((ticker, year, None), [])
    if len(unknown_q) == 1:
        return unknown_q

    return []


def flatten_transcripts(records: List[Dict[str, Any]], text_col: str, max_chars: int = 120_000) -> str:
    parts: List[str] = []
    for rec in records:
        txt = str(rec.get(text_col, "")).strip()
        if txt:
            parts.append(txt)
    merged = "\n\n".join(parts).strip()
    if len(merged) > max_chars:
        merged = merged[:max_chars]
    return merged


def call_claude_summary(
    transcript_text: str,
    api_key: str,
    model: str = DEFAULT_CLAUDE_MODEL,
) -> Dict[str, Any]:
    system = (
        "You summarize earnings call transcripts into concise, factual quarterly notes "
        "for quantitative trading research."
    )
    prompt = (
        "Return ONLY valid JSON with keys: "
        "summary, management_outlook, risks, key_drivers.\n"
        "Each value is a short paragraph (2-4 sentences), no bullet points.\n\n"
        f"TRANSCRIPT:\n{transcript_text}"
    )
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": 1300,
        "temperature": 0.0,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = requests.post(ANTHROPIC_API_URL, headers=headers, json=body, timeout=180)
    resp.raise_for_status()
    text = resp.json()["content"][0]["text"].strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    parsed = json.loads(text)
    out: Dict[str, Any] = {}
    for key in ("summary", "management_outlook", "risks", "key_drivers"):
        val = parsed.get(key)
        out[key] = str(val).strip() if val is not None else None
    return out


def init_finbert(device_str: Optional[str] = None):
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def split_for_sentiment(text: str, words_per_chunk: int = 420) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    for i in range(0, len(words), words_per_chunk):
        chunk = " ".join(words[i : i + words_per_chunk]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def finbert_score_text(
    text: str,
    tokenizer,
    model,
    device,
    batch_size: int = 16,
) -> Dict[str, Optional[float]]:
    chunks = split_for_sentiment(text)
    if not chunks:
        return {
            "chunk_count": 0,
            "mean_sentiment": None,
            "sentiment_volatility": None,
        }

    scores: List[float] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
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
        # [negative, neutral, positive]
        batch_scores = (probs[:, 2] - probs[:, 0]).detach().cpu().tolist()
        scores.extend(float(x) for x in batch_scores)

    mean = sum(scores) / len(scores)
    var = sum((x - mean) ** 2 for x in scores) / len(scores)
    return {
        "chunk_count": len(scores),
        "mean_sentiment": round(float(mean), 6),
        "sentiment_volatility": round(float(math.sqrt(var)), 6),
    }


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build quarterly earnings-call summaries and sentiment from Hugging Face transcripts.",
    )
    p.add_argument("--year", type=int, default=None, help="Year (single-quarter mode)")
    p.add_argument("--quarter", type=int, default=None, choices=[1, 2, 3, 4], help="Quarter number")
    p.add_argument("--start", type=str, default=None, help="Start quarter, e.g. 2024Q4")
    p.add_argument("--end", type=str, default=None, help="End quarter, e.g. 2025Q3")
    p.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers")
    p.add_argument("--supported", action="store_true", help="Use supported_tickers.yaml")
    p.add_argument("--dataset", type=str, default=DATASET_NAME, help="Hugging Face dataset ID")
    p.add_argument("--hf-token", type=str, default=None, help="Hugging Face token (or HF_TOKEN env var)")
    p.add_argument("--api-key", type=str, default=None, help="Anthropic API key (or ANTHROPIC_API_KEY env var)")
    p.add_argument("--model", type=str, default=DEFAULT_CLAUDE_MODEL, help="Claude model")
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Per-ticker output directory")
    p.add_argument("--force", action="store_true", help="Rebuild existing files")
    p.add_argument("--sleep-seconds", type=float, default=0.8, help="Delay between Claude calls")
    args = p.parse_args()

    if args.start and args.end:
        quarters = quarter_range_list(args.start, args.end)
    elif args.year and args.quarter:
        quarters = [(args.year, args.quarter)]
    else:
        p.error("Specify --year/--quarter or --start/--end")

    if args.supported:
        tickers = load_supported_tickers()
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        p.error("Either --tickers or --supported is required")

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not hf_token:
        print("ERROR: missing HF token. Set HF_TOKEN or pass --hf-token", file=sys.stderr)
        sys.exit(1)
    if not api_key:
        print("ERROR: missing Anthropic key. Set ANTHROPIC_API_KEY or pass --api-key", file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset: {args.dataset}", flush=True)
    ds_dict = load_dataset(args.dataset, token=hf_token)
    if "train" not in ds_dict:
        print("ERROR: dataset missing train split", file=sys.stderr)
        sys.exit(1)
    ds = ds_dict["train"]
    cols = detect_columns(ds)
    if not cols["symbol"] or not cols["year"] or not cols["text"]:
        print(
            "ERROR: dataset missing required columns (symbol/year/transcript-like text)",
            file=sys.stderr,
        )
        print(f"Columns detected: {ds.column_names}", file=sys.stderr)
        sys.exit(1)

    print("Building transcript index...", flush=True)
    transcript_index = build_transcript_index(ds, cols)
    print(f"Indexed {sum(len(v) for v in transcript_index.values())} records into {len(transcript_index)} groups", flush=True)

    print("Loading FinBERT...", flush=True)
    tokenizer, model, device = init_finbert()
    print(f"Using device: {device}", flush=True)

    output_dir = Path(args.output_dir)
    written = 0
    skipped = 0
    missing = 0

    for year, quarter in tqdm(quarters, desc="Quarters", unit="qtr"):
        qlabel = f"Q{quarter}"
        for ticker in tickers:
            out_path = output_dir / ticker / f"{year}_{qlabel}.json"
            if out_path.exists() and not args.force:
                skipped += 1
                continue

            records = lookup_records(transcript_index, ticker, year, quarter)
            if not records:
                missing += 1
                continue

            transcript_text = flatten_transcripts(records, cols["text"])
            if not transcript_text:
                missing += 1
                continue

            try:
                summary_obj = call_claude_summary(transcript_text, api_key=api_key, model=args.model)
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)
            except Exception as e:
                print(f"WARNING: Claude summary failed for {ticker} {year} {qlabel}: {e}", file=sys.stderr)
                summary_obj = {
                    "summary": None,
                    "management_outlook": None,
                    "risks": None,
                    "key_drivers": None,
                }

            sentiment_obj = finbert_score_text(transcript_text, tokenizer, model, device)

            payload: Dict[str, Any] = {
                "schema_version": "earnings_calls_v1",
                "ticker": ticker,
                "year": year,
                "quarter": qlabel,
                "source_dataset": args.dataset,
                "record_count": len(records),
                "features": {
                    **summary_obj,
                    **sentiment_obj,
                    "transcript_char_count": len(transcript_text),
                },
            }
            try:
                from provenance import inline_provenance

                payload.update(inline_provenance())
            except ImportError:
                pass

            atomic_write_json(out_path, payload)
            written += 1

    print(
        f"Done: wrote {written} file(s), skipped {skipped}, missing {missing} ticker-quarter(s).",
        flush=True,
    )


if __name__ == "__main__":
    main()
