#!/usr/bin/env python3
"""
Filing Summarization Pipeline — EDGAR raw text -> structured JSON summaries.

Reads raw filing .txt files produced by get_sec_data.py, sends narrative
sections to Claude for structured extraction, and saves validated JSON
summaries to finished_summaries/.

Architecture:
  raw_filings/{TICKER}/{YEAR}/{QUARTER}/{FORM_DATE}.txt
    -> Claude API (structured extraction, no hallucinated numbers)
    -> finished_summaries/{TICKER}/{YEAR}/{QUARTER}/{FORM}_summary.json

Examples:
  # 1. Summarize all tickers (reads from default raw_filings/ next to script)
  python filing_summarization_pipeline.py

  # 2. Specific tickers with API key from env
  ANTHROPIC_API_KEY=sk-ant-... python filing_summarization_pipeline.py \\
      --tickers AAPL,NVDA,MSFT

  # 3. Pass API key on command line
  python filing_summarization_pipeline.py \\
      --tickers AAPL \\
      --api-key sk-ant-...

  # 4. Custom input/output directories
  python filing_summarization_pipeline.py \\
      --raw-dir /data/edgar/raw_filings \\
      --out-dir /data/edgar/finished_summaries \\
      --tickers AAPL,NVDA

  # 5. Force re-summarize (overwrite existing summaries)
  python filing_summarization_pipeline.py \\
      --tickers AAPL \\
      --force

  # 6. Use a different Claude model
  python filing_summarization_pipeline.py \\
      --tickers AAPL \\
      --model claude-sonnet-4-20250514

  # 7. Full 8-ticker universe
  python filing_summarization_pipeline.py \\
      --raw-dir ./raw_filings \\
      --out-dir ./finished_summaries \\
      --tickers AAPL,NVDA,MSFT,GOOG,AMZN,META,JPM,GS

Design constraints:
  - No numeric extraction from filing text.
  - No table parsing.
  - All financial numbers come from structured APIs.
  - Deterministic: same input text -> same summary (temperature=0).
  - Atomic writes (tmp file -> rename).
  - Validated output schema.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# ==========================
# CONFIG
# ==========================

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096
TEMPERATURE = 0.0  # deterministic

# Claude API rate limit: conservative default
RATE_LIMIT_SECONDS = 1.0

# Maximum characters per chunk sent to Claude.  10-K filings can be
# 80k+ chars; we split into chunks and summarize each, then merge.
MAX_CHUNK_CHARS = 60_000

# ==========================
# SCHEMA
# ==========================

SUMMARY_SCHEMA = {
    "form": str,
    "filing_date": str,
    "accession": str,
    "mda": {
        "demand_signal": str,
        "margin_trend": str,
        "inventory_signal": str,
        "liquidity_commentary": str,
        "capex_trend": str,
        "tone_shift_score": (int, float, type(None)),
    },
    "risk_factors": {
        "new_risks_added": (bool, type(None)),
        "refinancing_risk": str,
        "customer_concentration_change": str,
    },
    "events": list,
}

EVENT_SCHEMA = {
    "type": str,
    "direction": str,
    "severity": (int, float, type(None)),
}


def validate_summary_schema(summary: dict) -> List[str]:
    """Validate a summary dict against the expected schema.

    Returns a list of error strings.  Empty list means valid.
    """
    errors: List[str] = []

    def _check(obj, schema, path=""):
        for key, expected in schema.items():
            full = f"{path}.{key}" if path else key
            if key not in obj:
                errors.append(f"missing key: {full}")
                continue
            val = obj[key]
            if isinstance(expected, dict):
                if not isinstance(val, dict):
                    errors.append(f"{full}: expected dict, got {type(val).__name__}")
                else:
                    _check(val, expected, full)
            elif isinstance(expected, tuple):
                if not isinstance(val, expected):
                    names = "/".join(t.__name__ for t in expected)
                    errors.append(f"{full}: expected {names}, got {type(val).__name__}")
            elif expected is list:
                if not isinstance(val, list):
                    errors.append(f"{full}: expected list, got {type(val).__name__}")
            elif not isinstance(val, expected):
                errors.append(f"{full}: expected {expected.__name__}, got {type(val).__name__}")

    _check(summary, SUMMARY_SCHEMA)

    # Validate events list entries
    if isinstance(summary.get("events"), list):
        for i, ev in enumerate(summary["events"]):
            if not isinstance(ev, dict):
                errors.append(f"events[{i}]: expected dict, got {type(ev).__name__}")
                continue
            for key, expected in EVENT_SCHEMA.items():
                full = f"events[{i}].{key}"
                if key not in ev:
                    errors.append(f"missing key: {full}")
                elif isinstance(expected, tuple):
                    if not isinstance(ev[key], expected):
                        names = "/".join(t.__name__ for t in expected)
                        errors.append(f"{full}: expected {names}, got {type(ev[key]).__name__}")
                elif not isinstance(ev[key], expected):
                    errors.append(f"{full}: expected {expected.__name__}, got {type(ev[key]).__name__}")

    return errors


# ==========================
# TEXT PARSING
# ==========================

def parse_filing_header(text: str) -> Dict[str, str]:
    """Extract FORM, FILING_DATE, ACCESSION from the structured text header.

    These headers are written by get_sec_data.py's format_text_output().
    """
    header: Dict[str, str] = {}
    for line in text.splitlines()[:10]:
        line = line.strip()
        if line.startswith("FORM:"):
            header["form"] = line[len("FORM:"):].strip()
        elif line.startswith("FILING_DATE:"):
            header["filing_date"] = line[len("FILING_DATE:"):].strip()
        elif line.startswith("ACCESSION:"):
            header["accession"] = line[len("ACCESSION:"):].strip()
    return header


def extract_sections(text: str) -> Dict[str, str]:
    """Split structured text into named sections.

    Returns dict of {section_name: section_text}.
    Sections are delimited by ==== SECTION: NAME ==== markers
    as written by get_sec_data.py.
    """
    sections: Dict[str, str] = {}
    current_name: Optional[str] = None
    current_lines: List[str] = []
    section_re = re.compile(r"^====\s*SECTION:\s*(\S+)\s*====$")

    for line in text.splitlines():
        m = section_re.match(line.strip())
        if m:
            if current_name is not None:
                sections[current_name] = "\n".join(current_lines).strip()
            current_name = m.group(1)
            current_lines = []
        elif current_name is not None:
            current_lines.append(line)

    if current_name is not None:
        sections[current_name] = "\n".join(current_lines).strip()

    return sections


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """Split text into chunks that fit within the API character limit.

    Splits on paragraph boundaries (double newlines) to avoid cutting
    mid-sentence.  Falls back to hard split if a single paragraph
    exceeds max_chars.
    """
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"\n\n+", text)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2  # account for \n\n separator
        if current_len + para_len > max_chars and current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        if para_len > max_chars:
            # Hard split oversized paragraph
            for i in range(0, len(para), max_chars):
                chunks.append(para[i:i + max_chars])
        else:
            current.append(para)
            current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


# ==========================
# PROMPTS
# ==========================

_SYSTEM_PROMPT = """\
You are a financial filing analyst. You extract structured signals from \
SEC filing text. You MUST follow these rules:

1. Return ONLY valid JSON. No prose, no markdown, no explanation outside the JSON.
2. NEVER hallucinate or invent financial numbers. If the text does not \
   contain information for a field, use null or "not discussed".
3. Use ONLY the provided filing text as your source.
4. All string fields should be brief qualitative assessments (1-2 sentences max).
5. tone_shift_score: float from -1.0 (much more negative vs prior) to \
   +1.0 (much more positive). Use 0.0 if neutral or insufficient context. \
   Use null if the section is missing entirely.
6. severity: float from 0.0 (negligible) to 1.0 (critical). null if unknown.
7. events: list material events mentioned. Empty list [] if none found."""

_10K_10Q_USER_PROMPT = """\
Analyze this SEC {form} filing text and return a JSON object with exactly \
this structure:

{{
  "mda": {{
    "demand_signal": "<qualitative demand outlook from MD&A>",
    "margin_trend": "<margin direction and drivers>",
    "inventory_signal": "<inventory buildup/drawdown if discussed>",
    "liquidity_commentary": "<cash position and liquidity outlook>",
    "capex_trend": "<capital expenditure direction>",
    "tone_shift_score": <float -1.0 to 1.0 or null>
  }},
  "risk_factors": {{
    "new_risks_added": <true/false/null>,
    "refinancing_risk": "<refinancing or debt maturity risk if discussed>",
    "customer_concentration_change": "<customer concentration changes>"
  }},
  "events": [
    {{
      "type": "<event category: guidance, restructuring, acquisition, divestiture, legal, regulatory, leadership, other>",
      "direction": "<positive/negative/neutral>",
      "severity": <float 0.0-1.0 or null>
    }}
  ]
}}

FILING TEXT:
{text}"""

_8K_USER_PROMPT = """\
Analyze this SEC 8-K filing text and return a JSON object with exactly \
this structure:

{{
  "mda": {{
    "demand_signal": "not applicable",
    "margin_trend": "not applicable",
    "inventory_signal": "not applicable",
    "liquidity_commentary": "not applicable",
    "capex_trend": "not applicable",
    "tone_shift_score": null
  }},
  "risk_factors": {{
    "new_risks_added": null,
    "refinancing_risk": "not applicable",
    "customer_concentration_change": "not applicable"
  }},
  "events": [
    {{
      "type": "<event category: earnings, guidance, restructuring, acquisition, divestiture, legal, regulatory, leadership, offering, other>",
      "direction": "<positive/negative/neutral>",
      "severity": <float 0.0-1.0 or null>
    }}
  ]
}}

Focus on identifying the material event(s) reported in this 8-K.

FILING TEXT:
{text}"""


def build_prompt(form: str, text: str) -> str:
    """Build the user prompt for a given form type and filing text."""
    base_form = form.upper().rstrip("/A").rstrip("/")
    if base_form == "8-K":
        return _8K_USER_PROMPT.format(text=text)
    return _10K_10Q_USER_PROMPT.format(form=form, text=text)


# ==========================
# CLAUDE API
# ==========================

def call_claude(
    api_key: str,
    system: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
) -> str:
    """Send a single message to the Claude API and return the text response.

    Raises on HTTP errors or missing content.
    """
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    resp = requests.post(ANTHROPIC_API_URL, headers=headers, json=body, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    content_blocks = data.get("content", [])
    if not content_blocks:
        raise ValueError("Claude returned empty content")
    return content_blocks[0].get("text", "")


def parse_json_response(raw: str) -> dict:
    """Extract and parse JSON from Claude's response.

    Handles responses that may include markdown code fences.
    """
    text = raw.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines)
    return json.loads(text)


# ==========================
# SUMMARIZATION LOGIC
# ==========================

def summarize_filing(
    api_key: str,
    form: str,
    text: str,
    model: str = DEFAULT_MODEL,
    rate_limit: float = RATE_LIMIT_SECONDS,
) -> dict:
    """Summarize a single filing's text via Claude.

    For long filings, chunks the text and merges chunk summaries.
    Returns the raw parsed summary dict (events merged, mda/risk
    from the first chunk that has substantive content).
    """
    chunks = chunk_text(text)

    if len(chunks) == 1:
        prompt = build_prompt(form, chunks[0])
        time.sleep(rate_limit)
        raw = call_claude(api_key, _SYSTEM_PROMPT, prompt, model=model)
        return parse_json_response(raw)

    # Multi-chunk: summarize each, then merge
    chunk_summaries: List[dict] = []
    for i, chunk in enumerate(chunks):
        prompt = build_prompt(form, chunk)
        time.sleep(rate_limit)
        print(f"    Chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)")
        raw = call_claude(api_key, _SYSTEM_PROMPT, prompt, model=model)
        chunk_summaries.append(parse_json_response(raw))

    return merge_chunk_summaries(chunk_summaries)


def merge_chunk_summaries(summaries: List[dict]) -> dict:
    """Merge summaries from multiple chunks of the same filing.

    Strategy:
    - mda: take first chunk with substantive content (not "not discussed")
    - risk_factors: take first chunk with substantive content
    - events: concatenate all events, deduplicate by type+direction
    """
    merged_mda = None
    merged_risk = None
    all_events: List[dict] = []

    for s in summaries:
        if merged_mda is None and "mda" in s:
            mda = s["mda"]
            if any(v not in (None, "not discussed", "not applicable")
                   for v in mda.values() if isinstance(v, str)):
                merged_mda = mda

        if merged_risk is None and "risk_factors" in s:
            rf = s["risk_factors"]
            if any(v not in (None, "not discussed", "not applicable")
                   for v in rf.values() if isinstance(v, str)):
                merged_risk = rf

        if "events" in s and isinstance(s["events"], list):
            all_events.extend(s["events"])

    # Deduplicate events by (type, direction)
    seen = set()
    unique_events: List[dict] = []
    for ev in all_events:
        key = (ev.get("type", ""), ev.get("direction", ""))
        if key not in seen:
            seen.add(key)
            unique_events.append(ev)

    if merged_mda is None and summaries:
        merged_mda = summaries[0].get("mda", {})
    if merged_risk is None and summaries:
        merged_risk = summaries[0].get("risk_factors", {})

    return {
        "mda": merged_mda or {},
        "risk_factors": merged_risk or {},
        "events": unique_events,
    }


# ==========================
# FILE I/O
# ==========================

def discover_filings(raw_dir: Path, tickers: Optional[List[str]] = None) -> List[Tuple[Path, str]]:
    """Walk raw_filings directory and return (filepath, ticker) pairs.

    If tickers is provided, only return filings for those tickers.
    Files must end with .txt.
    """
    results: List[Tuple[Path, str]] = []
    if not raw_dir.exists():
        return results

    for ticker_dir in sorted(raw_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        if tickers and ticker not in tickers:
            continue
        for txt_file in sorted(ticker_dir.rglob("*.txt")):
            results.append((txt_file, ticker))

    return results


def output_path_for(out_dir: Path, ticker: str, year: str, quarter: str, form: str) -> Path:
    """Build output path: out_dir/TICKER/YEAR/QUARTER/FORM_summary.json"""
    safe_form = form.replace("/", "-")
    return out_dir / ticker / year / quarter / f"{safe_form}_summary.json"


def save_summary(path: Path, summary: dict) -> None:
    """Atomically write summary JSON (tmp file -> rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    tmp.rename(path)


# ==========================
# PIPELINE
# ==========================

def process_filing(
    filepath: Path,
    ticker: str,
    out_dir: Path,
    api_key: str,
    model: str = DEFAULT_MODEL,
    force: bool = False,
) -> Optional[Path]:
    """Process a single raw filing text file into a structured summary.

    Returns the output path on success, None on skip or failure.
    """
    text = filepath.read_text(encoding="utf-8")

    header = parse_filing_header(text)
    form = header.get("form")
    filing_date = header.get("filing_date")
    accession = header.get("accession")

    if not form or not filing_date:
        print(f"  SKIP {filepath}: missing FORM or FILING_DATE header")
        return None

    # Derive year/quarter from the directory structure
    # Expected: raw_dir/TICKER/YEAR/QUARTER/filename.txt
    quarter = filepath.parent.name       # e.g. Q1
    year = filepath.parent.parent.name   # e.g. 2024

    out_path = output_path_for(out_dir, ticker, year, quarter, form)

    if out_path.exists() and not force:
        print(f"  SKIP {ticker} {form} {filing_date} (already exists)")
        return None

    # Extract sections
    sections = extract_sections(text)
    section_text = "\n\n".join(
        f"[{name}]\n{content}" for name, content in sections.items()
        if content and content != "[SECTION NOT FOUND]"
    )

    if not section_text.strip():
        print(f"  SKIP {ticker} {form} {filing_date}: no extractable sections")
        return None

    print(f"  Summarizing {ticker} {form} {filing_date} ({len(section_text)} chars)")

    try:
        raw_summary = summarize_filing(api_key, form, section_text, model=model)
    except Exception as e:
        print(f"  FAILED {ticker} {form} {filing_date}: {type(e).__name__}: {e}")
        return None

    # Attach metadata
    summary = {
        "form": form,
        "filing_date": filing_date,
        "accession": accession or "",
        **raw_summary,
    }

    # Ensure required keys exist with defaults
    summary.setdefault("mda", {})
    summary.setdefault("risk_factors", {})
    summary.setdefault("events", [])

    # Validate
    validation_errors = validate_summary_schema(summary)
    if validation_errors:
        print(f"  WARNING {ticker} {form}: schema issues: {validation_errors}")

    save_summary(out_path, summary)
    print(f"  Saved {out_path}")
    return out_path


def run_pipeline(
    raw_dir: Path,
    out_dir: Path,
    api_key: str,
    tickers: Optional[List[str]] = None,
    model: str = DEFAULT_MODEL,
    force: bool = False,
) -> Dict[str, Any]:
    """Run the full summarization pipeline.

    Returns a report dict with counts.
    """
    filings = discover_filings(raw_dir, tickers)
    print(f"Found {len(filings)} filing(s) to process")

    processed = 0
    skipped = 0

    for filepath, ticker in filings:
        result = process_filing(filepath, ticker, out_dir, api_key,
                                model=model, force=force)
        if result is None:
            skipped += 1
        else:
            processed += 1

    report = {
        "total_found": len(filings),
        "processed": processed,
        "skipped": skipped,
    }
    print(f"\nPipeline complete: {report}")
    return report


# ==========================
# CLI
# ==========================

def main():
    parser = argparse.ArgumentParser(
        description="Summarize SEC filing text via Claude API"
    )
    parser.add_argument(
        "--raw-dir", type=str,
        default=str(Path(__file__).resolve().parent / "raw_filings"),
        help="Directory containing raw filing .txt files",
    )
    parser.add_argument(
        "--out-dir", type=str,
        default=str(Path(__file__).resolve().parent / "finished_summaries"),
        help="Output directory for summary JSON files",
    )
    parser.add_argument(
        "--tickers", type=str, default=None,
        help="Comma-separated tickers to process (default: all)",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Claude model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="Overwrite existing summaries",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Anthropic API key required. Set ANTHROPIC_API_KEY or use --api-key.",
              file=sys.stderr)
        sys.exit(1)

    tickers = None
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    run_pipeline(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        api_key=api_key,
        tickers=tickers,
        model=args.model,
        force=args.force,
    )


if __name__ == "__main__":
    main()
