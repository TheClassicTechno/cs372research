#!/usr/bin/env python3
"""
Quarterly SEC Filing Summarization Pipeline.

Reads clean text files produced by get_sec_data.py, groups filings by
ticker/year/quarter, extracts high-signal sections, sends text to Claude
for structured factual extraction, and produces one summary JSON per quarter.

Input:  clean_text/{TICKER}/{TICKER}_{YEAR}_{QUARTER}_{FORM}_{DATE}.txt
Output: finished_summaries/{TICKER}/{YEAR}/{QUARTER}.json

Examples:
  # Summarize AAPL with API key from env
  ANTHROPIC_API_KEY=sk-ant-... python filing_summarization_pipeline.py --ticker AAPL

  # Specific tickers and years
  python filing_summarization_pipeline.py --ticker AAPL,NVDA --year 2025,2026

  # All tickers, force overwrite
  python filing_summarization_pipeline.py --all --force

  # Custom directories
  python filing_summarization_pipeline.py --ticker AAPL \
      --base-dir /data/edgar --out-dir /data/summaries

Design constraints:
  - No numeric extraction from filing text.
  - All financial numbers come from structured APIs (XBRL), not this pipeline.
  - Deterministic: temperature=0, same input -> same output.
  - Atomic writes (tmp file -> rename).
  - No Form 4. No enums. No sentiment scoring.
  - No dependency on other pipeline modules.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# ==========================
# CONFIG
# ==========================

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096
TEMPERATURE = 0.0
RATE_LIMIT_SECONDS = 1.0
MAX_CHUNK_CHARS = 60_000
MAX_RETRIES = 2
RETRY_DELAYS = [2, 5]

TARGET_FORMS = {"10-Q", "10-K", "8-K"}

SUMMARY_LIST_KEYS = [
    "financial_performance",
    "guidance_and_forward_outlook",
    "material_events",
    "risk_factors_emphasized",
    "explicit_uncertainties",
    "notable_causal_statements",
]


# ==========================
# SCHEMA VALIDATION
# ==========================

def validate_summary(summary: dict) -> List[str]:
    """Validate summary matches expected schema. Returns list of error strings."""
    errors = []

    if not isinstance(summary.get("ticker"), str):
        errors.append("ticker must be str")
    if not isinstance(summary.get("year"), int):
        errors.append("year must be int")
    if not isinstance(summary.get("quarter"), str):
        errors.append("quarter must be str")

    for key in SUMMARY_LIST_KEYS:
        if key not in summary:
            errors.append(f"missing key: {key}")
            continue
        val = summary[key]
        if not isinstance(val, list):
            errors.append(f"{key} must be list")
            continue
        for i, item in enumerate(val):
            if not isinstance(item, str):
                errors.append(f"{key}[{i}] must be str")
            elif len(item.split()) > 45:
                errors.append(f"{key}[{i}] exceeds 40 words ({len(item.split())} words)")

    expected = {"ticker", "year", "quarter"} | set(SUMMARY_LIST_KEYS)
    extra = set(summary.keys()) - expected
    for k in extra:
        errors.append(f"unexpected key: {k}")

    return errors


# ==========================
# TEXT PARSING
# ==========================

def parse_filing_header(text: str) -> Dict[str, str]:
    """Extract FORM, FILING_DATE, ACCESSION from the 3-line metadata header."""
    header = {}
    for line in text.splitlines()[:10]:
        line = line.strip()
        if line.startswith("FORM:"):
            header["form"] = line[len("FORM:"):].strip()
        elif line.startswith("FILING_DATE:"):
            header["filing_date"] = line[len("FILING_DATE:"):].strip()
        elif line.startswith("ACCESSION:"):
            header["accession"] = line[len("ACCESSION:"):].strip()
    return header


def extract_body(text: str) -> str:
    """Extract body text after the metadata header (first blank line)."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if i > 0 and line.strip() == "":
            body = "\n".join(lines[i + 1:])
            if body.strip():
                return body.strip()
    return text.strip()


# Item header regex — matches "Item 1.", "Item 1A.", "Item 7.", etc. at line start
_ITEM_HEADER_RE = re.compile(r"^\s*Item\s+\d", re.IGNORECASE)

# Section target patterns by form type.
# These require the full "Item N. Section Name" on a single line,
# which naturally skips TOC entries (split across two lines).
_10K_SECTIONS = [
    re.compile(r"Item\s+1A\.\s*Risk\s+Factors", re.IGNORECASE),
    re.compile(r"Item\s+7\.\s*Management.s\s+Discussion", re.IGNORECASE),
]

_10Q_SECTIONS = [
    re.compile(r"Item\s+2\.\s*Management.s\s+Discussion", re.IGNORECASE),
    re.compile(r"Item\s+1A\.\s*Risk\s+Factors", re.IGNORECASE),
]


def extract_sections(text: str, form: str) -> str:
    """Extract high-signal sections from 10-Q/10-K filing text.

    For 8-K, returns full text (small filings).
    If no target sections found, returns full text as fallback.

    Strategy: find target section headers that start with "Item N." on a
    single line, pick the occurrence with the most content following it
    (body section vs TOC entry), extract from header to the next Item header.
    """
    base_form = form.rstrip("/A").rstrip("/") if "/" in form else form

    if base_form == "8-K":
        return text

    if base_form == "10-K":
        target_patterns = _10K_SECTIONS
    elif base_form == "10-Q":
        target_patterns = _10Q_SECTIONS
    else:
        return text

    lines = text.splitlines()
    extracted = []

    for target_pat in target_patterns:
        best_start = -1
        best_end = len(lines)
        best_content_len = 0

        for i, line in enumerate(lines):
            if target_pat.search(line) and _ITEM_HEADER_RE.match(line.strip()):
                # Found potential section start. Find end: next Item header.
                end = len(lines)
                for j in range(i + 2, len(lines)):
                    if _ITEM_HEADER_RE.match(lines[j].strip()):
                        end = j
                        break

                content_len = sum(len(lines[k]) for k in range(i + 1, end))
                if content_len > best_content_len:
                    best_start = i
                    best_end = end
                    best_content_len = content_len

        if best_start >= 0 and best_content_len > 100:
            section = "\n".join(lines[best_start:best_end]).strip()
            extracted.append(section)

    if extracted:
        return "\n\n---\n\n".join(extracted)

    return text  # Fallback: full text


def strip_boilerplate(text: str) -> str:
    """Strip SEC header boilerplate and signature blocks.

    Removes:
    - SEC filing header (UNITED STATES, SECURITIES AND EXCHANGE COMMISSION...)
    - Signature blocks at the end
    """
    lines = text.splitlines()

    # Find start: first substantive section marker
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip().upper()
        if any(kw in stripped for kw in [
            "PART I", "PART II", "ITEM 1", "ITEM 2",
            "MANAGEMENT", "RISK FACTOR", "CURRENT REPORT",
        ]):
            start = i
            break

    # Find end: signature block (search backward from end, floor at midpoint)
    end = len(lines)
    floor = max(start, len(lines) // 2)
    for i in range(len(lines) - 1, floor, -1):
        stripped = lines[i].strip().upper()
        if stripped.startswith("SIGNATURE"):
            end = i
            break

    if start > 0 or end < len(lines):
        return "\n".join(lines[start:end]).strip()
    return text


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """Split text into chunks on paragraph boundaries.

    Falls back to hard split if a single paragraph exceeds max_chars.
    """
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"\n\n+", text)
    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2  # account for \n\n separator
        if current_len + para_len > max_chars and current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        if para_len > max_chars:
            for i in range(0, len(para), max_chars):
                chunks.append(para[i:i + max_chars])
        else:
            current.append(para)
            current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


# ==========================
# FILE DISCOVERY
# ==========================

def discover_quarterly_groups(
    base_dir: Path,
    tickers: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
) -> Dict[Tuple[str, int, str], List[Path]]:
    """Walk clean_text/ and group filing .txt files by (ticker, year, quarter).

    Only includes filings where base form is in TARGET_FORMS (10-Q, 10-K, 8-K).
    Reads the metadata header from each file to determine form and filing date.
    """
    clean_dir = base_dir / "clean_text"
    groups: Dict[Tuple[str, int, str], List[Path]] = defaultdict(list)

    if not clean_dir.exists():
        return dict(groups)

    for ticker_dir in sorted(clean_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        if tickers and ticker not in tickers:
            continue

        for txt_file in sorted(ticker_dir.glob("*.txt")):
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    head = f.read(500)
            except OSError:
                continue

            header = parse_filing_header(head)
            form = header.get("form", "")
            filing_date = header.get("filing_date", "")

            base_form = form.rstrip("/A").rstrip("/") if "/" in form else form
            if base_form not in TARGET_FORMS:
                continue

            if not filing_date:
                continue

            try:
                parsed = datetime.strptime(filing_date, "%Y-%m-%d")
            except ValueError:
                continue

            year = parsed.year
            quarter = f"Q{(parsed.month - 1) // 3 + 1}"

            if years and year not in years:
                continue

            groups[(ticker, year, quarter)].append(txt_file)

    return dict(groups)


def output_path_for(out_dir: Path, ticker: str, year: int, quarter: str) -> Path:
    """Build output path: out_dir/TICKER/YEAR/QUARTER.json"""
    return out_dir / ticker / str(year) / f"{quarter}.json"


def save_summary(path: Path, summary: dict) -> None:
    """Atomically write summary JSON (tmp -> rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    tmp.rename(path)


# ==========================
# PROMPTS
# ==========================

_SYSTEM_PROMPT = """\
You are a factual extractor for SEC filings. You extract specific \
statements and facts from filing text into structured JSON.

STRICT RULES:
1. Return ONLY valid JSON. No prose, no markdown fences, no explanation.
2. Each list contains short factual bullet statements.
3. Each bullet MUST be 40 words or fewer.
4. NO interpretation, opinion, or sentiment language.
5. NO prediction or speculation beyond what management explicitly states.
6. Extract ONLY what is explicitly stated in the text.
7. If a category has no relevant content, return an empty list [].
8. Do NOT invent or hallucinate any information.
9. Do NOT extract or cite specific financial numbers."""

_EXTRACTION_PROMPT = """\
Extract factual information from the following SEC filing text for \
{ticker} ({year} {quarter}).

Return a JSON object with EXACTLY this structure:

{{
  "financial_performance": ["<bullet>", ...],
  "guidance_and_forward_outlook": ["<bullet>", ...],
  "material_events": ["<bullet>", ...],
  "risk_factors_emphasized": ["<bullet>", ...],
  "explicit_uncertainties": ["<bullet>", ...],
  "notable_causal_statements": ["<bullet>", ...]
}}

Field definitions:
- financial_performance: Revenue trends, margin direction, segment \
results, earnings characterization. Qualitative statements only.
- guidance_and_forward_outlook: Management guidance, forward estimates, \
planned investments, strategic direction stated by management.
- material_events: Acquisitions, restructurings, leadership changes, \
legal actions, regulatory events, offerings, shareholder votes.
- risk_factors_emphasized: Specific risks highlighted, newly added \
risks, or risks given increased emphasis.
- explicit_uncertainties: Direct quotes or paraphrases where management \
explicitly acknowledges uncertainty or unpredictability.
- notable_causal_statements: Cause-and-effect statements by management \
(e.g., "revenue increased due to higher services demand").

Constraints:
- Each bullet <= 40 words.
- No interpretation. No sentiment. No opinion.
- Factual extraction only.

FILING TEXT:
{text}"""

_MERGE_PROMPT = """\
You have {n} partial extractions from SEC filings for {ticker} ({year} \
{quarter}). Merge them into ONE consolidated JSON object.

Rules:
- Deduplicate similar statements. Keep the most specific version.
- Each bullet <= 40 words.
- Return ONLY the merged JSON. No explanation.
- Same schema: financial_performance, guidance_and_forward_outlook, \
material_events, risk_factors_emphasized, explicit_uncertainties, \
notable_causal_statements.
- Each field is a list of strings. Empty list [] if nothing found.

PARTIAL EXTRACTIONS:
{partials}"""


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
    """Send a message to Claude API and return text response."""
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
    resp = requests.post(
        ANTHROPIC_API_URL, headers=headers, json=body, timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data.get("content", [])
    if not content:
        raise ValueError("Claude returned empty content")
    return content[0].get("text", "")


def parse_json_response(raw: str) -> dict:
    """Extract JSON from Claude's response, handling markdown code fences."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines)
    return json.loads(text)


def call_claude_json(
    api_key: str,
    system: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Call Claude and parse JSON response with retries.

    Retries on JSON parse errors and transient HTTP errors (429, 5xx).
    """
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            time.sleep(RATE_LIMIT_SECONDS)
            raw = call_claude(api_key, system, user_prompt, model=model)
            return parse_json_response(raw)
        except json.JSONDecodeError as e:
            last_error = e
            if attempt < MAX_RETRIES:
                print(f"      JSON parse error, retrying ({attempt + 1}/{MAX_RETRIES})...")
                time.sleep(RETRY_DELAYS[attempt])
        except requests.HTTPError as e:
            last_error = e
            status = e.response.status_code if e.response else 0
            if attempt < MAX_RETRIES and status in (429, 500, 502, 503):
                print(f"      HTTP {status}, retrying ({attempt + 1}/{MAX_RETRIES})...")
                time.sleep(RETRY_DELAYS[attempt])
            else:
                raise
    raise last_error


# ==========================
# SUMMARIZATION
# ==========================

def prepare_filing_text(filepath: Path) -> Optional[Tuple[str, str]]:
    """Read a filing, extract and clean high-signal text.

    Returns (form, cleaned_text) or None if no usable text.
    """
    text = filepath.read_text(encoding="utf-8")
    header = parse_filing_header(text)
    form = header.get("form", "unknown")
    body = extract_body(text)

    if not body.strip():
        return None

    stripped = strip_boilerplate(body)
    cleaned = extract_sections(stripped, form)

    if not cleaned.strip():
        return None

    return (form, cleaned.strip())


def summarize_quarter(
    api_key: str,
    ticker: str,
    year: int,
    quarter: str,
    filings: List[Path],
    model: str = DEFAULT_MODEL,
) -> dict:
    """Summarize all filings for one quarter into a single structured JSON."""

    # Prepare text from each filing
    prepared = []
    for fp in sorted(filings):
        result = prepare_filing_text(fp)
        if result:
            prepared.append(result)

    empty_result = {
        "ticker": ticker,
        "year": year,
        "quarter": quarter,
        **{k: [] for k in SUMMARY_LIST_KEYS},
    }

    if not prepared:
        return empty_result

    # Combine all filing texts with form-type separators
    combined_parts = []
    for form, text in prepared:
        combined_parts.append(f"=== {form} FILING ===\n{text}")
    combined = "\n\n".join(combined_parts)

    print(f"    Combined text: {len(combined)} chars from {len(prepared)} filing(s)")

    chunks = chunk_text(combined)

    if len(chunks) == 1:
        prompt = _EXTRACTION_PROMPT.format(
            ticker=ticker, year=year, quarter=quarter, text=chunks[0],
        )
        result = call_claude_json(api_key, _SYSTEM_PROMPT, prompt, model=model)
    else:
        # Multi-chunk: extract from each, then merge
        chunk_results = []
        for i, chunk in enumerate(chunks):
            print(f"    Chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)")
            prompt = _EXTRACTION_PROMPT.format(
                ticker=ticker, year=year, quarter=quarter, text=chunk,
            )
            chunk_results.append(
                call_claude_json(api_key, _SYSTEM_PROMPT, prompt, model=model)
            )

        print(f"    Merging {len(chunk_results)} chunk extractions...")
        partials_json = json.dumps(chunk_results, indent=2)
        merge_prompt = _MERGE_PROMPT.format(
            ticker=ticker, year=year, quarter=quarter,
            n=len(chunk_results), partials=partials_json,
        )
        result = call_claude_json(api_key, _SYSTEM_PROMPT, merge_prompt, model=model)

    # Attach metadata
    result["ticker"] = ticker
    result["year"] = year
    result["quarter"] = quarter

    # Ensure all list keys exist
    for key in SUMMARY_LIST_KEYS:
        result.setdefault(key, [])

    # Strip any unexpected keys the LLM may have added
    allowed = {"ticker", "year", "quarter"} | set(SUMMARY_LIST_KEYS)
    for key in list(result.keys()):
        if key not in allowed:
            del result[key]

    return result


# ==========================
# PIPELINE
# ==========================

def run_pipeline(
    base_dir: Path,
    out_dir: Path,
    api_key: str,
    tickers: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    model: str = DEFAULT_MODEL,
    force: bool = False,
) -> Dict[str, Any]:
    """Run the quarterly summarization pipeline."""
    groups = discover_quarterly_groups(base_dir, tickers, years)
    print(f"Found {len(groups)} quarterly group(s)")

    if not groups:
        print("No filings found to process.")
        return {"total": 0, "processed": 0, "skipped": 0}

    processed = 0
    skipped = 0

    for (ticker, year, quarter), filings in sorted(groups.items()):
        out_path = output_path_for(out_dir, ticker, year, quarter)

        if out_path.exists() and not force:
            print(f"  SKIP {ticker} {year} {quarter} (already exists)")
            skipped += 1
            continue

        # Show what filings are in this group
        forms = []
        for f in filings:
            with open(f, "r", encoding="utf-8") as fh:
                h = parse_filing_header(fh.read(500))
            forms.append(h.get("form", "?"))

        print(f"  {ticker} {year} {quarter}: {len(filings)} filing(s) [{', '.join(forms)}]")

        try:
            summary = summarize_quarter(
                api_key, ticker, year, quarter, filings, model=model,
            )
        except Exception as e:
            print(f"  FAILED {ticker} {year} {quarter}: {type(e).__name__}: {e}")
            skipped += 1
            continue

        errors = validate_summary(summary)
        if errors:
            print(f"  WARNING: {errors}")

        save_summary(out_path, summary)
        print(f"  Saved {out_path}")
        processed += 1

    report = {"total": len(groups), "processed": processed, "skipped": skipped}
    print(f"\nPipeline complete: {report}")
    return report


# ==========================
# CLI
# ==========================

def main():
    parser = argparse.ArgumentParser(
        description="Quarterly SEC filing summarization via Claude API",
    )
    parser.add_argument(
        "--base-dir", type=str,
        default=str(Path(__file__).resolve().parent),
        help="Base directory containing clean_text/ (default: EDGAR/)",
    )
    parser.add_argument(
        "--out-dir", type=str,
        default=str(Path(__file__).resolve().parent / "finished_summaries"),
        help="Output directory for quarterly JSON summaries",
    )
    parser.add_argument(
        "--ticker", type=str, default=None,
        help="Comma-separated tickers (e.g. AAPL,NVDA)",
    )
    parser.add_argument(
        "--year", type=str, default=None,
        help="Comma-separated years (e.g. 2025,2026)",
    )
    parser.add_argument(
        "--all", action="store_true", default=False,
        help="Process all tickers found in clean_text/",
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="Overwrite existing summaries",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Claude model (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "ERROR: Set ANTHROPIC_API_KEY or use --api-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    tickers = None
    if args.ticker:
        tickers = [t.strip().upper() for t in args.ticker.split(",") if t.strip()]

    years = None
    if args.year:
        years = [int(y.strip()) for y in args.year.split(",") if y.strip()]

    run_pipeline(
        base_dir=Path(args.base_dir),
        out_dir=Path(args.out_dir),
        api_key=api_key,
        tickers=tickers,
        years=years,
        model=args.model,
        force=args.force,
    )


if __name__ == "__main__":
    main()
