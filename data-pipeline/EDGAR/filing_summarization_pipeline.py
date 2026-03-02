#!/usr/bin/env python3
"""
Quarterly SEC Filing Summarization Pipeline.

Compresses clean SEC filing text into dense quarterly economic state
representations — structured JSON optimized for downstream agent consumption.

Input:  clean_text/{TICKER}/{TICKER}_{YEAR}_{QUARTER}_{FORM}_{DATE}.txt
Output: finished_summaries/{TICKER}/{YEAR}/{QUARTER}.json

Each output is a 200-375 word compressed state vector: six dense paragraphs
covering operating state, cost structure, material events, macro exposures,
forward outlook, and uncertainty profile.

Examples:
  ANTHROPIC_API_KEY=sk-ant-... python filing_summarization_pipeline.py --ticker AAPL
  python filing_summarization_pipeline.py --ticker AAPL,NVDA --year 2025,2026
  python filing_summarization_pipeline.py --all --force

Design constraints:
  - Deterministic: temperature=0, same input -> same output.
  - Compressed: 200-375 words per quarter, no bullet lists.
  - No numeric extraction from filing text.
  - Atomic writes (tmp file -> rename).
  - Stable JSON key ordering.
  - No dependency on other pipeline modules.
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml
from tqdm import tqdm

_SUPPORTED_TICKERS_PATH = Path(__file__).resolve().parent.parent / "supported_tickers.yaml"


def load_supported_tickers(yaml_path: Path = _SUPPORTED_TICKERS_PATH) -> list:
    """Load ticker symbols from supported_tickers.yaml."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return [entry["symbol"] for entry in data["supported_tickers"]]


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
MAX_WORKERS = 4

_rate_limit_lock = threading.Lock()

TARGET_FORMS = {"10-Q", "10-K", "8-K"}

MAX_SUMMARY_WORDS = 375
MIN_SUMMARY_WORDS = 150

# Form-specific word limits
WORD_LIMITS = {
    "10-K": {"target": 300, "max": 375},
    "10-Q": {"target": 275, "max": 350},
    "8-K": {"target": 225, "max": 350},
}
DEFAULT_WORD_LIMIT = {"target": 275, "max": 375}

PARAGRAPH_KEYS = [
    "operating_state",
    "cost_structure",
    "material_events",
    "macro_exposures",
    "forward_outlook",
    "uncertainty_profile",
]

META_KEYS = ["ticker", "form", "filing_date", "fiscal_period", "period_type"]

REQUIRED_KEYS = META_KEYS + PARAGRAPH_KEYS + ["word_count"]

# Stable key ordering for output JSON
KEY_ORDER = [
    "ticker", "form", "filing_date", "fiscal_period", "period_type",
    "operating_state", "cost_structure", "material_events",
    "macro_exposures", "forward_outlook", "uncertainty_profile",
    "word_count",
    "pipeline_run_id", "generated_at_utc",
]


# ==========================
# HELPERS
# ==========================

def count_words(text: str) -> int:
    """Count words in a string."""
    return len(text.split())


def compute_total_words(summary: dict) -> int:
    """Sum word counts across all paragraph fields."""
    total = 0
    for key in PARAGRAPH_KEYS:
        val = summary.get(key, "")
        if isinstance(val, str):
            total += count_words(val)
    return total


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace, strip leading/trailing."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


# ==========================
# SCHEMA VALIDATION
# ==========================

_BULLET_RE = re.compile(r"^\s*[-*•]\s", re.MULTILINE)


def validate_summary(summary: dict) -> List[str]:
    """Validate summary matches the compressed state vector schema.

    Returns list of error strings. Empty means valid.
    """
    errors = []

    # Required keys
    for key in REQUIRED_KEYS:
        if key not in summary:
            errors.append(f"missing key: {key}")

    # Metadata types
    for key in ("ticker", "form", "filing_date", "fiscal_period", "period_type"):
        val = summary.get(key)
        if val is not None and not isinstance(val, str):
            errors.append(f"{key} must be str")

    if "word_count" in summary and not isinstance(summary["word_count"], int):
        errors.append("word_count must be int")

    # Paragraph fields: must be str, no bullet chars
    for key in PARAGRAPH_KEYS:
        val = summary.get(key)
        if val is None:
            continue
        if not isinstance(val, str):
            errors.append(f"{key} must be str, got {type(val).__name__}")
            continue
        if _BULLET_RE.search(val):
            errors.append(f"{key} contains bullet characters")

    # Word count bounds
    wc = summary.get("word_count", 0)
    if isinstance(wc, int) and wc > MAX_SUMMARY_WORDS:
        errors.append(f"word_count {wc} exceeds {MAX_SUMMARY_WORDS}")

    # No extra keys (provenance fields are allowed)
    expected = set(REQUIRED_KEYS) | {"pipeline_run_id", "generated_at_utc"}
    extra = set(summary.keys()) - expected
    for k in sorted(extra):
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


_ITEM_HEADER_RE = re.compile(r"^\s*Item\s+\d", re.IGNORECASE)

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

    For 8-K, returns full text. If no sections found, returns full text.
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

    return text


def strip_boilerplate(text: str) -> str:
    """Strip SEC header boilerplate and signature blocks."""
    lines = text.splitlines()

    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip().upper()
        if any(kw in stripped for kw in [
            "PART I", "PART II", "ITEM 1", "ITEM 2",
            "MANAGEMENT", "RISK FACTOR", "CURRENT REPORT",
        ]):
            start = i
            break

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
    """Split text into chunks on paragraph boundaries."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"\n\n+", text)
    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2
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

# Matches filenames like AAPL_2024_Q4_10-K_2024-11-01.txt
_FILENAME_RE = re.compile(
    r"^[A-Z]+_(\d{4})_(Q[1-4])_[^_]+_\d{4}-\d{2}-\d{2}\.txt$"
)


def _parse_fiscal_quarter_from_filename(
    filepath: Path,
) -> Optional[Tuple[int, str]]:
    """Extract (year, quarter) from the fiscal-aligned filename.

    Filenames are written by get_sec_data.py using reportDate-based
    fiscal alignment: TICKER_YEAR_QUARTER_FORM_FILINGDATE.txt
    The YEAR and QUARTER in the filename represent the fiscal period,
    NOT the filing date.
    """
    m = _FILENAME_RE.match(filepath.name)
    if m:
        return int(m.group(1)), m.group(2)
    return None


def discover_quarterly_groups(
    base_dir: Path,
    tickers: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
) -> Dict[Tuple[str, int, str], List[Path]]:
    """Walk clean_text/ and group filing .txt files by (ticker, year, quarter).

    Year/quarter are derived from the fiscal-aligned filename (set by
    get_sec_data.py using reportDate), NOT from the filing date.
    Only includes filings where base form is in TARGET_FORMS.
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
            # Extract fiscal year/quarter from filename
            fiscal = _parse_fiscal_quarter_from_filename(txt_file)
            if fiscal is None:
                # Skip old accession-based filenames
                continue

            year, quarter = fiscal

            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    head = f.read(500)
            except OSError:
                continue

            header = parse_filing_header(head)
            form = header.get("form", "")

            base_form = form.rstrip("/A").rstrip("/") if "/" in form else form
            if base_form not in TARGET_FORMS:
                continue

            if years and year not in years:
                continue

            groups[(ticker, year, quarter)].append(txt_file)

    return dict(groups)


def output_path_for(out_dir: Path, ticker: str, year: int, quarter: str) -> Path:
    """Build output path: out_dir/TICKER/YEAR/QUARTER.json"""
    return out_dir / ticker / str(year) / f"{quarter}.json"


def save_summary(path: Path, summary: dict) -> None:
    """Atomically write summary JSON with stable key ordering."""
    try:
        from provenance import inline_provenance
        summary.update(inline_provenance())
    except ImportError:
        pass

    ordered = {}
    for k in KEY_ORDER:
        if k in summary:
            ordered[k] = summary[k]
    for k in summary:
        if k not in ordered:
            ordered[k] = summary[k]

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2)
    tmp.rename(path)


# ==========================
# PROMPTS
# ==========================

_SYSTEM_PROMPT = """\
You compress SEC filing text into structured quarterly economic state \
representations. You write dense institutional language suitable for a \
90-second portfolio manager briefing.

RULES:
1. Return ONLY valid JSON. No prose outside the JSON object.
2. Each field is a dense paragraph of 2-5 sentences. NO bullet lists. \
NO dashes. NO numbered items. Prose paragraphs only.
3. NEVER exceed the specified word maximum across all six fields.
4. Compress aggressively: merge similar statements, eliminate repetition.
5. Focus on DELTA: what changed this period, what shifted materially.
6. Preserve causal chains: "X increased driven by Y and Z."
7. Remove generic boilerplate risk language UNLESS it reflects a NEW or \
materially CHANGED exposure this period.
8. NO per-region percentage breakdowns. Merge regional performance into \
one sentence using directional language ("broadly positive with \
double-digit growth in Europe and Japan").
9. NO per-line-item dollar amounts. Anchor numbers ONLY at consolidated \
total level (total company revenue, total gross margin percentage). NO \
segment revenue dollars, NO operating expense line-item dollars, NO \
purchase obligation breakdowns. Describe components directionally.
10. NO accounting standard code enumeration (ASU numbers). Reference \
pending accounting changes generically unless a standard has material \
near-term impact.
11. NO redundant risk stacking. If risks share a common mechanism, \
state the mechanism once with its variants.
12. No interpretation. No sentiment. No editorializing. Neutral factual tone.
13. No "may/could/might" hedging unless reflecting a specific new \
material disclosure."""

_EXTRACTION_PROMPT = """\
Compress the following SEC filing text for {ticker} ({fiscal_period}) \
into a quarterly economic state representation.

Return a JSON object with EXACTLY this structure:

{{
  "operating_state": "<2-5 sentences>",
  "cost_structure": "<2-5 sentences>",
  "material_events": "<2-5 sentences>",
  "macro_exposures": "<2-5 sentences>",
  "forward_outlook": "<2-5 sentences>",
  "uncertainty_profile": "<2-5 sentences>"
}}

WORD BUDGET: {word_target} words ideal, {word_max} maximum across all fields.

Field definitions:
- operating_state: Revenue direction and primary growth drivers. Segment \
performance as aggregate patterns ("broadly positive across regions with \
particular strength in..."). Key causal chains. Anchor numbers at total \
revenue level only.
- cost_structure: Gross margin direction and mix dynamics (products vs \
services). Operating expense trend directionally with primary drivers. \
No per-line-item dollar amounts.
- material_events: Capital actions (buybacks, dividends), regulatory \
developments, product launches, leadership changes. Only events that \
actually occurred. One sentence per distinct event.
- macro_exposures: ONLY new or materially escalated macro risks this \
period. Merge related trade/tariff risks into one statement. Skip \
standard boilerplate carried forward unchanged.
- forward_outlook: Management guidance, strategic direction, capital \
allocation plans. What management committed to or signaled about the future.
- uncertainty_profile: Unresolved contingencies, pending regulatory or \
legal matters. Reference accounting standard changes generically without \
enumerating codes. Merge related uncertainties.

Constraints:
- Dense prose paragraphs. No bullets. No lists.
- If a field has no relevant content, write one sentence noting the absence.

FILING TEXT:
{text}"""

_MERGE_PROMPT = """\
You have {n} partial compressions from SEC filings for {ticker} \
({fiscal_period}). Merge into ONE consolidated JSON object.

Rules:
- Deduplicate: keep the most specific version of each signal.
- Merge aggressively. No repeated information across fields.
- Same six-field JSON structure (operating_state, cost_structure, \
material_events, macro_exposures, forward_outlook, uncertainty_profile).
- Each field: 2-5 dense prose sentences. No bullets.
- Word budget: {word_target} words ideal, {word_max} maximum.
- No per-region percentage breakdowns. No per-line-item dollar amounts. \
No accounting standard code enumeration.
- Return ONLY the merged JSON.

PARTIAL COMPRESSIONS:
{partials}"""

_RECOMPRESS_PROMPT = """\
The following quarterly summary for {ticker} ({fiscal_period}) has \
{current_words} words, exceeding the {target} word maximum. You MUST \
reduce to {word_target} words or fewer.

CUT THESE AGGRESSIVELY:
- Remove ALL per-segment dollar amounts (keep only total company revenue).
- Remove ALL per-region percentages. Use directional language instead.
- Remove ALL operating expense line-item dollar amounts.
- Merge sentences describing similar risks or trends into one.
- Drop purchase obligation specifics beyond one summary sentence.
- Reference accounting changes in one generic sentence without codes.
- Remove market share commentary and generic competitive language.

KEEP:
- Causal chains explaining why key metrics moved.
- Total/headline numbers (total revenue, total gross margin percentage).
- Material capital actions at aggregate level (total buybacks, dividend changes).

Return ONLY the compressed JSON. Same six-field structure. Dense prose, \
no bullets. Maximum {target} words.

CURRENT SUMMARY:
{current_json}"""


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
    """Call Claude and parse JSON response with retries."""
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            with _rate_limit_lock:
                time.sleep(RATE_LIMIT_SECONDS)
            raw = call_claude(api_key, system, user_prompt, model=model)
            return parse_json_response(raw)
        except json.JSONDecodeError as e:
            last_error = e
            if attempt < MAX_RETRIES:
                tqdm.write(f"      JSON parse error, retrying ({attempt + 1}/{MAX_RETRIES})...")
                time.sleep(RETRY_DELAYS[attempt])
        except requests.HTTPError as e:
            last_error = e
            status = e.response.status_code if e.response else 0
            if attempt < MAX_RETRIES and status in (429, 500, 502, 503):
                tqdm.write(f"      HTTP {status}, retrying ({attempt + 1}/{MAX_RETRIES})...")
                time.sleep(RETRY_DELAYS[attempt])
            else:
                raise
    raise last_error


# ==========================
# SUMMARIZATION
# ==========================

# Form priority for picking primary filing metadata: 10-K > 10-Q > 8-K
_FORM_PRIORITY = {"10-K": 0, "10-Q": 1, "8-K": 2}


def _base_form(form: str) -> str:
    return form.rstrip("/A").rstrip("/") if "/" in form else form


def _derive_metadata(
    filings: List[Path],
    year: int,
    quarter: str,
) -> Tuple[str, str, str, str]:
    """Derive (form, filing_date, fiscal_period, period_type) from filings.

    Uses the fiscal year/quarter from the caller (originally parsed from
    the filename, which reflects reportDate-based fiscal alignment).
    Picks the primary filing by priority: 10-K > 10-Q > 8-K.
    """
    candidates = []
    for fp in filings:
        with open(fp, "r", encoding="utf-8") as f:
            head = f.read(500)
        header = parse_filing_header(head)
        form = header.get("form", "")
        filing_date = header.get("filing_date", "")
        bf = _base_form(form)
        candidates.append((bf, form, filing_date))

    candidates.sort(key=lambda x: _FORM_PRIORITY.get(x[0], 99))
    bf, form, filing_date = candidates[0]

    if bf == "10-K":
        fiscal_period = f"FY{year}"
        period_type = "annual"
    else:
        fiscal_period = f"{year}-{quarter}"
        period_type = "quarterly"

    return form, filing_date, fiscal_period, period_type


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


def _normalize_paragraphs(result: dict) -> dict:
    """Apply whitespace normalization to all paragraph fields."""
    for key in PARAGRAPH_KEYS:
        val = result.get(key, "")
        if isinstance(val, str):
            result[key] = normalize_whitespace(val)
    return result


def summarize_quarter(
    api_key: str,
    ticker: str,
    year: int,
    quarter: str,
    filings: List[Path],
    model: str = DEFAULT_MODEL,
) -> dict:
    """Compress all filings for one quarter into a single state vector."""

    form, filing_date, fiscal_period, period_type = _derive_metadata(filings, year, quarter)

    # Resolve form-specific word limits
    bf = _base_form(form)
    limits = WORD_LIMITS.get(bf, DEFAULT_WORD_LIMIT)
    word_target = limits["target"]
    word_max = limits["max"]

    # Prepare text from each filing
    prepared = []
    for fp in sorted(filings):
        result = prepare_filing_text(fp)
        if result:
            prepared.append(result)

    empty_result = {
        "ticker": ticker,
        "form": form,
        "filing_date": filing_date,
        "fiscal_period": fiscal_period,
        "period_type": period_type,
        **{k: "No relevant content in filing text." for k in PARAGRAPH_KEYS},
        "word_count": 0,
    }
    empty_result["word_count"] = compute_total_words(empty_result)

    if not prepared:
        return empty_result

    # Combine filing texts with separators
    combined_parts = []
    for filing_form, text in prepared:
        combined_parts.append(f"=== {filing_form} FILING ===\n{text}")
    combined = "\n\n".join(combined_parts)

    tqdm.write(f"    [{ticker}] Combined text: {len(combined)} chars from {len(prepared)} filing(s)")

    chunks = chunk_text(combined)

    if len(chunks) == 1:
        tqdm.write(f"    [{ticker}] Compressing {len(combined)} chars via Claude...")
        prompt = _EXTRACTION_PROMPT.format(
            ticker=ticker, fiscal_period=fiscal_period,
            word_target=word_target, word_max=word_max, text=chunks[0],
        )
        result = call_claude_json(api_key, _SYSTEM_PROMPT, prompt, model=model)
    else:
        tqdm.write(f"    [{ticker}] Compressing {len(chunks)} chunks in parallel...")
        chunk_prompts = []
        for i, chunk in enumerate(chunks):
            prompt = _EXTRACTION_PROMPT.format(
                ticker=ticker, fiscal_period=fiscal_period,
                word_target=word_target, word_max=word_max, text=chunk,
            )
            chunk_prompts.append((i, prompt))

        chunk_results = [None] * len(chunks)
        with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
            futures = {
                pool.submit(call_claude_json, api_key, _SYSTEM_PROMPT, prompt, model):
                    idx
                for idx, prompt in chunk_prompts
            }
            for future in as_completed(futures):
                idx = futures[future]
                chunk_results[idx] = future.result()
                tqdm.write(f"    [{ticker}] Chunk {idx + 1}/{len(chunks)} done ({len(chunks[idx])} chars)")

        tqdm.write(f"    [{ticker}] Merging {len(chunk_results)} chunk compressions...")
        partials_json = json.dumps(chunk_results, indent=2)
        merge_prompt = _MERGE_PROMPT.format(
            ticker=ticker, fiscal_period=fiscal_period,
            n=len(chunk_results), partials=partials_json,
            word_target=word_target, word_max=word_max,
        )
        result = call_claude_json(api_key, _SYSTEM_PROMPT, merge_prompt, model=model)

    # Normalize whitespace
    result = _normalize_paragraphs(result)

    # Ensure all paragraph keys exist
    for key in PARAGRAPH_KEYS:
        result.setdefault(key, "No relevant content in filing text.")

    # Strip unexpected keys from LLM output
    allowed_llm = set(PARAGRAPH_KEYS)
    for key in list(result.keys()):
        if key not in allowed_llm:
            del result[key]

    # Check word count — recompress if over limit (up to 2 passes)
    total = compute_total_words(result)
    for recompress_pass in range(2):
        if total <= word_max:
            break
        tqdm.write(f"    [{ticker}] Word count {total} exceeds {word_max}, recompressing (pass {recompress_pass + 1})...")
        recompress_prompt = _RECOMPRESS_PROMPT.format(
            ticker=ticker, fiscal_period=fiscal_period,
            current_words=total, target=word_max,
            word_target=word_target,
            current_json=json.dumps(result, indent=2),
        )
        result = call_claude_json(api_key, _SYSTEM_PROMPT, recompress_prompt, model=model)
        result = _normalize_paragraphs(result)
        for key in PARAGRAPH_KEYS:
            result.setdefault(key, "No relevant content in filing text.")
        for key in list(result.keys()):
            if key not in allowed_llm:
                del result[key]
        total = compute_total_words(result)

    # Attach metadata
    result["ticker"] = ticker
    result["form"] = form
    result["filing_date"] = filing_date
    result["fiscal_period"] = fiscal_period
    result["period_type"] = period_type
    result["word_count"] = total

    return result


# ==========================
# PIPELINE
# ==========================

def _process_one_quarter(
    ticker: str,
    year: int,
    quarter: str,
    filings: List[Path],
    out_dir: Path,
    api_key: str,
    model: str,
    force: bool,
) -> str:
    """Process a single quarter. Returns 'processed', 'skipped', or 'failed'."""
    out_path = output_path_for(out_dir, ticker, year, quarter)

    if out_path.exists() and not force:
        tqdm.write(f"  [{ticker}] SKIP {year} {quarter} (already exists)")
        return "skipped"

    forms = []
    for f in filings:
        with open(f, "r", encoding="utf-8") as fh:
            h = parse_filing_header(fh.read(500))
        forms.append(h.get("form", "?"))

    tqdm.write(f"  [{ticker}] {year} {quarter}: {len(filings)} filing(s) [{', '.join(forms)}]")

    try:
        summary = summarize_quarter(
            api_key, ticker, year, quarter, filings, model=model,
        )
    except requests.Timeout:
        tqdm.write(f"  [{ticker}] FAIL {year} {quarter} — API timeout (120s)")
        return "failed"
    except requests.ConnectionError as e:
        tqdm.write(f"  [{ticker}] FAIL {year} {quarter} — connection error: {e}")
        return "failed"
    except requests.HTTPError as e:
        status = e.response.status_code if e.response else "unknown"
        tqdm.write(f"  [{ticker}] FAIL {year} {quarter} — HTTP {status}: {e}")
        return "failed"
    except json.JSONDecodeError as e:
        tqdm.write(f"  [{ticker}] FAIL {year} {quarter} — invalid JSON from Claude: {e}")
        return "failed"
    except Exception as e:
        tqdm.write(f"  [{ticker}] FAIL {year} {quarter} — {type(e).__name__}: {e}")
        return "failed"

    errors = validate_summary(summary)
    if errors:
        tqdm.write(f"  [{ticker}] WARNING {year} {quarter}: {errors}")

    save_summary(out_path, summary)
    wc = summary.get("word_count", "?")
    tqdm.write(f"  [{ticker}] Saved {year} {quarter} ({wc} words)")
    return "processed"


def run_pipeline(
    base_dir: Path,
    out_dir: Path,
    api_key: str,
    tickers: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    model: str = DEFAULT_MODEL,
    force: bool = False,
    workers: int = 1,
) -> Dict[str, Any]:
    """Run the quarterly summarization pipeline."""
    groups = discover_quarterly_groups(base_dir, tickers, years)
    print(f"Found {len(groups)} quarterly group(s)", flush=True)

    if not groups:
        print("No filings found to process.")
        return {"total": 0, "processed": 0, "skipped": 0, "failed": 0}

    processed = 0
    skipped = 0
    failed = 0

    sorted_groups = sorted(groups.items())
    label = f"Summarizing ({workers} worker{'s' if workers > 1 else ''})"

    if workers == 1:
        pbar = tqdm(sorted_groups, desc=label, unit="quarter")
        for (ticker, year, quarter), filings in pbar:
            pbar.set_postfix_str(f"{ticker} {year} {quarter}")
            result = _process_one_quarter(
                ticker, year, quarter, filings, out_dir, api_key, model, force,
            )
            if result == "processed":
                processed += 1
            elif result == "skipped":
                skipped += 1
            else:
                failed += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _process_one_quarter,
                    ticker, year, quarter, filings,
                    out_dir, api_key, model, force,
                ): (ticker, year, quarter)
                for (ticker, year, quarter), filings in sorted_groups
            }
            for future in tqdm(
                as_completed(futures), desc=label, unit="quarter", total=len(futures),
            ):
                result = future.result()
                if result == "processed":
                    processed += 1
                elif result == "skipped":
                    skipped += 1
                else:
                    failed += 1

    report = {"total": len(groups), "processed": processed, "skipped": skipped, "failed": failed}
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
        "--supported", action="store_true", default=False,
        help="Use all tickers from supported_tickers.yaml",
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
    parser.add_argument(
        "--parallel", action="store_true",
        help="Process quarters in parallel",
    )
    parser.add_argument(
        "--workers", type=int, default=3,
        help=f"Number of parallel workers (default: 3, max: {MAX_WORKERS})",
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
    if args.supported:
        tickers = load_supported_tickers()
    elif args.ticker:
        tickers = [t.strip().upper() for t in args.ticker.split(",") if t.strip()]

    years = None
    if args.year:
        years = [int(y.strip()) for y in args.year.split(",") if y.strip()]

    workers = min(args.workers, MAX_WORKERS) if args.parallel else 1

    run_pipeline(
        base_dir=Path(args.base_dir),
        out_dir=Path(args.out_dir),
        api_key=api_key,
        tickers=tickers,
        years=years,
        model=args.model,
        force=args.force,
        workers=workers,
    )


if __name__ == "__main__":
    main()
