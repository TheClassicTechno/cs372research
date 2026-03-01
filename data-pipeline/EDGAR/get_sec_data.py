#!/usr/bin/env python3
"""
SEC Filing Text Extractor — Download filings from EDGAR and extract narrative text.

Examples:
  # 1. Last 4 completed quarters for a few tickers (auto-maps Q4 to 10-K)
  python get_sec_data.py --tickers AAPL,NVDA,MSFT --last-n 4

  # 2. Specific year + quarters (10-Q filings)
  python get_sec_data.py --tickers AAPL --years 2024 --quarters Q1,Q2,Q3

  # 3. Annual filings only (10-K, matches any quarter in the year)
  python get_sec_data.py --tickers AAPL,GOOG --years 2023,2024 --quarters ANNUAL

  # 4. Multiple years, all quarters
  python get_sec_data.py --tickers JPM,GS --years 2023,2024,2025 --quarters Q1,Q2,Q3,Q4

  # 5. Custom form types (8-K and Form 4)
  python get_sec_data.py --tickers AAPL --last-n 4 --forms 8-K,4

  # 6. Full institutional bundle (10-K, 10-Q, 8-K, Form 4, SC 13D/G, etc.)
  python get_sec_data.py --tickers AAPL --last-n 8 --bundle core

  # 7. Include amendment filings (10-K/A, 10-Q/A, etc.)
  python get_sec_data.py --tickers NVDA --years 2024 --quarters Q1,Q2,Q3,Q4 --include-amendments

  # 8. Parallel download with 3 workers
  python get_sec_data.py --tickers AAPL,NVDA,MSFT --last-n 4 --parallel --workers 3

  # 9. Force re-extract everything (overwrite existing)
  python get_sec_data.py --tickers AAPL --last-n 2 --force-refresh

  # 10. Stateless run — ignore index, don't write cache
  python get_sec_data.py --tickers AAPL --last-n 2 --no-cache --output /tmp/sec_filings
"""

import argparse
import json
import os
import re
import requests
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from html import unescape as html_unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Dict

# ==========================
# CONFIG
# ==========================

SEC_HEADERS = {
    "User-Agent": "Your Name your.email@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.edgar.gov"
}

BASE_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
BASE_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"

RATE_LIMIT_SECONDS = 0.2  # 5 requests/second max (SEC compliant)

# Default institutional form bundle for --bundle core
CORE_FORMS = [
    "10-K", "10-Q", "8-K", "4", "SC 13D", "SC 13G",
    "NT 10-Q", "NT 10-K", "S-3", "6-K", "424B*",
]

MAX_WORKERS = 5  # Cap parallel threads for SEC politeness

SUBMISSIONS_CACHE_TTL = 86400  # 24 hours — how long cached submissions JSON stays fresh

# ==========================
# UTILITIES
# ==========================

# Lock serializes rate_limit() calls across threads so parallel workers
# never burst requests beyond SEC's per-second limit.
_rate_limit_lock = threading.Lock()

def rate_limit():
    """SEC-compliant rate limiter. Lock ensures parallel threads never burst."""
    with _rate_limit_lock:
        time.sleep(RATE_LIMIT_SECONDS)


def normalize_cik(cik: str) -> str:
    return cik.zfill(10)


def quarter_from_month(month: int) -> str:
    """1-12 → 'Q1'-'Q4'."""
    if not 1 <= month <= 12:
        raise ValueError(f"month must be 1-12, got {month}")
    return f"Q{(month - 1) // 3 + 1}"


def form_type_for_quarter(quarter: str) -> str:
    """'Q4' → '10-K', else '10-Q'."""
    return "10-K" if quarter == "Q4" else "10-Q"


def is_amendment(form: str) -> bool:
    """True if form ends with '/A'."""
    return form.endswith("/A")


def matches_form(form: str, allowed_forms: List[str]) -> bool:
    """Check if a filing form matches any allowed form pattern.

    Supports exact match and prefix match (trailing '*').
    Case-insensitive, whitespace-trimmed.
    """
    form_normalized = form.strip().upper()
    for pattern in allowed_forms:
        p = pattern.strip().upper()
        if p.endswith("*"):
            if form_normalized.startswith(p[:-1]):
                return True
        else:
            if form_normalized == p:
                return True
    return False


def resolve_form_types(forms_arg=None, bundle=None):
    """Resolve form types from --forms or --bundle.

    Returns None to signal default 10-K/10-Q behavior.
    --forms takes precedence over --bundle.
    """
    if forms_arg:
        return forms_arg
    if bundle == "core":
        return list(CORE_FORMS)
    return None


def compute_last_n_quarters(n: int, today: date | None = None) -> list[tuple[int, str]]:
    """Last N completed quarters as [(year, 'Q1'), ...]. Current quarter excluded."""
    if today is None:
        today = date.today()
    current_q = (today.month - 1) // 3 + 1
    year = today.year
    q = current_q - 1
    if q == 0:
        q = 4
        year -= 1
    result = []
    for _ in range(n):
        result.append((year, f"Q{q}"))
        q -= 1
        if q == 0:
            q = 4
            year -= 1
    return result


def build_output_path(base_dir, ticker, year, quarter, form, filing_date) -> Path:
    """base/TICKER/YEAR/QUARTER/FORM_DATE (no extension — caller adds .txt).
    Sanitize '/' in form to '-' for filesystem safety."""
    safe_form = form.replace("/", "-")
    return Path(base_dir) / ticker / str(year) / quarter / f"{safe_form}_{filing_date}"


def load_index(index_path: Path) -> dict:
    """Load _index.json from disk. Returns empty structure on missing/corrupt file."""
    if not index_path.exists():
        return {"cik": "", "last_updated": "", "filings": {}}
    try:
        data = json.loads(index_path.read_text())
        if not isinstance(data.get("filings"), dict):
            raise ValueError("filings must be a dict")
        return data
    except (json.JSONDecodeError, ValueError):
        print(f"  Warning: corrupted index at {index_path}, starting fresh")
        return {"cik": "", "last_updated": "", "filings": {}}


def save_index(index_path: Path, index_data: dict) -> None:
    """Atomically write _index.json (temp file → rename)."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = index_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(index_data, f, indent=2)
    tmp.rename(index_path)


def get_company_filings_cached(
    cik: str, cache_path: Path,
    force_refresh: bool = False, no_cache: bool = False,
) -> Dict:
    """Get submissions JSON, using disk cache unless bypassed.

    Default:          use cache if < 24 hours old.
    --force-refresh:  always fetch fresh from SEC.
    --no-cache:       always fetch fresh, don't read or write cache.
    """
    # Try reading cache in default mode only
    if not no_cache and not force_refresh and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            fetched_at = datetime.fromisoformat(cached["fetched_at"])
            age = (datetime.now(timezone.utc) - fetched_at).total_seconds()
            if age < SUBMISSIONS_CACHE_TTL:
                return cached["data"]
        except (json.JSONDecodeError, KeyError, ValueError):
            pass  # corrupted cache — fall through to fresh fetch

    data = get_company_filings(cik)

    # Write cache unless --no-cache
    if not no_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        wrapper = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        tmp = cache_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(wrapper, f)
        tmp.rename(cache_path)

    return data


def schedule_downloads(
    filtered: List[Dict], index: dict,
    force_refresh: bool = False, no_cache: bool = False,
) -> List[Dict]:
    """Determine which filings to process based on cache mode.

    Default:          skip filings already in index with text_saved=True.
    --force-refresh:  schedule all filtered filings (re-extract everything).
    --no-cache:       schedule all filtered filings (index ignored entirely).
    """
    if no_cache or force_refresh:
        return list(filtered)
    indexed = index.get("filings", {})
    return [
        f for f in filtered
        if f["accession"] not in indexed
        or not indexed[f["accession"]].get("text_saved", False)
    ]


def get_ticker_cik_map() -> Dict[str, str]:
    url = "https://www.sec.gov/files/company_tickers.json"
    rate_limit()
    resp = requests.get(url, headers=SEC_HEADERS)
    resp.raise_for_status()
    data = resp.json()

    mapping = {}
    for entry in data.values():
        mapping[entry["ticker"].upper()] = str(entry["cik_str"]).zfill(10)

    return mapping


def get_company_filings(cik: str) -> Dict:
    url = BASE_SUBMISSIONS_URL.format(cik=cik)
    rate_limit()
    resp = requests.get(url, headers=SEC_HEADERS)
    resp.raise_for_status()
    return resp.json()


def build_filing_url(cik: str, accession: str, primary_doc: str) -> str:
    accession_no_dash = accession.replace("-", "")
    cik_no_leading = str(int(cik))
    return f"{BASE_ARCHIVES_URL}/{cik_no_leading}/{accession_no_dash}/{primary_doc}"


def download_html(url: str) -> str | None:
    """Fetch HTML filing from SEC. Returns HTML string or None on failure.

    Rate limiting enforced per request for SEC compliance.
    """
    try:
        rate_limit()
        resp = requests.get(url, headers=SEC_HEADERS)
        if resp.status_code == 200:
            return resp.text
        return None
    except Exception:
        return None


# ==========================
# TEXT EXTRACTION
# ==========================

# This script extracts narrative text only.  Financial tables, XBRL data,
# and numeric values are intentionally excluded.  All numeric fundamentals
# should come from structured APIs (e.g., SEC company_facts endpoint).


class _TextExtractor(HTMLParser):
    """Extract visible text from HTML.  Strips <script>, <style>, etc."""

    _SKIP_TAGS = frozenset({"script", "style", "noscript"})
    _BLOCK_TAGS = frozenset({
        "p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "tr", "td", "th", "section", "article", "blockquote",
    })

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        lines = [" ".join(line.split()) for line in raw.splitlines()]
        text = "\n".join(lines)
        return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_html_to_text(raw_html: str) -> str:
    """Convert HTML filing to clean plain text.

    Strips script/style tags, converts block elements to line breaks,
    normalizes whitespace.  Uses stdlib html.parser — no external deps.
    """
    extractor = _TextExtractor()
    extractor.feed(raw_html)
    return html_unescape(extractor.get_text())


# Section heading patterns — case-insensitive.
# 10-K uses Item 7 for MD&A; 10-Q uses Item 2.  Both use Item 1A for risks.
_SECTION_HEADINGS = {
    "MDA": [
        r"item\s+7[.\s\-\u2014:]*management.{0,5}s\s+discussion",
        r"item\s+2[.\s\-\u2014:]*management.{0,5}s\s+discussion",
    ],
    "RISK_FACTORS": [
        r"item\s+1a[.\s\-\u2014:]*risk\s+factors",
    ],
}

_NEXT_ITEM_RE = re.compile(r"\n\s*item\s+\d+[a-z]?[.\s\-\u2014:]", re.IGNORECASE)


def _extract_one_section(text: str, patterns: list[str]) -> str | None:
    """Find section by heading pattern, extract until next Item heading."""
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start = match.end()
            next_item = _NEXT_ITEM_RE.search(text, start)
            end = next_item.start() if next_item else len(text)
            section_text = text[start:end].strip()
            if section_text:
                return section_text
    return None


def extract_sections(form: str, text: str) -> dict:
    """Extract narrative sections from cleaned filing text.

    10-K/10-Q: MD&A and Risk Factors.
    8-K:       entire body (no item-level parsing).

    NOTE: No table parsing. No XBRL. No numeric extraction.
    """
    base_form = form.upper().rstrip("/A").rstrip("/")

    if base_form == "8-K":
        return {"BODY": text}

    sections = {}
    for section_name, patterns in _SECTION_HEADINGS.items():
        sections[section_name] = _extract_one_section(text, patterns)
    return sections


def format_text_output(form: str, filing_date: str, accession: str,
                       sections: dict) -> str:
    """Build structured plain-text output for a single filing."""
    lines = [
        f"FORM: {form}",
        f"FILING_DATE: {filing_date}",
        f"ACCESSION: {accession}",
        "",
    ]
    for name, content in sections.items():
        lines.append(f"==== SECTION: {name} ====")
        lines.append(content if content else "[SECTION NOT FOUND]")
        lines.append("")
    return "\n".join(lines)


# ==========================
# CORE LOGIC
# ==========================

def filter_filings(
    filings: Dict,
    targets: list[tuple[int, str | None, str]],
    include_amendments: bool = False,
) -> List[Dict]:

    results = []

    recent = filings.get("filings", {}).get("recent", {})
    for i in range(len(recent.get("form", []))):

        form = recent["form"][i]
        filing_date = recent["filingDate"][i]
        accession = recent["accessionNumber"][i]
        primary_doc = recent["primaryDocument"][i]

        if is_amendment(form) and not include_amendments:
            continue

        base_form = form.rstrip("/A").rstrip("/") if is_amendment(form) else form

        parsed = datetime.strptime(filing_date, "%Y-%m-%d")
        filing_year = parsed.year
        filing_quarter = quarter_from_month(parsed.month)

        for target_year, target_quarter, target_form in targets:
            if not matches_form(base_form, [target_form]):
                continue
            if filing_year != target_year:
                continue
            if target_quarter is not None and filing_quarter != target_quarter:
                continue

            matched_quarter = target_quarter if target_quarter is not None else filing_quarter

            results.append({
                "form": form,
                "filing_date": filing_date,
                "accession": accession,
                "primary_doc": primary_doc,
                "matched_year": target_year,
                "matched_quarter": matched_quarter,
            })
            break

    return results


def build_targets_from_args(years=None, quarters=None, last_n=None, form_types=None):
    """Convert CLI args into 3-tuple targets: [(year, quarter_or_None, form_type), ...].

    When form_types is provided (via --forms or --bundle), all specified forms
    are matched for every year/quarter combination. When None, the original
    10-K/10-Q auto-mapping is preserved for backward compatibility.
    """
    if last_n:
        quarter_pairs = compute_last_n_quarters(last_n)
        if form_types:
            return [(y, q, f) for y, q in quarter_pairs for f in form_types]
        return [(y, q, form_type_for_quarter(q)) for y, q in quarter_pairs]
    if quarters == ["ANNUAL"]:
        forms = form_types or ["10-K"]
        return [(y, None, f) for y in years for f in forms]
    forms = form_types or ["10-Q"]
    return [(y, q, f) for y in years for q in quarters for f in forms]


def download_filing(
    ticker: str, cik: str, filing: dict, output_dir: Path,
) -> tuple[str, bool]:
    """Download HTML filing, extract text, and save structured .txt. Thread-safe.

    Pipeline: download_html → clean_html_to_text → extract_sections →
              format_text_output → atomic write .txt

    Skip/overwrite decisions are made by the main thread via schedule_downloads().
    This function unconditionally attempts the download.

    Returns (accession, success). Index mutation happens in the main thread.
    Rate limiting is enforced per-request via rate_limit() inside download_html().
    """
    out_base = build_output_path(
        output_dir, ticker,
        filing["matched_year"], filing["matched_quarter"],
        filing["form"], filing["filing_date"],
    )
    out_base.parent.mkdir(parents=True, exist_ok=True)

    filing_url = build_filing_url(cik, filing["accession"], filing["primary_doc"])

    raw_html = download_html(filing_url)
    if raw_html is None:
        print(f"  FAILED {filing['form']} {filing['filing_date']}")
        return (filing["accession"], False)

    plain_text = clean_html_to_text(raw_html)
    sections = extract_sections(filing["form"], plain_text)
    output_text = format_text_output(
        filing["form"], filing["filing_date"], filing["accession"], sections,
    )

    output_path = out_base.parent / (out_base.name + ".txt")
    tmp_path = output_path.with_suffix(".txt.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(output_text)
    tmp_path.rename(output_path)

    print(f"  Extracted {filing['form']} {filing['filing_date']}")
    return (filing["accession"], True)


def process_ticker(
    ticker: str,
    cik: str,
    targets: list[tuple[int, str | None, str]],
    output_dir: Path,
    include_amendments: bool = False,
    parallel: bool = False,
    workers: int = 1,
    force_refresh: bool = False,
    no_cache: bool = False,
):
    """Download and extract text from filings for a single ticker.

    Cache modes (main thread controls all skip/download decisions):
      Default:          Skip filings already in _index.json with text_saved=True.
      --force-refresh:  Re-extract all matching filings. Update index.
      --no-cache:       Ignore _index.json entirely. Stateless run.

    Workers only download files and return (accession, success). Index
    mutation happens exclusively in the main thread after workers complete.

    When parallel=True and workers>1, filings are downloaded concurrently
    using a ThreadPoolExecutor. Rate limiting is enforced per-request
    via a global threading lock. Parallelism is per-ticker only.
    """
    print(f"\nProcessing {ticker} (CIK: {cik})")

    ticker_dir = output_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    # --- Load submissions JSON (respect cache policy) ---
    submissions_cache_path = ticker_dir / "_submissions.json"
    filings_data = get_company_filings_cached(
        cik, submissions_cache_path,
        force_refresh=force_refresh, no_cache=no_cache,
    )

    filtered = filter_filings(filings_data, targets, include_amendments)

    # --- Load index (skip entirely in --no-cache mode) ---
    index_path = ticker_dir / "_index.json"
    if no_cache:
        index = {"cik": cik, "last_updated": "", "filings": {}}
    else:
        index = load_index(index_path)
        if not index.get("cik"):
            index["cik"] = cik

    # --- Main thread determines download schedule ---
    filings_to_download = schedule_downloads(
        filtered, index, force_refresh=force_refresh, no_cache=no_cache,
    )

    skipped = len(filtered) - len(filings_to_download)
    if skipped > 0:
        print(f"  Skipping {skipped} already-downloaded filing(s)")
    if not filings_to_download:
        print(f"  Nothing to download for {ticker}")
        return

    # Pre-create all output directories before spawning threads
    for filing in filings_to_download:
        out_base = build_output_path(
            output_dir, ticker,
            filing["matched_year"], filing["matched_quarter"],
            filing["form"], filing["filing_date"],
        )
        out_base.parent.mkdir(parents=True, exist_ok=True)

    # --- Download filings (workers return results, no index mutation) ---
    results = []
    if parallel and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_filing = {
                executor.submit(download_filing, ticker, cik, f, output_dir): f
                for f in filings_to_download
            }
            for future in future_to_filing:
                filing = future_to_filing[future]
                accession, success = future.result()
                results.append((filing, accession, success))
    else:
        for filing in filings_to_download:
            accession, success = download_filing(ticker, cik, filing, output_dir)
            results.append((filing, accession, success))

    # --- Update index in main thread (skip in --no-cache mode) ---
    if not no_cache:
        for filing, accession, success in results:
            if success:
                index["filings"][accession] = {
                    "form": filing["form"],
                    "filing_date": filing["filing_date"],
                    "text_saved": True,
                    "last_extracted": datetime.now(timezone.utc).isoformat(),
                }
        index["last_updated"] = datetime.now(timezone.utc).isoformat()
        save_index(index_path, index)


# ==========================
# MAIN
# ==========================

def main():

    parser = argparse.ArgumentParser(description="SEC Filing Text Extractor")

    parser.add_argument("--tickers", required=True,
                        help="Comma-separated list of tickers")

    parser.add_argument("--years",
                        help="Comma-separated list of years (e.g. 2023,2024)")

    parser.add_argument("--quarters",
                        help="Comma-separated Q1,Q2,Q3,Q4 or ANNUAL")

    parser.add_argument("--last-n", type=int,
                        help="Download last N completed quarters (rolling window)")

    parser.add_argument("--forms",
                        help="Comma-separated form types (e.g. 10-Q,8-K,4)")

    parser.add_argument("--bundle", choices=["core"],
                        help="Use a predefined form bundle (core: institutional set)")

    parser.add_argument("--include-amendments", action="store_true", default=False,
                        help="Include amendment filings (e.g. 10-K/A)")

    parser.add_argument("--parallel", action="store_true", default=False,
                        help="Enable parallel downloading of filings per ticker")

    parser.add_argument("--workers", type=int, default=1,
                        help="Number of download threads (default: 1, max: 5)")

    parser.add_argument("--force-refresh", action="store_true", default=False,
                        help="Re-download all matching filings, overwriting existing")

    parser.add_argument("--no-cache", action="store_true", default=False,
                        help="Ignore _index.json entirely — fully stateless run")

    parser.add_argument("--output", default="./sec_filings",
                        help="Output directory")

    args = parser.parse_args()

    if args.force_refresh and args.no_cache:
        parser.error("--force-refresh and --no-cache are mutually exclusive")

    if args.last_n and (args.years or args.quarters):
        parser.error("--last-n is mutually exclusive with --years/--quarters")

    if not args.last_n:
        if not args.years or not args.quarters:
            parser.error("--years and --quarters are required when not using --last-n")

    # Cap workers at MAX_WORKERS for SEC politeness
    workers = args.workers
    if workers > MAX_WORKERS:
        print(f"Warning: capping --workers from {workers} to {MAX_WORKERS} for SEC compliance")
        workers = MAX_WORKERS

    tickers = [t.strip().upper() for t in args.tickers.split(",")]

    years = [int(y.strip()) for y in args.years.split(",")] if args.years else None
    quarters = [q.strip().upper() for q in args.quarters.split(",")] if args.quarters else None

    # --forms takes precedence over --bundle; None preserves default 10-K/10-Q behavior
    forms_arg = [f.strip() for f in args.forms.split(",")] if args.forms else None
    form_types = resolve_form_types(forms_arg=forms_arg, bundle=args.bundle)

    targets = build_targets_from_args(
        years=years, quarters=quarters, last_n=args.last_n, form_types=form_types,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    ticker_map = get_ticker_cik_map()

    for ticker in tickers:
        if ticker not in ticker_map:
            print(f"Ticker not found: {ticker}")
            continue

        cik = ticker_map[ticker]
        process_ticker(
            ticker, cik, targets, output_dir,
            include_amendments=args.include_amendments,
            parallel=args.parallel,
            workers=workers,
            force_refresh=args.force_refresh,
            no_cache=args.no_cache,
        )


if __name__ == "__main__":
    main()