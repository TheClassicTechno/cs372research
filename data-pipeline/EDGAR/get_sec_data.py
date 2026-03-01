#!/usr/bin/env python3
"""
SEC Filing Downloader — Pure ingestion layer for EDGAR filings.

Downloads raw HTML filings and converts them to clean plain text.
No section parsing, no metadata formatting — that belongs in the
summarization pipeline.

Output structure:
  {base_dir}/raw_html/{TICKER}/{TICKER}_{YEAR}_{QUARTER}_{FORM}_{FILINGDATE}.html
  {base_dir}/clean_text/{TICKER}/{TICKER}_{YEAR}_{QUARTER}_{FORM}_{FILINGDATE}.txt

Examples:
  # 1. Last 4 completed quarters for a few tickers
  python get_sec_data.py --tickers AAPL,NVDA,MSFT --last-n 4

  # 2. Specific year + quarters
  python get_sec_data.py --tickers AAPL --years 2024 --quarters Q1,Q2,Q3

  # 3. Annual filings only (matches any quarter in the year)
  python get_sec_data.py --tickers AAPL,GOOG --years 2023,2024 --quarters ANNUAL

  # 4. Multiple years, all quarters
  python get_sec_data.py --tickers JPM,GS --years 2023,2024,2025 --quarters Q1,Q2,Q3,Q4

  # 5. Custom form types (override defaults)
  python get_sec_data.py --tickers AAPL --last-n 4 --forms 10-K,10-Q

  # 6. Include amendment filings (10-K/A, 10-Q/A, etc.)
  python get_sec_data.py --tickers NVDA --years 2024 --quarters Q1,Q2,Q3,Q4 --include-amendments

  # 7. Parallel download with 3 workers
  python get_sec_data.py --tickers AAPL,NVDA,MSFT --last-n 4 --parallel --workers 3

  # 8. Force re-extract everything (overwrite existing)
  python get_sec_data.py --tickers AAPL --last-n 2 --force-refresh

  # 9. Stateless run — ignore index, don't write cache
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
from pathlib import Path
from typing import List, Dict, Set

import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

import yaml

# ==========================
# CONFIG
# ==========================

SEC_HEADERS = {
    "User-Agent": "Veljko Skarich vskarich@stanford.edu",
    "Accept-Encoding": "gzip, deflate",
}

BASE_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
BASE_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"

RATE_LIMIT_SECONDS = 0.2  # 5 requests/second max (SEC compliant)

DEFAULT_FORMS: Set[str] = {"10-K", "10-Q", "8-K", "424B2"}

MAX_WORKERS = 5  # Cap parallel threads for SEC politeness

SUBMISSIONS_CACHE_TTL = 8640000  # 2400 hours — how long cached submissions JSON stays fresh

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
    """1-12 -> 'Q1'-'Q4'."""
    if not 1 <= month <= 12:
        raise ValueError(f"month must be 1-12, got {month}")
    return f"Q{(month - 1) // 3 + 1}"


def is_amendment(form: str) -> bool:
    """True if form ends with '/A'."""
    return form.endswith("/A")


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


def _filing_stem(ticker: str, year: int, quarter: str, form: str, filing_date: str) -> str:
    """Build human-readable filename stem: TICKER_YEAR_QUARTER_FORM_FILINGDATE."""
    safe_form = form.replace("/", "-")
    return f"{ticker}_{year}_{quarter}_{safe_form}_{filing_date}"


def build_html_path(base_dir: Path, ticker: str, year: int, quarter: str,
                    form: str, filing_date: str) -> Path:
    """base_dir/raw_html/TICKER/TICKER_YEAR_QUARTER_FORM_FILINGDATE.html"""
    stem = _filing_stem(ticker, year, quarter, form, filing_date)
    return base_dir / "raw_html" / ticker / f"{stem}.html"


def build_text_path(base_dir: Path, ticker: str, year: int, quarter: str,
                    form: str, filing_date: str) -> Path:
    """base_dir/clean_text/TICKER/TICKER_YEAR_QUARTER_FORM_FILINGDATE.txt"""
    stem = _filing_stem(ticker, year, quarter, form, filing_date)
    return base_dir / "clean_text" / ticker / f"{stem}.txt"


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
    """Atomically write _index.json (temp file -> rename)."""
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


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace within lines and excessive blank lines."""
    lines = [" ".join(line.split()) for line in text.splitlines()]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_html_to_text(raw_html: str) -> str:
    """Convert HTML filing to clean plain text using BeautifulSoup.

    Strips script/style/noscript tags and inline XBRL (iXBRL) metadata,
    extracts visible text with newline separators, normalizes whitespace.
    Uses lxml parser for robust handling of broken HTML common in SEC filings.
    """
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "lxml")

    # Strip script/style/noscript
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Strip inline XBRL (iXBRL) hidden metadata blocks.
    # SEC filings use ix:header and ix:hidden to embed XBRL taxonomy data
    # that produces noise like "xbrli:shares", "iso4217:USD", etc.
    for tag in soup.find_all(re.compile(r"^ix:", re.IGNORECASE)):
        if tag.name and tag.name.lower() in ("ix:header", "ix:hidden", "ix:references"):
            tag.decompose()

    text = soup.get_text(separator="\n")
    return _normalize_whitespace(text)


# ==========================
# CORE LOGIC
# ==========================

def filter_filings(
    filings: Dict,
    targets: list[tuple[int, str | None]],
    allowed_forms: Set[str],
    include_amendments: bool = False,
) -> List[Dict]:
    """Filter SEC filings by time window and form type.

    Time filtering (targets) and form filtering (allowed_forms) are independent.

    Args:
        filings: Raw submissions JSON from SEC.
        targets: List of (year, quarter_or_None) time windows.
        allowed_forms: Set of form types to include (e.g. {"10-K", "10-Q"}).
        include_amendments: Whether to include /A amendment filings.
    """
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

        # Form filter — independent of time
        if base_form not in allowed_forms:
            continue

        parsed = datetime.strptime(filing_date, "%Y-%m-%d")
        filing_year = parsed.year
        filing_quarter = quarter_from_month(parsed.month)

        # Time filter — independent of form
        for target_year, target_quarter in targets:
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


def build_targets_from_args(years=None, quarters=None, last_n=None):
    """Convert CLI args into 2-tuple time targets: [(year, quarter_or_None), ...].

    Form filtering is handled independently by the allowed_forms set.
    """
    if last_n:
        return compute_last_n_quarters(last_n)
    if quarters == ["ANNUAL"]:
        return [(y, None) for y in years]
    return [(y, q) for y in years for q in quarters]


def download_filing(
    ticker: str, cik: str, filing: dict, base_dir: Path,
) -> tuple[str, bool]:
    """Download HTML filing, save raw HTML + clean text. Thread-safe.

    Pipeline: download_html -> save raw HTML -> clean_html_to_text -> save clean text

    Returns (accession, success). Index mutation happens in the main thread.
    """
    accession = filing["accession"]
    year = filing["matched_year"]
    quarter = filing["matched_quarter"]
    form = filing["form"]
    filing_date = filing["filing_date"]

    html_path = build_html_path(base_dir, ticker, year, quarter, form, filing_date)
    text_path = build_text_path(base_dir, ticker, year, quarter, form, filing_date)

    html_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)

    filing_url = build_filing_url(cik, accession, filing["primary_doc"])

    raw_html = download_html(filing_url)
    if raw_html is None:
        print(f"  FAILED {filing['form']} {filing['filing_date']}")
        return (accession, False)

    # Save raw HTML (atomic write)
    tmp_html = html_path.with_suffix(".html.tmp")
    with open(tmp_html, "w", encoding="utf-8") as f:
        f.write(raw_html)
    tmp_html.rename(html_path)

    # Convert to clean text (full document, no slicing)
    plain_text = clean_html_to_text(raw_html)

    # Save clean text with minimal metadata header
    output_text = (
        f"FORM: {filing['form']}\n"
        f"FILING_DATE: {filing['filing_date']}\n"
        f"ACCESSION: {accession}\n"
        f"\n"
        f"{plain_text}"
    )

    tmp_text = text_path.with_suffix(".txt.tmp")
    with open(tmp_text, "w", encoding="utf-8") as f:
        f.write(output_text)
    tmp_text.rename(text_path)

    print(f"  Saved {filing['form']} {filing['filing_date']} ({accession})")
    return (accession, True)


def process_ticker(
    ticker: str,
    cik: str,
    targets: list[tuple[int, str | None]],
    allowed_forms: Set[str],
    base_dir: Path,
    include_amendments: bool = False,
    parallel: bool = False,
    workers: int = 1,
    force_refresh: bool = False,
    no_cache: bool = False,
):
    """Download and save filings for a single ticker.

    Cache modes (main thread controls all skip/download decisions):
      Default:          Skip filings already in _index.json with text_saved=True.
      --force-refresh:  Re-extract all matching filings. Update index.
      --no-cache:       Ignore _index.json entirely. Stateless run.
    """
    print(f"\nProcessing {ticker} (CIK: {cik})")

    cache_dir = base_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Pre-create output directories
    (base_dir / "raw_html" / ticker).mkdir(parents=True, exist_ok=True)
    (base_dir / "clean_text" / ticker).mkdir(parents=True, exist_ok=True)

    # --- Load submissions JSON (respect cache policy) ---
    submissions_cache_path = cache_dir / f"{ticker}_submissions.json"
    filings_data = get_company_filings_cached(
        cik, submissions_cache_path,
        force_refresh=force_refresh, no_cache=no_cache,
    )

    filtered = filter_filings(filings_data, targets, allowed_forms, include_amendments)

    # --- Load index (skip entirely in --no-cache mode) ---
    index_path = cache_dir / f"{ticker}_index.json"
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

    # --- Download filings (workers return results, no index mutation) ---
    results = []
    if parallel and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_filing = {
                executor.submit(download_filing, ticker, cik, f, base_dir): f
                for f in filings_to_download
            }
            for future in future_to_filing:
                filing = future_to_filing[future]
                accession, success = future.result()
                results.append((filing, accession, success))
    else:
        for filing in filings_to_download:
            accession, success = download_filing(ticker, cik, filing, base_dir)
            results.append((filing, accession, success))

    # --- Update index in main thread (skip in --no-cache mode) ---
    if not no_cache:
        for filing, accession, success in results:
            if success:
                index["filings"][accession] = {
                    "form": filing["form"],
                    "filing_date": filing["filing_date"],
                    "text_saved": True,
                    "html_saved": True,
                    "last_extracted": datetime.now(timezone.utc).isoformat(),
                }
        index["last_updated"] = datetime.now(timezone.utc).isoformat()
        save_index(index_path, index)


_SUPPORTED_TICKERS_PATH = Path(__file__).resolve().parent.parent / "supported_tickers.yaml"


def load_supported_tickers(yaml_path: Path = _SUPPORTED_TICKERS_PATH) -> list:
    """Load ticker symbols from supported_tickers.yaml."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return [entry["symbol"] for entry in data["supported_tickers"]]


# ==========================
# MAIN
# ==========================

def main():

    parser = argparse.ArgumentParser(description="SEC Filing Downloader — pure ingestion layer")

    parser.add_argument("--tickers", default=None,
                        help="Comma-separated list of tickers")

    parser.add_argument("--supported", action="store_true", default=False,
                        help="Use all tickers from supported_tickers.yaml")

    parser.add_argument("--years",
                        help="Comma-separated list of years (e.g. 2023,2024)")

    parser.add_argument("--quarters",
                        help="Comma-separated Q1,Q2,Q3,Q4 or ANNUAL")

    parser.add_argument("--last-n", type=int,
                        help="Download last N completed quarters (rolling window)")

    parser.add_argument("--forms",
                        help="Comma-separated form types to download "
                             f"(default: {','.join(sorted(DEFAULT_FORMS))})")

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

    parser.add_argument("--output",
                        default=str(Path(__file__).resolve().parent),
                        help="Output base directory (default: EDGAR/ dir next to this script)")

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

    if args.supported:
        tickers = load_supported_tickers()
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        parser.error("either --tickers or --supported is required")

    years = [int(y.strip()) for y in args.years.split(",")] if args.years else None
    quarters = [q.strip().upper() for q in args.quarters.split(",")] if args.quarters else None

    # --forms overrides default set; otherwise use DEFAULT_FORMS
    if args.forms:
        allowed_forms = {f.strip() for f in args.forms.split(",")}
    else:
        allowed_forms = set(DEFAULT_FORMS)

    targets = build_targets_from_args(
        years=years, quarters=quarters, last_n=args.last_n,
    )

    base_dir = Path(args.output)
    base_dir.mkdir(parents=True, exist_ok=True)

    ticker_map = get_ticker_cik_map()

    for ticker in tickers:
        if ticker not in ticker_map:
            print(f"Ticker not found: {ticker}")
            continue

        cik = ticker_map[ticker]
        process_ticker(
            ticker, cik, targets, allowed_forms, base_dir,
            include_amendments=args.include_amendments,
            parallel=args.parallel,
            workers=workers,
            force_refresh=args.force_refresh,
            no_cache=args.no_cache,
        )


if __name__ == "__main__":
    main()
