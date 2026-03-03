#!/usr/bin/env python3
"""
SEC Filing Downloader — Pure ingestion layer for EDGAR filings.

ENHANCED VERSION:
- Aligns to fiscal quarter using reportDate when available
- Checks one-quarter-ahead filing window
- Robust to non-standard fiscal years (e.g., Apple)
- Maintains economic quarter labeling

Output structure:
  {base_dir}/raw_html/{TICKER}/{TICKER}_{YEAR}_{QUARTER}_{FORM}_{FILINGDATE}.html
  {base_dir}/clean_text/{TICKER}/{TICKER}_{YEAR}_{QUARTER}_{FORM}_{FILINGDATE}.txt
"""

import argparse
import json
import re
import requests
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Dict, Set

import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

import yaml
from tqdm import tqdm

# ==========================
# CONFIG
# ==========================

SEC_HEADERS = {
    "User-Agent": "Veljko Skarich vskarich@stanford.edu",
    "Accept-Encoding": "gzip, deflate",
}

BASE_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
BASE_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"

RATE_LIMIT_SECONDS = 0.2
DEFAULT_FORMS: Set[str] = {"10-K", "10-Q", "8-K"}
MAX_WORKERS = 5
SUBMISSIONS_CACHE_TTL = 8640000
MAX_RETRY_ROUNDS = 5
RETRY_BACKOFF_BASE = 2.0  # seconds; doubles each round

_rate_limit_lock = threading.Lock()

# ==========================
# UTILITIES
# ==========================

def rate_limit():
    with _rate_limit_lock:
        time.sleep(RATE_LIMIT_SECONDS)


def quarter_from_month(month: int) -> str:
    return f"Q{(month - 1) // 3 + 1}"


def next_quarter(year: int, quarter: str) -> tuple[int, str]:
    q_num = int(quarter[1])
    if q_num < 4:
        return (year, f"Q{q_num + 1}")
    else:
        return (year + 1, "Q1")


def is_amendment(form: str) -> bool:
    return form.endswith("/A")


def compute_last_n_quarters(n: int, today: date | None = None):
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


# ==========================
# SEC API
# ==========================

def get_ticker_cik_map():
    print("Fetching ticker → CIK map from SEC...")
    url = "https://www.sec.gov/files/company_tickers.json"
    rate_limit()
    resp = requests.get(url, headers=SEC_HEADERS)
    resp.raise_for_status()
    data = resp.json()
    mapping = {
        entry["ticker"].upper(): str(entry["cik_str"]).zfill(10)
        for entry in data.values()
    }
    print(f"  Loaded {len(mapping)} tickers")
    return mapping


def get_company_filings(cik: str):
    url = BASE_SUBMISSIONS_URL.format(cik=cik)
    rate_limit()
    resp = requests.get(url, headers=SEC_HEADERS)
    resp.raise_for_status()
    return resp.json()


def build_filing_url(cik: str, accession: str, primary_doc: str):
    accession_no_dash = accession.replace("-", "")
    cik_no_leading = str(int(cik))
    return f"{BASE_ARCHIVES_URL}/{cik_no_leading}/{accession_no_dash}/{primary_doc}"


def download_html(url: str) -> tuple:
    """Download HTML from URL. Returns (html_text, error_reason) tuple."""
    try:
        rate_limit()
        resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
        if resp.status_code == 200:
            return (resp.text, None)
        return (None, f"HTTP {resp.status_code}")
    except requests.Timeout:
        return (None, "timeout (30s)")
    except requests.ConnectionError as e:
        return (None, f"connection error: {e}")
    except Exception as e:
        return (None, str(e))


# ==========================
# TEXT EXTRACTION
# ==========================

def clean_html_to_text(raw_html: str):
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [" ".join(line.split()) for line in text.splitlines()]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ==========================
# CORE LOGIC (ENHANCED)
# ==========================

def filter_filings(
    filings: Dict,
    targets: list[tuple[int, str | None]],
    allowed_forms: Set[str],
    include_amendments: bool = False,
) -> List[Dict]:

    results = []
    seen_accessions = set()
    recent = filings.get("filings", {}).get("recent", {})

    for i in range(len(recent.get("form", []))):

        form = recent["form"][i]
        filing_date = recent["filingDate"][i]
        accession = recent["accessionNumber"][i]
        primary_doc = recent["primaryDocument"][i]

        report_date = recent.get("reportDate", [None] * len(recent["form"]))[i]

        if is_amendment(form) and not include_amendments:
            continue

        base_form = form.rstrip("/A") if is_amendment(form) else form
        if base_form not in allowed_forms:
            continue

        # 🔥 KEY FIX: Use reportDate (fiscal period end) if available
        if report_date:
            parsed = datetime.strptime(report_date, "%Y-%m-%d")
        else:
            parsed = datetime.strptime(filing_date, "%Y-%m-%d")

        fiscal_year = parsed.year
        fiscal_quarter = quarter_from_month(parsed.month)

        # Two-pass matching: direct match always wins over spillover.
        # This prevents 10-K filings from companies with non-calendar
        # fiscal year-ends (e.g., WMT Jan 31) from being matched to the
        # wrong quarter via spillover when a direct match also exists.
        matched = None

        # Pass 1: direct match
        for target_year, target_quarter in targets:
            if target_quarter is None:
                if fiscal_year == target_year:
                    matched = (target_year, fiscal_quarter)
                    break
            elif fiscal_year == target_year and fiscal_quarter == target_quarter:
                matched = (target_year, target_quarter)
                break

        # Pass 2: spillover (10-K only) — only if no direct match found
        if matched is None:
            for target_year, target_quarter in targets:
                if target_quarter is None:
                    continue
                next_year_, next_q = next_quarter(target_year, target_quarter)
                if (base_form == "10-K"
                        and fiscal_year == next_year_
                        and fiscal_quarter == next_q):
                    matched = (target_year, target_quarter)
                    break

        if matched is not None and accession not in seen_accessions:
            seen_accessions.add(accession)
            results.append({
                "form": form,
                "filing_date": filing_date,
                "accession": accession,
                "primary_doc": primary_doc,
                "matched_year": matched[0],
                "matched_quarter": matched[1],
            })

    return results


# ==========================
# FILE WRITE
# ==========================

_failed_accessions: Set[str] = set()


def download_filing(ticker, cik, filing, base_dir):
    accession = filing["accession"]
    if accession in _failed_accessions:
        return (accession, False)
    year = filing["matched_year"]
    quarter = filing["matched_quarter"]
    form = filing["form"]
    filing_date = filing["filing_date"]

    raw_dir = base_dir / "raw_html" / ticker
    txt_dir = base_dir / "clean_text" / ticker
    raw_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{ticker}_{year}_{quarter}_{form.replace('/', '-')}_{filing_date}"
    html_path = raw_dir / f"{stem}.html"
    text_path = txt_dir / f"{stem}.txt"

    if text_path.exists() and html_path.exists():
        tqdm.write(f"    [{ticker}] SKIP {form} {filing_date} (already exists)")
        return (accession, True)

    # Cross-run dedup: check if same filing exists under a different quarter
    form_sanitized = form.replace("/", "-")
    existing = list(txt_dir.glob(f"{ticker}_*_{form_sanitized}_{filing_date}.txt"))
    if existing:
        tqdm.write(f"    [{ticker}] SKIP {form} {filing_date} (exists as {existing[0].name})")
        return (accession, True)

    url = build_filing_url(cik, accession, filing["primary_doc"])
    raw_html, error = download_html(url)
    if raw_html is None:
        tqdm.write(f"    [{ticker}] FAIL {form} {filing_date} — {error}")
        _failed_accessions.add(accession)
        return (accession, False)

    html_path.write_text(raw_html, encoding="utf-8")
    clean_text = clean_html_to_text(raw_html)

    text_path.write_text(
        f"FORM: {form}\nFILING_DATE: {filing_date}\nACCESSION: {accession}\n\n{clean_text}",
        encoding="utf-8"
    )

    tqdm.write(f"    [{ticker}] Saved {form} {filing_date} ({accession})")
    return (accession, True)


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
    parser = argparse.ArgumentParser()

    parser.add_argument("--tickers")
    parser.add_argument("--supported", action="store_true", default=False,
                        help="Use all tickers from supported_tickers.yaml")
    parser.add_argument("--years")
    parser.add_argument("--quarters")
    parser.add_argument("--last-n", type=int)
    parser.add_argument("--forms")
    parser.add_argument("--include-amendments", action="store_true")
    parser.add_argument("--parallel", action="store_true",
                        help="Download filings in parallel across tickers")
    parser.add_argument("--workers", type=int, default=3,
                        help="Number of parallel download threads (default: 3, max: 5)")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent))

    args = parser.parse_args()

    if args.last_n:
        targets = compute_last_n_quarters(args.last_n)
    else:
        years = [int(y) for y in args.years.split(",")]
        quarters = [q.strip().upper() for q in args.quarters.split(",")]
        targets = [(y, q) for y in years for q in quarters]

    allowed_forms = set(args.forms.split(",")) if args.forms else DEFAULT_FORMS

    if args.supported:
        tickers = load_supported_tickers()
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        parser.error("either --tickers or --supported is required")
    base_dir = Path(args.output)
    base_dir.mkdir(parents=True, exist_ok=True)

    workers = min(args.workers, MAX_WORKERS) if args.parallel else 1

    ticker_map = get_ticker_cik_map()

    # Collect all download tasks: (ticker, cik, filing) across all tickers
    all_tasks = []
    for ticker in tqdm(tickers, desc="Scanning", unit="ticker"):
        if ticker not in ticker_map:
            tqdm.write(f"  Ticker not found: {ticker}")
            continue

        cik = ticker_map[ticker]
        filings = get_company_filings(cik)

        filtered = filter_filings(
            filings,
            targets,
            allowed_forms,
            include_amendments=args.include_amendments
        )

        for filing in filtered:
            all_tasks.append((ticker, cik, filing))

    if not all_tasks:
        tqdm.write("No filings to download.")
        return

    tqdm.write(f"\n{len(all_tasks)} filing(s) to process ({workers} worker{'s' if workers > 1 else ''})\n")

    def _run_downloads(tasks, desc, num_workers):
        """Run download tasks, return list of failed (ticker, cik, filing) tuples."""
        failed_tasks = []
        if num_workers == 1:
            for ticker, cik, filing in tqdm(tasks, desc=desc, unit="filing"):
                accession, ok = download_filing(ticker, cik, filing, base_dir)
                if not ok:
                    failed_tasks.append((ticker, cik, filing))
        else:
            task_map = {}
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                futures = []
                for ticker, cik, filing in tasks:
                    f = pool.submit(download_filing, ticker, cik, filing, base_dir)
                    task_map[f] = (ticker, cik, filing)
                    futures.append(f)
                for future in tqdm(
                    as_completed(futures), desc=desc, unit="filing", total=len(futures),
                ):
                    accession, ok = future.result()
                    if not ok:
                        failed_tasks.append(task_map[future])
        return failed_tasks

    # Initial pass
    failed = _run_downloads(all_tasks, "Downloading", workers)

    # Retry failed downloads with exponential backoff
    for retry_round in range(1, MAX_RETRY_ROUNDS + 1):
        if not failed:
            break

        backoff = RETRY_BACKOFF_BASE * (2 ** (retry_round - 1))
        tqdm.write(f"\n{len(failed)} filing(s) failed. "
                   f"Retry {retry_round}/{MAX_RETRY_ROUNDS} after {backoff:.0f}s backoff...")
        time.sleep(backoff)

        # Clear failed accessions so download_filing will attempt them again
        for _, _, filing in failed:
            _failed_accessions.discard(filing["accession"])

        failed = _run_downloads(failed, f"Retry {retry_round}", 1)

    if failed:
        tqdm.write(f"\n{len(failed)} filing(s) still failed after {MAX_RETRY_ROUNDS} retries:")
        for ticker, _, filing in failed:
            tqdm.write(f"  [{ticker}] {filing['form']} {filing['filing_date']} ({filing['accession']})")
    else:
        tqdm.write(f"\nAll downloads complete.")


if __name__ == "__main__":
    main()