#!/usr/bin/env python3
"""
Build a combined ticker snapshot: SEC filings + XBRL fundamentals.

Downloads the most recent 10-Q, 10-K, and 8-K filings for a ticker,
then fetches structured financial fundamentals from the SEC XBRL API.

ARCHITECTURAL RULE:
  Financial numbers come ONLY from the SEC XBRL companyfacts API.
  HTML filings are ONLY for narrative text (MD&A, risk factors, etc.).
  NO number parsing from HTML. Ever.

Usage:
  python build_snapshot.py --ticker AAPL
  python build_snapshot.py --ticker AAPL --output /tmp/snapshots
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone as _utc_mod
from pathlib import Path

_utc = _utc_mod.utc
from typing import Dict, List, Optional

import requests

# Import ingestion layer functions
sys.path.insert(0, str(Path(__file__).resolve().parent))
import get_sec_data as sec


# ==========================
# CONFIG
# ==========================

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# XBRL taxonomy -> human-readable field mapping
# These are the official us-gaap tags for each metric.
XBRL_FIELDS = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "net_income": [
        "NetIncomeLoss",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
    ],
    "eps_basic": [
        "EarningsPerShareBasic",
    ],
    "eps_diluted": [
        "EarningsPerShareDiluted",
    ],
    "total_assets": [
        "Assets",
    ],
    "total_liabilities": [
        "Liabilities",
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
    ],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
    ],
}


# ==========================
# FILING DISCOVERY
# ==========================

def find_most_recent_filings(
    submissions: Dict,
    form_types: List[str],
) -> List[Dict]:
    """Find the most recent filing of each requested form type.

    Scans the submissions JSON and returns the first (most recent)
    match for each form type in form_types.
    """
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    found = {}
    for i in range(len(forms)):
        form = forms[i]
        base_form = form.rstrip("/A").rstrip("/") if sec.is_amendment(form) else form

        if base_form in form_types and base_form not in found:
            filing_date = dates[i]
            parsed = datetime.strptime(filing_date, "%Y-%m-%d")
            quarter = sec.quarter_from_month(parsed.month)

            found[base_form] = {
                "form": form,
                "filing_date": filing_date,
                "accession": accessions[i],
                "primary_doc": primary_docs[i],
                "matched_year": parsed.year,
                "matched_quarter": quarter,
            }

        if len(found) == len(form_types):
            break

    return list(found.values())


# ==========================
# XBRL FUNDAMENTALS
# ==========================

def fetch_companyfacts(cik: str) -> Dict:
    """Fetch structured XBRL data from SEC companyfacts API."""
    url = COMPANYFACTS_URL.format(cik=cik)
    sec.rate_limit()
    resp = requests.get(url, headers=sec.SEC_HEADERS)
    resp.raise_for_status()
    return resp.json()


def _get_latest_value(facts: Dict, xbrl_tags: List[str]) -> Optional[float]:
    """Extract the most recent value for a set of XBRL tags.

    Tries each tag in order. For each tag, looks for 10-Q/10-K filings
    in USD units, returns the most recent value.
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    for tag in xbrl_tags:
        concept = us_gaap.get(tag)
        if not concept:
            continue

        units = concept.get("units", {})
        # EPS uses USD/shares, everything else uses USD
        values = units.get("USD") or units.get("USD/shares")
        if not values:
            continue

        # Filter to 10-Q and 10-K filings only, sort by end date descending
        quarterly = [
            v for v in values
            if v.get("form") in ("10-Q", "10-K")
            and v.get("end")
        ]
        if not quarterly:
            continue

        quarterly.sort(key=lambda v: v["end"], reverse=True)
        entry = quarterly[0]
        return {
            "value": entry["val"],
            "period_end": entry["end"],
            "form": entry["form"],
        }

    return None


def extract_fundamentals(facts: Dict) -> Dict:
    """Extract structured fundamentals from companyfacts JSON.

    Returns dict with each metric containing value + period metadata.
    All numbers come from the SEC XBRL API — never from HTML.
    """
    result = {"source": "SEC_XBRL_API"}
    latest_quarter = {}

    for field_name, xbrl_tags in XBRL_FIELDS.items():
        entry = _get_latest_value(facts, xbrl_tags)
        if entry:
            latest_quarter[field_name] = entry["value"]
            # Track the period for reference
            if "period_end" not in result:
                result["period_end"] = entry["period_end"]
        else:
            latest_quarter[field_name] = None

    result["latest_quarter"] = latest_quarter
    return result


# ==========================
# SNAPSHOT BUILDER
# ==========================

def build_snapshot(
    ticker: str,
    base_dir: Path,
) -> Dict:
    """Build a complete ticker snapshot: filings + XBRL fundamentals.

    1. Resolve CIK from ticker
    2. Fetch submissions metadata
    3. Find most recent 10-Q, 10-K, 8-K
    4. Download raw HTML + clean text for each
    5. Fetch XBRL companyfacts
    6. Extract structured fundamentals
    7. Return combined JSON
    """
    print(f"Building snapshot for {ticker}")
    print("=" * 50)

    # --- Resolve CIK ---
    print(f"  Resolving CIK for {ticker}...")
    ticker_map = sec.get_ticker_cik_map()
    if ticker not in ticker_map:
        raise ValueError(f"Ticker not found: {ticker}")
    cik = ticker_map[ticker]
    print(f"  CIK: {cik}")

    # --- Fetch submissions ---
    print(f"  Fetching submissions metadata...")
    cache_dir = base_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    submissions_cache = cache_dir / f"{ticker}_submissions.json"
    submissions = sec.get_company_filings_cached(cik, submissions_cache)

    # --- Find most recent filings ---
    target_forms = ["10-Q", "10-K", "8-K"]
    print(f"  Finding most recent {', '.join(target_forms)}...")
    filings = find_most_recent_filings(submissions, target_forms)

    if not filings:
        raise ValueError(f"No filings found for {ticker}")

    for f in filings:
        print(f"    {f['form']} filed {f['filing_date']} (accession: {f['accession']})")

    # --- Download filings ---
    print(f"  Downloading filings...")
    (base_dir / "raw_html" / ticker).mkdir(parents=True, exist_ok=True)
    (base_dir / "clean_text" / ticker).mkdir(parents=True, exist_ok=True)

    filing_records = []
    for filing in filings:
        accession, success = sec.download_filing(ticker, cik, filing, base_dir)
        if success:
            html_path = sec.build_html_path(
                base_dir, ticker, filing["matched_year"],
                filing["matched_quarter"], filing["form"], filing["filing_date"],
            )
            text_path = sec.build_text_path(
                base_dir, ticker, filing["matched_year"],
                filing["matched_quarter"], filing["form"], filing["filing_date"],
            )
            filing_records.append({
                "form": filing["form"],
                "filing_date": filing["filing_date"],
                "year": filing["matched_year"],
                "quarter": filing["matched_quarter"],
                "accession": filing["accession"],
                "text_path": str(text_path.relative_to(base_dir)),
                "html_path": str(html_path.relative_to(base_dir)),
            })

    # --- Fetch XBRL fundamentals ---
    print(f"  Fetching XBRL companyfacts...")
    facts = fetch_companyfacts(cik)

    # Cache the raw companyfacts
    facts_cache = cache_dir / f"{ticker}_companyfacts.json"
    tmp = facts_cache.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(facts, f)
    tmp.rename(facts_cache)
    print(f"  Cached companyfacts ({facts_cache})")

    fundamentals = extract_fundamentals(facts)
    print(f"  Extracted fundamentals (period ending: {fundamentals.get('period_end', 'N/A')})")

    # --- Build snapshot ---
    snapshot = {
        "ticker": ticker,
        "cik": cik,
        "snapshot_date": datetime.now(tz=_utc).isoformat(),
        "filings": filing_records,
        "fundamentals": fundamentals,
    }

    return snapshot


# ==========================
# CLI
# ==========================

def main():
    parser = argparse.ArgumentParser(
        description="Build combined ticker snapshot (filings + XBRL fundamentals)"
    )
    parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g. AAPL)")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent),
        help="Output base directory (default: EDGAR/ dir next to this script)",
    )
    args = parser.parse_args()

    ticker = args.ticker.strip().upper()
    base_dir = Path(args.output)
    base_dir.mkdir(parents=True, exist_ok=True)

    snapshot = build_snapshot(ticker, base_dir)

    # Save snapshot JSON
    snapshot_dir = base_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"{ticker}_snapshot.json"
    tmp = snapshot_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(snapshot, f, indent=2)
    tmp.rename(snapshot_path)

    print(f"\n{'=' * 50}")
    print(f"Snapshot saved: {snapshot_path}")
    print(f"{'=' * 50}")
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
