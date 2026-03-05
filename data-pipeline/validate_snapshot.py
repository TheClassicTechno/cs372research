#!/usr/bin/env python3
"""
Snapshot validation — cross-check snapshot data against all source files.

Runs offline at any time (not just during pipeline execution).  Checks
completeness, temporal integrity, source cross-reference, cross-sectional
sanity, fiscal alignment, and optional provenance manifest verification.

Usage:

  # Validate a single quarter
  python validate_snapshot.py --year 2025 --quarter Q1

  # Validate a range
  python validate_snapshot.py --start 2024Q4 --end 2025Q3

  # Verbose (print every check)
  python validate_snapshot.py --year 2025 --quarter Q1 --verbose
"""

import argparse
import datetime as dt
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_SUPPORTED_TICKERS_PATH = _SCRIPT_DIR / "supported_tickers.yaml"
_SUMMARIES_DIR = _SCRIPT_DIR / "EDGAR" / "finished_summaries"
_SENTIMENT_DIR = _SCRIPT_DIR / "sentiment" / "data"
_MACRO_DIR = _SCRIPT_DIR / "macro" / "data"
_ASSETS_DIR = _SCRIPT_DIR / "quarterly_asset_details" / "data"
_SNAPSHOT_DIR = _SCRIPT_DIR / "final_snapshots" / "json_data"
_MEMO_DIR = _SCRIPT_DIR / "final_snapshots" / "memo_data"
_PROVENANCE_DIR = _SCRIPT_DIR / "provenance"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

QUARTER_ENDS = {
    "Q1": (3, 31),
    "Q2": (6, 30),
    "Q3": (9, 30),
    "Q4": (12, 31),
}


def quarter_end_date(year: int, quarter: str) -> dt.date:
    month, day = QUARTER_ENDS[quarter]
    return dt.date(year, month, day)


def parse_quarter_string(qstr: str) -> Tuple[int, str]:
    year = int(qstr[:4])
    q = qstr[4:]
    if q not in ("Q1", "Q2", "Q3", "Q4"):
        raise ValueError(f"Invalid quarter: {q}")
    return year, q


def next_quarter(year: int, quarter: str) -> Tuple[int, str]:
    labels = ["Q1", "Q2", "Q3", "Q4"]
    idx = labels.index(quarter)
    if idx < 3:
        return year, labels[idx + 1]
    return year + 1, "Q1"


def quarter_range_list(start: str, end: str) -> List[Tuple[int, str]]:
    y, q = parse_quarter_string(start)
    end_y, end_q = parse_quarter_string(end)
    result = []
    while True:
        result.append((y, q))
        if y == end_y and q == end_q:
            break
        y, q = next_quarter(y, q)
    return result


def load_supported_tickers() -> List[str]:
    with open(_SUPPORTED_TICKERS_PATH, "r") as f:
        data = yaml.safe_load(f)
    return [t["symbol"] for t in data["supported_tickers"]]


def load_fiscal_year_ends() -> Dict[str, str]:
    with open(_SUPPORTED_TICKERS_PATH, "r") as f:
        data = yaml.safe_load(f)
    return {
        t["symbol"]: t.get("fiscal_year_end", "12-31")
        for t in data["supported_tickers"]
    }


# ---------------------------------------------------------------------------
# Validation context
# ---------------------------------------------------------------------------

class ValidationResult:
    def __init__(self):
        self.ok = 0
        self.warn = 0
        self.fail = 0
        self.messages: List[Tuple[str, str]] = []  # (level, message)

    def _log(self, level: str, msg: str):
        self.messages.append((level, msg))

    def passed(self, msg: str):
        self.ok += 1
        self._log("OK", msg)

    def warning(self, msg: str):
        self.warn += 1
        self._log("WARN", msg)

    def error(self, msg: str):
        self.fail += 1
        self._log("FAIL", msg)

    def print_section(self, title: str, verbose: bool):
        section_msgs = []
        for level, msg in self.messages:
            if verbose or level != "OK":
                section_msgs.append((level, msg))
        if section_msgs:
            print(f"\n{title}")
            for level, msg in section_msgs:
                print(f"  {level:4s}  {msg}")


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

def check_completeness(
    doc: dict,
    supported_tickers: List[str],
    verbose: bool,
) -> ValidationResult:
    r = ValidationResult()
    snapshot_tickers = set(doc.get("tickers", []))

    present = [t for t in supported_tickers if t in snapshot_tickers]
    missing = [t for t in supported_tickers if t not in snapshot_tickers]

    if missing:
        for t in missing:
            r.warning(f"{t} missing from snapshot")
    else:
        r.passed(f"All {len(supported_tickers)} supported tickers present")

    # Coverage
    ticker_data = doc.get("ticker_data", {})
    fs_count = sum(1 for t in present if ticker_data.get(t, {}).get("filing_summary", {}).get("periodic"))
    sent_count = sum(1 for t in present if ticker_data.get(t, {}).get("news_sentiment"))
    asset_count = sum(1 for t in present if ticker_data.get(t, {}).get("asset_features"))

    r.passed(f"Coverage: {fs_count} filing_summary, {sent_count} sentiment, {asset_count} asset_features")

    r.print_section("COMPLETENESS", verbose)
    return r


def check_filing_cross_reference(
    doc: dict,
    verbose: bool,
) -> ValidationResult:
    r = ValidationResult()
    ticker_data = doc.get("ticker_data", {})
    verified = 0
    total = 0

    paragraph_keys = [
        "operating_state", "cost_structure", "material_events",
        "macro_exposures", "forward_outlook", "uncertainty_profile",
    ]

    for ticker, td in ticker_data.items():
        fs = td.get("filing_summary", {})
        periodic = fs.get("periodic") if isinstance(fs, dict) else None
        if not periodic:
            continue
        total += 1

        # Find source file
        found = False
        for json_file in _SUMMARIES_DIR.joinpath(ticker).rglob("*.json"):
            try:
                with open(json_file, "r") as f:
                    source = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            if (source.get("filing_date") == periodic.get("filing_date")
                    and source.get("form") == periodic.get("form")):
                # Compare paragraph fields
                mismatches = []
                for key in paragraph_keys:
                    if source.get(key) != periodic.get(key):
                        mismatches.append(key)
                if mismatches:
                    r.error(f"{ticker} periodic filing mismatch in: {', '.join(mismatches)} (vs {json_file.name})")
                else:
                    r.passed(f"{ticker} periodic ({periodic.get('form')} {periodic.get('filing_date')}) matches source")
                    verified += 1
                found = True
                break

        if not found:
            r.warning(f"{ticker} periodic filing source not found ({periodic.get('form')} {periodic.get('filing_date')})")

    if total > 0:
        r.passed(f"{verified}/{total} periodic filings verified")

    r.print_section("FILING CROSS-REFERENCE", verbose)
    return r


def check_temporal_integrity(
    doc: dict,
    year: int,
    quarter: str,
    verbose: bool,
) -> ValidationResult:
    r = ValidationResult()
    rebal = quarter_end_date(year, quarter)
    cutoff = rebal.isoformat()
    cutoff_90 = (rebal - dt.timedelta(days=90)).isoformat()

    ticker_data = doc.get("ticker_data", {})
    periodic_violations = []
    event_violations = []

    for ticker, td in ticker_data.items():
        fs = td.get("filing_summary", {})
        if not isinstance(fs, dict):
            continue

        periodic = fs.get("periodic")
        if periodic and isinstance(periodic, dict):
            fd = periodic.get("filing_date", "")
            if fd and fd > cutoff:
                periodic_violations.append(f"{ticker}: periodic filing_date {fd} > {cutoff}")

        for ev in fs.get("event_filings", []):
            fd = ev.get("filing_date", "")
            if fd and fd > cutoff:
                event_violations.append(f"{ticker}: 8-K filing_date {fd} > {cutoff}")
            elif fd and fd < cutoff_90:
                event_violations.append(f"{ticker}: 8-K filing_date {fd} outside 90-day window")

    if periodic_violations:
        for v in periodic_violations:
            r.error(v)
    else:
        r.passed(f"All periodic filing_dates <= {cutoff}")

    if event_violations:
        for v in event_violations:
            r.error(v)
    else:
        r.passed(f"All 8-K filing_dates within 90-day window")

    r.print_section("TEMPORAL INTEGRITY", verbose)
    return r


def check_freshness(
    doc: dict,
    year: int,
    quarter: str,
    verbose: bool,
) -> ValidationResult:
    r = ValidationResult()
    rebal = quarter_end_date(year, quarter)
    cutoff = rebal.isoformat()
    ticker_data = doc.get("ticker_data", {})
    periodic_forms = {"10-Q", "10-K", "10-Q/A", "10-K/A"}
    stale = []

    for ticker, td in ticker_data.items():
        fs = td.get("filing_summary", {})
        snapshot_periodic = fs.get("periodic") if isinstance(fs, dict) else None
        if not snapshot_periodic:
            continue

        snapshot_fd = snapshot_periodic.get("filing_date", "")

        # Scan all source summaries for this ticker
        ticker_dir = _SUMMARIES_DIR / ticker
        if not ticker_dir.exists():
            continue

        most_recent_fd = ""
        most_recent_form = ""
        for json_file in ticker_dir.rglob("*.json"):
            try:
                with open(json_file, "r") as f:
                    source = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if source.get("form", "") not in periodic_forms:
                continue
            fd = source.get("filing_date", "")
            if fd and fd <= cutoff and fd > most_recent_fd:
                most_recent_fd = fd
                most_recent_form = source.get("form", "")

        if most_recent_fd and most_recent_fd > snapshot_fd:
            stale.append(
                f"{ticker}: snapshot uses {snapshot_periodic.get('form')} from {snapshot_fd} "
                f"but {most_recent_form} from {most_recent_fd} exists and is eligible"
            )

    if stale:
        for s in stale:
            r.warning(s)
    else:
        r.passed("All tickers using most recent eligible periodic filing")

    r.print_section("FRESHNESS", verbose)
    return r


def check_sentiment_cross_reference(
    doc: dict,
    year: int,
    quarter: str,
    verbose: bool,
) -> ValidationResult:
    r = ValidationResult()
    ticker_data = doc.get("ticker_data", {})
    verified = 0
    total = 0

    for ticker, td in ticker_data.items():
        sent = td.get("news_sentiment")
        if not sent:
            continue
        total += 1

        source_path = _SENTIMENT_DIR / ticker / f"{year}_{quarter}.json"
        if not source_path.exists():
            r.warning(f"{ticker} sentiment source not found: {source_path.name}")
            continue

        try:
            with open(source_path, "r") as f:
                source = json.load(f)
            source_feats = source.get("features", {})
        except (json.JSONDecodeError, OSError):
            r.warning(f"{ticker} sentiment source unreadable")
            continue

        mismatches = []
        for key in ("article_count", "mean_sentiment"):
            sv = source_feats.get(key) if source_feats else None
            dv = sent.get(key)
            if sv != dv:
                mismatches.append(key)

        if mismatches:
            r.error(f"{ticker} sentiment mismatch: {', '.join(mismatches)}")
        else:
            r.passed(f"{ticker} sentiment matches source")
            verified += 1

    if total > 0:
        r.passed(f"{verified}/{total} sentiment records verified")

    r.print_section("SENTIMENT CROSS-REFERENCE", verbose)
    return r


def check_asset_cross_reference(
    doc: dict,
    year: int,
    quarter: str,
    verbose: bool,
) -> ValidationResult:
    r = ValidationResult()
    ticker_data = doc.get("ticker_data", {})
    verified = 0
    total = 0

    spot_check_keys = ("close", "ret_60d", "beta_1y")

    for ticker, td in ticker_data.items():
        af = td.get("asset_features")
        if not af or "error" in af:
            continue
        total += 1

        source_path = _ASSETS_DIR / ticker / f"{year}_{quarter}.json"
        if not source_path.exists():
            r.warning(f"{ticker} asset source not found: {source_path.name}")
            continue

        try:
            with open(source_path, "r") as f:
                source = json.load(f)
            source_feats = source.get("features", {})
        except (json.JSONDecodeError, OSError):
            r.warning(f"{ticker} asset source unreadable")
            continue

        if source_feats and "error" in source_feats:
            continue

        mismatches = []
        for key in spot_check_keys:
            sv = source_feats.get(key) if source_feats else None
            dv = af.get(key)
            if sv != dv:
                mismatches.append(key)

        if mismatches:
            r.error(f"{ticker} asset mismatch: {', '.join(mismatches)}")
        else:
            r.passed(f"{ticker} asset features match source")
            verified += 1

    if total > 0:
        r.passed(f"{verified}/{total} asset records verified")

    r.print_section("ASSET CROSS-REFERENCE", verbose)
    return r


def check_macro_cross_reference(
    doc: dict,
    year: int,
    quarter: str,
    verbose: bool,
) -> ValidationResult:
    r = ValidationResult()
    macro = doc.get("macro_regime")
    if not macro:
        r.warning("No macro_regime in snapshot")
        r.print_section("MACRO CROSS-REFERENCE", verbose)
        return r

    source_path = _MACRO_DIR / f"macro_{year}_{quarter}.json"
    if not source_path.exists():
        r.warning(f"Macro source not found: {source_path.name}")
        r.print_section("MACRO CROSS-REFERENCE", verbose)
        return r

    try:
        with open(source_path, "r") as f:
            source = json.load(f)
    except (json.JSONDecodeError, OSError):
        r.warning("Macro source unreadable")
        r.print_section("MACRO CROSS-REFERENCE", verbose)
        return r

    # Compare L1 metrics
    source_l1 = source.get("layers", {}).get("L1", {}).get("metrics", {})
    snapshot_mm = macro.get("macro_metrics", {})

    mismatches = []
    for key in source_l1:
        sv = source_l1[key].get("value") if isinstance(source_l1[key], dict) else None
        dv = snapshot_mm.get(key, {}).get("value") if isinstance(snapshot_mm.get(key), dict) else None
        if sv != dv:
            mismatches.append(key)

    if mismatches:
        r.error(f"Macro L1 metric mismatch: {', '.join(mismatches[:5])}")
    else:
        r.passed(f"Macro metrics match {source_path.name}")

    r.print_section("MACRO CROSS-REFERENCE", verbose)
    return r


def check_cross_sectional(
    doc: dict,
    verbose: bool,
) -> ValidationResult:
    r = ValidationResult()
    ticker_data = doc.get("ticker_data", {})

    # relative_strength_60d is median-centered, so median should be ~0
    rs_vals = []
    for td in ticker_data.values():
        af = td.get("asset_features")
        if af and "error" not in af and af.get("relative_strength_60d") is not None:
            rs_vals.append(af["relative_strength_60d"])

    if rs_vals:
        rs_sorted = sorted(rs_vals)
        n = len(rs_sorted)
        rs_median = (
            rs_sorted[n // 2]
            if n % 2 == 1
            else (rs_sorted[n // 2 - 1] + rs_sorted[n // 2]) / 2
        )
        if abs(rs_median) < 0.01:
            r.passed(f"relative_strength_60d median={rs_median:.4f} (tolerance 0.01)")
        else:
            r.error(f"relative_strength_60d median={rs_median:.4f} (expected ~0)")

    # cross_sectional_z should sum to ~0 and std ~1
    z_vals = []
    for td in ticker_data.values():
        sent = td.get("news_sentiment")
        if sent and sent.get("cross_sectional_z") is not None:
            z_vals.append(sent["cross_sectional_z"])

    if len(z_vals) >= 2:
        z_sum = sum(z_vals)
        z_mean = z_sum / len(z_vals)
        z_std = math.sqrt(sum((x - z_mean) ** 2 for x in z_vals) / len(z_vals))

        if abs(z_sum) < 0.01:
            r.passed(f"cross_sectional_z sums to {z_sum:.4f}")
        else:
            r.error(f"cross_sectional_z sums to {z_sum:.4f} (expected ~0)")

        if abs(z_std - 1.0) < 0.1:
            r.passed(f"cross_sectional_z std={z_std:.4f}")
        else:
            r.warning(f"cross_sectional_z std={z_std:.4f} (expected ~1.0)")

    r.print_section("CROSS-SECTIONAL", verbose)
    return r


def check_provenance(
    doc: dict,
    year: int,
    quarter: str,
    verbose: bool,
) -> ValidationResult:
    r = ValidationResult()

    if not _PROVENANCE_DIR.exists():
        r.print_section("PROVENANCE", verbose)
        return r

    # Find the most recent manifest
    manifests = sorted(_PROVENANCE_DIR.glob("manifest_*.json"), reverse=True)
    if not manifests:
        r.print_section("PROVENANCE", verbose)
        return r

    try:
        from provenance import hash_file
    except ImportError:
        r.warning("provenance module not importable — skipping hash checks")
        r.print_section("PROVENANCE", verbose)
        return r

    # Find manifest that produced this snapshot
    snapshot_filename = f"final_snapshots/json_data/snapshot_{year}_{quarter}.json"
    target_manifest = None
    for mf in manifests:
        try:
            with open(mf, "r") as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        stages = manifest.get("stages", {})
        for stage_data in stages.values():
            for dep in stage_data.get("dependencies", []):
                if dep.get("output", "").endswith(f"snapshot_{year}_{quarter}.json"):
                    target_manifest = manifest
                    break
            if target_manifest:
                break
        if target_manifest:
            r.passed(f"Found manifest: {mf.name}")
            break

    if not target_manifest:
        r.print_section("PROVENANCE", verbose)
        return r

    # Verify input file hashes
    verified = 0
    changed = []
    stages = target_manifest.get("stages", {})
    for stage_data in stages.values():
        for dep in stage_data.get("dependencies", []):
            for inp in dep.get("inputs", []):
                path = inp.get("path")
                expected_hash = inp.get("sha256")
                if not path or not expected_hash:
                    continue
                full_path = _SCRIPT_DIR / path
                actual = hash_file(full_path)
                if actual == expected_hash:
                    verified += 1
                elif actual is None:
                    changed.append(f"{path} (file missing)")
                else:
                    changed.append(f"{path} (hash changed)")

    if verified > 0:
        r.passed(f"{verified} input file hashes verified")
    if changed:
        r.warning(f"{len(changed)} files changed since manifest:")
        for c in changed[:5]:
            r.warning(f"  {c}")

    r.print_section("PROVENANCE", verbose)
    return r


def check_fiscal_config(verbose: bool) -> ValidationResult:
    """Validate that supported_tickers.yaml has fiscal_year_end for every ticker."""
    r = ValidationResult()
    try:
        with open(_SUPPORTED_TICKERS_PATH, "r") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        r.error("Cannot read supported_tickers.yaml")
        r.print_section("FISCAL CONFIG", verbose)
        return r

    entries = data.get("supported_tickers", [])
    bad = 0
    for entry in entries:
        symbol = entry.get("symbol", "?")
        fye = entry.get("fiscal_year_end")
        if not fye:
            r.error(f"{symbol} missing fiscal_year_end")
            bad += 1
        elif not re.match(r"^\d{2}-\d{2}$", fye):
            r.error(f"{symbol} bad FYE format: {fye}")
            bad += 1

    if bad == 0:
        r.passed(f"All {len(entries)} tickers have valid fiscal_year_end")

    r.print_section("FISCAL CONFIG", verbose)
    return r


def check_fiscal_alignment(
    doc: dict,
    year: int,
    quarter: str,
    verbose: bool,
) -> ValidationResult:
    r = ValidationResult()
    fye_map = load_fiscal_year_ends()
    rebal = quarter_end_date(year, quarter)
    ticker_data = doc.get("ticker_data", {})

    non_standard = {t: fye for t, fye in fye_map.items() if fye != "12-31"}

    # --- Non-standard FYE: filing_date and fiscal year-end checks ---

    for ticker, fye in non_standard.items():
        td = ticker_data.get(ticker)
        if not td:
            continue

        fs = td.get("filing_summary", {})
        periodic = fs.get("periodic") if isinstance(fs, dict) else None
        if not periodic:
            continue

        fd = periodic.get("filing_date", "")
        fiscal_period = periodic.get("fiscal_period", "")
        period_type = periodic.get("period_type", "")

        # Check filing_date <= as_of_date
        if fd and fd > rebal.isoformat():
            r.error(f"{ticker}: filing_date {fd} > as-of {rebal.isoformat()}")
        else:
            r.passed(f"{ticker}: filing_date {fd} <= as-of {rebal.isoformat()}")

        # For annual filings: check fiscal year-end date <= rebalance_date
        if period_type == "annual" and fiscal_period.startswith("FY"):
            try:
                fy_year = int(fiscal_period[2:])
                fye_month = int(fye[:2])
                fye_day = int(fye[3:])
                fye_date = dt.date(fy_year, fye_month, fye_day)

                month_name = {
                    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
                }[fye_month]

                if fye_date <= rebal:
                    r.passed(
                        f"{ticker}: {periodic.get('form')} {fiscal_period} "
                        f"(FYE {month_name} {fye_day}) filed {fd} — "
                        f"fiscal year-end {fye_date.isoformat()} <= as-of {rebal.isoformat()}"
                    )
                else:
                    r.error(
                        f"{ticker}: {periodic.get('form')} {fiscal_period} "
                        f"(FYE {month_name} {fye_day}) — fiscal year-end "
                        f"{fye_date.isoformat()} > as-of {rebal.isoformat()} LEAKAGE"
                    )
            except (ValueError, KeyError):
                r.warning(f"{ticker}: could not parse fiscal year-end from {fiscal_period}")

    # --- Period consistency for all tickers ---

    for ticker, td in ticker_data.items():
        fs = td.get("filing_summary", {})
        periodic = fs.get("periodic") if isinstance(fs, dict) else None
        if not periodic:
            continue

        form = periodic.get("form", "")
        pt = periodic.get("period_type", "")
        fp = periodic.get("fiscal_period", "")

        if not fp:
            r.warning(f"{ticker}: periodic filing missing fiscal_period")
        if not pt:
            r.warning(f"{ticker}: periodic filing missing period_type")

        if form in ("10-K", "10-K/A"):
            if pt and pt != "annual":
                r.error(f"{ticker}: 10-K has period_type={pt}, expected annual")
            if fp and not fp.startswith("FY"):
                r.error(f"{ticker}: 10-K fiscal_period={fp}, expected FY*")
        elif form in ("10-Q", "10-Q/A"):
            if pt and pt != "quarterly":
                r.error(f"{ticker}: 10-Q has period_type={pt}, expected quarterly")
            if fp and not re.match(r"^\d{4}-Q[1-4]$", fp):
                r.error(f"{ticker}: 10-Q fiscal_period={fp}, expected YYYY-Q#")

    r.print_section("FISCAL ALIGNMENT", verbose)
    return r


def check_memo_fiscal_annotations(
    year: int,
    quarter: str,
    verbose: bool,
) -> ValidationResult:
    """Check that non-standard FYE tickers have FYE annotations in memos."""
    r = ValidationResult()
    memo_path = _MEMO_DIR / f"memo_{year}_{quarter}.txt"

    if not memo_path.exists():
        r.print_section("MEMO FISCAL ANNOTATIONS", verbose)
        return r

    memo_text = memo_path.read_text()
    fye_map = load_fiscal_year_ends()
    non_standard = {t: fye for t, fye in fye_map.items() if fye != "12-31"}

    for ticker, fye in non_standard.items():
        ticker_pattern = f"TICKER: {ticker}"
        if ticker_pattern not in memo_text:
            continue

        # Extract this ticker's section
        idx = memo_text.index(ticker_pattern)
        next_ticker = memo_text.find("TICKER:", idx + len(ticker_pattern))
        end_memo = memo_text.find("END OF MEMO", idx)
        boundaries = [x for x in [next_ticker, end_memo, len(memo_text)] if x > 0]
        section = memo_text[idx:min(boundaries)]

        # Check for filing summary presence
        if "Filing Summary" not in section:
            continue

        # Check for FYE annotation
        if "Filing Summary (" in section:
            filing_line = [l for l in section.splitlines() if "Filing Summary (" in l]
            if filing_line and "not available" not in filing_line[0]:
                if "FYE:" in section:
                    r.passed(f"{ticker}: FYE annotation present in memo")
                else:
                    r.warning(f"{ticker}: non-standard FYE ticker missing FYE annotation")

    r.print_section("MEMO FISCAL ANNOTATIONS", verbose)
    return r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate_quarter(
    year: int,
    quarter: str,
    verbose: bool,
    tickers: Optional[List[str]] = None,
) -> Tuple[int, int, int]:
    as_of = quarter_end_date(year, quarter).isoformat()
    print(f"\nSnapshot Validation: {year} {quarter} (as-of {as_of})")
    print("=" * 64)

    snapshot_path = _SNAPSHOT_DIR / f"snapshot_{year}_{quarter}.json"
    if not snapshot_path.exists():
        # Try alternate location (snapshot files might be in final_snapshots/ directly)
        alt_path = _SCRIPT_DIR / "final_snapshots" / f"snapshot_{year}_{quarter}.json"
        if alt_path.exists():
            snapshot_path = alt_path
        else:
            print(f"  SKIP: snapshot not found at {snapshot_path}")
            return (0, 0, 1)

    with open(snapshot_path, "r") as f:
        doc = json.load(f)

    # Use scenario-specific tickers if provided, otherwise all supported
    expected = tickers if tickers is not None else load_supported_tickers()

    checks = [
        check_completeness(doc, expected, verbose),
        check_filing_cross_reference(doc, verbose),
        check_temporal_integrity(doc, year, quarter, verbose),
        check_freshness(doc, year, quarter, verbose),
        check_sentiment_cross_reference(doc, year, quarter, verbose),
        check_asset_cross_reference(doc, year, quarter, verbose),
        check_macro_cross_reference(doc, year, quarter, verbose),
        check_cross_sectional(doc, verbose),
        check_provenance(doc, year, quarter, verbose),
        check_fiscal_alignment(doc, year, quarter, verbose),
        check_memo_fiscal_annotations(year, quarter, verbose),
    ]

    total_ok = sum(c.ok for c in checks)
    total_warn = sum(c.warn for c in checks)
    total_fail = sum(c.fail for c in checks)

    print(f"\n{'=' * 64}")
    status = "PASS" if total_fail == 0 else "FAIL"
    print(f"Result: {total_ok} passed, {total_warn} warnings, {total_fail} errors — {status}")

    return (total_ok, total_warn, total_fail)


def main():
    p = argparse.ArgumentParser(description="Validate quarterly snapshot data")
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--quarter", default=None, choices=["Q1", "Q2", "Q3", "Q4"])
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--tickers", type=str, default=None,
                   help="Comma-separated tickers to check (default: all supported)")
    p.add_argument("--verbose", action="store_true", help="Print all checks, not just failures")
    args = p.parse_args()

    ticker_list = (
        [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if args.tickers else None
    )

    if args.start and args.end:
        quarters = quarter_range_list(args.start, args.end)
    elif args.year and args.quarter:
        quarters = [(args.year, args.quarter)]
    else:
        p.error("specify --year/--quarter or --start/--end")

    total_ok = 0
    total_warn = 0
    total_fail = 0

    # Global checks (run once)
    config_result = check_fiscal_config(args.verbose)
    total_ok += config_result.ok
    total_warn += config_result.warn
    total_fail += config_result.fail

    for year, quarter in quarters:
        ok, warn, fail = validate_quarter(year, quarter, args.verbose, ticker_list)
        total_ok += ok
        total_warn += warn
        total_fail += fail

    if len(quarters) > 1:
        print(f"\n{'=' * 64}")
        print(f"Overall: {total_ok} passed, {total_warn} warnings, {total_fail} errors")

    sys.exit(1 if total_fail > 0 else 0)


if __name__ == "__main__":
    main()
