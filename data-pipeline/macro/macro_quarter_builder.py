#!/usr/bin/env python3
"""
Macro Market State Builder — Quarter-Aware, No-Leak.

Builds general macro economic state (rates, inflation, vol) as-of a
specific quarter end. No ticker-specific data. Backtest safe.

Output:
  data-pipeline/macro/data/macro_{YEAR}_{QUARTER}.json

Examples:

  # Single quarter
  python macro_quarter_builder.py --year 2025 --quarter Q2

  # Quarter range (Q4 2024 through Q3 2025)
  python macro_quarter_builder.py --start 2024Q4 --end 2025Q3

  # Custom output path (single quarter only)
  python macro_quarter_builder.py --year 2025 --quarter Q2 --out /tmp/macro.json
"""

import argparse
import datetime as dt
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

FRED_RATE_LIMIT = 0.25  # seconds between FRED API requests
_fred_lock = threading.Lock()

try:
    import yfinance as yf
except Exception:
    print("ERROR: yfinance required. pip install yfinance", file=sys.stderr)
    raise

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA_DIR = _SCRIPT_DIR / "data"

# ----------------------------
# Utilities
# ----------------------------

def iso(d: dt.date) -> str:
    return d.isoformat()

def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def bps(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return (b - a) * 100.0

def q_end_dates(year: int) -> Dict[str, dt.date]:
    return {
        "Q1": dt.date(year, 3, 31),
        "Q2": dt.date(year, 6, 30),
        "Q3": dt.date(year, 9, 30),
        "Q4": dt.date(year, 12, 31),
    }

def prev_q_end(q: str, year: int) -> dt.date:
    if q == "Q1":
        return dt.date(year - 1, 12, 31)
    if q == "Q2":
        return dt.date(year, 3, 31)
    if q == "Q3":
        return dt.date(year, 6, 30)
    if q == "Q4":
        return dt.date(year, 9, 30)
    raise ValueError("Invalid quarter")

def next_quarter(year: int, quarter: str) -> Tuple[int, str]:
    order = ["Q1", "Q2", "Q3", "Q4"]
    idx = order.index(quarter)
    if idx < 3:
        return year, order[idx + 1]
    return year + 1, "Q1"

def parse_quarter_string(qstr: str) -> Tuple[int, str]:
    """'2025Q2' -> (2025, 'Q2')"""
    year = int(qstr[:4])
    quarter = qstr[4:]
    if quarter not in ["Q1", "Q2", "Q3", "Q4"]:
        raise ValueError(f"Invalid quarter: {quarter}")
    return year, quarter

def quarter_range(start: str, end: str) -> List[Tuple[int, str]]:
    """Generate list of (year, quarter) from start to end inclusive."""
    y, q = parse_quarter_string(start)
    end_y, end_q = parse_quarter_string(end)
    result = []
    while True:
        result.append((y, q))
        if y == end_y and q == end_q:
            break
        y, q = next_quarter(y, q)
    return result

# ----------------------------
# FRED
# ----------------------------

@dataclass
class FredSeries:
    id: str
    name: str
    units: str

class FredClient:
    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key
        self.base = "https://api.stlouisfed.org/fred/series/observations"

    def fetch(self, series_id: str, start: dt.date, end: dt.date) -> pd.Series:
        params = {
            "series_id": series_id,
            "api_key": self.api_key or "",
            "file_type": "json",
            "observation_start": iso(start),
            "observation_end": iso(end),
        }
        try:
            with _fred_lock:
                time.sleep(FRED_RATE_LIMIT)
            r = requests.get(self.base, params=params, timeout=30)
            r.raise_for_status()
        except requests.Timeout:
            tqdm.write(f"    FAIL FRED {series_id} — timeout (30s)")
            return pd.Series(dtype=float)
        except requests.ConnectionError as e:
            tqdm.write(f"    FAIL FRED {series_id} — connection error: {e}")
            return pd.Series(dtype=float)
        except requests.HTTPError as e:
            status = getattr(e.response, 'status_code', None) or 'unknown'
            tqdm.write(f"    FAIL FRED {series_id} — HTTP {status} (check API key?)")
            return pd.Series(dtype=float)
        js = r.json()
        obs = js.get("observations", [])
        if not obs:
            tqdm.write(f"    WARN FRED {series_id} — no observations returned")
            return pd.Series(dtype=float)
        dates = []
        vals = []
        for o in obs:
            dates.append(pd.to_datetime(o["date"]))
            try:
                vals.append(float(o["value"]))
            except Exception:
                vals.append(np.nan)
        return pd.Series(vals, index=pd.DatetimeIndex(dates)).sort_index()

    def value_on_or_before(self, series: pd.Series, date: dt.date) -> Optional[float]:
        if series.empty:
            return None
        eligible = series[series.index <= pd.to_datetime(date)]
        if eligible.empty:
            return None
        return safe_float(eligible.dropna().iloc[-1])

# ----------------------------
# Yahoo Finance
# ----------------------------

def yf_history(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=iso(start),
        end=iso(end + dt.timedelta(days=1)),
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# ----------------------------
# Layer 1 (Macro)
# ----------------------------

def build_layer1_macro(fred: FredClient,
                       year: int,
                       quarter: str,
                       start_date: dt.date,
                       end_date: dt.date) -> Dict[str, Any]:

    q_end = q_end_dates(year)[quarter]
    prev_q = prev_q_end(quarter, year)
    yoy_date = dt.date(q_end.year - 1, q_end.month, q_end.day)

    series_ids = {
        "FF": "DFF",
        "2Y": "DGS2",
        "10Y": "DGS10",
        "30Y": "DGS30",
        "REAL10": "DFII10",
        "CPI": "CPIAUCSL",
        "CORE": "CPILFESL",
        "INDPRO": "INDPRO",
        "HY": "BAMLH0A0HYM2",
        "UNRATE": "UNRATE",
        "DXY": "DTWEXBGS",
        "WTI": "DCOILWTICO",
    }

    tqdm.write(f"  [L1] Fetching {len(series_ids)} FRED series in parallel...")
    pulled = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(fred.fetch, sid, start_date, end_date): k
            for k, sid in series_ids.items()
        }
        for future in as_completed(futures):
            k = futures[future]
            try:
                pulled[k] = future.result()
            except Exception as e:
                tqdm.write(f"    FAIL FRED {k} — {type(e).__name__}: {e}")
                pulled[k] = pd.Series(dtype=float)
    tqdm.write(f"  [L1] FRED fetch complete ({len(pulled)} series)")

    def val(k, d):
        return fred.value_on_or_before(pulled[k], d)

    y10 = val("10Y", q_end)
    y2 = val("2Y", q_end)
    curve = None if y10 is None or y2 is None else y10 - y2

    out = {
        "layer": "L1",
        "year": year,
        "quarter": quarter,
        "asof": iso(q_end),
        "metrics": {
            "FF": {"value": val("FF", q_end)},
            "2Y": {"value": y2, "delta_bps_qoq": bps(val("2Y", prev_q), y2)},
            "10Y": {"value": y10, "delta_bps_qoq": bps(val("10Y", prev_q), y10)},
            "30Y": {"value": val("30Y", q_end)},
            "CURVE": {"value": curve},
            "REAL10": {"value": val("REAL10", q_end)},
            "CPI_YOY": {"value": None if val("CPI", yoy_date) is None else (val("CPI", q_end)/val("CPI", yoy_date)-1)*100},
            "CORE_YOY": {"value": None if val("CORE", yoy_date) is None else (val("CORE", q_end)/val("CORE", yoy_date)-1)*100},
            "INDPRO": {"value": val("INDPRO", q_end)},
            "HY_OAS": {"value": val("HY", q_end)},
            "UNRATE": {"value": val("UNRATE", q_end)},
            "DXY": {"value": val("DXY", q_end)},
            "WTI": {"value": val("WTI", q_end)},
        },
    }

    return out

# ----------------------------
# Layer 4 (Vol)
# ----------------------------

def build_layer4_vol(fred: FredClient,
                     year: int,
                     quarter: str,
                     start_date: dt.date,
                     end_date: dt.date):

    q_end = q_end_dates(year)[quarter]

    tqdm.write(f"  [L4] Fetching VIX, SPY, TLT in parallel...")
    with ThreadPoolExecutor(max_workers=3) as pool:
        f_vix = pool.submit(fred.fetch, "VIXCLS", start_date, end_date)
        f_spy = pool.submit(yf_history, "SPY", q_end - dt.timedelta(days=120), q_end)
        f_tlt = pool.submit(yf_history, "TLT", q_end - dt.timedelta(days=120), q_end)

    try:
        vix = f_vix.result()
    except Exception as e:
        tqdm.write(f"    FAIL FRED VIXCLS — {type(e).__name__}: {e}")
        vix = pd.Series(dtype=float)
    try:
        spy = f_spy.result()
    except Exception as e:
        tqdm.write(f"    FAIL Yahoo SPY — {type(e).__name__}: {e}")
        spy = pd.DataFrame()
    try:
        tlt = f_tlt.result()
    except Exception as e:
        tqdm.write(f"    FAIL Yahoo TLT — {type(e).__name__}: {e}")
        tlt = pd.DataFrame()
    tqdm.write(f"  [L4] Vol data fetched")

    corr = None
    if not spy.empty and not tlt.empty:
        sret = np.log(spy["Close"]).diff()
        tret = np.log(tlt["Close"]).diff()
        df = pd.concat([sret, tret], axis=1).dropna()
        if len(df) >= 30:
            corr = float(df.iloc[:,0].corr(df.iloc[:,1]))

    return {
        "layer": "L4",
        "year": year,
        "quarter": quarter,
        "asof": iso(q_end),
        "metrics": {
            "VIX": {"value": fred.value_on_or_before(vix, q_end)},
            "EQ_BOND_CORR": {"value": corr},
        },
    }

# ----------------------------
# Orchestrator
# ----------------------------

def build_macro_state(year: int,
                      quarter: str,
                      fred_key: Optional[str],
                      back_years: int = 2):

    q_end = q_end_dates(year)[quarter]
    start_date = dt.date(q_end.year - back_years, 1, 1)
    end_date = q_end

    fred = FredClient(api_key=fred_key)

    doc = {
        "schema_version": "macro_state_v4_quarter_aware",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "year": year,
        "quarter": quarter,
        "as_of": iso(q_end),
        "layers": {}
    }

    print(f"Building macro state for {year} {quarter} (as-of {iso(q_end)})", flush=True)
    print(f"  Lookback: {iso(start_date)} → {iso(end_date)}", flush=True)

    # Both layers are independent — run in parallel
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_l1 = pool.submit(build_layer1_macro, fred, year, quarter, start_date, end_date)
        f_l4 = pool.submit(build_layer4_vol, fred, year, quarter, start_date, end_date)

        for name, future in [("L1", f_l1), ("L4", f_l4)]:
            try:
                doc["layers"][name] = future.result()
            except Exception as e:
                tqdm.write(f"  FAIL layer {name} — {type(e).__name__}: {e}")
                doc["layers"][name] = {"error": str(e)}

    print(f"All layers complete.")
    return doc

# ----------------------------
# CLI
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Build macro economic state for one or more quarters")
    # Single quarter mode
    p.add_argument("--year", type=int, default=None, help="Year (single quarter mode)")
    p.add_argument("--quarter", type=str, default=None, choices=["Q1","Q2","Q3","Q4"],
                   help="Quarter (single quarter mode)")
    # Range mode
    p.add_argument("--start", type=str, default=None, help="Start quarter, e.g. 2024Q4")
    p.add_argument("--end", type=str, default=None, help="End quarter, e.g. 2025Q3")
    p.add_argument("--fred-key", type=str, default=None)
    p.add_argument("--out", type=str, default=None,
                   help="Output path (single quarter only; default: macro/data/macro_YEAR_QUARTER.json)")
    p.add_argument("--back-years", type=int, default=2)
    return p.parse_args()

def main():
    args = parse_args()
    fred_key = args.fred_key or os.environ.get("FRED_API_KEY")

    if not fred_key:
        print("WARNING: No FRED API key. Set FRED_API_KEY or use --fred-key.", file=sys.stderr)

    # Determine quarters to build
    if args.start and args.end:
        quarters = quarter_range(args.start, args.end)
    elif args.year and args.quarter:
        quarters = [(args.year, args.quarter)]
    else:
        print("ERROR: specify --year/--quarter or --start/--end", file=sys.stderr)
        sys.exit(1)

    print(f"Building macro state for {len(quarters)} quarter(s)", flush=True)

    for year, quarter in tqdm(quarters, desc="Quarters", unit="quarter"):
        doc = build_macro_state(
            year=year,
            quarter=quarter,
            fred_key=fred_key,
            back_years=args.back_years,
        )

        if args.out and len(quarters) == 1:
            out_path = Path(args.out)
        else:
            out_path = _DEFAULT_DATA_DIR / f"macro_{year}_{quarter}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(doc, f, indent=2)

        tqdm.write(f"Wrote: {out_path}")

    print(f"Done: {len(quarters)} quarter(s) complete.")

if __name__ == "__main__":
    main()