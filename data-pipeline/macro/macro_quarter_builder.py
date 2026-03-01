#!/usr/bin/env python3
"""
AUGMENTED MARKET STATE — Builder v4 (Quarter-Aware, No-Leak)

This version is QUARTER-AWARE.

Key changes:
- Requires --year and --quarter
- Builds state ONLY as-of that quarter end
- No future-quarter leakage
- Backtest safe

Example:

  python macro_quarter_builder.py --year 2025 --quarter Q2 --tickers AAPL,NVDA
"""

import argparse
import datetime as dt
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except Exception:
    print("ERROR: yfinance required. pip install yfinance", file=sys.stderr)
    raise

_SCRIPT_DIR = Path(__file__).resolve().parent

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

def annualized_vol(daily_returns: pd.Series, trading_days: int = 252) -> Optional[float]:
    if len(daily_returns.dropna()) < 10:
        return None
    return float(daily_returns.std(ddof=1) * math.sqrt(trading_days))

def max_drawdown(prices: pd.Series) -> Optional[float]:
    if len(prices.dropna()) < 10:
        return None
    peak = prices.cummax()
    dd = prices / peak - 1.0
    return float(dd.min())

def sma(series: pd.Series, window: int) -> Optional[float]:
    if len(series.dropna()) < window:
        return None
    return float(series.rolling(window).mean().iloc[-1])

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
        r = requests.get(self.base, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        obs = js.get("observations", [])
        if not obs:
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
        "ISM": "NAPM",
        "HY": "BAMLH0A0HYM2",
        "UNRATE": "UNRATE",
        "DXY": "DTWEXBGS",
        "WTI": "DCOILWTICO",
    }

    pulled = {}
    for k, sid in series_ids.items():
        try:
            pulled[k] = fred.fetch(sid, start_date, end_date)
        except Exception:
            pulled[k] = pd.Series(dtype=float)

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
            "ISM": {"value": val("ISM", q_end)},
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

    vix = fred.fetch("VIXCLS", start_date, end_date)
    move = fred.fetch("MOVE", start_date, end_date)

    spy = yf_history("SPY", q_end - dt.timedelta(days=120), q_end)
    tlt = yf_history("TLT", q_end - dt.timedelta(days=120), q_end)

    corr = None
    if not spy.empty and not tlt.empty:
        sret = np.log(spy["Adj Close"]).diff()
        tret = np.log(tlt["Adj Close"]).diff()
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
            "MOVE": {"value": fred.value_on_or_before(move, q_end)},
            "EQ_BOND_CORR": {"value": corr},
        },
    }

# ----------------------------
# Layer 6 (Ticker prices)
# ----------------------------

def build_layer6_prices(year: int,
                        quarter: str,
                        tickers: List[str],
                        start_date: dt.date,
                        end_date: dt.date):

    q_end = q_end_dates(year)[quarter]
    out = {"layer": "L6", "year": year, "quarter": quarter, "tickers": {}}

    for t in tickers:
        df = yf_history(t, start_date, end_date)
        if df.empty:
            out["tickers"][t] = {"error": "no_data"}
            continue

        px = df["Adj Close"].dropna()
        px = px[px.index <= pd.to_datetime(q_end)]
        if len(px) < 10:
            out["tickers"][t] = {"error": "insufficient_data"}
            continue

        px60 = px.iloc[-61:]

        out["tickers"][t] = {
            "asof": iso(q_end),
            "metrics": {
                "PRICE": float(px.iloc[-1]),
                "RET60": float(px60.iloc[-1]/px60.iloc[0]-1)*100 if len(px60)>=2 else None,
                "VOL60": annualized_vol(np.log(px60).diff()),
                "DD60": max_drawdown(px60),
                "SMA20": sma(px,20),
                "SMA50": sma(px,50),
            }
        }

    return out

# ----------------------------
# Orchestrator
# ----------------------------

def build_augmented_market_state(year: int,
                                 quarter: str,
                                 tickers: List[str],
                                 fred_key: Optional[str],
                                 back_years: int = 2):

    q_end = q_end_dates(year)[quarter]
    start_date = dt.date(q_end.year - back_years, 1, 1)
    end_date = q_end

    fred = FredClient(api_key=fred_key)

    doc = {
        "schema_version": "augmented_market_state_v4_quarter_aware",
        "generated_at_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "year": year,
        "quarter": quarter,
        "as_of": iso(q_end),
        "tickers": tickers,
        "layers": {}
    }

    doc["layers"]["L1"] = build_layer1_macro(fred, year, quarter, start_date, end_date)
    doc["layers"]["L4"] = build_layer4_vol(fred, year, quarter, start_date, end_date)
    doc["layers"]["L6"] = build_layer6_prices(year, quarter, tickers, start_date, end_date)

    return doc

# ----------------------------
# CLI
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--quarter", type=str, required=True, choices=["Q1","Q2","Q3","Q4"])
    p.add_argument("--tickers", type=str, default="SPY")
    p.add_argument("--fred-key", type=str, default=None)
    p.add_argument("--out", type=str, default="macro_quarter.json")
    p.add_argument("--back-years", type=int, default=2)
    return p.parse_args()

def main():
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    fred_key = args.fred_key or os.environ.get("FRED_API_KEY")

    doc = build_augmented_market_state(
        year=args.year,
        quarter=args.quarter,
        tickers=tickers,
        fred_key=fred_key,
        back_years=args.back_years
    )

    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2)

    print(f"Wrote: {args.out}")

if __name__ == "__main__":
    main()