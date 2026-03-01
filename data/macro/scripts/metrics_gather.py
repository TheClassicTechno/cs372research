#!/usr/bin/env python3
"""
AUGMENTED MARKET STATE — Builder v3 (Layers 1,4,5,6) -> JSON

Goal
- Produce a JSON object that covers:
  Layer 1 (Macro): rates, curve, real yields, liquidity, inflation, growth, credit (+ a few common adds)
  Layer 4 (Vol & risk pricing): VIX, MOVE, SKEW (if available), equity–bond corr (computed)
  Layer 5 (Market internals): % > 200DMA (approx via SP500 constituents), concentration proxy (top-10 mcap share),
                             equal-weight vs cap-weight (RSP vs SPY)
  Layer 6 (Price summary): for each ticker: current price, 60D return/vol/maxDD, SMA20/SMA50

What you need
- Optional: FRED API key (recommended). Set env var FRED_API_KEY or pass --fred-key.
- yfinance must be installed (pip install yfinance).
- requests, pandas, numpy are standard in most DS envs.

Example
  python augmented_market_state_v3.py --year 2025 --tickers AAPL,MSFT,NVDA --out out_2025.json

Notes
- Some series are fetched from FRED; if a series is unavailable, we record an error for that metric.
- Breadth and concentration use S&P 500 constituents from Wikipedia + yfinance. This is inherently approximate and can be slow.
  Use --breadth-sample to reduce runtime (e.g., 150). Use --breadth-max to cap.
"""

import argparse
import datetime as dt
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except Exception as e:
    print("ERROR: yfinance is required. Install with: pip install yfinance", file=sys.stderr)
    raise

# ----------------------------
# Utilities
# ----------------------------

def iso(d: dt.date) -> str:
    return d.isoformat()

def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (np.floating, np.integer)):
            return float(x)
        if isinstance(x, (float, int)):
            return float(x)
        if isinstance(x, str) and x.strip() == "":
            return None
        return float(x)
    except Exception:
        return None

def pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    # percent change b over a
    if a is None or b is None:
        return None
    if a == 0:
        return None
    return (b - a) / a * 100.0

def bps(a: Optional[float], b: Optional[float]) -> Optional[float]:
    # basis points change b - a
    if a is None or b is None:
        return None
    return (b - a) * 100.0

def annualized_vol(daily_returns: pd.Series, trading_days: int = 252) -> Optional[float]:
    if daily_returns is None or len(daily_returns.dropna()) < 10:
        return None
    return float(daily_returns.dropna().std(ddof=1) * math.sqrt(trading_days))

def max_drawdown(prices: pd.Series) -> Optional[float]:
    if prices is None or len(prices.dropna()) < 10:
        return None
    p = prices.dropna().astype(float)
    peak = p.cummax()
    dd = (p / peak) - 1.0
    return float(dd.min())

def sma(series: pd.Series, window: int) -> Optional[float]:
    if series is None or len(series.dropna()) < window:
        return None
    return float(series.dropna().rolling(window).mean().iloc[-1])

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
    raise ValueError(f"Unknown quarter: {q}")

# ----------------------------
# FRED client
# ----------------------------

@dataclass
class FredSeries:
    id: str
    name: str
    units: str  # for human reference

class FredClient:
    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key
        self.base = "https://api.stlouisfed.org/fred/series/observations"

    def fetch_series(
        self,
        series_id: str,
        start: dt.date,
        end: dt.date,
        frequency: Optional[str] = None,
    ) -> pd.Series:
        params = {
            "series_id": series_id,
            "api_key": self.api_key or "",
            "file_type": "json",
            "observation_start": iso(start),
            "observation_end": iso(end),
        }
        if frequency:
            params["frequency"] = frequency
        r = requests.get(self.base, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        obs = js.get("observations", [])
        if not obs:
            return pd.Series(dtype=float)
        dates = []
        vals = []
        for o in obs:
            d = o.get("date")
            v = o.get("value")
            try:
                val = float(v)
            except Exception:
                val = np.nan
            dates.append(pd.to_datetime(d))
            vals.append(val)
        s = pd.Series(vals, index=pd.DatetimeIndex(dates)).sort_index()
        return s

    def value_on_or_before(self, series: pd.Series, date: dt.date) -> Optional[float]:
        if series is None or series.empty:
            return None
        idx = series.index
        target = pd.to_datetime(date)
        eligible = series[idx <= target]
        if eligible.empty:
            return None
        v = eligible.dropna().iloc[-1] if not eligible.dropna().empty else None
        return safe_float(v)

# ----------------------------
# Yahoo Finance helpers
# ----------------------------

def yf_history(ticker: str, start: dt.date, end: dt.date, interval: str = "1d") -> pd.DataFrame:
    # yfinance end is exclusive-ish; add one day to be safe
    end_plus = end + dt.timedelta(days=1)
    df = yf.download(
        tickers=ticker,
        start=iso(start),
        end=iso(end_plus),
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    # If multiple tickers returned, yfinance uses multiindex columns. For single ticker it may still.
    if isinstance(df.columns, pd.MultiIndex):
        # pick the first ticker level
        df = df.xs(ticker, axis=1, level=0, drop_level=True)
    return df

def get_market_cap(ticker: str) -> Optional[float]:
    try:
        info = yf.Ticker(ticker).fast_info
        # fast_info sometimes includes market_cap
        mc = info.get("market_cap", None)
        if mc is not None:
            return safe_float(mc)
    except Exception:
        pass
    # fallback
    try:
        info = yf.Ticker(ticker).info
        mc = info.get("marketCap", None)
        return safe_float(mc)
    except Exception:
        return None

# ----------------------------
# S&P 500 constituent list (Wikipedia)
# ----------------------------

def fetch_sp500_tickers() -> List[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, timeout=30).text
    tables = pd.read_html(html)
    # First table has constituents
    df = tables[0]
    syms = df["Symbol"].astype(str).tolist()
    # Wikipedia uses dots for BRK.B; yfinance uses hyphen (BRK-B)
    syms = [s.replace(".", "-") for s in syms]
    # Remove weird artifacts
    syms = [s.strip() for s in syms if s and s.strip()]
    return syms

# ----------------------------
# Layer builders
# ----------------------------

def build_layer1_macro(fred: FredClient, year: int) -> Dict[str, Any]:
    """
    Layer 1: Macro regime (levels + QoQ/YoY deltas)
    We compute per-quarter levels at quarter-end and QoQ/YoY deltas.

    Series map (FRED):
      - DFF: Effective Federal Funds Rate
      - DGS2, DGS10, DGS30: Treasury yields (%)
      - DFII10: 10-Year TIPS (real yield, %)
      - CPIAUCSL: CPI index (we compute YoY % change using last value)
      - CPILFESL: Core CPI index (YoY %)
      - NAPM: ISM PMI (Manufacturing)
      - NAPMNOI: ISM Non-Manufacturing (Services) index (if missing, we log error)
      - BAMLH0A0HYM2: HY OAS (percent points)
      - BAMLH0A0HYM2EY: HY effective yield? (optional)
      - BAMLH0A3HYC: HY default rate proxy? (often not available). We'll try alternative:
          - DRALACBN: Delinquency Rate on All Loans and Leases at Commercial Banks (not HY default)
        Since "HY default rate" is not a clean public FRED series, we treat it as optional:
          - We'll try "BAMLH0A3HYC" (if exists) else "BAMLH0A0HYM2" only and mark missing.
      Liquidity (net liquidity proxy):
        - WALCL (Fed balance sheet)
        - WDTGAL (Treasury General Account)
        - RRPONTSYD (ON RRP)
        NetLiquidity = WALCL - WDTGAL - RRPONTSYD
      Adds:
        - UNRATE (Unemployment)
        - ICSA (Initial claims)
        - DTWEXBGS (Broad dollar index)
        - DCOILWTICO (WTI)
    """
    q_ends = q_end_dates(year)

    series_defs = {
        "L1-MON-FF": FredSeries("DFF", "Fed Funds (EFFR)", "%"),
        "L1-MON-2Y": FredSeries("DGS2", "2Y Treasury", "%"),
        "L1-MON-10Y": FredSeries("DGS10", "10Y Treasury", "%"),
        "L1-MON-30Y": FredSeries("DGS30", "30Y Treasury", "%"),
        "L1-MON-REAL": FredSeries("DFII10", "10Y Real Yield (TIPS)", "%"),
        "L1-INF-CPI_IDX": FredSeries("CPIAUCSL", "CPI Index (SA)", "index"),
        "L1-INF-CORE_IDX": FredSeries("CPILFESL", "Core CPI Index (SA)", "index"),
        "L1-GRW-ISM": FredSeries("NAPM", "ISM Manufacturing PMI", "index"),
        "L1-GRW-SRV": FredSeries("NAPMNOI", "ISM Services (NMI) Index", "index"),
        "L1-CRED-HY_OAS": FredSeries("BAMLH0A0HYM2", "HY OAS", "%"),
        # Optional default proxy (may fail):
        "L1-CRED-HY_DEF": FredSeries("BAMLH0A3HYC", "HY CCC and Lower OAS (proxy)", "%"),
        # Liquidity components:
        "_WALCL": FredSeries("WALCL", "Fed Total Assets", "USD"),
        "_TGA": FredSeries("WDTGAL", "Treasury General Account", "USD"),
        "_RRP": FredSeries("RRPONTSYD", "ON RRP", "USD"),
        # Adds:
        "L1-LAB-UNRATE": FredSeries("UNRATE", "Unemployment Rate", "%"),
        "L1-LAB-ICSA": FredSeries("ICSA", "Initial Jobless Claims", "count"),
        "L1-FX-DXY": FredSeries("DTWEXBGS", "Broad USD Index", "index"),
        "L1-COM-WTI": FredSeries("DCOILWTICO", "WTI Oil", "USD/bbl"),
    }

    # determine pull window: include prior-year Q4 for QoQ, and prior-year same quarter for YoY
    start = dt.date(year - 1, 1, 1)
    end = dt.date(year, 12, 31)

    pulled: Dict[str, pd.Series] = {}
    errors: Dict[str, str] = {}

    for key, sdef in series_defs.items():
        if key.startswith("_"):
            # internal series, still fetched
            pass
        try:
            pulled[key] = fred.fetch_series(sdef.id, start=start, end=end)
        except Exception as e:
            pulled[key] = pd.Series(dtype=float)
            errors[key] = f"fetch_failed: {type(e).__name__}: {e}"

    # compute net liquidity series
    net_liq_series = None
    try:
        walcl = pulled["_WALCL"]
        tga = pulled["_TGA"]
        rrp = pulled["_RRP"]
        # align on daily-ish index
        df = pd.concat([walcl, tga, rrp], axis=1)
        df.columns = ["WALCL", "TGA", "RRP"]
        net = df["WALCL"] - df["TGA"] - df["RRP"]
        net_liq_series = net
    except Exception as e:
        net_liq_series = pd.Series(dtype=float)
        errors["L1-LIQ-NET"] = f"net_liquidity_failed: {type(e).__name__}: {e}"

    def level_q(series_key: str, q: str) -> Optional[float]:
        s = pulled.get(series_key, pd.Series(dtype=float))
        return fred.value_on_or_before(s, q_ends[q])

    def level_q_netliq(q: str) -> Optional[float]:
        if net_liq_series is None or net_liq_series.empty:
            return None
        return fred.value_on_or_before(net_liq_series, q_ends[q])

    def level_on(series_key: str, date: dt.date) -> Optional[float]:
        s = pulled.get(series_key, pd.Series(dtype=float))
        return fred.value_on_or_before(s, date)

    # CPI YoY and Core YoY computed from index
    def yoy_from_index(series_key: str, q: str) -> Optional[float]:
        end_date = q_ends[q]
        prev_date = dt.date(end_date.year - 1, end_date.month, end_date.day)
        v_end = level_on(series_key, end_date)
        v_prev = level_on(series_key, prev_date)
        if v_end is None or v_prev is None:
            return None
        if v_prev == 0:
            return None
        return (v_end / v_prev - 1.0) * 100.0

    out = {
        "layer": "L1",
        "year": year,
        "quarters": {},
        "errors": errors,
        "notes": {
            "curve_10y_2y": "Computed as (10Y - 2Y) in percentage points; bps deltas reflect *100",
            "cpi_yoy": "Computed from CPI index vs same date one year prior",
            "core_cpi_yoy": "Computed from Core CPI index vs same date one year prior",
            "net_liquidity": "Proxy = WALCL - WDTGAL - RRPONTSYD (all USD)",
        },
    }

    for q in ["Q1", "Q2", "Q3", "Q4"]:
        # levels
        ff = level_q("L1-MON-FF", q)
        y2 = level_q("L1-MON-2Y", q)
        y10 = level_q("L1-MON-10Y", q)
        y30 = level_q("L1-MON-30Y", q)
        real10 = level_q("L1-MON-REAL", q)
        curve = None if (y10 is None or y2 is None) else (y10 - y2)
        netliq = level_q_netliq(q)
        cpi_yoy = yoy_from_index("L1-INF-CPI_IDX", q)
        core_yoy = yoy_from_index("L1-INF-CORE_IDX", q)
        ism = level_q("L1-GRW-ISM", q)
        srv = level_q("L1-GRW-SRV", q)
        hy_oas = level_q("L1-CRED-HY_OAS", q)
        hy_def = level_q("L1-CRED-HY_DEF", q)  # proxy / may be None
        unrate = level_q("L1-LAB-UNRATE", q)
        icsa = level_q("L1-LAB-ICSA", q)
        dxy = level_q("L1-FX-DXY", q)
        wti = level_q("L1-COM-WTI", q)

        # deltas
        pq = prev_q_end(q, year)
        yq = dt.date(year - 1, q_ends[q].month, q_ends[q].day)

        def delta_bps(val_key: str, val_now: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
            v_q = level_on(val_key, pq)
            v_y = level_on(val_key, yq)
            return bps(v_q, val_now), bps(v_y, val_now)

        def delta_pp_yoy(val_key: str, val_now: Optional[float]) -> Optional[float]:
            v_y = level_on(val_key, yq)
            if v_y is None or val_now is None:
                return None
            return val_now - v_y

        d10_q, d10_y = delta_bps("L1-MON-10Y", y10)
        d2_q, d2_y = delta_bps("L1-MON-2Y", y2)
        d30_q, d30_y = delta_bps("L1-MON-30Y", y30)
        dreal_q, dreal_y = delta_bps("L1-MON-REAL", real10)
        dcurve_q, dcurve_y = (bps((level_on("L1-MON-10Y", pq) - level_on("L1-MON-2Y", pq)) if (level_on("L1-MON-10Y", pq) is not None and level_on("L1-MON-2Y", pq) is not None) else None,
                                 curve),
                             bps((level_on("L1-MON-10Y", yq) - level_on("L1-MON-2Y", yq)) if (level_on("L1-MON-10Y", yq) is not None and level_on("L1-MON-2Y", yq) is not None) else None,
                                 curve))
        # liquidity delta in USD (not bps)
        netliq_q = fred.value_on_or_before(net_liq_series, pq) if net_liq_series is not None and not net_liq_series.empty else None
        netliq_y = fred.value_on_or_before(net_liq_series, yq) if net_liq_series is not None and not net_liq_series.empty else None
        dnetliq_q = None if (netliq is None or netliq_q is None) else float(netliq - netliq_q)
        dnetliq_y = None if (netliq is None or netliq_y is None) else float(netliq - netliq_y)

        # credit OAS deltas in bps
        dhy_q, dhy_y = delta_bps("L1-CRED-HY_OAS", hy_oas)

        out["quarters"][q] = {
            "asof": iso(q_ends[q]),
            "metrics": {
                "L1-MON-FF": {"value": ff, "units": "%"},
                "L1-MON-2Y": {"value": y2, "units": "%", "delta_bps_qoq": d2_q, "delta_bps_yoy": d2_y},
                "L1-MON-10Y": {"value": y10, "units": "%", "delta_bps_qoq": d10_q, "delta_bps_yoy": d10_y},
                "L1-MON-30Y": {"value": y30, "units": "%", "delta_bps_qoq": d30_q, "delta_bps_yoy": d30_y},
                "L1-MON-CURVE": {"value": curve, "units": "pp", "delta_bps_qoq": dcurve_q, "delta_bps_yoy": dcurve_y},
                "L1-MON-REAL": {"value": real10, "units": "%", "delta_bps_qoq": dreal_q, "delta_bps_yoy": dreal_y},
                "L1-LIQ-NET": {"value": netliq, "units": "USD", "delta_qoq": dnetliq_q, "delta_yoy": dnetliq_y},
                "L1-INF-CPI": {"value": cpi_yoy, "units": "% YoY"},
                "L1-INF-CORE": {"value": core_yoy, "units": "% YoY"},
                "L1-GRW-ISM": {"value": ism, "units": "index"},
                "L1-GRW-SRV": {"value": srv, "units": "index"},
                "L1-CRED-HY": {"value": hy_oas, "units": "%", "delta_bps_qoq": dhy_q, "delta_bps_yoy": dhy_y},
                "L1-CRED-DEF_PROXY": {"value": hy_def, "units": "% (proxy)"},
                # common adds (still Layer 1)
                "L1-LAB-UNRATE": {"value": unrate, "units": "%"},
                "L1-LAB-ICSA": {"value": icsa, "units": "count"},
                "L1-FX-DXY": {"value": dxy, "units": "index"},
                "L1-COM-WTI": {"value": wti, "units": "USD/bbl"},
            },
        }

    return out

def build_layer4_vol(fred: FredClient, year: int) -> Dict[str, Any]:
    """
    Layer 4: Vol surface & risk pricing
      - VIX (VIXCLS)
      - MOVE (MOVE)
      - SKEW (SKEW) if available
      - Equity-bond correlation (computed from SPY and TLT daily returns over last ~60 trading days of each quarter)
    """
    q_ends = q_end_dates(year)
    start = dt.date(year - 1, 1, 1)
    end = dt.date(year, 12, 31)

    series_defs = {
        "L4-VIX": FredSeries("VIXCLS", "VIX", "index"),
        "L4-MOVE": FredSeries("MOVE", "MOVE", "index"),
        "L4-SKEW": FredSeries("SKEW", "CBOE SKEW", "index"),
    }
    pulled: Dict[str, pd.Series] = {}
    errors: Dict[str, str] = {}

    for key, sdef in series_defs.items():
        try:
            pulled[key] = fred.fetch_series(sdef.id, start=start, end=end)
        except Exception as e:
            pulled[key] = pd.Series(dtype=float)
            errors[key] = f"fetch_failed: {type(e).__name__}: {e}"

    def level(series_key: str, q: str) -> Optional[float]:
        return fred.value_on_or_before(pulled.get(series_key, pd.Series(dtype=float)), q_ends[q])

    out = {
        "layer": "L4",
        "year": year,
        "quarters": {},
        "errors": errors,
        "notes": {
            "equity_bond_corr": "Computed from daily log returns of SPY and TLT over last ~60 trading days in the quarter",
        },
    }

    for q in ["Q1", "Q2", "Q3", "Q4"]:
        end_date = q_ends[q]
        start_date = end_date - dt.timedelta(days=120)  # enough to get ~60 trading days
        spy = yf_history("SPY", start_date, end_date)
        tlt = yf_history("TLT", start_date, end_date)
        corr = None
        if not spy.empty and not tlt.empty:
            sret = np.log(spy["Adj Close"].astype(float)).diff()
            tret = np.log(tlt["Adj Close"].astype(float)).diff()
            df = pd.concat([sret, tret], axis=1).dropna()
            df.columns = ["spy", "tlt"]
            if len(df) >= 30:
                corr = float(df["spy"].corr(df["tlt"]))
        out["quarters"][q] = {
            "asof": iso(end_date),
            "metrics": {
                "L4-VIX": {"value": level("L4-VIX", q), "units": "index"},
                "L4-MOVE": {"value": level("L4-MOVE", q), "units": "index"},
                "L4-SKEW": {"value": level("L4-SKEW", q), "units": "index"},
                "L4-CORR": {"value": corr, "units": "corr"},
            },
        }

    return out

def build_layer5_internals(year: int, breadth_sample: int, breadth_max: int) -> Dict[str, Any]:
    """
    Layer 5: Market internals (approx, public-data)
      - % of S&P 500 constituents above 200DMA at each quarter end
      - Concentration proxy: top-10 market cap share of constituents (at quarter end; mcap via yfinance info)
      - Equal-weight underperformance proxy: RSP vs SPY total return over the quarter

    Notes:
      - This uses Wikipedia tickers and yfinance daily prices. It's approximate and can be slow.
      - breadth_sample limits number of tickers used (random-ish deterministic slice).
      - breadth_max caps the number even if sample is large.
    """
    q_ends = q_end_dates(year)
    errors: Dict[str, str] = {}
    out = {
        "layer": "L5",
        "year": year,
        "quarters": {},
        "errors": errors,
        "notes": {
            "breadth_200dma": "Approx: % of sampled SP500 tickers with last close > 200DMA at quarter end",
            "concentration_top10": "Approx: top-10 market cap share using yfinance marketCap (fast_info/info)",
            "equal_weight_vs_cap_weight": "Proxy: RSP (equal-weight SP500 ETF) vs SPY quarterly total return",
        },
        "sampling": {
            "breadth_sample": breadth_sample,
            "breadth_max": breadth_max,
        },
    }

    try:
        tickers = fetch_sp500_tickers()
    except Exception as e:
        errors["SP500_TICKERS"] = f"fetch_failed: {type(e).__name__}: {e}"
        tickers = []

    # deterministic sampling: take every k-th ticker to spread across list
    if tickers:
        n = min(len(tickers), max(1, breadth_sample))
        step = max(1, len(tickers) // n)
        sampled = tickers[::step][: min(breadth_max, n)]
    else:
        sampled = []

    # Pre-fetch daily history for sampled tickers per quarter (needs at least 220 trading days lookback)
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        end_date = q_ends[q]
        start_date = end_date - dt.timedelta(days=420)  # ~1.6 years calendar to ensure 200DMA
        above = 0
        total = 0

        # breadth: compute for sampled tickers
        for tkr in sampled:
            try:
                df = yf_history(tkr, start_date, end_date)
                if df.empty or "Adj Close" not in df.columns:
                    continue
                px = df["Adj Close"].astype(float).dropna()
                if len(px) < 210:
                    continue
                last = float(px.iloc[-1])
                ma200 = float(px.rolling(200).mean().iloc[-1])
                total += 1
                if last > ma200:
                    above += 1
            except Exception:
                continue

        pct_above = None if total == 0 else float(above / total * 100.0)

        # concentration: market cap share (top 10) — we use market cap snapshot (not quarter end exact)
        # This is an approximation; to make it quarter-specific we'd need historical shares/price.
        mcaps: List[Tuple[str, float]] = []
        for tkr in sampled[: min(150, len(sampled))]:
            mc = get_market_cap(tkr)
            if mc is not None and mc > 0:
                mcaps.append((tkr, mc))
        mcaps_sorted = sorted(mcaps, key=lambda x: x[1], reverse=True)
        top10 = mcaps_sorted[:10]
        total_mcap = sum(m for _, m in mcaps_sorted) if mcaps_sorted else 0.0
        top10_share = None if total_mcap <= 0 else float(sum(m for _, m in top10) / total_mcap * 100.0)

        # equal-weight vs cap-weight (RSP vs SPY) quarterly return
        q_start = prev_q_end(q, year) + dt.timedelta(days=1)
        try:
            rsp = yf_history("RSP", q_start - dt.timedelta(days=5), end_date)
            spy = yf_history("SPY", q_start - dt.timedelta(days=5), end_date)
            rsp_ret = None
            spy_ret = None
            diff = None
            if not rsp.empty and not spy.empty:
                rsp_px = rsp["Adj Close"].dropna().astype(float)
                spy_px = spy["Adj Close"].dropna().astype(float)
                if len(rsp_px) >= 2 and len(spy_px) >= 2:
                    rsp_ret = float(rsp_px.iloc[-1] / rsp_px.iloc[0] - 1.0) * 100.0
                    spy_ret = float(spy_px.iloc[-1] / spy_px.iloc[0] - 1.0) * 100.0
                    diff = rsp_ret - spy_ret
        except Exception as e:
            errors[f"L5-EW_{q}"] = f"calc_failed: {type(e).__name__}: {e}"
            rsp_ret = spy_ret = diff = None

        out["quarters"][q] = {
            "asof": iso(end_date),
            "metrics": {
                "L5-200DMA": {"value": pct_above, "units": "%"},
                "L5-CONC": {"value": top10_share, "units": "% (top10 mcap share, approx)"},
                "L5-EW": {
                    "value": diff,
                    "units": "pp (RSP% - SPY%)",
                    "rsp_return_pct": rsp_ret,
                    "spy_return_pct": spy_ret,
                },
            },
        }

    return out

def build_layer6_prices(year: int, tickers: List[str]) -> Dict[str, Any]:
    """
    Layer 6: Daily price data summary + compact bars per quarter for each ticker
      - current price (quarter end close)
      - 60D return
      - 60D vol (annualized)
      - max drawdown over 60D
      - SMA20, SMA50 at quarter end
      - daily bars for the quarter (date, close)
    """
    q_ends = q_end_dates(year)
    out = {
        "layer": "L6",
        "year": year,
        "tickers": {},
        "errors": {},
        "notes": {
            "bars": "Daily bars are quarter-only (date, close).",
            "return_60d": "Computed using last close vs close ~60 trading days prior within lookback window.",
            "vol_60d": "Annualized vol from daily log returns over ~60 trading days.",
            "dd_60d": "Max drawdown over ~60 trading days.",
        },
    }

    for tkr in tickers:
        out["tickers"][tkr] = {"quarters": {}}
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            end_date = q_ends[q]
            q_start = prev_q_end(q, year) + dt.timedelta(days=1)
            # fetch quarter bars plus some lookback for 60D stats
            lookback_start = q_start - dt.timedelta(days=120)
            try:
                df = yf_history(tkr, lookback_start, end_date)
                if df.empty or "Adj Close" not in df.columns:
                    out["tickers"][tkr]["quarters"][q] = {"error": "no_price_data"}
                    continue
                px = df["Adj Close"].dropna().astype(float)
                # quarter-only bars (from q_start)
                qdf = df[df.index >= pd.to_datetime(q_start)]
                bars = []
                if not qdf.empty:
                    closes = qdf["Adj Close"].dropna().astype(float)
                    for d, c in closes.items():
                        bars.append({"timestamp": d.date().isoformat(), "close": float(c)})

                current = float(px.loc[px.index <= pd.to_datetime(end_date)].iloc[-1]) if not px.empty else None

                # 60D stats: use last ~60 trading days up to end_date
                px60 = px.loc[px.index <= pd.to_datetime(end_date)].iloc[-61:]
                ret60 = None
                vol60 = None
                dd60 = None
                sma20v = None
                sma50v = None

                if len(px60) >= 20:
                    ret60 = float(px60.iloc[-1] / px60.iloc[0] - 1.0) * 100.0
                    r = np.log(px60).diff()
                    vol60 = annualized_vol(r)
                    dd60 = max_drawdown(px60)
                # SMA at quarter end using full available history in df
                sma20v = sma(px.loc[px.index <= pd.to_datetime(end_date)], 20)
                sma50v = sma(px.loc[px.index <= pd.to_datetime(end_date)], 50)

                out["tickers"][tkr]["quarters"][q] = {
                    "asof": iso(end_date),
                    "metrics": {
                        "CURRENT_PRICE": {"value": current, "units": "USD"},
                        "PX-RET60": {"value": ret60, "units": "%"},
                        "PX-VOL60": {"value": vol60, "units": "ann_vol"},
                        "PX-DD60": {"value": dd60, "units": "fraction"},
                        "PX-SMA20": {"value": sma20v, "units": "USD"},
                        "PX-SMA50": {"value": sma50v, "units": "USD"},
                    },
                    "daily_bars": bars,
                }
            except Exception as e:
                out["errors"][f"{tkr}:{q}"] = f"calc_failed: {type(e).__name__}: {e}"
                out["tickers"][tkr]["quarters"][q] = {"error": f"{type(e).__name__}: {e}"}

    return out

# ----------------------------
# Orchestrator
# ----------------------------

def build_augmented_market_state(year: int, tickers: List[str], fred_key: Optional[str],
                                 breadth_sample: int, breadth_max: int) -> Dict[str, Any]:
    fred = FredClient(api_key=fred_key)
    doc = {
        "schema_version": "augmented_market_state_v3",
        "generated_at_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "year": year,
        "tickers": tickers,
        "layers": {},
    }

    # Layer 1
    doc["layers"]["L1"] = build_layer1_macro(fred, year)

    # Layer 4
    doc["layers"]["L4"] = build_layer4_vol(fred, year)

    # Layer 5
    doc["layers"]["L5"] = build_layer5_internals(year, breadth_sample=breadth_sample, breadth_max=breadth_max)

    # Layer 6
    doc["layers"]["L6"] = build_layer6_prices(year, tickers=tickers)

    # Basic coverage report (counts)
    # We count metric keys present (not None) for quick debugging.
    coverage = {}
    try:
        # L1
        l1_count = 0
        l1_nonnull = 0
        for q, qobj in doc["layers"]["L1"]["quarters"].items():
            for k, v in qobj["metrics"].items():
                l1_count += 1
                if v.get("value") is not None:
                    l1_nonnull += 1
        coverage["L1"] = {"metric_slots": l1_count, "nonnull": l1_nonnull}

        # L4
        l4_count = 0
        l4_nonnull = 0
        for q, qobj in doc["layers"]["L4"]["quarters"].items():
            for k, v in qobj["metrics"].items():
                l4_count += 1
                if v.get("value") is not None:
                    l4_nonnull += 1
        coverage["L4"] = {"metric_slots": l4_count, "nonnull": l4_nonnull}

        # L5
        l5_count = 0
        l5_nonnull = 0
        for q, qobj in doc["layers"]["L5"]["quarters"].items():
            for k, v in qobj["metrics"].items():
                l5_count += 1
                if v.get("value") is not None:
                    l5_nonnull += 1
        coverage["L5"] = {"metric_slots": l5_count, "nonnull": l5_nonnull}

        # L6 (per ticker)
        l6_count = 0
        l6_nonnull = 0
        for tkr, tobj in doc["layers"]["L6"]["tickers"].items():
            for q, qobj in tobj["quarters"].items():
                if "metrics" not in qobj:
                    continue
                for k, v in qobj["metrics"].items():
                    l6_count += 1
                    if v.get("value") is not None:
                        l6_nonnull += 1
        coverage["L6"] = {"metric_slots": l6_count, "nonnull": l6_nonnull}
    except Exception:
        pass

    doc["coverage"] = coverage
    return doc

# ----------------------------
# CLI
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, required=True, help="Year to build quarters for (e.g., 2025)")
    p.add_argument("--tickers", type=str, default="SPY", help="Comma-separated tickers for Layer 6 (e.g., AAPL,MSFT,NVDA)")
    p.add_argument("--fred-key", type=str, default=None, help="FRED API key (or set env FRED_API_KEY)")
    p.add_argument("--out", type=str, default="augmented_market_state_v3.json", help="Output JSON path")
    p.add_argument("--breadth-sample", type=int, default=150, help="How many SP500 tickers to sample for breadth/concentration")
    p.add_argument("--breadth-max", type=int, default=200, help="Hard cap for sampled tickers")
    return p.parse_args()

def main():
    args = parse_args()
    year = args.year
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    fred_key = args.fred_key or os.environ.get("FRED_API_KEY")

    doc = build_augmented_market_state(
        year=year,
        tickers=tickers,
        fred_key=fred_key,
        breadth_sample=args.breadth_sample,
        breadth_max=args.breadth_max,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"Wrote: {args.out}")

if __name__ == "__main__":
    main()