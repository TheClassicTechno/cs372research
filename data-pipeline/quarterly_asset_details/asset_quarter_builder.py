#!/usr/bin/env python3
"""
Asset Quarter Builder — Per-ticker price + fundamental features for a quarter.

Fetches price history from Yahoo Finance and computes price, returns,
volatility, drawdown, moving averages, exposure, fundamental, and
cross-sectional metrics for each ticker as-of quarter end.

All rolling windows end exactly on quarter end. No future leakage.

Output:
  data-pipeline/quarterly_asset_details/data/assets_{YEAR}_{QUARTER}.json

Examples:

  # Single quarter
  python asset_quarter_builder.py --year 2025 --quarter Q2 --tickers AAPL,NVDA

  # Quarter range with supported tickers
  python asset_quarter_builder.py --start 2024Q4 --end 2025Q3 --supported

  # Custom output path (single quarter only)
  python asset_quarter_builder.py --year 2025 --quarter Q2 --tickers AAPL --out /tmp/assets.json
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA_DIR = _SCRIPT_DIR / "data"
_SUPPORTED_TICKERS_PATH = _SCRIPT_DIR.parent / "supported_tickers.yaml"


def load_supported_tickers(yaml_path: Path = _SUPPORTED_TICKERS_PATH) -> list:
    """Load ticker symbols from supported_tickers.yaml."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return [entry["symbol"] for entry in data["supported_tickers"]]


def iso(d): return d.isoformat()

def q_end_dates(year):
    """Quarter end dates — all rolling windows must end on these dates."""
    return {
        "Q1": dt.date(year, 3, 31),
        "Q2": dt.date(year, 6, 30),
        "Q3": dt.date(year, 9, 30),
        "Q4": dt.date(year, 12, 31),
    }

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
# Price metric helpers
# ----------------------------

def _ret(px: pd.Series, window: int) -> Optional[float]:
    """Simple return over `window` trading days. Decimal form."""
    if len(px) < window + 1:
        return None
    return float(px.iloc[-1] / px.iloc[-(window + 1)] - 1)


def _ann_vol(px: pd.Series, window: int) -> Optional[float]:
    """Annualized volatility from log returns over `window` trading days."""
    if len(px) < window + 1:
        return None
    log_ret = np.log(px.iloc[-(window + 1):]).diff().dropna()
    if len(log_ret) < 10:
        return None
    return float(log_ret.std() * np.sqrt(252))


def _downside_vol(px: pd.Series, window: int) -> Optional[float]:
    """Annualized std dev of NEGATIVE daily log returns only.

    Measures downside risk — volatility contributed only by losing days.
    Annualized with sqrt(252).
    """
    if len(px) < window + 1:
        return None
    log_ret = np.log(px.iloc[-(window + 1):]).diff().dropna()
    neg_ret = log_ret[log_ret < 0]
    if len(neg_ret) < 5:
        return None
    return float(neg_ret.std() * np.sqrt(252))


def _sma(px: pd.Series, window: int) -> Optional[float]:
    """Simple moving average over `window` trading days."""
    if len(px) < window:
        return None
    return float(px.iloc[-window:].mean())


def _max_drawdown(px: pd.Series) -> Optional[float]:
    """Maximum drawdown (decimal, negative) over the given price series."""
    if len(px) < 2:
        return None
    peak = px.cummax()
    return float((px / peak - 1).min())


def _sharpe(px: pd.Series, window: int) -> Optional[float]:
    """Annualized Sharpe ratio over `window` trading days, risk-free = 0.

    sharpe = (mean_daily_logret / std_daily_logret) * sqrt(252)
    """
    if len(px) < window + 1:
        return None
    log_ret = np.log(px.iloc[-(window + 1):]).diff().dropna()
    if len(log_ret) < 10 or log_ret.std() == 0:
        return None
    return float((log_ret.mean() / log_ret.std()) * np.sqrt(252))


def _beta(ticker_px: pd.Series, spy_px: pd.Series, window: int) -> Optional[float]:
    """OLS beta vs SPY over `window` trading days.

    beta = cov(r_ticker, r_spy) / var(r_spy)
    """
    if len(ticker_px) < window + 1 or len(spy_px) < window + 1:
        return None
    t_ret = np.log(ticker_px.iloc[-(window + 1):]).diff().dropna()
    s_ret = np.log(spy_px.iloc[-(window + 1):]).diff().dropna()
    aligned = pd.concat([t_ret, s_ret], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        return None
    y = aligned.iloc[:, 0].values  # ticker
    x = aligned.iloc[:, 1].values  # SPY
    cov = np.cov(y, x, ddof=1)
    if cov[1, 1] == 0:
        return None
    return float(cov[0, 1] / cov[1, 1])


def _idiosyncratic_momentum(
    ticker_px: pd.Series, spy_px: pd.Series, window: int,
) -> Optional[float]:
    """Residual return after removing beta component over `window` days.

    Procedure:
      1. Regress daily ticker returns on SPY returns (OLS with intercept)
      2. Compute residual series: e_t = r_ticker_t - (a + b * r_spy_t)
      3. Return sum of residuals = cumulative idiosyncratic return
    """
    if len(ticker_px) < window + 1 or len(spy_px) < window + 1:
        return None
    t_ret = np.log(ticker_px.iloc[-(window + 1):]).diff().dropna()
    s_ret = np.log(spy_px.iloc[-(window + 1):]).diff().dropna()
    aligned = pd.concat([t_ret, s_ret], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        return None
    y = aligned.iloc[:, 0].values
    x = aligned.iloc[:, 1].values
    # OLS: y = alpha + beta * x + epsilon
    X = np.column_stack([np.ones(len(x)), x])
    try:
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ coeffs
        return float(residuals.sum())
    except np.linalg.LinAlgError:
        return None


def _momentum_12_1(px: pd.Series) -> Optional[float]:
    """12-month return skipping the most recent month (21 trading days).

    momentum_12_1 = price_21d_ago / price_274d_ago - 1
    Requires ~274 trading days of data. Decimal form.
    """
    if len(px) < 253 + 21:
        return None
    return float(px.iloc[-22] / px.iloc[-253] - 1)


def _trend_consistency(px: pd.Series, window: int) -> Optional[float]:
    """Fraction of positive daily return days over `window` trading days.

    Values near 1.0 = strong uptrend, near 0.0 = strong downtrend,
    near 0.5 = no trend / choppy.
    """
    if len(px) < window + 1:
        return None
    log_ret = np.log(px.iloc[-(window + 1):]).diff().dropna()
    if len(log_ret) == 0:
        return None
    return float((log_ret > 0).sum() / len(log_ret))


def _avg_dollar_volume(
    close: pd.Series, volume: pd.Series, window: int,
) -> Optional[float]:
    """Mean daily dollar volume (close * volume) over `window` trading days."""
    dv = (close * volume).dropna()
    if len(dv) < window:
        return None
    return float(dv.iloc[-window:].mean())


def _round_or_none(val: Optional[float], decimals: int) -> Optional[float]:
    """Round a value, returning None if input is None or NaN."""
    if val is None:
        return None
    if np.isnan(val) or np.isinf(val):
        return None
    return round(val, decimals)


# ----------------------------
# Fundamental data helpers
# ----------------------------

def _safe_get(df: pd.DataFrame, col, row_names: list) -> Optional[float]:
    """Try multiple row names to extract a value from a financial statement column."""
    if df is None or df.empty or col is None:
        return None
    for name in row_names:
        if name in df.index:
            val = df.loc[name, col]
            if pd.notna(val):
                return float(val)
    return None


def _latest_valid_col(df: pd.DataFrame, cutoff: pd.Timestamp):
    """Return the most recent column date <= cutoff from a financial statement."""
    if df is None or df.empty:
        return None
    valid = sorted([c for c in df.columns if c <= cutoff], reverse=True)
    return valid[0] if valid else None


def _fetch_fundamentals(ticker_symbol: str, q_end: dt.date) -> dict:
    """Fetch fundamental data from yfinance financial statements.

    Uses quarterly_income_stmt, quarterly_balance_sheet, quarterly_cashflow,
    and earnings_dates. All filtered to fiscal periods <= q_end for
    point-in-time safety.

    Returns dict with: shares_outstanding, book_value_per_share,
    gross_margin, roe, fcf, debt_to_equity, earnings_surprise_pct.
    """
    result: Dict[str, Optional[float]] = {}
    cutoff = pd.Timestamp(q_end)

    try:
        yt = yf.Ticker(ticker_symbol)
    except Exception:
        return result

    # --- Financial statements (quarterly, dated columns) ---
    try:
        income = yt.quarterly_income_stmt
    except Exception:
        income = None
    try:
        balance = yt.quarterly_balance_sheet
    except Exception:
        balance = None
    try:
        cashflow = yt.quarterly_cashflow
    except Exception:
        cashflow = None

    inc_col = _latest_valid_col(income, cutoff)
    bal_col = _latest_valid_col(balance, cutoff)

    # --- Shares + book value from balance sheet (point-in-time safe) ---
    shares = _safe_get(balance, bal_col, [
        "Ordinary Shares Number", "Share Issued",
        "Common Stock Shares Outstanding",
    ])
    result["shares_outstanding"] = shares

    equity = _safe_get(balance, bal_col, [
        "Stockholders Equity", "Common Stock Equity",
        "Total Equity Gross Minority Interest",
    ])
    if equity is not None and shares is not None and shares > 0:
        result["book_value_per_share"] = equity / shares
    else:
        result["book_value_per_share"] = None

    cf_col = _latest_valid_col(cashflow, cutoff)

    # Gross margin = gross_profit / revenue
    gp = _safe_get(income, inc_col, ["Gross Profit"])
    rev = _safe_get(income, inc_col, ["Total Revenue", "Revenue"])
    if gp is not None and rev is not None and rev != 0:
        result["gross_margin"] = gp / rev

    # ROE = net_income / stockholder_equity
    ni = _safe_get(income, inc_col, [
        "Net Income", "Net Income Common Stockholders",
    ])
    if ni is not None and equity is not None and equity != 0:
        result["roe"] = ni / equity

    # Debt to equity = total_debt / stockholder_equity
    debt = _safe_get(balance, bal_col, ["Total Debt", "Long Term Debt"])
    if debt is not None and equity is not None and equity != 0:
        result["debt_to_equity"] = debt / equity

    # Free cash flow (try direct row, fall back to operating - capex)
    fcf = _safe_get(cashflow, cf_col, ["Free Cash Flow"])
    if fcf is None:
        opcf = _safe_get(cashflow, cf_col, [
            "Operating Cash Flow",
            "Cash Flow From Continuing Operating Activities",
        ])
        capex = _safe_get(cashflow, cf_col, [
            "Capital Expenditure", "Capital Expenditures",
        ])
        if opcf is not None and capex is not None:
            fcf = opcf + capex  # capex is typically negative in yfinance
    result["fcf"] = fcf

    # --- Earnings surprise ---
    # (actual_eps - estimated_eps) / |estimated_eps|
    # Uses most recent earnings report dated <= q_end.
    # Fallback: None if analyst estimate unavailable.
    try:
        ed = yt.earnings_dates
        if ed is not None and not ed.empty:
            # Normalize timezone for comparison
            ed_idx = ed.index.tz_localize(None) if ed.index.tz else ed.index
            valid_mask = ed_idx <= cutoff
            valid = ed[valid_mask]
            if not valid.empty:
                row = valid.iloc[0]  # most recent
                actual = row.get("Reported EPS")
                estimate = row.get("EPS Estimate")
                if (pd.notna(actual) and pd.notna(estimate)
                        and float(estimate) != 0):
                    result["earnings_surprise_pct"] = (
                        (float(actual) - float(estimate)) / abs(float(estimate))
                    )
    except Exception:
        pass

    return result


# ----------------------------
# Quarter builder
# ----------------------------

def build_asset_state(year, quarter, tickers):

    q_end = q_end_dates(year)[quarter]
    # 600-day lookback covers 252 trading days + 21-day skip for momentum_12_1
    start = q_end - dt.timedelta(days=600)

    out = {
        "schema_version": "asset_state_v2",
        "year": year,
        "quarter": quarter,
        "as_of": iso(q_end),
        "metadata": {
            "frequency": "daily",
            "vol_annualized": True,
            "return_units": "decimal",
            "windows": {
                "ret_20d": 20,
                "ret_60d": 60,
                "ret_120d": 120,
                "ret_252d": 252,
            },
        },
        "tickers": {}
    }

    print(f"Building asset state for {year} {quarter} (as-of {iso(q_end)})", flush=True)
    print(f"  {len(tickers)} ticker(s): {', '.join(tickers)}", flush=True)

    # ── Batch price download ──────────────────────────────────────────
    # yfinance has a global state race condition that corrupts results
    # when parallel yf.download() calls run concurrently, so we do one
    # batch call and slice per-ticker afterwards.
    all_symbols = list(tickers) + (["SPY"] if "SPY" not in tickers else [])
    print(f"  Downloading price data ({len(all_symbols)} symbols)...", flush=True)
    try:
        raw = yf.download(
            all_symbols, start=iso(start), end=iso(q_end),
            group_by="ticker", progress=False,
        )
    except Exception as e:
        print(f"  FAIL batch download — {type(e).__name__}: {e}", flush=True)
        raw = pd.DataFrame()

    # Extract SPY prices for beta / idiosyncratic momentum calculations
    spy_px = None
    if not raw.empty:
        try:
            if len(all_symbols) == 1:
                spy_df = raw.copy()
                if isinstance(spy_df.columns, pd.MultiIndex):
                    spy_df.columns = spy_df.columns.get_level_values(0)
            else:
                spy_df = raw["SPY"]
            spy_px = spy_df["Close"].dropna()
            spy_px = spy_px[spy_px.index <= pd.to_datetime(q_end)]
        except Exception:
            tqdm.write("  [SPY] WARN — could not extract SPY data for beta calc")

    # ── Fetch fundamentals ────────────────────────────────────────────
    # Financial statements, earnings dates, shares outstanding.
    # Sequential — yfinance Ticker objects are not thread-safe.
    print(f"  Fetching fundamentals...", flush=True)
    ticker_funds: Dict[str, dict] = {}
    for t in tqdm(tickers, desc="Fundamentals", unit="ticker", leave=False):
        try:
            ticker_funds[t] = _fetch_fundamentals(t, q_end)
        except Exception:
            ticker_funds[t] = {}

    # ── Per-ticker metric computation ─────────────────────────────────
    for t in tqdm(tickers, desc="Computing metrics", unit="ticker"):
        try:
            if raw.empty:
                tqdm.write(f"  [{t}] WARN — no price data (batch download failed)")
                out["tickers"][t] = {"error": "no_data"}
                continue

            # Slice this ticker's DataFrame from the batch download
            if len(all_symbols) == 1:
                df = raw.copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
            else:
                try:
                    df = raw[t].copy()
                except KeyError:
                    tqdm.write(f"  [{t}] WARN — ticker not in download results")
                    out["tickers"][t] = {"error": "no_data"}
                    continue

            if df.empty or df["Close"].dropna().empty:
                tqdm.write(f"  [{t}] WARN — no price data")
                out["tickers"][t] = {"error": "no_data"}
                continue

            # ── Filter to quarter end (no future leakage) ─────────
            px = df["Close"].dropna()
            px = px[px.index <= pd.to_datetime(q_end)]
            vol_series = df["Volume"].dropna()
            vol_series = vol_series[vol_series.index <= pd.to_datetime(q_end)]

            if len(px) < 20:
                tqdm.write(f"  [{t}] WARN — insufficient data ({len(px)} points)")
                out["tickers"][t] = {"error": "insufficient_data"}
                continue

            close = float(px.iloc[-1])
            funds = ticker_funds.get(t, {})

            # ── Moving averages & momentum ────────────────────────
            sma_200 = _sma(px, 200)
            # momentum_200d = close / sma_200d (price relative to trend)
            momentum_200d = None
            if sma_200 is not None and sma_200 != 0:
                momentum_200d = close / sma_200

            # ── Drawdown windows ──────────────────────────────────
            dd_60_px = px.iloc[-61:] if len(px) >= 61 else px
            dd_1y_px = px.iloc[-253:] if len(px) >= 253 else px

            # ── Exposure: beta vs SPY (252 trading days) ──────────
            beta_1y = _beta(px, spy_px, 252) if spy_px is not None else None

            # ── Exposure: log market cap = ln(close * shares) ─────
            size_log_mcap = None
            shares = funds.get("shares_outstanding")
            if shares and close:
                size_log_mcap = float(np.log(close * shares))

            # ── Exposure: book-to-market = book_per_share / close ─
            value_book_to_market = None
            bvps = funds.get("book_value_per_share")
            if bvps and close and close > 0:
                value_book_to_market = float(bvps / close)

            # ── Free cash flow yield = FCF / market_cap ───────────
            fcf_yield = None
            fcf = funds.get("fcf")
            if fcf is not None and shares and close and close > 0:
                mcap = close * shares
                if mcap > 0:
                    fcf_yield = fcf / mcap

            # ── Build result dict ─────────────────────────────────
            # Rounding happens only here at serialization stage.
            out["tickers"][t] = {
                # Price
                "close":                _round_or_none(close, 2),
                # Returns (decimal, 4dp)
                "ret_20d":              _round_or_none(_ret(px, 20), 4),
                "ret_60d":              _round_or_none(_ret(px, 60), 4),
                "ret_120d":             _round_or_none(_ret(px, 120), 4),
                "ret_252d":             _round_or_none(_ret(px, 252), 4),
                # Volatility (annualized decimal, 4dp)
                "vol_20d":              _round_or_none(_ann_vol(px, 20), 4),
                "vol_60d":              _round_or_none(_ann_vol(px, 60), 4),
                "vol_120d":             _round_or_none(_ann_vol(px, 120), 4),
                "downside_vol_60d":     _round_or_none(_downside_vol(px, 60), 4),
                # Drawdown (decimal, 4dp)
                "drawdown_60d":         _round_or_none(_max_drawdown(dd_60_px), 4),
                "max_drawdown_1y":      _round_or_none(_max_drawdown(dd_1y_px), 4),
                # Moving averages (2dp)
                "sma_20d":              _round_or_none(_sma(px, 20), 2),
                "sma_50d":              _round_or_none(_sma(px, 50), 2),
                "sma_200d":             _round_or_none(sma_200, 2),
                # Momentum (4dp)
                "momentum_200d":        _round_or_none(momentum_200d, 4),
                "momentum_12_1":        _round_or_none(_momentum_12_1(px), 4),
                "idiosyncratic_momentum": _round_or_none(
                    _idiosyncratic_momentum(px, spy_px, 60)
                    if spy_px is not None else None, 4,
                ),
                "trend_consistency":    _round_or_none(_trend_consistency(px, 60), 4),
                # Risk-adjusted (4dp)
                "sharpe_60d":           _round_or_none(_sharpe(px, 60), 4),
                # Exposure (4dp)
                "beta_1y":              _round_or_none(beta_1y, 4),
                "size_log_mcap":        _round_or_none(size_log_mcap, 4),
                "value_book_to_market": _round_or_none(value_book_to_market, 4),
                # Liquidity (2dp — dollar value)
                "avg_dollar_volume_20d": _round_or_none(
                    _avg_dollar_volume(px, vol_series, 20), 2,
                ),
                # Fundamentals (4dp — ratios)
                "gross_margin":         _round_or_none(funds.get("gross_margin"), 4),
                "roe":                  _round_or_none(funds.get("roe"), 4),
                "free_cash_flow_yield": _round_or_none(fcf_yield, 4),
                "debt_to_equity":       _round_or_none(funds.get("debt_to_equity"), 4),
                "earnings_surprise_pct": _round_or_none(
                    funds.get("earnings_surprise_pct"), 4,
                ),
                }
            tqdm.write(f"  [{t}] OK — ${close:.2f}")

        except Exception as e:
            tqdm.write(f"  [{t}] FAIL — {type(e).__name__}: {e}")
            out["tickers"][t] = {"error": str(e)}

    ok = sum(1 for v in out["tickers"].values() if "error" not in v)
    print(f"Asset state complete: {ok}/{len(tickers)} tickers OK")
    return out


# ----------------------------
# CLI
# ----------------------------

def main():
    p = argparse.ArgumentParser(
        description="Build per-ticker asset state for one or more quarters",
    )
    # Single quarter mode
    p.add_argument("--year", type=int, default=None,
                   help="Year (single quarter mode)")
    p.add_argument("--quarter", default=None, choices=["Q1","Q2","Q3","Q4"],
                   help="Quarter (single quarter mode)")
    # Range mode
    p.add_argument("--start", type=str, default=None,
                   help="Start quarter, e.g. 2024Q4")
    p.add_argument("--end", type=str, default=None,
                   help="End quarter, e.g. 2025Q3")
    # Tickers
    p.add_argument("--tickers", default=None,
                   help="Comma-separated tickers")
    p.add_argument("--supported", action="store_true", default=False,
                   help="Use all tickers from supported_tickers.yaml")
    args = p.parse_args()

    # Resolve quarters
    if args.start and args.end:
        quarters = quarter_range(args.start, args.end)
    elif args.year and args.quarter:
        quarters = [(args.year, args.quarter)]
    else:
        p.error("specify --year/--quarter or --start/--end")

    # Resolve tickers
    if args.supported:
        tickers = load_supported_tickers()
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        p.error("either --tickers or --supported is required")

    print(f"Building asset state for {len(quarters)} quarter(s), "
          f"{len(tickers)} ticker(s)", flush=True)

    for year, quarter in tqdm(quarters, desc="Quarters", unit="quarter"):
        doc = build_asset_state(year, quarter, tickers)
        as_of = doc["as_of"]

        for t in tickers:
            ticker_dir = _DEFAULT_DATA_DIR / t
            ticker_dir.mkdir(parents=True, exist_ok=True)
            out_path = ticker_dir / f"{year}_{quarter}.json"

            ticker_data = doc["tickers"].get(t)
            wrapped = {
                "schema_version": "asset_state_v2",
                "ticker": t,
                "year": year,
                "quarter": quarter,
                "as_of": as_of,
                "features": ticker_data,
            }
            try:
                from provenance import inline_provenance
                wrapped.update(inline_provenance())
            except ImportError:
                pass
            with open(out_path, "w") as f:
                json.dump(wrapped, f, indent=2)

        tqdm.write(f"  Wrote {len(tickers)} ticker file(s) for {year} {quarter}")

    print(f"Done: {len(quarters)} quarter(s) complete.")


if __name__ == "__main__":
    main()
