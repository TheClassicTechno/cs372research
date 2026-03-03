#!/usr/bin/env python3
"""
Generate agent-readable memo from a quarterly snapshot JSON.

Reads snapshot_{YEAR}_{QUARTER}.json and produces a structured plain-text
memo designed for injection into an agent prompt.

Output:
  final_snapshots/memo_data/memo_{YEAR}_{QUARTER}.txt

Examples:

  # All tickers in snapshot, single quarter
  python generate_quarterly_memo.py --year 2025 --quarter Q1

  # Quarter range, all 19 supported tickers
  python generate_quarterly_memo.py --start 2024Q4 --end 2025Q3 --supported

  # Specific tickers only
  python generate_quarterly_memo.py --start 2025Q1 --end 2025Q3 --tickers AAPL,NVDA,MSFT

  # Custom input/output dirs
  python generate_quarterly_memo.py --year 2025 --quarter Q1 --input-dir /tmp --output-dir /tmp
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_PIPELINE_DIR = _SCRIPT_DIR.parent
_DEFAULT_INPUT_DIR = _SCRIPT_DIR / "json_data"
_DEFAULT_OUTPUT_DIR = _SCRIPT_DIR / "memo_data"
_SUPPORTED_TICKERS_PATH = _PIPELINE_DIR / "supported_tickers.yaml"


def load_supported_tickers() -> List[str]:
    with open(_SUPPORTED_TICKERS_PATH, "r") as f:
        data = yaml.safe_load(f)
    return [t["symbol"] for t in data["supported_tickers"]]


def load_fiscal_year_ends() -> Dict[str, str]:
    """Return {ticker: fiscal_year_end} map, e.g. {"AAPL": "09-27", "NVDA": "01-26"}."""
    with open(_SUPPORTED_TICKERS_PATH, "r") as f:
        data = yaml.safe_load(f)
    return {
        t["symbol"]: t.get("fiscal_year_end", "12-31")
        for t in data["supported_tickers"]
    }


# Month names for FYE display
_MONTH_NAMES = {
    "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr",
    "05": "May", "06": "Jun", "07": "Jul", "08": "Aug",
    "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec",
}


# ==========================
# Quarter utilities
# ==========================

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


# ==========================
# Formatting helpers
# ==========================

def _pct(val: Optional[float], decimals: int = 1) -> str:
    """Signed percent — use for returns and changes."""
    if val is None:
        return "n/a"
    return f"{val * 100:+.{decimals}f}%"


def _pct_level(val: Optional[float], decimals: int = 1) -> str:
    """Unsigned percent — use for levels (vol, margins, rates)."""
    if val is None:
        return "n/a"
    return f"{val * 100:.{decimals}f}%"


def _f(val: Optional[float], decimals: int = 2) -> str:
    if val is None:
        return "n/a"
    return f"{val:.{decimals}f}"


def _delta_bps(val: Optional[float]) -> str:
    if val is None:
        return ""
    sign = "+" if val >= 0 else ""
    return f" ({sign}{val:.0f}bps QoQ)"


def _arrow(val: Optional[float]) -> str:
    if val is None:
        return ""
    if val > 0:
        return " ↑"
    if val < 0:
        return " ↓"
    return ""



# ==========================
# Memo sections
# ==========================

def write_header(lines: list, doc: dict) -> None:
    year = doc["year"]
    quarter = doc["quarter"]
    as_of = doc["as_of_date"]
    n_tickers = len(doc["tickers"])

    lines.append(f"QUARTERLY SNAPSHOT MEMO — {year} {quarter}")
    lines.append(f"As-of date: {as_of}")
    lines.append(f"Tickers: {n_tickers}")
    lines.append("")
    lines.append("This document is the complete data payload for agent deliberation.")
    lines.append("It contains macro regime, per-ticker price/fundamental metrics,")
    lines.append("SEC filing summaries, and news sentiment for the quarter.")
    lines.append("")
    lines.append("Agents must:")
    lines.append("  - Cite evidence IDs (e.g. [L1-10Y], [AAPL-RET60]) when making claims.")
    lines.append("  - Integrate across layers (macro -> filings -> sentiment -> price).")
    lines.append("  - Treat all interpretations as non-binding hypotheses.")


def write_macro(lines: list, macro: Optional[dict]) -> None:
    lines.append("")
    lines.append("=" * 70)
    lines.append("LAYER 1 — MACRO REGIME")
    lines.append("=" * 70)

    if not macro:
        lines.append("  No macro data available.")
        return

    mm = macro.get("macro_metrics", {})
    vm = macro.get("vol_metrics", {})

    # Rates
    lines.append("")
    lines.append("Rates & Yields:")

    ff = mm.get("FF", {})
    if ff.get("value") is not None:
        lines.append(f"  [L1-FF]    Fed Funds: {_f(ff['value'])}%")

    y2 = mm.get("2Y", {})
    if y2.get("value") is not None:
        lines.append(f"  [L1-2Y]    2Y Treasury: {_f(y2['value'])}%{_delta_bps(y2.get('delta_bps_qoq'))}")

    y10 = mm.get("10Y", {})
    if y10.get("value") is not None:
        lines.append(f"  [L1-10Y]   10Y Treasury: {_f(y10['value'])}%{_delta_bps(y10.get('delta_bps_qoq'))}")

    y30 = mm.get("30Y", {})
    if y30.get("value") is not None:
        lines.append(f"  [L1-30Y]   30Y Treasury: {_f(y30['value'])}%")

    curve = mm.get("CURVE", {})
    if curve.get("value") is not None:
        v = curve["value"]
        status = "inverted" if v < 0 else "positive"
        lines.append(f"  [L1-CURVE] 10Y-2Y Spread: {_f(v)}% ({status})")

    real10 = mm.get("REAL10", {})
    if real10.get("value") is not None:
        lines.append(f"  [L1-REAL]  10Y Real Yield: {_f(real10['value'])}%")

    # Inflation
    lines.append("")
    lines.append("Inflation:")

    cpi = mm.get("CPI_YOY", {})
    if cpi.get("value") is not None:
        lines.append(f"  [L1-CPI]   CPI YoY: {_f(cpi['value'])}%")

    core = mm.get("CORE_YOY", {})
    if core.get("value") is not None:
        lines.append(f"  [L1-CORE]  Core CPI YoY: {_f(core['value'])}%")

    # Growth & Labor
    lines.append("")
    lines.append("Growth & Labor:")

    indpro = mm.get("INDPRO", {})
    if indpro.get("value") is not None:
        lines.append(f"  [L1-INDPRO] Industrial Production Index: {_f(indpro['value'])}")

    unrate = mm.get("UNRATE", {})
    if unrate.get("value") is not None:
        lines.append(f"  [L1-UNEMP] Unemployment: {_f(unrate['value'])}%")

    # Credit
    lines.append("")
    lines.append("Credit & Commodities:")

    hy = mm.get("HY_OAS", {})
    if hy.get("value") is not None:
        lines.append(f"  [L1-HY]    HY OAS Spread: {_f(hy['value'])}%")

    dxy = mm.get("DXY", {})
    if dxy.get("value") is not None:
        lines.append(f"  [L1-DXY]   Dollar Index: {_f(dxy['value'])}")

    wti = mm.get("WTI", {})
    if wti.get("value") is not None:
        lines.append(f"  [L1-WTI]   WTI Crude: ${_f(wti['value'])}")

    # Vol
    lines.append("")
    lines.append("Volatility:")

    vix = vm.get("VIX", {})
    if vix.get("value") is not None:
        lines.append(f"  [L1-VIX]   VIX: {_f(vix['value'])}")

    eqbc = vm.get("EQ_BOND_CORR", {})
    if eqbc.get("value") is not None:
        lines.append(f"  [L1-EQBC]  Equity-Bond Correlation: {_f(eqbc['value'])}")


def write_ticker(lines: list, ticker: str, data: dict, fiscal_year_end: str = "12-31") -> None:
    lines.append("")
    lines.append("=" * 70)
    lines.append(f"TICKER: {ticker}")
    lines.append("=" * 70)

    # -- Asset features --
    af = data.get("asset_features")
    if af:
        lines.append("")
        lines.append(f"  Price & Returns:")
        lines.append(f"    [{ticker}-PX]       Close: ${_f(af.get('close'))}")
        lines.append(f"    [{ticker}-RET20]    20D Return: {_pct(af.get('ret_20d'))}")
        lines.append(f"    [{ticker}-RET60]    60D Return: {_pct(af.get('ret_60d'))}")
        lines.append(f"    [{ticker}-RET120]   120D Return: {_pct(af.get('ret_120d'))}")
        lines.append(f"    [{ticker}-RET252]   1Y Return: {_pct(af.get('ret_252d'))}")

        lines.append("")
        lines.append(f"  Volatility & Risk:")
        lines.append(f"    [{ticker}-VOL20]    20D Vol (ann): {_pct_level(af.get('vol_20d'))}")
        lines.append(f"    [{ticker}-VOL60]    60D Vol (ann): {_pct_level(af.get('vol_60d'))}")
        lines.append(f"    [{ticker}-DVOL60]   60D Downside Vol: {_pct_level(af.get('downside_vol_60d'))}")
        lines.append(f"    [{ticker}-DD60]     60D Drawdown: {_pct(af.get('drawdown_60d'))}")
        lines.append(f"    [{ticker}-MDD1Y]    1Y Max Drawdown: {_pct(af.get('max_drawdown_1y'))}")
        lines.append(f"    [{ticker}-SHARPE]   60D Sharpe: {_f(af.get('sharpe_60d'))}")
        lines.append(f"    [{ticker}-BETA]     1Y Beta: {_f(af.get('beta_1y'))}")

        lines.append("")
        lines.append(f"  Trend & Momentum:")
        lines.append(f"    [{ticker}-SMA20]    SMA 20D: ${_f(af.get('sma_20d'))}")
        lines.append(f"    [{ticker}-SMA50]    SMA 50D: ${_f(af.get('sma_50d'))}")
        lines.append(f"    [{ticker}-SMA200]   SMA 200D: ${_f(af.get('sma_200d'))}")
        lines.append(f"    [{ticker}-MOM200]   200D Momentum: {_f(af.get('momentum_200d'))}")
        lines.append(f"    [{ticker}-MOM12_1]  12-1M Momentum: {_pct(af.get('momentum_12_1'))}")
        lines.append(f"    [{ticker}-IDMOM]    Idiosyncratic Momentum 60D: {_f(af.get('idiosyncratic_momentum'), 4)}")
        lines.append(f"    [{ticker}-TREND]    Trend Consistency: {_f(af.get('trend_consistency'))}")
        lines.append(f"    [{ticker}-RS60]     Relative Strength 60D: {_pct(af.get('relative_strength_60d'))}")

        lines.append("")
        lines.append(f"  Fundamentals:")
        lines.append(f"    [{ticker}-MCAP]     Log Market Cap: {_f(af.get('size_log_mcap'))}")
        lines.append(f"    [{ticker}-BM]       Book-to-Market: {_f(af.get('value_book_to_market'), 4)}")
        lines.append(f"    [{ticker}-GM]       Gross Margin: {_pct_level(af.get('gross_margin'))}")
        lines.append(f"    [{ticker}-ROE]      ROE: {_pct(af.get('roe'))}")
        lines.append(f"    [{ticker}-FCFY]     FCF Yield: {_pct(af.get('free_cash_flow_yield'))}")
        lines.append(f"    [{ticker}-DE]       Debt/Equity: {_f(af.get('debt_to_equity'))}")
        lines.append(f"    [{ticker}-ESURP]    Earnings Surprise: {_pct(af.get('earnings_surprise_pct'))}")

        dvol = af.get("avg_dollar_volume_20d")
        if dvol is not None:
            lines.append(f"    [{ticker}-ADVOL]    Avg Dollar Volume 20D: ${dvol:,.0f}")
        else:
            lines.append(f"    [{ticker}-ADVOL]    Avg Dollar Volume 20D: n/a")
    else:
        lines.append("")
        lines.append("  Asset features: not available")

    # -- Sentiment --
    sent = data.get("news_sentiment")
    lines.append("")
    if sent:
        z = sent.get("cross_sectional_z")
        lines.append(f"  Sentiment:")
        lines.append(f"    [{ticker}-SENT]     Mean Sentiment: {_f(sent.get('mean_sentiment'), 4)}")
        lines.append(f"    [{ticker}-SVOL]     Sentiment Volatility: {_f(sent.get('sentiment_volatility'), 4)}")
        surprise = sent.get("surprise_sentiment")
        if surprise is not None:
            lines.append(f"    [{ticker}-SSURP]    Surprise (QoQ): {_f(surprise, 4)}{_arrow(surprise)}")
        else:
            lines.append(f"    [{ticker}-SSURP]    Surprise (QoQ): n/a (first quarter)")
        lines.append(f"    [{ticker}-CSZ]      Cross-Sectional Z: {_f(z, 4)}")
        lines.append(f"    [{ticker}-ACNT]     Article Count: {sent.get('article_count', 'n/a')}")
    else:
        lines.append("  Sentiment: not available")

    # -- Filing summary --
    fs = data.get("filing_summary", {})
    periodic = fs.get("periodic") if fs else None
    events = fs.get("event_filings", []) if fs else []

    lines.append("")
    if periodic:
        form = periodic.get("form", "?")
        fdate = periodic.get("filing_date", "?")
        fiscal = periodic.get("fiscal_period", "")
        # Annotate non-standard fiscal year-end companies
        if fiscal and fiscal_year_end != "12-31":
            month_str = _MONTH_NAMES.get(fiscal_year_end[:2], fiscal_year_end[:2])
            day_str = fiscal_year_end[3:]
            lines.append(f"  Filing Summary ({form}, filed {fdate}, fiscal period: {fiscal} [FYE: {month_str} {day_str}]):")
        elif fiscal:
            lines.append(f"  Filing Summary ({form}, filed {fdate}, fiscal period: {fiscal}):")
        else:
            lines.append(f"  Filing Summary ({form}, filed {fdate}):")

        fields = [
            ("operating_state", "Operations"),
            ("cost_structure", "Costs/Margins"),
            ("material_events", "Material Events"),
            ("macro_exposures", "Macro Exposures"),
            ("forward_outlook", "Forward Outlook"),
            ("uncertainty_profile", "Risks/Uncertainty"),
        ]
        tag_idx = 1
        for field_key, label in fields:
            text = periodic.get(field_key)
            if text:
                lines.append(f"    [{ticker}-F{tag_idx}] {label}: {text}")
                tag_idx += 1
    else:
        lines.append("  Filing summary: not available")

    if events:
        lines.append("")
        lines.append(f"  Recent 8-K Events ({len(events)}):")
        for i, ev in enumerate(events):
            form = ev.get("form", "8-K")
            fdate = ev.get("filing_date", "?")
            lines.append(f"    [{ticker}-EV{i+1}] {form} filed {fdate}")
            for field_key in ("material_events", "operating_state"):
                text = ev.get(field_key)
                if text:
                    lines.append(f"      {text}")
                    break


def write_ticker_summary_table(lines: list, tickers: list, ticker_data: dict) -> None:
    """Compact cross-sectional comparison table."""
    lines.append("")
    lines.append("=" * 70)
    lines.append("CROSS-SECTIONAL SUMMARY")
    lines.append("=" * 70)
    lines.append("")

    header = f"  {'TICKER':<7} {'CLOSE':>8} {'60D RET':>8} {'60D VOL':>8} {'BETA':>6} {'SHARPE':>7} {'SENT_Z':>7} {'GM':>7}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for t in tickers:
        td = ticker_data.get(t, {})
        af = td.get("asset_features") or {}
        sent = td.get("news_sentiment") or {}

        close = af.get("close")
        ret60 = af.get("ret_60d")
        vol60 = af.get("vol_60d")
        beta = af.get("beta_1y")
        sharpe = af.get("sharpe_60d")
        csz = sent.get("cross_sectional_z")
        gm = af.get("gross_margin")

        row = f"  {t:<7}"
        row += f" {_f(close, 2):>8}" if close is not None else f" {'n/a':>8}"
        row += f" {_pct(ret60):>8}" if ret60 is not None else f" {'n/a':>8}"
        row += f" {_pct_level(vol60):>8}" if vol60 is not None else f" {'n/a':>8}"
        row += f" {_f(beta):>6}" if beta is not None else f" {'n/a':>6}"
        row += f" {_f(sharpe):>7}" if sharpe is not None else f" {'n/a':>7}"
        row += f" {_f(csz, 2):>7}" if csz is not None else f" {'n/a':>7}"
        row += f" {_pct_level(gm):>7}" if gm is not None else f" {'n/a':>7}"
        lines.append(row)


# ==========================
# Main memo builder
# ==========================

def build_memo(doc: dict, filter_tickers: Optional[List[str]] = None) -> str:
    lines: list = []

    # Load fiscal year-end data for annotations
    try:
        fye_map = load_fiscal_year_ends()
    except (FileNotFoundError, KeyError):
        fye_map = {}

    # If filter specified, only include those tickers (preserving order)
    all_tickers = doc.get("tickers", [])
    if filter_tickers:
        tickers = [t for t in filter_tickers if t in doc.get("ticker_data", {})]
        missing = [t for t in filter_tickers if t not in doc.get("ticker_data", {})]
        if missing:
            lines.append(f"WARNING: tickers not in snapshot data: {', '.join(missing)}")
            lines.append("")
    else:
        tickers = all_tickers

    # Patch doc for header to reflect filtered count
    filtered_doc = {**doc, "tickers": tickers}

    write_header(lines, filtered_doc)
    write_macro(lines, doc.get("macro_regime"))

    ticker_data = doc.get("ticker_data", {})

    write_ticker_summary_table(lines, tickers, ticker_data)

    for t in tickers:
        td = ticker_data.get(t, {})
        if td:
            fye = td.get("fiscal_year_end") or fye_map.get(t, "12-31")
            write_ticker(lines, t, td, fiscal_year_end=fye)

    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF MEMO")
    lines.append("=" * 70)
    lines.append("")

    return "\n".join(lines)


# ==========================
# CLI
# ==========================

def main():
    p = argparse.ArgumentParser(
        description="Generate agent-readable memo from quarterly snapshot JSON",
    )
    p.add_argument("--year", type=int, default=None,
                   help="Year (single quarter mode)")
    p.add_argument("--quarter", default=None, choices=["Q1", "Q2", "Q3", "Q4"],
                   help="Quarter (single quarter mode)")
    p.add_argument("--start", type=str, default=None,
                   help="Start quarter, e.g. 2024Q4")
    p.add_argument("--end", type=str, default=None,
                   help="End quarter, e.g. 2025Q3")
    p.add_argument("--tickers", type=str, default=None,
                   help="Comma-separated tickers to include (default: all in snapshot)")
    p.add_argument("--supported", action="store_true",
                   help="Use all 19 supported tickers from supported_tickers.yaml")
    p.add_argument("--input-dir", type=str, default=str(_DEFAULT_INPUT_DIR))
    p.add_argument("--output-dir", type=str, default=str(_DEFAULT_OUTPUT_DIR))
    args = p.parse_args()

    if args.start and args.end:
        quarters = quarter_range_list(args.start, args.end)
    elif args.year and args.quarter:
        quarters = [(args.year, args.quarter)]
    else:
        p.error("specify --year/--quarter or --start/--end")

    # Resolve ticker filter
    filter_tickers: Optional[List[str]] = None
    if args.supported:
        filter_tickers = load_supported_tickers()
    elif args.tickers:
        filter_tickers = [t.strip().upper() for t in args.tickers.split(",")]

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    for year, quarter in quarters:
        snapshot_path = input_dir / f"snapshot_{year}_{quarter}.json"
        if not snapshot_path.exists():
            print(f"  SKIP: {snapshot_path} not found", file=sys.stderr)
            continue

        with open(snapshot_path, "r") as f:
            doc = json.load(f)

        memo = build_memo(doc, filter_tickers=filter_tickers)

        out_path = output_dir / f"memo_{year}_{quarter}.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(memo)

        n_tickers = len(filter_tickers) if filter_tickers else len(doc.get("tickers", []))
        print(f"  Wrote: {out_path} ({n_tickers} tickers)")

    print("Done.")


if __name__ == "__main__":
    main()
