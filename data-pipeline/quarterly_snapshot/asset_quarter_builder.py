#!/usr/bin/env python3

import argparse
import datetime as dt
import json
import numpy as np
import pandas as pd
import yfinance as yf


def iso(d): return d.isoformat()

def q_end_dates(year):
    return {
        "Q1": dt.date(year, 3, 31),
        "Q2": dt.date(year, 6, 30),
        "Q3": dt.date(year, 9, 30),
        "Q4": dt.date(year, 12, 31),
    }

def annualized_vol(r):
    if len(r.dropna()) < 10: return None
    return float(r.std() * np.sqrt(252))

def max_drawdown(px):
    peak = px.cummax()
    return float((px / peak - 1).min())

def sma(px, w):
    if len(px) < w: return None
    return float(px.rolling(w).mean().iloc[-1])


def build_asset_state(year, quarter, tickers):

    q_end = q_end_dates(year)[quarter]
    start = q_end - dt.timedelta(days=400)

    out = {
        "schema_version": "asset_state_v1",
        "year": year,
        "quarter": quarter,
        "as_of": iso(q_end),
        "tickers": {}
    }

    for t in tickers:
        df = yf.download(t, start=iso(start), end=iso(q_end), progress=False)

        if df.empty:
            out["tickers"][t] = {"error": "no_data"}
            continue

        px = df["Adj Close"].dropna()
        px = px[px.index <= pd.to_datetime(q_end)]

        if len(px) < 20:
            out["tickers"][t] = {"error": "insufficient_data"}
            continue

        px60 = px.iloc[-61:]

        out["tickers"][t] = {
            "PRICE": float(px.iloc[-1]),
            "RET60": float(px60.iloc[-1] / px60.iloc[0] - 1) * 100,
            "VOL60": annualized_vol(np.log(px60).diff()),
            "DD60": max_drawdown(px60),
            "SMA20": sma(px, 20),
            "SMA50": sma(px, 50),
        }

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--quarter", required=True, choices=["Q1","Q2","Q3","Q4"])
    p.add_argument("--tickers", required=True)
    p.add_argument("--out", default="asset_state.json")
    args = p.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")]

    doc = build_asset_state(args.year, args.quarter, tickers)

    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2)

    print("Wrote:", args.out)


if __name__ == "__main__":
    main()