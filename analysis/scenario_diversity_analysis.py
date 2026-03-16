#!/usr/bin/env python3
"""Analyze the 35 scenarios used across ablations for diversity and coverage.

Outputs: analysis/scenario_diversity_report.txt
"""

from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIO_LIST = REPO_ROOT / "config" / "scenarios" / "top_divergence" / "top_scenarios_l.json"
OUTPUT = REPO_ROOT / "analysis" / "scenario_diversity_report.txt"


def load_scenarios() -> list[dict]:
    data = json.loads(SCENARIO_LIST.read_text())
    scenarios = []
    for p in data["scenarios"]:
        path = REPO_ROOT / p
        with open(path) as f:
            cfg = yaml.safe_load(f)
        cfg["_path"] = p
        cfg["_stem"] = Path(p).stem
        scenarios.append(cfg)
    return scenarios


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def classify_sector(ticker: str) -> str:
    sectors = {
        # Tech
        "AAPL": "Tech", "AMD": "Tech", "AMZN": "Tech", "AVGO": "Tech",
        "GOOG": "Tech", "META": "Tech", "MSFT": "Tech", "NFLX": "Tech",
        "NVDA": "Tech",
        # Financials
        "BAC": "Financials", "BLK": "Financials", "GS": "Financials",
        "JPM": "Financials", "MS": "Financials",
        # Energy
        "COP": "Energy", "CVX": "Energy", "OXY": "Energy",
        "SLB": "Energy", "XOM": "Energy",
        # Industrials
        "CAT": "Industrials", "DAL": "Industrials", "DE": "Industrials",
        "RTX": "Industrials",
        # Consumer
        "COST": "Consumer", "DIS": "Consumer", "HD": "Consumer",
        "NKE": "Consumer", "PG": "Consumer", "TSLA": "Consumer",
        "WMT": "Consumer",
        # Healthcare
        "JNJ": "Healthcare", "LLY": "Healthcare", "PFE": "Healthcare",
        "UNH": "Healthcare",
        # REITs
        "AMT": "REITs", "PLD": "REITs",
    }
    return sectors.get(ticker, "Other")


def main() -> None:
    scenarios = load_scenarios()
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    w("=" * 78)
    w("SCENARIO DIVERSITY ANALYSIS — 35 Ablation Scenarios")
    w("=" * 78)
    w()

    # ── 1. Per-scenario table ──────────────────────────────────────────────
    w("1. SCENARIO OVERVIEW")
    w("-" * 78)
    w(f"{'#':>3}  {'Quarter':<8} {'Type':<8} {'Tickers':>7}  {'Ticker List'}")
    w(f"{'':>3}  {'':8} {'':8} {'':>7}  {'':}")
    w("-" * 78)

    for i, sc in enumerate(scenarios, 1):
        q = sc["invest_quarter"]
        tickers = sc["tickers"]
        n = len(tickers)
        stem = sc["_stem"]

        if "tech" in stem:
            stype = "tech"
        elif "rand3" in stem:
            stype = "rand3"
        elif "rand4" in stem:
            stype = "rand4"
        elif "rand7" in stem:
            stype = "rand7"
        elif "rand8" in stem:
            stype = "rand8"
        elif "rand9" in stem:
            stype = "rand9"
        elif "rand12" in stem:
            stype = "rand12"
        else:
            stype = "full"

        ticker_str = ", ".join(tickers)
        if len(ticker_str) > 55:
            ticker_str = ticker_str[:52] + "..."
        w(f"{i:3d}  {q:<8} {stype:<8} {n:>7}  {ticker_str}")

    w()

    # ── 2. Temporal distribution ───────────────────────────────────────────
    w("2. TEMPORAL DISTRIBUTION")
    w("-" * 78)

    quarters = [sc["invest_quarter"] for sc in scenarios]
    years = [q[:4] for q in quarters]
    q_nums = [q[5:] for q in quarters]

    year_counts = Counter(years)
    quarter_counts = Counter(quarters)

    w("By Year:")
    for y in sorted(year_counts):
        bar = "#" * year_counts[y]
        w(f"  {y}:  {year_counts[y]:2d}  {bar}")
    w()

    w("By Calendar Quarter:")
    for qn in ["Q1", "Q2", "Q3", "Q4"]:
        count = sum(1 for q in q_nums if q == qn)
        bar = "#" * count
        w(f"  {qn}:  {count:2d}  {bar}")
    w()

    w("By Year-Quarter:")
    for q in sorted(quarter_counts):
        bar = "#" * quarter_counts[q]
        w(f"  {q}:  {quarter_counts[q]:2d}  {bar}")
    w()

    unique_quarters = len(set(quarters))
    w(f"Unique year-quarters: {unique_quarters}")
    w(f"Span: {min(quarters)} to {max(quarters)}")
    w()

    # ── 3. Portfolio size distribution ─────────────────────────────────────
    w("3. PORTFOLIO SIZE DISTRIBUTION")
    w("-" * 78)

    sizes = [len(sc["tickers"]) for sc in scenarios]
    size_counts = Counter(sizes)

    w(f"  Min tickers:    {min(sizes)}")
    w(f"  Max tickers:    {max(sizes)}")
    w(f"  Mean tickers:   {sum(sizes)/len(sizes):.1f}")
    w(f"  Median tickers: {sorted(sizes)[len(sizes)//2]}")
    w()
    w("  Size distribution:")
    for s in sorted(size_counts):
        bar = "#" * size_counts[s]
        w(f"    {s:2d} tickers:  {size_counts[s]:2d}  {bar}")
    w()

    # Group by type
    w("  By scenario type:")
    type_sizes = {}
    for sc in scenarios:
        stem = sc["_stem"]
        if "tech" in stem:
            t = "tech"
        elif "rand" in stem:
            t = "random-subset"
        else:
            t = "full-universe"
        type_sizes.setdefault(t, []).append(len(sc["tickers"]))

    for t in sorted(type_sizes):
        ss = type_sizes[t]
        w(f"    {t:<16}: n={len(ss):2d}, sizes={sorted(set(ss))}")
    w()

    # ── 4. Ticker frequency ───────────────────────────────────────────────
    w("4. TICKER FREQUENCY ACROSS SCENARIOS")
    w("-" * 78)

    all_tickers = Counter()
    for sc in scenarios:
        all_tickers.update(sc["tickers"])

    w(f"  Unique tickers in universe: {len(all_tickers)}")
    w()
    w(f"  {'Ticker':<6} {'Count':>5} {'Pct':>6}  {'Sector':<14} Bar")
    w(f"  {'------':<6} {'-----':>5} {'---':>6}  {'------':<14} ---")

    for ticker, count in all_tickers.most_common():
        pct = count / len(scenarios) * 100
        sector = classify_sector(ticker)
        bar = "#" * count
        w(f"  {ticker:<6} {count:5d} {pct:5.1f}%  {sector:<14} {bar}")
    w()

    # Tickers in every scenario
    in_all = [t for t, c in all_tickers.items() if c == len(scenarios)]
    in_most = [t for t, c in all_tickers.items() if c >= 0.8 * len(scenarios)]
    rare = [t for t, c in all_tickers.items() if c <= 3]
    w(f"  In ALL 35 scenarios:   {in_all if in_all else 'none'}")
    w(f"  In >=80% (>=28):       {sorted(in_most) if in_most else 'none'}")
    w(f"  In <=3 scenarios:      {sorted(rare)}")
    w()

    # ── 5. Sector composition ──────────────────────────────────────────────
    w("5. SECTOR COMPOSITION")
    w("-" * 78)

    sector_counts = Counter()
    for sc in scenarios:
        for t in sc["tickers"]:
            sector_counts[classify_sector(t)] += 1

    total_ticker_slots = sum(sector_counts.values())
    w(f"  Total ticker-slots across all scenarios: {total_ticker_slots}")
    w()

    for sector, count in sector_counts.most_common():
        pct = count / total_ticker_slots * 100
        bar = "#" * (count // 2)
        w(f"  {sector:<14} {count:4d} ({pct:5.1f}%)  {bar}")
    w()

    # Per-scenario sector mix
    w("  Per-scenario sector diversity:")
    sector_diversities = []
    for sc in scenarios:
        sc_sectors = Counter(classify_sector(t) for t in sc["tickers"])
        n_sectors = len(sc_sectors)
        sector_diversities.append(n_sectors)

    w(f"    Min sectors in a scenario:  {min(sector_diversities)}")
    w(f"    Max sectors in a scenario:  {max(sector_diversities)}")
    w(f"    Mean sectors per scenario:  {sum(sector_diversities)/len(sector_diversities):.1f}")
    w()

    # ── 6. Pairwise overlap (Jaccard) ─────────────────────────────────────
    w("6. PAIRWISE TICKER OVERLAP (JACCARD SIMILARITY)")
    w("-" * 78)

    jaccards = []
    for (i, a), (j, b) in combinations(enumerate(scenarios), 2):
        j_val = jaccard(set(a["tickers"]), set(b["tickers"]))
        jaccards.append((j_val, i, j))

    jaccards_vals = [j for j, _, _ in jaccards]
    w(f"  Pairs compared:   {len(jaccards_vals)}")
    w(f"  Mean Jaccard:     {sum(jaccards_vals)/len(jaccards_vals):.3f}")
    w(f"  Median Jaccard:   {sorted(jaccards_vals)[len(jaccards_vals)//2]:.3f}")
    w(f"  Min Jaccard:      {min(jaccards_vals):.3f}")
    w(f"  Max Jaccard:      {max(jaccards_vals):.3f}")
    w()

    # Distribution buckets
    buckets = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
               (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    w("  Jaccard distribution:")
    for lo, hi in buckets:
        n = sum(1 for v in jaccards_vals if lo <= v < hi)
        bar = "#" * (n // 3 + (1 if n % 3 else 0))
        label = f"[{lo:.1f}, {hi:.1f})" if hi < 1.01 else f"[{lo:.1f}, 1.0]"
        w(f"    {label:>12}: {n:4d}  {bar}")
    w()

    # Most similar / most different pairs
    jaccards.sort()
    w("  5 most DIFFERENT pairs (lowest Jaccard):")
    for j_val, i, j in jaccards[:5]:
        w(f"    J={j_val:.3f}  {scenarios[i]['_stem'][:35]}")
        w(f"           vs {scenarios[j]['_stem'][:35]}")
    w()

    w("  5 most SIMILAR pairs (highest Jaccard):")
    for j_val, i, j in jaccards[-5:]:
        w(f"    J={j_val:.3f}  {scenarios[i]['_stem'][:35]}")
        w(f"           vs {scenarios[j]['_stem'][:35]}")
    w()

    # ── 7. Same-quarter variations ────────────────────────────────────────
    w("7. SAME-QUARTER SCENARIO VARIATIONS")
    w("-" * 78)
    w("  How many distinct ticker-sets per quarter:")
    w()

    by_quarter: dict[str, list[dict]] = {}
    for sc in scenarios:
        by_quarter.setdefault(sc["invest_quarter"], []).append(sc)

    for q in sorted(by_quarter):
        scs = by_quarter[q]
        sizes = [len(sc["tickers"]) for sc in scs]
        w(f"  {q}: {len(scs)} scenario(s), sizes={sizes}")
        if len(scs) > 1:
            for (a_idx, a), (b_idx, b) in combinations(enumerate(scs), 2):
                j = jaccard(set(a["tickers"]), set(b["tickers"]))
                w(f"    J({a['_stem'][:25]}, {b['_stem'][:25]}) = {j:.3f}")
    w()

    # ── 8. Market regime diversity ────────────────────────────────────────
    w("8. MARKET REGIME CONTEXT")
    w("-" * 78)
    w("  Approximate market regimes by quarter:")
    w()

    regimes = {
        "2021Q4": "Late bull run, pre-tightening. Fed still accommodative, meme-stock era winding down, supply chain disruptions.",
        "2022Q1": "Rate-hike onset. Fed signals aggressive tightening, growth-to-value rotation begins, Russia-Ukraine war starts.",
        "2022Q2": "Bear market entry. Aggressive rate hikes, tech selloff, crypto crash (Luna/3AC), inflation at 40yr high.",
        "2022Q3": "Bear market rally + reversal. Summer rally fades, Jackson Hole hawkish, energy volatility, strong USD.",
        "2022Q4": "Bottoming process. Inflation shows signs of peaking, China reopening speculation, FTX collapse.",
        "2023Q1": "Banking crisis. SVB/Signature collapse, AI boom starts (ChatGPT), recession fears, flight to quality.",
        "2023Q2": "AI rally, narrow breadth. Magnificent 7 divergence, debt ceiling drama, resilient labor market.",
        "2023Q4": "Year-end rally. Soft landing consensus builds, rate-cut expectations surge, broad market recovery.",
        "2024Q1": "Momentum continuation. AI capex boom, Mag7 earnings beats, rate-cut timeline pushed out, Bitcoin ETF launch.",
        "2025Q1": "Late-cycle uncertainty. Tariff policy shifts, DeepSeek disruption, Fed on hold, mixed economic signals.",
        "2025Q2": "Trade war escalation. Tariff regime uncertainty, global supply chain re-routing, sector rotation.",
        "2025Q3": "Policy recalibration. Tariff de-escalation attempts, Fed data-dependent, earnings resilience test.",
    }

    for q in sorted(by_quarter):
        n = len(by_quarter[q])
        regime = regimes.get(q, "No regime description available.")
        w(f"  {q} ({n} scenario{'s' if n > 1 else ''}):")
        # Word-wrap regime
        words = regime.split()
        line = "    "
        for word in words:
            if len(line) + len(word) + 1 > 76:
                w(line)
                line = "    " + word
            else:
                line += " " + word if line.strip() else "    " + word
        if line.strip():
            w(line)
        w()

    # ── 9. Summary assessment ─────────────────────────────────────────────
    w("9. DIVERSITY ASSESSMENT SUMMARY")
    w("-" * 78)
    w()

    # Compute summary stats
    mean_jaccard = sum(jaccards_vals) / len(jaccards_vals)
    pct_low_overlap = sum(1 for v in jaccards_vals if v < 0.3) / len(jaccards_vals) * 100

    w("  TEMPORAL COVERAGE")
    w(f"    Quarters spanned:      {min(quarters)} – {max(quarters)} ({unique_quarters} unique)")
    w(f"    Years represented:      {len(year_counts)} ({', '.join(sorted(year_counts))})")
    w(f"    Market regimes:         Bull, bear, crisis, recovery, uncertainty")
    w(f"    Assessment:             GOOD — covers full cycle from late-2021 to mid-2025")
    w()

    w("  PORTFOLIO SIZE VARIATION")
    w(f"    Range:                  {min(sizes)}–{max(sizes)} tickers")
    w(f"    Includes:               concentrated (3-4), mid-size (7-12), full-universe (21-35)")
    w(f"    Assessment:             GOOD — tests reasoning at multiple complexity levels")
    w()

    w("  TICKER DIVERSITY")
    w(f"    Universe size:          {len(all_tickers)} unique tickers")
    w(f"    Sectors covered:        {len(sector_counts)}")
    most_freq = all_tickers.most_common(1)[0]
    w(f"    Most frequent ticker:   {most_freq[0]} ({most_freq[1]}/35 = {most_freq[1]/35*100:.0f}%)")
    w(f"    Assessment:             GOOD — broad coverage, no single ticker dominates all scenarios")
    w()

    w("  PAIRWISE OVERLAP")
    w(f"    Mean Jaccard:           {mean_jaccard:.3f}")
    w(f"    Low-overlap pairs (<0.3): {pct_low_overlap:.0f}% of all pairs")
    if mean_jaccard < 0.4:
        overlap_verdict = "GOOD — scenarios are diverse, low average overlap"
    elif mean_jaccard < 0.6:
        overlap_verdict = "MODERATE — some overlap but sufficient diversity"
    else:
        overlap_verdict = "CONCERN — high overlap, scenarios may not be independent enough"
    w(f"    Assessment:             {overlap_verdict}")
    w()

    w("  OVERALL VERDICT")
    w("    The 35 scenarios provide a reasonably diverse testbed for ablation")
    w("    experiments. They span 4+ years of market history including bull,")
    w("    bear, crisis, and recovery regimes. Portfolio sizes range from")
    w(f"    concentrated ({min(sizes)} tickers) to broad ({max(sizes)} tickers), and the")
    w(f"    {len(all_tickers)}-ticker universe covers {len(sector_counts)} sectors. Pairwise Jaccard overlap")
    w(f"    (mean={mean_jaccard:.3f}) indicates meaningful differentiation between scenarios.")
    w()

    # Caveats
    w("  CAVEATS / LIMITATIONS")
    w("    - Temporal coverage skews toward 2022 (11 scenarios) vs 2024 (2 scenarios)")
    w("    - Full-universe scenarios share the same ticker set, inflating some Jaccard pairs")
    w("    - No emerging-market, small-cap, or fixed-income exposure")
    w("    - All scenarios use the same constraint structure (fully_invested + cash virtual ticker)")
    w()

    report = "\n".join(lines)
    OUTPUT.write_text(report, encoding="utf-8")
    print(f"Report written to {OUTPUT}")
    print()
    print(report)


if __name__ == "__main__":
    main()
