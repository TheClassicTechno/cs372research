"""Load quarterly memos and snapshot JSONs into Case objects for allocation mode.

Two cases are produced per invest quarter:

  1. **Decision case** — agents see prior-quarter memo + entry prices, debate,
     and output allocation weights.
  2. **Mark-to-market case** — empty context, invest-quarter exit prices. The
     broker updates ``_last_prices`` so that ``_build_summary()`` computes
     correct P&L.

If the invest-quarter snapshot doesn't exist yet (current quarter), only the
decision case is returned and a warning is logged.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from models.case import Case, CaseData, CaseDataItem, StockData

logger = logging.getLogger(__name__)


def prev_quarter(year: int, quarter: str) -> tuple[int, str]:
    """Return the previous calendar quarter.

    >>> prev_quarter(2025, "Q1")
    (2024, 'Q4')
    >>> prev_quarter(2025, "Q3")
    (2025, 'Q2')
    """
    labels = ["Q1", "Q2", "Q3", "Q4"]
    idx = labels.index(quarter)
    if idx == 0:
        return year - 1, "Q4"
    return year, labels[idx - 1]


def _parse_invest_quarter(invest_quarter: str) -> tuple[int, str]:
    """Parse '2025Q1' → (2025, 'Q1')."""
    year = int(invest_quarter[:4])
    q = invest_quarter[4:]
    if q not in ("Q1", "Q2", "Q3", "Q4"):
        raise ValueError(f"Invalid quarter in invest_quarter: {invest_quarter!r}")
    return year, q


def _load_snapshot_json(base_dir: Path, year: int, quarter: str) -> dict | None:
    """Load snapshot JSON, returning None if file doesn't exist."""
    path = base_dir / "json_data" / f"snapshot_{year}_{quarter}.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def _load_memo_text(base_dir: Path, year: int, quarter: str) -> str | None:
    """Load memo text, returning None if file doesn't exist."""
    path = base_dir / "memo_data" / f"memo_{year}_{quarter}.txt"
    if not path.exists():
        return None
    return path.read_text()


def _extract_prices(snapshot: dict, tickers: list[str]) -> dict[str, float]:
    """Extract close prices for the given tickers from a snapshot JSON."""
    ticker_data = snapshot.get("ticker_data", {})
    prices: dict[str, float] = {}
    for t in tickers:
        td = ticker_data.get(t, {})
        af = td.get("asset_features", {})
        close = af.get("close")
        if close is not None:
            prices[t] = float(close)
        else:
            logger.warning("No close price for %s in snapshot — using 0.0", t)
            prices[t] = 0.0
    return prices


def _build_stock_data(
    tickers: list[str],
    prices: dict[str, float],
) -> dict[str, StockData]:
    """Build per-ticker StockData with close price and empty daily_bars."""
    return {
        t: StockData(ticker=t, current_price=prices.get(t, 0.0), daily_bars=[])
        for t in tickers
    }


def load_memo_cases(
    dataset_path: str,
    invest_quarter: str,
    memo_format: str,
    tickers: list[str],
) -> list[Case]:
    """Load decision + mark-to-market cases for memo-based allocation.

    Parameters
    ----------
    dataset_path:
        Path to ``data-pipeline/final_snapshots/`` (or equivalent).
    invest_quarter:
        Quarter the agents are investing for, e.g. ``"2025Q1"``.
        Agents see the prior quarter's data.
    memo_format:
        ``"text"`` to load ``.txt`` memo, ``"json"`` to load snapshot JSON
        and serialize it as context text.
    tickers:
        Allocation universe (from config).

    Returns
    -------
    List of 1–2 Case objects: [decision_case] or [decision_case, mtm_case].
    """
    base_dir = Path(dataset_path)
    inv_year, inv_q = _parse_invest_quarter(invest_quarter)
    prior_year, prior_q = prev_quarter(inv_year, inv_q)

    # --- Load prior-quarter context ---
    if memo_format == "text":
        memo_text = _load_memo_text(base_dir, prior_year, prior_q)
        if memo_text is None:
            raise FileNotFoundError(
                f"Memo not found: {base_dir / 'memo_data' / f'memo_{prior_year}_{prior_q}.txt'}"
            )
        context = memo_text
    elif memo_format == "json":
        prior_snap = _load_snapshot_json(base_dir, prior_year, prior_q)
        if prior_snap is None:
            raise FileNotFoundError(
                f"Snapshot not found: {base_dir / 'json_data' / f'snapshot_{prior_year}_{prior_q}.json'}"
            )
        context = json.dumps(prior_snap, indent=2)
    else:
        raise ValueError(f"Unknown memo_format: {memo_format!r}")

    # Always load prior-quarter snapshot for entry prices
    prior_snap = _load_snapshot_json(base_dir, prior_year, prior_q)
    if prior_snap is None:
        raise FileNotFoundError(
            f"Snapshot JSON required for prices: "
            f"{base_dir / 'json_data' / f'snapshot_{prior_year}_{prior_q}.json'}"
        )

    entry_prices = _extract_prices(prior_snap, tickers)
    as_of = prior_snap.get("as_of_date", f"{prior_year}-{prior_q}")

    # --- Build decision case ---
    decision_case = Case(
        case_data=CaseData(items=[CaseDataItem(kind="other", content=context)]),
        stock_data=_build_stock_data(tickers, entry_prices),
        case_id=f"memo/{invest_quarter}",
        information_cutoff_timestamp=as_of,
    )
    cases = [decision_case]

    logger.info(
        "Loaded decision case for %s (prior=%s%s, %d tickers, as_of=%s)",
        invest_quarter, prior_year, prior_q, len(tickers), as_of,
    )

    # --- Build mark-to-market case (for exit prices / P&L) ---
    inv_snap = _load_snapshot_json(base_dir, inv_year, inv_q)
    if inv_snap is not None:
        exit_prices = _extract_prices(inv_snap, tickers)
        mtm_case = Case(
            case_data=CaseData(items=[]),
            stock_data=_build_stock_data(tickers, exit_prices),
            case_id=f"mtm/{invest_quarter}",
        )
        cases.append(mtm_case)
        logger.info(
            "Loaded MTM case for %s (%d tickers)", invest_quarter, len(tickers),
        )
    else:
        logger.warning(
            "No invest-quarter snapshot for %s%s — skipping MTM case. "
            "P&L will not be computed.",
            inv_year, inv_q,
        )

    return cases
