"""Dataset loading and Case construction.

The case loader reads pre-computed case data from disk as ``Case`` objects
(with placeholder runtime fields) and provides ``build_case`` to stamp in
the live portfolio and IDs at simulation time.

Supported formats
-----------------
* **Directory (with optional sub-directories)**: the loader recursively
  finds all ``.json`` files under the dataset path.  Cases are sorted by
  their path relative to the dataset root, so a structure like
  ``AAPL/2024_Q1.json`` naturally orders cases by ticker, then by quarter.
* **Single JSON-lines file**: each line is one case.
* **Single JSON file**: a JSON array of cases.

On-disk cases need only contain ``case_data`` and ``stock_data``; the
runtime-only fields (``portfolio``, ``case_id``, ``decision_point_idx``)
default to placeholders and are overwritten by ``build_case``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from models.case import Case
from models.portfolio import PortfolioSnapshot

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Building a run-time Case from a loaded template
# ------------------------------------------------------------------

def build_case(
    template: Case,
    portfolio: PortfolioSnapshot,
    case_id: str,
    decision_point_idx: int = 0,
) -> Case:
    """Stamp a template ``Case`` with run-time portfolio and IDs.

    Returns a new ``Case`` via ``model_copy`` — the original is not mutated.
    """
    return template.model_copy(
        update={
            "portfolio": portfolio,
            "case_id": case_id,
            "decision_point_idx": decision_point_idx,
        },
    )


# ------------------------------------------------------------------
# Loading from disk
# ------------------------------------------------------------------

def load_case_templates(
    dataset_path: str,
    top_n_news: int | None = None,
) -> list[Case]:
    """Load cases from *dataset_path*.

    Accepts a directory of ``.json`` files, a single ``.json`` file containing
    a JSON array, or a ``.jsonl`` file (one case per line).  Returns cases in
    decision-point order.

    If *top_n_news* is set, each case's case_data.items are filtered to the
    top N by abs(impact_score) descending.  Items without impact_score are
    included after scored items.  The agent never sees impact_score.
    """
    path = Path(dataset_path)

    if path.is_dir():
        cases = _load_from_directory(path)
    elif path.is_file() and path.suffix in (".jsonl", ".json"):
        if path.suffix == ".jsonl":
            cases = _load_from_jsonl(path)
        else:
            cases = _load_from_json_array(path)
    else:
        raise FileNotFoundError(
            f"Dataset path '{dataset_path}' is neither a directory nor a "
            f"supported file (.json, .jsonl)."
        )

    if top_n_news is not None:
        cases = [_apply_top_n(c, top_n_news) for c in cases]
        logger.info("Applied top_n_news=%d to %d cases.", top_n_news, len(cases))

    return cases


def _apply_top_n(case: Case, n: int) -> Case:
    """Filter case_data.items to top N by abs(impact_score)."""
    items = case.case_data.items
    if len(items) <= n:
        return case

    def sort_key(item):
        score = item.impact_score
        if score is None:
            return (1, 0.0)  # Items without score go last
        return (0, -abs(score))

    sorted_items = sorted(items, key=sort_key)
    filtered = sorted_items[:n]
    return case.model_copy(
        update={"case_data": case.case_data.model_copy(update={"items": filtered})}
    )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _load_from_directory(directory: Path) -> list[Case]:
    """Load one case per JSON file, recursively searching sub-directories.

    Files are sorted by their path relative to *directory* so that a
    per-ticker layout like ``AAPL/2024_Q1.json`` orders cases by ticker
    first, then chronologically within each ticker.
    """
    files = sorted(
        directory.rglob("*.json"),
        key=lambda f: f.relative_to(directory),
    )
    if not files:
        raise FileNotFoundError(f"No .json files found under '{directory}'.")
    cases = [
        Case.model_validate(json.loads(f.read_text(encoding="utf-8")))
        for f in files
    ]
    logger.info("Loaded %d cases from directory '%s'.", len(cases), directory)
    return cases


def _load_from_jsonl(path: Path) -> list[Case]:
    """Load one case per line from a JSON-lines file."""
    cases: list[Case] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            cases.append(Case.model_validate(json.loads(line)))
    logger.info("Loaded %d cases from '%s'.", len(cases), path)
    return cases


def _load_from_json_array(path: Path) -> list[Case]:
    """Load cases from a JSON file containing a list."""
    raw_list = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_list, list):
        raise ValueError(
            f"Expected a JSON array in '{path}', got {type(raw_list).__name__}."
        )
    cases = [Case.model_validate(item) for item in raw_list]
    logger.info("Loaded %d cases from '%s'.", len(cases), path)
    return cases
