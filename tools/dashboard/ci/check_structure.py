#!/usr/bin/env python3
"""Structure drift and rewrite guard for dashboard JS files.

Modes:
    --check-structure   Fail if files were added or removed vs baseline.
    --check-rewrite     Fail if any file's line count changed drastically.
    --update            Regenerate the baseline from current state.

Baseline: ci/baselines/structure.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent.parent
JS_ROOT = DASHBOARD_DIR / "static" / "js"
BASELINE_FILE = Path(__file__).resolve().parent / "baselines" / "structure.json"

# Rewrite guard: flag if line count changes by more than this fraction
REWRITE_THRESHOLD = 0.50


def _scan_files() -> dict[str, int]:
    """Return {relative_path: line_count} for all JS files."""
    result = {}
    for path in sorted(JS_ROOT.rglob("*.js")):
        rel = str(path.relative_to(JS_ROOT)).replace("\\", "/")
        result[rel] = len(path.read_text(encoding="utf-8").splitlines())
    return result


def _load_baseline() -> dict[str, int]:
    if not BASELINE_FILE.exists():
        print(f"ERROR: No baseline file at {BASELINE_FILE}")
        print("Run with --update to create one.")
        sys.exit(1)
    data = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    return data["files"]


def _save_baseline(files: dict[str, int]) -> None:
    BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_FILE.write_text(
        json.dumps({"files": files}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def check_structure() -> bool:
    """Compare current file list against baseline. Return True if clean."""
    baseline = _load_baseline()
    current = _scan_files()

    baseline_set = set(baseline.keys())
    current_set = set(current.keys())

    added = sorted(current_set - baseline_set)
    removed = sorted(baseline_set - current_set)

    if not added and not removed:
        print(f"Structure check PASSED ({len(current)} files match baseline)")
        return True

    print("Structure drift detected:")
    for f in added:
        print(f"  + {f}  (added)")
    for f in removed:
        print(f"  - {f}  (removed)")
    print(f"\n{len(added)} added, {len(removed)} removed.")
    print("Run with --update to accept the new structure.")
    return False


def check_rewrite() -> bool:
    """Compare line counts against baseline. Return True if clean."""
    baseline = _load_baseline()
    current = _scan_files()

    violations = []
    for path, cur_lines in current.items():
        if path not in baseline:
            continue
        base_lines = baseline[path]
        if base_lines == 0:
            continue
        delta = abs(cur_lines - base_lines)
        ratio = delta / base_lines
        if ratio > REWRITE_THRESHOLD and delta > 15:
            direction = "grew" if cur_lines > base_lines else "shrank"
            violations.append(
                f"  {path}: {base_lines} -> {cur_lines} lines "
                f"({direction} by {ratio:.0%})"
            )

    if not violations:
        print(f"Rewrite guard PASSED ({len(current)} files within threshold)")
        return True

    print("Potential full rewrites detected (>{:.0%} change):".format(REWRITE_THRESHOLD))
    for v in violations:
        print(v)
    print("\nIf intentional, run with --update to accept new line counts.")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check-structure", action="store_true")
    group.add_argument("--check-rewrite", action="store_true")
    group.add_argument("--update", action="store_true")
    args = parser.parse_args()

    if args.update:
        files = _scan_files()
        _save_baseline(files)
        print(f"Baseline updated: {len(files)} files saved to {BASELINE_FILE.name}")
        sys.exit(0)

    if args.check_structure:
        sys.exit(0 if check_structure() else 1)

    if args.check_rewrite:
        sys.exit(0 if check_rewrite() else 1)


if __name__ == "__main__":
    main()
