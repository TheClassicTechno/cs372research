#!/usr/bin/env python3
"""
USAGE EXAMPLES
==============

1) Basic runtime validation (read-only; no schema required; no backup; no file modification):
   python runtime_validate_debate_json.py --input_file examples/nvda_posthoc_example_v2.json

2) Dry run (read-only; prints what it WOULD do; no backup; no writes):
   python runtime_validate_debate_json.py --input_file examples/nvda_posthoc_example_v2.json --dry_run

3) Normalize ordering/format in-place (canonical JSON formatting) WITHOUT backup (backups off by default):
   python runtime_validate_debate_json.py --input_file examples/nvda_posthoc_example_v2.json --normalize

4) Normalize in-place WITH backup (creates timestamped backup, then writes):
   python runtime_validate_debate_json.py --input_file examples/nvda_posthoc_example_v2.json --normalize --create_backup

5) Normalize + dry run + backup flag (shows backup path it WOULD create; no writes):
   python runtime_validate_debate_json.py --input_file examples/nvda_posthoc_example_v2.json --normalize --create_backup --dry_run

6) Quiet mode (suppress non-essential narration):
   python runtime_validate_debate_json.py --input_file examples/nvda_posthoc_example_v2.json --quiet
"""

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ============================
# OPTIONAL HARD-CODED OVERRIDES
# ============================
# If these constants are set (not None), they override command-line arguments.
# If they are None, argparse values are used instead.

INPUT_FILE: Optional[str] = None


# ============================
# ERROR CODES (STABLE CONTRACT)
# ============================

E_SPEAKER_UNKNOWN = "E_SPEAKER_UNKNOWN"
E_DUPLICATE_TURN_INDEX = "E_DUPLICATE_TURN_INDEX"
E_NEGATIVE_TURN_INDEX = "E_NEGATIVE_TURN_INDEX"
E_DUPLICATE_TURN_ID = "E_DUPLICATE_TURN_ID"
E_TURN_ID_EMPTY = "E_TURN_ID_EMPTY"
E_SPEAKER_ID_EMPTY = "E_SPEAKER_ID_EMPTY"
E_ATTEMPT_INDEX_INVALID = "E_ATTEMPT_INDEX_INVALID"
E_RECOMMENDATION_RANGE_INVALID = "E_RECOMMENDATION_RANGE_INVALID"
E_MODE_MISMATCH = "E_MODE_MISMATCH"

W_NO_STRUCTURED_FINAL_ANSWER = "W_NO_STRUCTURED_FINAL_ANSWER"
W_ROUND_INDEX_LT_1 = "W_ROUND_INDEX_LT_1"
W_ATTEMPTS_PRESENT_POSTHOC = "W_ATTEMPTS_PRESENT_POSTHOC"
W_TURN_INDEX_NOT_MONOTONIC = "W_TURN_INDEX_NOT_MONOTONIC"


@dataclass
class Finding:
    code: str
    severity: str  # "HARD" | "SOFT" | "INFO"
    message: str
    path: str


@dataclass
class RunMetrics:
    turns_processed: int = 0
    attempts_processed: int = 0
    errors: List[Finding] = field(default_factory=list)
    warnings: List[Finding] = field(default_factory=list)
    normalized: bool = False
    modified_files: int = 0
    skipped_files: int = 0


def vprint(message: str, quiet: bool) -> None:
    if not quiet:
        print(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Runtime validator for debate artifacts: semantic integrity checks not enforceable by JSON Schema. "
                    "Optionally normalize JSON formatting in-place."
    )

    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to the input debate JSON file to validate."
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, no files are modified and no backups are written. Prints what WOULD happen."
    )

    parser.add_argument(
        "--create_backup",
        action="store_true",
        help="If set AND an in-place edit is performed, create a timestamped backup in the same directory before writing."
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
        help="If set, rewrite the input JSON in-place using canonical formatting (indent=2, ensure_ascii=False). "
             "This is the ONLY transform this script performs."
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="If set, suppress non-essential verbose narration."
    )

    # Optional knobs that often matter in practice. Safe defaults.
    parser.add_argument(
        "--enforce_monotonic_turn_index",
        action="store_true",
        help="If set, require turn_index to be strictly increasing. Otherwise, non-monotonic order is a warning."
    )

    parser.add_argument(
        "--max_retries_per_turn",
        type=int,
        default=None,
        help="Optional safety check: if provided, enforce that attempts per turn do not exceed (max_retries_per_turn + 1)."
    )

    return parser.parse_args()


def resolve_input_path(args: argparse.Namespace) -> Path:
    input_path = Path(INPUT_FILE) if INPUT_FILE is not None else Path(args.input_file) if args.input_file else None
    if input_path is None:
        print("❌ --input_file must be provided (unless INPUT_FILE override is set).")
        sys.exit(1)
    if not input_path.exists():
        print(f"❌ Input file does not exist: {input_path}")
        sys.exit(1)
    return input_path


def load_json(path: Path, quiet: bool) -> Dict[str, Any]:
    vprint(f"Loading JSON: {path}", quiet)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to read JSON file: {path}")
        print(f"   Error: {e}")
        sys.exit(1)


def compute_backup_path(input_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return input_path.with_name(f"{input_path.name}.backup.{timestamp}")


def maybe_backup_file(input_path: Path, create_backup: bool, dry_run: bool, quiet: bool) -> Optional[Path]:
    backup_path = compute_backup_path(input_path)

    if not create_backup:
        vprint("Backups: DISABLED (set --create_backup to enable).", quiet)
        return None

    if dry_run:
        vprint(f"[DRY RUN] Would create backup at: {backup_path}", quiet)
        return backup_path

    vprint(f"Creating backup at: {backup_path}", quiet)
    try:
        shutil.copy2(input_path, backup_path)
        vprint("✅ Backup created.", quiet)
        print(f"Backup path: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"❌ Failed to create backup file: {backup_path}")
        print(f"   Error: {e}")
        sys.exit(1)


def maybe_write_normalized_json(
    input_path: Path,
    data: Dict[str, Any],
    normalize: bool,
    create_backup: bool,
    dry_run: bool,
    quiet: bool,
    metrics: RunMetrics
) -> Tuple[Optional[Path], bool]:
    if not normalize:
        vprint("Normalize: OFF (no file edits will occur).", quiet)
        metrics.skipped_files += 1
        return None, False

    vprint("Normalize: ON (will rewrite the input file in-place using canonical formatting).", quiet)

    backup_path = maybe_backup_file(input_path, create_backup=create_backup, dry_run=dry_run, quiet=quiet)

    if dry_run:
        if create_backup:
            vprint(f"[DRY RUN] Would create backup at: {backup_path}", quiet)
        else:
            vprint("[DRY RUN] No backup would be created (backups disabled).", quiet)
        vprint(f"[DRY RUN] Would rewrite file in-place: {input_path}", quiet)
        metrics.normalized = True
        return backup_path, False

    vprint(f"Writing normalized JSON in-place: {input_path}", quiet)
    try:
        with open(input_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        vprint("✅ In-place normalization write complete.", quiet)
        metrics.normalized = True
        metrics.modified_files += 1
        return backup_path, True
    except Exception as e:
        print(f"❌ Failed to write normalized JSON to: {input_path}")
        print(f"   Error: {e}")
        sys.exit(1)


def add_error(metrics: RunMetrics, code: str, message: str, path: str) -> None:
    metrics.errors.append(Finding(code=code, severity="HARD", message=message, path=path))


def add_warning(metrics: RunMetrics, code: str, message: str, path: str) -> None:
    metrics.warnings.append(Finding(code=code, severity="SOFT", message=message, path=path))


def safe_get_list(data: Dict[str, Any], key: str) -> List[Any]:
    val = data.get(key, [])
    return val if isinstance(val, list) else []


def runtime_validate(
    debate: Dict[str, Any],
    metrics: RunMetrics,
    enforce_monotonic_turn_index: bool,
    max_retries_per_turn: Optional[int],
    quiet: bool
) -> Dict[str, Any]:
    """
    Perform semantic integrity checks not covered by JSON Schema.
    Returns a small derived summary (e.g., resolved final attempt indices) without modifying the debate object.
    """
    vprint("Step: Runtime semantic validation (referential integrity, uniqueness, ordering, semantic range checks)...", quiet)

    participants = safe_get_list(debate, "participants")
    participant_ids: Set[str] = set()
    for i, p in enumerate(participants):
        agent_id = (p or {}).get("agent_id")
        if isinstance(agent_id, str) and agent_id.strip():
            participant_ids.add(agent_id.strip())

    turns = safe_get_list(debate, "turns")

    # Turn-level uniqueness
    seen_turn_indices: Set[int] = set()
    seen_turn_ids: Set[str] = set()
    last_turn_index: Optional[int] = None

    evaluation_mode = debate.get("evaluation_mode")  # "posthoc" | "in_loop" or missing

    resolved_final_attempt_indices: Dict[str, int] = {}

    for t_i, turn in enumerate(turns):
        metrics.turns_processed += 1
        turn_path = f"turns[{t_i}]"

        turn_id = (turn or {}).get("turn_id")
        if not isinstance(turn_id, str) or not turn_id.strip():
            add_error(metrics, E_TURN_ID_EMPTY, "turn_id must be a non-empty string.", f"{turn_path}.turn_id")
            turn_id = f"<missing:{t_i}>"
        else:
            if turn_id in seen_turn_ids:
                add_error(metrics, E_DUPLICATE_TURN_ID, f"Duplicate turn_id detected: {turn_id}", f"{turn_path}.turn_id")
            seen_turn_ids.add(turn_id)

        # turn_index checks
        turn_index = (turn or {}).get("turn_index")
        if not isinstance(turn_index, int):
            add_error(metrics, E_DUPLICATE_TURN_INDEX, "turn_index must be an integer.", f"{turn_path}.turn_index")
        else:
            if turn_index < 0:
                add_error(metrics, E_NEGATIVE_TURN_INDEX, "turn_index must be >= 0.", f"{turn_path}.turn_index")
            if turn_index in seen_turn_indices:
                add_error(metrics, E_DUPLICATE_TURN_INDEX, f"Duplicate turn_index detected: {turn_index}", f"{turn_path}.turn_index")
            seen_turn_indices.add(turn_index)

            if last_turn_index is not None and turn_index <= last_turn_index:
                if enforce_monotonic_turn_index:
                    add_error(metrics, E_DUPLICATE_TURN_INDEX, f"turn_index not strictly increasing (prev={last_turn_index}, curr={turn_index}).", f"{turn_path}.turn_index")
                else:
                    add_warning(metrics, W_TURN_INDEX_NOT_MONOTONIC, f"turn_index not increasing (prev={last_turn_index}, curr={turn_index}).", f"{turn_path}.turn_index")
            last_turn_index = turn_index

        # speaker_id referential integrity
        speaker_id = (turn or {}).get("speaker_id")
        if not isinstance(speaker_id, str) or not speaker_id.strip():
            add_error(metrics, E_SPEAKER_ID_EMPTY, "speaker_id must be a non-empty string.", f"{turn_path}.speaker_id")
        else:
            if participant_ids and speaker_id not in participant_ids:
                add_error(
                    metrics,
                    E_SPEAKER_UNKNOWN,
                    f"speaker_id '{speaker_id}' not found in participants.agent_id set.",
                    f"{turn_path}.speaker_id"
                )

        # round_index sanity (soft)
        round_index = (turn or {}).get("round_index")
        if isinstance(round_index, int) and round_index < 1:
            add_warning(metrics, W_ROUND_INDEX_LT_1, "round_index is < 1; expected >= 1 when provided.", f"{turn_path}.round_index")

        # recommendation min/max sanity
        rec = (turn or {}).get("recommendation")
        if isinstance(rec, dict):
            ps_min = rec.get("position_size_pct_min")
            ps_max = rec.get("position_size_pct_max")
            h_min = rec.get("horizon_days_min")
            h_max = rec.get("horizon_days_max")
            if isinstance(ps_min, (int, float)) and isinstance(ps_max, (int, float)) and ps_min > ps_max:
                add_error(metrics, E_RECOMMENDATION_RANGE_INVALID, "position_size_pct_min > position_size_pct_max", f"{turn_path}.recommendation")
            if isinstance(h_min, int) and isinstance(h_max, int) and h_min > h_max:
                add_error(metrics, E_RECOMMENDATION_RANGE_INVALID, "horizon_days_min > horizon_days_max", f"{turn_path}.recommendation")

        # attempts checks
        attempts = (turn or {}).get("attempts")
        attempts_list = attempts if isinstance(attempts, list) else None

        if evaluation_mode == "posthoc" and attempts_list and len(attempts_list) > 1:
            add_warning(metrics, W_ATTEMPTS_PRESENT_POSTHOC, "evaluation_mode=posthoc but attempts has length > 1.", f"{turn_path}.attempts")

        if evaluation_mode == "posthoc" and attempts_list and len(attempts_list) > 1:
            # also a semantic mismatch check (not always fatal; you may have mixed data)
            add_warning(metrics, E_MODE_MISMATCH, "Mode mismatch: posthoc with multi-attempt turn. Consider setting evaluation_mode=in_loop.", "evaluation_mode")

        if max_retries_per_turn is not None and attempts_list is not None:
            allowed_attempts = max_retries_per_turn + 1
            if len(attempts_list) > allowed_attempts:
                add_error(
                    metrics,
                    E_ATTEMPT_INDEX_INVALID,
                    f"attempts length {len(attempts_list)} exceeds allowed {allowed_attempts} (max_retries_per_turn={max_retries_per_turn}).",
                    f"{turn_path}.attempts"
                )

        # attempt_index integrity + resolved final attempt
        final_attempt_index = 0
        if attempts_list is not None and len(attempts_list) > 0:
            attempt_indices: Set[int] = set()
            max_idx = -1
            last_idx: Optional[int] = None

            for a_i, attempt in enumerate(attempts_list):
                metrics.attempts_processed += 1
                attempt_path = f"{turn_path}.attempts[{a_i}]"

                aidx = (attempt or {}).get("attempt_index")
                if not isinstance(aidx, int) or aidx < 0:
                    add_error(metrics, E_ATTEMPT_INDEX_INVALID, "attempt_index must be an integer >= 0.", f"{attempt_path}.attempt_index")
                    continue

                if aidx in attempt_indices:
                    add_error(metrics, E_ATTEMPT_INDEX_INVALID, f"Duplicate attempt_index detected: {aidx}", f"{attempt_path}.attempt_index")
                attempt_indices.add(aidx)

                if last_idx is not None and aidx <= last_idx:
                    add_error(metrics, E_ATTEMPT_INDEX_INVALID, f"attempt_index not strictly increasing (prev={last_idx}, curr={aidx}).", f"{attempt_path}.attempt_index")
                last_idx = aidx

                if aidx > max_idx:
                    max_idx = aidx

            final_attempt_index = max_idx if max_idx >= 0 else 0

        resolved_final_attempt_indices[turn_id] = final_attempt_index

        # final answer resolution warning (soft)
        has_attempt_final = False
        if attempts_list:
            for attempt in attempts_list:
                fa = (attempt or {}).get("final_answer")
                if isinstance(fa, str) and fa.strip():
                    has_attempt_final = True
                    break

        rec_present = isinstance((turn or {}).get("recommendation"), dict)
        if not has_attempt_final and not rec_present:
            # In posthoc without recommendation, evaluators will infer from content; warn but don't fail.
            add_warning(
                metrics,
                W_NO_STRUCTURED_FINAL_ANSWER,
                "No attempt.final_answer found and recommendation is missing; evaluators must infer final answer from content.",
                f"{turn_path}"
            )

    # Mode existence sanity (soft; schema may require it but runtime validator shouldn't assume)
    if evaluation_mode not in (None, "posthoc", "in_loop"):
        add_warning(metrics, E_MODE_MISMATCH, f"Unknown evaluation_mode '{evaluation_mode}'. Expected 'posthoc' or 'in_loop'.", "evaluation_mode")

    return {
        "resolved_final_attempt_indices": resolved_final_attempt_indices,
        "participants_count": len(participants),
        "turns_count": len(turns),
    }


def print_final_report(
    input_path: Path,
    args: argparse.Namespace,
    metrics: RunMetrics,
    backup_path: Optional[Path],
    derived: Dict[str, Any],
    elapsed: float
) -> None:
    print("\n================ FINAL REPORT ================")
    print(f"Execution Mode                 : {'DRY RUN' if args.dry_run else 'REAL EXECUTION'}")
    print(f"Input File                     : {input_path}")
    print(f"Normalize Requested            : {'YES' if args.normalize else 'NO'}")
    print(f"Create Backup Enabled          : {'YES' if args.create_backup else 'NO'}")

    if args.normalize:
        if args.create_backup:
            if args.dry_run:
                print(f"Backup Path (would create)     : {backup_path}")
            else:
                print(f"Backup Path (created)          : {backup_path}")
        else:
            print("Backup Path                    : None (backups disabled)")
    else:
        print("Backup Path                    : None (no edits performed)")

    print(f"Turns Processed                : {metrics.turns_processed}")
    print(f"Attempts Processed             : {metrics.attempts_processed}")
    print(f"Errors (HARD)                  : {len(metrics.errors)}")
    print(f"Warnings (SOFT)                : {len(metrics.warnings)}")
    print(f"Enforce Monotonic turn_index   : {'YES' if args.enforce_monotonic_turn_index else 'NO'}")
    print(f"Max Retries Per Turn           : {args.max_retries_per_turn if args.max_retries_per_turn is not None else 'None'}")
    print(f"Files Modified Count           : {metrics.modified_files}")
    print(f"Files Skipped Count            : {metrics.skipped_files}")
    print(f"Total Runtime (sec)            : {round(elapsed, 4)}")

    # Derived info
    print("\n--- DERIVED OUTPUTS (NON-MUTATING) ---")
    print(f"Participants Count             : {derived.get('participants_count')}")
    print(f"Turns Count                    : {derived.get('turns_count')}")
    print("Resolved Final Attempt Indices :")
    r = derived.get("resolved_final_attempt_indices", {})
    if isinstance(r, dict) and r:
        # Keep readable but not enormous.
        for k, v in list(r.items())[:25]:
            print(f"  - {k}: {v}")
        if len(r) > 25:
            print(f"  ... ({len(r) - 25} more)")
    else:
        print("  (none)")

    # Findings
    if metrics.errors:
        print("\n--- ERRORS (HARD) ---")
        for f in metrics.errors:
            print(f"[{f.code}] {f.path} :: {f.message}")

    if metrics.warnings:
        print("\n--- WARNINGS (SOFT) ---")
        for f in metrics.warnings:
            print(f"[{f.code}] {f.path} :: {f.message}")

    print("==============================================\n")


def main() -> None:
    start = time.time()
    metrics = RunMetrics()

    args = parse_args()
    input_path = resolve_input_path(args)

    vprint("\n--- BEGIN RUNTIME VALIDATION ---", args.quiet)
    vprint(f"Input file resolved to : {input_path}", args.quiet)
    vprint(f"Mode                  : {'DRY RUN' if args.dry_run else 'REAL EXECUTION'}", args.quiet)
    vprint(f"Create backup         : {'ON' if args.create_backup else 'OFF'}", args.quiet)
    vprint(f"Normalize             : {'ON' if args.normalize else 'OFF'}", args.quiet)
    vprint(f"Monotonic turn_index  : {'ENFORCED' if args.enforce_monotonic_turn_index else 'NOT ENFORCED (warning only)'}", args.quiet)
    if args.max_retries_per_turn is not None:
        vprint(f"Max retries per turn  : {args.max_retries_per_turn}", args.quiet)

    debate = load_json(input_path, args.quiet)

    # Semantic validation (non-mutating)
    derived = runtime_validate(
        debate=debate,
        metrics=metrics,
        enforce_monotonic_turn_index=args.enforce_monotonic_turn_index,
        max_retries_per_turn=args.max_retries_per_turn,
        quiet=args.quiet,
    )

    # Optional normalization (the only transform)
    backup_path = None
    did_write = False
    if args.normalize:
        vprint("Step: Preparing for in-place edit (normalization).", args.quiet)
        backup_path, did_write = maybe_write_normalized_json(
            input_path=input_path,
            data=debate,
            normalize=args.normalize,
            create_backup=args.create_backup,
            dry_run=args.dry_run,
            quiet=args.quiet,
            metrics=metrics,
        )
    else:
        # Explicitly narrate backup behavior in read-only mode
        if args.create_backup:
            backup_would_be = compute_backup_path(input_path)
            if args.dry_run:
                vprint(f"[DRY RUN] Backups enabled but no edits requested. No backup would be created. (Would-have path: {backup_would_be})", args.quiet)
            else:
                vprint("Backups enabled but no edits requested. No backup created because script is read-only without --normalize.", args.quiet)
        else:
            vprint("Backups disabled and no edits requested. Read-only validation only.", args.quiet)

    elapsed = time.time() - start

    print_final_report(
        input_path=input_path,
        args=args,
        metrics=metrics,
        backup_path=backup_path if args.normalize and args.create_backup else None,
        derived=derived,
        elapsed=elapsed,
    )

    # Exit codes:
    # 0 = success (no HARD errors)
    # 2 = validation failed (HARD errors)
    if metrics.errors:
        sys.exit(2)


if __name__ == "__main__":
    main()