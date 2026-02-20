#!/usr/bin/env python3
"""
USAGE EXAMPLES
==============

1) Basic validation (read-only; no backup; no file modification):
   python validate_debate_json.py --schema_file schemas/debate_output_v2.schema.json --input_file examples/nvda_posthoc_example_v2.json

2) Dry run (still read-only; prints what it WOULD do; no backup; no writes):
   python validate_debate_json.py --schema_file schemas/debate_output_v2.schema.json --input_file examples/nvda_posthoc_example_v2.json --dry_run

3) Enable backup + normalize formatting in-place (writes input file; creates timestamped backup):
   python validate_debate_json.py --schema_file schemas/debate_output_v2.schema.json --input_file examples/nvda_posthoc_example_v2.json --normalize --create_backup

4) Enable backup + normalize formatting, but dry run (shows backup path it WOULD create; no writes):
   python validate_debate_json.py --schema_file schemas/debate_output_v2.schema.json --input_file examples/nvda_posthoc_example_v2.json --normalize --create_backup --dry_run

5) Quiet mode (suppresses non-essential narration):
   python validate_debate_json.py --schema_file schemas/debate_output_v2.schema.json --input_file examples/nvda_posthoc_example_v2.json --quiet

6) Using hard-coded overrides (set INPUT_FILE and/or SCHEMA_FILE below; then run with no args):
   python validate_debate_json.py
"""

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError


# ============================
# OPTIONAL HARD-CODED OVERRIDES
# ============================
# If these constants are set (not None), they override command-line arguments.
# If they are None, argparse values are used instead.

INPUT_FILE: Optional[str] = None
SCHEMA_FILE: Optional[str] = None


@dataclass
class RunMetrics:
    schema_errors: int = 0
    instance_errors: int = 0
    normalized: bool = False
    modified_files: int = 0
    skipped_files: int = 0
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


def vprint(message: str, quiet: bool) -> None:
    """Verbose print (ON by default)."""
    if not quiet:
        print(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a debate JSON file against a Draft 2020-12 JSON Schema. Optionally normalize formatting in-place."
    )

    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to the input debate JSON file."
    )
    parser.add_argument(
        "--schema_file",
        type=str,
        help="Path to the JSON Schema file (Draft 2020-12)."
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
        help="If set, rewrite the input JSON in-place using a canonical pretty-printed format (indent=2, ensure_ascii=False, sort_keys=False). "
             "This is the ONLY edit/transform performed by this script."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="If set, suppress non-essential verbose narration."
    )

    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    input_path = Path(INPUT_FILE) if INPUT_FILE is not None else Path(args.input_file) if args.input_file else None
    schema_path = Path(SCHEMA_FILE) if SCHEMA_FILE is not None else Path(args.schema_file) if args.schema_file else None

    if input_path is None or schema_path is None:
        print("❌ Both --input_file and --schema_file must be provided (unless hard-coded overrides are set).")
        sys.exit(1)

    if not input_path.exists():
        print(f"❌ Input file does not exist: {input_path}")
        sys.exit(1)

    if not schema_path.exists():
        print(f"❌ Schema file does not exist: {schema_path}")
        sys.exit(1)

    return input_path, schema_path


def load_json(path: Path, quiet: bool) -> Dict[str, Any]:
    vprint(f"Loading JSON: {path}", quiet)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to read JSON file: {path}")
        print(f"   Error: {e}")
        sys.exit(1)


def check_schema(schema: Dict[str, Any], quiet: bool) -> None:
    vprint("Step: Checking schema validity (Draft 2020-12)...", quiet)
    try:
        Draft202012Validator.check_schema(schema)
        vprint("✅ Schema is structurally valid.", quiet)
    except SchemaError as e:
        print("❌ Schema is INVALID:")
        print(e)
        sys.exit(1)


def validate_instance(schema: Dict[str, Any], instance: Dict[str, Any], quiet: bool) -> List[str]:
    vprint("Step: Validating instance against schema...", quiet)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.path))

    messages: List[str] = []
    if errors:
        for err in errors:
            path_str = "/".join(map(str, err.path)) if err.path else "<root>"
            messages.append(f"Path {path_str}: {err.message}")
    else:
        vprint("✅ Instance conforms to schema.", quiet)

    return messages


def compute_backup_path(input_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return input_path.with_name(f"{input_path.name}.backup.{timestamp}")


def maybe_backup_file(input_path: Path, create_backup: bool, dry_run: bool, quiet: bool) -> Optional[Path]:
    """
    Create a backup only if:
      - we are going to edit the file in-place, AND
      - --create_backup is set, AND
      - --dry_run is NOT set
    The caller is responsible for deciding whether an edit is happening.
    """
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
    instance: Dict[str, Any],
    normalize: bool,
    create_backup: bool,
    dry_run: bool,
    quiet: bool,
    metrics: RunMetrics
) -> Tuple[Optional[Path], bool]:
    """
    If --normalize is set, rewrite the JSON in-place (canonical pretty formatting).
    Backup only occurs if --create_backup is set (and not dry_run).
    """
    if not normalize:
        vprint("Normalize: OFF (no file edits will occur).", quiet)
        metrics.skipped_files += 1
        return None, False

    vprint("Normalize: ON (will rewrite the input file in-place using canonical formatting).", quiet)

    # Backup decision (only relevant because we're editing)
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
            json.dump(instance, f, indent=2, ensure_ascii=False)
            f.write("\n")
        vprint("✅ In-place normalization write complete.", quiet)
        metrics.normalized = True
        metrics.modified_files += 1
        return backup_path, True
    except Exception as e:
        print(f"❌ Failed to write normalized JSON to: {input_path}")
        print(f"   Error: {e}")
        sys.exit(1)


def main() -> None:
    start = time.time()
    metrics = RunMetrics()

    args = parse_args()
    input_path, schema_path = resolve_paths(args)

    vprint("\n--- BEGIN VALIDATION PROCESS ---", args.quiet)
    vprint(f"Input file resolved to : {input_path}", args.quiet)
    vprint(f"Schema file resolved to: {schema_path}", args.quiet)
    vprint(f"Mode                  : {'DRY RUN' if args.dry_run else 'REAL EXECUTION'}", args.quiet)
    vprint(f"Create backup         : {'ON' if args.create_backup else 'OFF'}", args.quiet)
    vprint(f"Normalize             : {'ON' if args.normalize else 'OFF'}", args.quiet)

    # Load
    schema = load_json(schema_path, args.quiet)
    instance = load_json(input_path, args.quiet)

    # Validate schema structure
    check_schema(schema, args.quiet)

    # Validate instance
    error_messages = validate_instance(schema, instance, args.quiet)
    if error_messages:
        metrics.instance_errors = len(error_messages)
        vprint("❌ Instance validation FAILED with the following errors:", args.quiet)
        for msg in error_messages:
            print(f"  - {msg}")

    # Optional in-place normalization (the only transform)
    backup_path = None
    did_write = False
    if args.normalize:
        vprint("Step: Preparing for in-place edit (normalization).", args.quiet)
        backup_path, did_write = maybe_write_normalized_json(
            input_path=input_path,
            instance=instance,
            normalize=args.normalize,
            create_backup=args.create_backup,
            dry_run=args.dry_run,
            quiet=args.quiet,
            metrics=metrics
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

    elapsed = round(time.time() - start, 4)

    # ============================
    # FINAL REPORT
    # ============================
    print("\n================ FINAL REPORT ================")
    print(f"Execution Mode           : {'DRY RUN' if args.dry_run else 'REAL EXECUTION'}")
    print(f"Input File               : {input_path}")
    print(f"Schema File              : {schema_path}")
    print(f"Schema Structure Valid    : YES")
    print(f"Instance Validation       : {'PASS' if metrics.instance_errors == 0 else 'FAIL'}")
    print(f"Instance Error Count      : {metrics.instance_errors}")
    print(f"Normalize Requested       : {'YES' if args.normalize else 'NO'}")
    print(f"File Modified             : {'YES' if did_write else 'NO'}")
    print(f"Create Backup Enabled     : {'YES' if args.create_backup else 'NO'}")

    if args.normalize:
        if args.create_backup:
            if args.dry_run:
                print(f"Backup Path (would create): {backup_path}")
            else:
                print(f"Backup Path (created)     : {backup_path}")
        else:
            print("Backup Path              : None (backups disabled)")
    else:
        print("Backup Path              : None (no edits performed)")

    print(f"Files Modified Count      : {metrics.modified_files}")
    print(f"Files Skipped Count       : {metrics.skipped_files}")
    if metrics.warnings:
        print(f"Warnings                  : {len(metrics.warnings)}")
        for w in metrics.warnings:
            print(f"  - {w}")
    else:
        print("Warnings                  : 0")
    print(f"Total Runtime (sec)       : {elapsed}")
    print("==============================================\n")

    # Exit codes:
    # 0 = success
    # 2 = instance validation failed
    if metrics.instance_errors != 0:
        sys.exit(2)


if __name__ == "__main__":
    main()