#!/usr/bin/env python3
"""
VALIDATE DEBATE OUTPUT

Now performs:

1) JSON Schema validation (structure-level)
2) Runtime semantic validation (research invariants)
3) Optional normalization (format only)

Exit Codes:
0 = success
2 = HARD validation failure (schema or runtime)
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

# NEW: schema validation
try:
    import jsonschema
except ImportError:
    print("❌ jsonschema library not installed. Run: uv add jsonschema")
    sys.exit(1)

# ============================
# CONFIG
# ============================

SCHEMA_PATH = Path("global/schemas/debate_output.schema.json")
INPUT_FILE: Optional[str] = None

# ============================
# ERROR CODES
# ============================

E_SCHEMA_VALIDATION_FAILED = "E_SCHEMA_VALIDATION_FAILED"
E_POSTHOC_MULTI_ATTEMPT = "E_POSTHOC_MULTI_ATTEMPT"
E_SPEAKER_UNKNOWN = "E_SPEAKER_UNKNOWN"
E_DUPLICATE_TURN_INDEX = "E_DUPLICATE_TURN_INDEX"
E_NEGATIVE_TURN_INDEX = "E_NEGATIVE_TURN_INDEX"
E_DUPLICATE_TURN_ID = "E_DUPLICATE_TURN_ID"
E_TURN_ID_EMPTY = "E_TURN_ID_EMPTY"
E_SPEAKER_ID_EMPTY = "E_SPEAKER_ID_EMPTY"
E_ATTEMPT_INDEX_INVALID = "E_ATTEMPT_INDEX_INVALID"
E_RECOMMENDATION_RANGE_INVALID = "E_RECOMMENDATION_RANGE_INVALID"

W_NO_STRUCTURED_FINAL_ANSWER = "W_NO_STRUCTURED_FINAL_ANSWER"
W_ROUND_INDEX_LT_1 = "W_ROUND_INDEX_LT_1"
W_TURN_INDEX_NOT_MONOTONIC = "W_TURN_INDEX_NOT_MONOTONIC"

# ============================
# DATA STRUCTURES
# ============================

@dataclass
class Finding:
    code: str
    severity: str  # "HARD" | "SOFT"
    message: str
    path: str

@dataclass
class RunMetrics:
    turns_processed: int = 0
    attempts_processed: int = 0
    errors: List[Finding] = field(default_factory=list)
    warnings: List[Finding] = field(default_factory=list)

# ============================
# UTIL
# ============================

def add_error(metrics: RunMetrics, code: str, message: str, path: str):
    metrics.errors.append(Finding(code, "HARD", message, path))

def add_warning(metrics: RunMetrics, code: str, message: str, path: str):
    metrics.warnings.append(Finding(code, "SOFT", message, path))

# ============================
# SCHEMA VALIDATION
# ============================

def validate_schema(debate: Dict[str, Any], metrics: RunMetrics):
    if not SCHEMA_PATH.exists():
        print(f"❌ Schema file not found: {SCHEMA_PATH}")
        sys.exit(1)

    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    validator = jsonschema.Draft202012Validator(schema)

    for error in validator.iter_errors(debate):
        path = ".".join(str(x) for x in error.absolute_path)
        add_error(metrics, E_SCHEMA_VALIDATION_FAILED, error.message, path)

# ============================
# RUNTIME VALIDATION
# ============================

def runtime_validate(debate: Dict[str, Any], metrics: RunMetrics):
    participants = debate.get("participants", [])
    participant_ids = {p.get("agent_id") for p in participants if isinstance(p, dict)}

    turns = debate.get("turns", [])
    evaluation_mode = debate.get("evaluation_mode")

    seen_turn_indices = set()
    seen_turn_ids = set()

    for i, turn in enumerate(turns):
        metrics.turns_processed += 1
        path = f"turns[{i}]"

        # turn_id
        turn_id = turn.get("turn_id")
        if not isinstance(turn_id, str) or not turn_id.strip():
            add_error(metrics, E_TURN_ID_EMPTY, "turn_id must be non-empty string", f"{path}.turn_id")
        elif turn_id in seen_turn_ids:
            add_error(metrics, E_DUPLICATE_TURN_ID, "Duplicate turn_id", f"{path}.turn_id")
        seen_turn_ids.add(turn_id)

        # turn_index
        turn_index = turn.get("turn_index")
        if not isinstance(turn_index, int):
            add_error(metrics, E_DUPLICATE_TURN_INDEX, "turn_index must be integer", f"{path}.turn_index")
        else:
            if turn_index < 0:
                add_error(metrics, E_NEGATIVE_TURN_INDEX, "turn_index must be >= 0", f"{path}.turn_index")
            if turn_index in seen_turn_indices:
                add_error(metrics, E_DUPLICATE_TURN_INDEX, "Duplicate turn_index", f"{path}.turn_index")
            seen_turn_indices.add(turn_index)

        # speaker_id
        speaker_id = turn.get("speaker_id")
        if not isinstance(speaker_id, str) or not speaker_id.strip():
            add_error(metrics, E_SPEAKER_ID_EMPTY, "speaker_id must be non-empty string", f"{path}.speaker_id")
        elif participant_ids and speaker_id not in participant_ids:
            add_error(metrics, E_SPEAKER_UNKNOWN, "speaker_id not found in participants", f"{path}.speaker_id")

        # recommendation sanity
        rec = turn.get("recommendation")
        if isinstance(rec, dict):
            ps_min = rec.get("position_size_pct_min")
            ps_max = rec.get("position_size_pct_max")
            if isinstance(ps_min, (int, float)) and isinstance(ps_max, (int, float)) and ps_min > ps_max:
                add_error(metrics, E_RECOMMENDATION_RANGE_INVALID, "position_size_pct_min > position_size_pct_max", f"{path}.recommendation")

        # attempts
        attempts = turn.get("attempts", [])
        if not isinstance(attempts, list):
            continue

        # 🔥 HARD RULE: Post-hoc must have exactly one attempt
        if evaluation_mode == "posthoc":
            if len(attempts) != 1:
                add_error(
                    metrics,
                    E_POSTHOC_MULTI_ATTEMPT,
                    "In posthoc mode, each turn must contain exactly one attempt.",
                    f"{path}.attempts"
                )

        # attempt validation
        last_idx = -1
        seen_attempt_indices = set()

        for j, attempt in enumerate(attempts):
            metrics.attempts_processed += 1
            a_path = f"{path}.attempts[{j}]"

            aidx = attempt.get("attempt_index")
            if not isinstance(aidx, int) or aidx < 0:
                add_error(metrics, E_ATTEMPT_INDEX_INVALID, "attempt_index must be integer >= 0", f"{a_path}.attempt_index")
            elif aidx in seen_attempt_indices:
                add_error(metrics, E_ATTEMPT_INDEX_INVALID, "Duplicate attempt_index", f"{a_path}.attempt_index")
            elif aidx <= last_idx:
                add_error(metrics, E_ATTEMPT_INDEX_INVALID, "attempt_index must be strictly increasing", f"{a_path}.attempt_index")

            seen_attempt_indices.add(aidx)
            last_idx = aidx

# ============================
# MAIN
# ============================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--create_backup", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        debate = json.load(f)

    metrics = RunMetrics()

    # 1️⃣ Schema validation
    validate_schema(debate, metrics)

    # 2️⃣ Runtime validation
    runtime_validate(debate, metrics)

    # Report
    print("\n================ VALIDATION REPORT ================")
    print(f"Turns Processed   : {metrics.turns_processed}")
    print(f"Attempts Processed: {metrics.attempts_processed}")
    print(f"HARD Errors       : {len(metrics.errors)}")
    print(f"Warnings          : {len(metrics.warnings)}")

    if metrics.errors:
        print("\n--- HARD ERRORS ---")
        for e in metrics.errors:
            print(f"[{e.code}] {e.path} :: {e.message}")

    if metrics.warnings:
        print("\n--- WARNINGS ---")
        for w in metrics.warnings:
            print(f"[{w.code}] {w.path} :: {w.message}")

    print("===================================================\n")

    if metrics.errors:
        sys.exit(2)

if __name__ == "__main__":
    main()