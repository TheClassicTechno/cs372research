#!/usr/bin/env python3
"""Extract ablation run data into test fixtures.

Usage:
    python tests/integration/scripts/extract_ablation_fixture.py \
        --ablation 7 --run-id run_2026-03-11_16-12-13 --dest ablation7_baseline

    # Generate corrupt fixture from an existing clean fixture:
    python tests/integration/scripts/extract_ablation_fixture.py \
        --generate-corrupt ablation7_baseline --dest ablation7_corrupt

    # Generate misrouted fixture from an existing clean fixture:
    python tests/integration/scripts/extract_ablation_fixture.py \
        --generate-misrouted ablation7_baseline --dest ablation7_misrouted

    # Generate prompt hashes for all fixtures:
    python tests/integration/scripts/extract_ablation_fixture.py \
        --generate-hashes ablation7_baseline ablation8_baseline ablation10_treatment
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # cs372research/
LOGGING_DIR = REPO_ROOT / "logging" / "runs"
FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"

PROMPT_DELIMITER = "=== USER PROMPT ==="

# Phase directory → phase key mapping
PHASE_DIR_MAP = {
    "proposals": "proposal",
    "critiques": "critique",
    "revisions": "revision",
    "CRIT": "crit",
}


def _hash_strict(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _split_prompt(prompt_text: str) -> dict:
    """Split a prompt.txt into system and user parts."""
    if PROMPT_DELIMITER in prompt_text:
        parts = prompt_text.split(PROMPT_DELIMITER, 1)
        system = parts[0]
        # Strip the === SYSTEM PROMPT === header if present
        system = re.sub(r"^=== SYSTEM PROMPT ===\s*", "", system)
        user = parts[1]
        return {"system": system.strip(), "user": user.strip()}
    return {"system": prompt_text.strip(), "user": ""}


# ── Clean fixture extraction ──────────────────────────────────────────

def extract_clean(ablation: int, run_id: str, dest_name: str) -> Path:
    src = LOGGING_DIR / f"vskarich_ablation_{ablation}" / run_id
    if not src.exists():
        sys.exit(f"Source not found: {src}")

    dst = FIXTURES_DIR / dest_name
    if dst.exists():
        shutil.rmtree(dst)

    shutil.copytree(src, dst)

    # Sanitize manifest.json: strip absolute paths, remove run_command
    manifest_path = dst / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest.pop("run_command", None)
        manifest["config_paths"] = [
            Path(p).name for p in manifest.get("config_paths", [])
        ]
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    # Remove _dashboard/
    dashboard = dst / "_dashboard"
    if dashboard.exists():
        shutil.rmtree(dashboard)

    print(f"Extracted {src.name} -> {dst}")
    return dst


# ── Corrupt fixture generation ────────────────────────────────────────

def generate_corrupt(source_name: str, dest_name: str) -> Path:
    src = FIXTURES_DIR / source_name
    if not src.exists():
        sys.exit(f"Source fixture not found: {src}")

    dst = FIXTURES_DIR / dest_name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    round_dir = dst / "rounds" / "round_001"
    roles = _get_roles(dst)
    first_role = roles[0]

    # 1. Remove claim_id from first claim in a proposal
    _corrupt_remove_claim_id(round_dir / "proposals" / first_role / "response.txt")

    # 2. Set allocation sum to 0.5 in a revision
    _corrupt_allocation_sum(round_dir / "revisions" / first_role / "response.txt")

    # 3. Set invalid target_role in a critique
    second_role = roles[1] if len(roles) > 1 else first_role
    _corrupt_target_role(round_dir / "critiques" / second_role / "response.json")

    # 4. Remove pillar_scores from a CRIT response
    _corrupt_remove_pillars(round_dir / "CRIT" / first_role / "response.txt")

    print(f"Generated corrupt fixture: {dst}")
    return dst


def _get_roles(fixture_dir: Path) -> list:
    manifest = json.loads((fixture_dir / "manifest.json").read_text())
    return manifest["roles"]


def _corrupt_remove_claim_id(response_path: Path):
    """Remove claim_id from the first claim in a proposal response."""
    text = response_path.read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try extracting JSON from text
        data = _extract_json_from_text(text)
    if data and "claims" in data and len(data["claims"]) > 0:
        data["claims"][0].pop("claim_id", None)
        response_path.write_text(json.dumps(data, indent=2))


def _corrupt_allocation_sum(response_path: Path):
    """Set allocation weights to sum to ~0.5."""
    text = response_path.read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = _extract_json_from_text(text)
    if data and "allocation" in data:
        alloc = data["allocation"]
        for k in alloc:
            alloc[k] = alloc[k] * 0.5
        response_path.write_text(json.dumps(data, indent=2))


def _corrupt_target_role(response_path: Path):
    """Set target_role to a nonexistent agent."""
    text = response_path.read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = _extract_json_from_text(text)
    if data:
        critiques = data.get("critiques", [data] if "target_role" in data else [])
        if critiques:
            critiques[0]["target_role"] = "nonexistent_agent"
        if "critiques" in data:
            data["critiques"] = critiques
        response_path.write_text(json.dumps(data, indent=2))


def _corrupt_remove_pillars(response_path: Path):
    """Remove pillar_scores from CRIT response."""
    text = response_path.read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = _extract_json_from_text(text)
    if data:
        data.pop("pillar_scores", None)
        response_path.write_text(json.dumps(data, indent=2))


def _extract_json_from_text(text: str) -> dict | None:
    """Try to extract JSON object from text that may have markdown fences."""
    # Try raw parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from markdown fences
    m = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try finding first { ... } block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ── Misrouted fixture generation ──────────────────────────────────────

def generate_misrouted(source_name: str, dest_name: str) -> Path:
    src = FIXTURES_DIR / source_name
    if not src.exists():
        sys.exit(f"Source fixture not found: {src}")

    dst = FIXTURES_DIR / dest_name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    round_dir = dst / "rounds" / "round_001"
    roles = _get_roles(dst)

    if len(roles) >= 2:
        # Role swap: proposals/macro/prompt.txt <-> proposals/technical/prompt.txt
        p0 = round_dir / "proposals" / roles[0] / "prompt.txt"
        p1 = round_dir / "proposals" / roles[1] / "prompt.txt"
        if p0.exists() and p1.exists():
            t0, t1 = p0.read_text(), p1.read_text()
            p0.write_text(t1)
            p1.write_text(t0)
            print(f"  Swapped role prompts: {roles[0]} <-> {roles[1]}")

    # Phase swap: proposals/first_role/prompt.txt <-> critiques/first_role/prompt.txt
    prop = round_dir / "proposals" / roles[0] / "prompt.txt"
    crit = round_dir / "critiques" / roles[0] / "prompt.txt"
    if prop.exists() and crit.exists():
        tp, tc = prop.read_text(), crit.read_text()
        prop.write_text(tc)
        crit.write_text(tp)
        print(f"  Swapped phase prompts: propose/{roles[0]} <-> critique/{roles[0]}")

    print(f"Generated misrouted fixture: {dst}")
    return dst


# ── Hash generation ───────────────────────────────────────────────────

def generate_hashes(fixture_names: list[str]) -> Path:
    """Generate expected_prompt_hashes.json for all fixtures."""
    all_hashes = {}

    for name in fixture_names:
        fixture_dir = FIXTURES_DIR / name
        if not fixture_dir.exists():
            print(f"WARNING: fixture not found: {name}, skipping")
            continue

        # Find all round directories
        rounds_dir = fixture_dir / "rounds"
        if not rounds_dir.exists():
            continue

        for round_path in sorted(rounds_dir.iterdir()):
            if not round_path.is_dir() or not round_path.name.startswith("round_"):
                continue
            round_num = int(round_path.name.split("_")[1])

            # Standard phases
            for phase_dir_name, phase_key in PHASE_DIR_MAP.items():
                phase_dir = round_path / phase_dir_name
                if not phase_dir.exists():
                    continue
                for role_dir in sorted(phase_dir.iterdir()):
                    if not role_dir.is_dir():
                        continue
                    prompt_file = role_dir / "prompt.txt"
                    if not prompt_file.exists():
                        continue
                    parts = _split_prompt(prompt_file.read_text())
                    # Map phase_key back to pipeline phase name
                    phase_name = {
                        "proposal": "propose",
                        "critique": "critique",
                        "revision": "revise",
                        "crit": "crit",
                    }[phase_key]
                    key = f"{name}/{round_num}/{phase_name}/{role_dir.name}/0"
                    all_hashes[key] = {
                        "system": _hash_strict(parts["system"]),
                        "user": _hash_strict(parts["user"]),
                    }

            # Retry directories
            for retry_dir in sorted(round_path.iterdir()):
                m = re.match(r"revisions_retry_(\d+)", retry_dir.name)
                if not m or not retry_dir.is_dir():
                    continue
                call_index = int(m.group(1))
                for role_dir in sorted(retry_dir.iterdir()):
                    if not role_dir.is_dir():
                        continue
                    prompt_file = role_dir / "prompt.txt"
                    if not prompt_file.exists():
                        continue
                    parts = _split_prompt(prompt_file.read_text())
                    key = f"{name}/{round_num}/revise/{role_dir.name}/{call_index}"
                    all_hashes[key] = {
                        "system": _hash_strict(parts["system"]),
                        "user": _hash_strict(parts["user"]),
                    }

            # Judge prompt
            judge_prompt = fixture_dir / "final" / "judge_prompt.txt"
            if judge_prompt.exists():
                parts = _split_prompt(judge_prompt.read_text())
                key = f"{name}/{round_num}/judge/judge/0"
                all_hashes[key] = {
                    "system": _hash_strict(parts["system"]),
                    "user": _hash_strict(parts["user"]),
                }

    out_path = FIXTURES_DIR / "expected_prompt_hashes.json"
    out_path.write_text(json.dumps(all_hashes, indent=2, sort_keys=True) + "\n")
    print(f"Generated {len(all_hashes)} prompt hashes -> {out_path}")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract ablation fixtures")
    parser.add_argument("--ablation", type=int, help="Ablation number (7, 8, 10)")
    parser.add_argument("--run-id", help="Run ID (e.g. run_2026-03-11_16-12-13)")
    parser.add_argument("--dest", help="Destination fixture name")
    parser.add_argument("--generate-corrupt", metavar="SOURCE",
                        help="Generate corrupt fixture from SOURCE")
    parser.add_argument("--generate-misrouted", metavar="SOURCE",
                        help="Generate misrouted fixture from SOURCE")
    parser.add_argument("--generate-hashes", nargs="+", metavar="FIXTURE",
                        help="Generate prompt hashes for fixtures")
    args = parser.parse_args()

    if args.generate_hashes:
        generate_hashes(args.generate_hashes)
    elif args.generate_corrupt:
        if not args.dest:
            sys.exit("--dest required with --generate-corrupt")
        generate_corrupt(args.generate_corrupt, args.dest)
    elif args.generate_misrouted:
        if not args.dest:
            sys.exit("--dest required with --generate-misrouted")
        generate_misrouted(args.generate_misrouted, args.dest)
    elif args.ablation and args.run_id and args.dest:
        extract_clean(args.ablation, args.run_id, args.dest)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
