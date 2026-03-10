#!/usr/bin/env python3
"""Generate ablation 2 debate config YAML files.

Ablation 2 tests the BEST vs WORST 3-agent combinations from ablation 1:
  - Best:  no_risk   (macro + technical + value)  — avg +0.48%
  - Worst: no_technical (macro + risk + value)     — avg -2.04%

Axes:
  - 2 agent combos × 4 profile variants × 2 PID modes = 16 configs
  - 3 rounds per debate (up from 2 in ablation 1)
  - 3 scenarios (2022Q1, 2023Q3, 2025Q1) → 48 total runs

Usage:
    python vskarich_ablations/ablation_2/generate_debate_config_ablations.py
"""

from __future__ import annotations

import shutil
from itertools import product
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Target directory — wiped and regenerated on every run
# ---------------------------------------------------------------------------
TARGET_DIR = Path(__file__).resolve().parent / "debate_configs"

# ---------------------------------------------------------------------------
# Ablation axes
# ---------------------------------------------------------------------------

# Role combinations: best and worst 3-agent combos from ablation 1
ROLE_COMBOS = [
    # Best 3-agent: no_risk (macro + technical + value)
    ["macro", "technical", "value"],
    # Worst 3-agent: no_technical (macro + risk + value)
    ["macro", "risk", "value"],
]

# Profile suffix variants (appended to "{role}_enriched")
PROFILE_VARIANTS = [
    "",                 # standard
    "_intense",         # intense role
    "_light",           # light contract
    "_intense_light",   # intense role + light contract
]

# Build all rosters: each role combo × each profile variant
AGENT_ROSTERS = []
for roles in ROLE_COMBOS:
    for suffix in PROFILE_VARIANTS:
        roster = {role: f"{role}_enriched{suffix}" for role in roles}
        AGENT_ROSTERS.append(roster)

JUDGE_PROFILES = [
    "judge_standard",
]

LLM_MODELS = [
    "gpt-5-mini",
]

TEMPERATURES = [
    0.3,
]

MAX_ROUNDS = [
    2,  # Reduced from 3 to 2
]

PID_ENABLED = [
    True,
]

# ---------------------------------------------------------------------------
# PID controller configurations (same as ablation 1)
# ---------------------------------------------------------------------------

CONSERVATIVE_PID = {
    "name": "pid_conservative",
    "pid_kp": 0.20,
    "pid_ki": 0.015,
    "pid_kd": 0.10,
    "pid_rho_star": 0.83,
    "pid_initial_beta": 0.35,
    "pid_epsilon": 0.02,
    "pid_gamma_beta": 0.35,
}

AGGRESSIVE_PID = {
    "name": "pid_aggressive",
    "pid_kp": 0.55,
    "pid_ki": 0.06,
    "pid_kd": 0.05,
    "pid_rho_star": 0.83,
    "pid_initial_beta": 0.40,
    "pid_epsilon": 0.015,
    "pid_gamma_beta": 0.65,
}

PID_CONFIGS = [
    CONSERVATIVE_PID,
    AGGRESSIVE_PID,
]

INITIAL_CASH = [
    100000.0,
]

# ---------------------------------------------------------------------------
# Fixed settings (not ablated)
# ---------------------------------------------------------------------------

FIXED = {
    "case_format": "memo",
    "dataset_path": "data-pipeline/final_snapshots",
    "memo_format": "text",
    "num_episodes": 1,
}

FIXED_DEBATE = {
    "agent_system": "multi_agent_debate",
    "llm_provider": "openai",
    "max_retries": 3,
    "propose_only": False,
    "logging_mode": "debug",
    "parallel_agents": True,
    "llm_stagger_ms": 500,
    "max_concurrent_llm": 3,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_filename(roster, model, temp, rounds, pid_cfg, judge, cash) -> str:
    roles = sorted(roster.keys())

    # Name by missing role (3-agent combos)
    missing = {"value", "risk", "technical", "macro"} - set(roles)
    if len(missing) == 1:
        role_tag = "no_" + "_".join(sorted(missing))
    elif len(missing) == 0:
        role_tag = "all_4"
    else:
        role_tag = "_".join(r[0] for r in roles)

    # Detect profile variants
    is_intense = any("intense" in v for v in roster.values())
    is_light = any("light" in v for v in roster.values())

    variant = "standard"
    if is_intense and is_light:
        variant = "intense_light"
    elif is_intense:
        variant = "intense"
    elif is_light:
        variant = "light"

    parts = [
        role_tag,
        variant,
        pid_cfg["name"],
        f"r{rounds}",
        f"t{temp}",
    ]

    return "debate_" + "_".join(parts) + ".yaml"


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate() -> None:

    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)

    TARGET_DIR.mkdir(parents=True)

    axes = list(product(
        AGENT_ROSTERS,
        LLM_MODELS,
        TEMPERATURES,
        MAX_ROUNDS,
        PID_CONFIGS,
        JUDGE_PROFILES,
        INITIAL_CASH,
    ))

    print(f"Generating {len(axes)} config files in {TARGET_DIR}/")

    for roster, model, temp, rounds, pid_cfg, judge, cash in axes:

        filename = build_filename(roster, model, temp, rounds, pid_cfg, judge, cash)

        config = dict(FIXED)
        config["agents"] = dict(roster)
        config["judge_profile"] = judge

        debate = dict(FIXED_DEBATE)
        debate["llm_model"] = model
        debate["temperature"] = temp
        debate["max_rounds"] = rounds
        debate["pid_enabled"] = True
        debate["crit_llm_model"] = "gpt-5"
        debate["experiment_name"] = "vskarich_ablation_2"

        # Inject PID parameters
        debate.update({
            "pid_kp": pid_cfg["pid_kp"],
            "pid_ki": pid_cfg["pid_ki"],
            "pid_kd": pid_cfg["pid_kd"],
            "pid_rho_star": pid_cfg["pid_rho_star"],
            "pid_initial_beta": pid_cfg["pid_initial_beta"],
            "pid_epsilon": pid_cfg["pid_epsilon"],
            "pid_gamma_beta": pid_cfg["pid_gamma_beta"],
        })

        config["debate_setup"] = debate
        config["broker"] = {"initial_cash": cash}

        out_path = TARGET_DIR / filename
        out_path.write_text(
            yaml.dump(config, default_flow_style=False, sort_keys=False)
        )

        print(f"  {filename}")

    print(f"Done. {len(axes)} files written.")


if __name__ == "__main__":
    generate()
