#!/usr/bin/env python3
"""Generate ablation debate config YAML files.

Produces the Cartesian product of all settings lists below.
Each combination gets its own YAML file in the target directory.

Usage:
    python config/debate/vskarich_ablations/generate_debate_config_ablations.py
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
# Ablation axes — edit these lists to control the sweep
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Roster generation — all role pairs and triples × all profile variants
# ---------------------------------------------------------------------------
ALL_ROLES = ["value", "risk", "technical", "macro"]

# Profile suffix variants (appended to "{role}_enriched")
PROFILE_VARIANTS = [
    "",                 # standard
    "_intense",         # intense role
    "_light",           # light contract
    "_intense_light",   # intense role + light contract
]

# Role combinations to ablate: all 3-agent and 2-agent subsets
from itertools import combinations

ROLE_COMBOS = [
    list(combo) for combo in combinations(ALL_ROLES, 3)
] + [
    list(combo) for combo in combinations(ALL_ROLES, 2)
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
    2,
]

PID_ENABLED = [
    True,
]

# ---------------------------------------------------------------------------
# PID controller configurations
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

    # For 2-agent combos, name by roles present; for 3-agent, name by missing role
    if len(roles) == 2:
        role_tag = "_".join(r[0] for r in roles)  # e.g. "r_v" for risk+value
    elif len(roles) == 4:
        role_tag = "all_4"
    else:
        missing = {"value", "risk", "technical", "macro"} - set(roles)
        role_tag = "no_" + "_".join(sorted(missing))

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
        debate["experiment_name"] = "vskarich_ablation_1"

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