"""Baseline configuration, sweep group definitions, ticker/role presets, and scenarios.

This is a stability-first ablation for control-theoretic validation of
inference-time PID regulation. Primary goals:
  1) Identify stable PID parameter regions
  2) Measure convergence behavior (ρ and β)
  3) Characterize quadrant dynamics and transitions
  4) Detect oscillation / overshoot / instability
  5) Distinguish control stability from behavioral stability
"""

from __future__ import annotations

from multi_agent.models import (
    Constraints,
    MarketState,
    Observation,
    PortfolioState,
)

# =============================================================================
# CONSTANTS
# =============================================================================

# Default number of replicates per config (LLMs are stochastic)
DEFAULT_REPLICATES = 3

# Convergence window: JS < epsilon for W consecutive rounds
CONVERGENCE_WINDOW = 2

# Oscillation detection: sign changes > K → oscillation flag
OSCILLATION_K = 3

# Settling band: |beta_t - beta_final| < EPSILON_BAND
EPSILON_BAND = 0.05

# High correction threshold (|u_t| above this counts as escalation)
HIGH_CORRECTION_THRESHOLD = 0.3

# Number of random gain samples
NUM_RANDOM_GAIN_SAMPLES = 25

# Temperature regime boundary
STOCHASTIC_REGIME_THRESHOLD = 0.7

# Quarterly episodes: 4 quarters in 2025
INVEST_QUARTERS = ["2025Q1", "2025Q2", "2025Q3", "2025Q4"]

# Default dataset path for memo-based quarterly data
DEFAULT_DATASET_PATH = "data-pipeline/final_snapshots"

# Default memo format
DEFAULT_MEMO_FORMAT = "text"

# =============================================================================
# BASELINE
# =============================================================================

BASELINE: dict = {
    "Kp": 0.15,
    "Ki": 0.01,
    "Kd": 0.03,
    "rho_star": 0.8,
    "gamma_beta": 0.9,
    "delta_js": 0.05,
    "delta_s": 0.05,
    "delta_beta": 0.1,
    "epsilon": 0.01,
    "mu": 1.0,
    "initial_beta": 0.5,
    "max_rounds": 10,
    "temperature": 0.3,
    "model_name": "gpt-4o-mini",
    "tickers": ["AAPL", "NVDA", "MSFT", "GOOG", "JPM"],
    "roles": ["macro", "value", "risk", "technical"],
    "agreeableness": 0.3,
    "pid_propose": False,
    "pid_critique": True,
    "pid_revise": True,
    "enable_adversarial": False,
}

# =============================================================================
# SWEEP GROUPS
# =============================================================================

SWEEP_GROUPS: dict[str, list[dict]] = {
    # --- Group 1: PID gains (15 runs) ---
    "gains": [
        {"Kp": v} for v in [0.0, 0.05, 0.1, 0.2, 0.3]
    ] + [
        {"Ki": v} for v in [0.0, 0.005, 0.02, 0.05, 0.1]
    ] + [
        {"Kd": v} for v in [0.0, 0.01, 0.05, 0.1, 0.2]
    ],

    # --- Group 2: Quality target (4 runs) ---
    "quality": [
        {"rho_star": v} for v in [0.5, 0.6, 0.7, 0.9]
    ],

    # --- Group 3: Beta dynamics (10 runs) ---
    "dynamics": [
        {"gamma_beta": v} for v in [0.5, 0.7, 0.8, 0.9, 0.95]
    ] + [
        {"initial_beta": v} for v in [0.1, 0.3, 0.5, 0.7, 0.9]
    ],

    # --- Group 4: Thresholds (15 runs) ---
    "thresholds": [
        {"delta_js": v} for v in [0.01, 0.03, 0.05, 0.1, 0.2]
    ] + [
        {"delta_s": v} for v in [0.01, 0.03, 0.05, 0.1, 0.2]
    ] + [
        {"delta_beta": v} for v in [0.05, 0.1, 0.15, 0.2, 0.3]
    ],

    # --- Group 5: Sycophancy & convergence (10 runs) ---
    "sycophancy": [
        {"mu": v} for v in [0.0, 0.5, 1.0, 2.0, 3.0]
    ] + [
        {"epsilon": v} for v in [0.001, 0.005, 0.01, 0.05, 0.1]
    ],

    # --- Group 6: Phase toggles (4 runs) ---
    "phases": [
        {"pid_propose": True, "pid_critique": True, "pid_revise": True},
        {"pid_propose": False, "pid_critique": True, "pid_revise": False},
        {"pid_propose": False, "pid_critique": False, "pid_revise": True},
        {"pid_propose": True, "pid_critique": False, "pid_revise": False},
    ],

    # --- Group 7: LLM models (5 runs) ---
    "models": [
        {"model_name": "gpt-4o"},
        {"model_name": "gpt-4.1-mini"},
        {"model_name": "claude-haiku-4-5-20251001"},
        {"model_name": "claude-sonnet-4-20250514"},
        {"model_name": "claude-sonnet-4-6-20250819"},
    ],

    # --- Group 8: Temperature (4 runs) ---
    "temperature": [
        {"temperature": v} for v in [0.1, 0.5, 0.7, 1.0]
    ],

    # --- Group 9: Ticker universe (5 runs) ---
    "tickers": [
        {"tickers": ["AAPL", "JPM", "XOM"]},
        {"tickers": ["XOM", "CAT", "UNH", "COST", "AMT"]},
        {"tickers": ["AAPL", "NVDA", "MSFT", "GOOG", "META",
                      "JPM", "UNH", "XOM", "COST", "CAT"]},
        {"tickers": ["AAPL", "NVDA", "MSFT", "GOOG", "META", "AMZN", "TSLA",
                      "JPM", "GS", "BAC", "UNH", "LLY", "JNJ", "XOM", "COST"]},
        {"tickers": ["AAPL", "NVDA", "MSFT", "GOOG", "META", "NFLX",
                      "AMZN", "TSLA", "COST", "WMT", "JPM", "GS", "BAC",
                      "UNH", "LLY", "JNJ", "XOM", "CAT", "DAL", "AMT"]},
    ],

    # --- Group 10: Agent roster (5 runs) ---
    "agents": [
        {"roles": ["macro", "risk"]},
        {"roles": ["macro", "value", "risk"]},
        {"roles": ["macro", "value", "risk", "technical"],
         "enable_adversarial": True},
        {"roles": ["macro", "value", "risk", "technical", "sentiment"]},
        {"roles": ["macro", "value", "risk", "technical", "sentiment"],
         "enable_adversarial": True},
    ],

    # --- Group 11: Interaction combos (8 runs) ---
    "interactions": [
        {"Kp": 0.3, "Ki": 0.01, "Kd": 0.03},
        {"Kp": 0.15, "Ki": 0.05, "Kd": 0.03},
        {"Kp": 0.15, "Ki": 0.01, "Kd": 0.15},
        {"Kp": 0.15, "Ki": 0.01, "Kd": 0.03, "rho_star": 0.9},
        {"Kp": 0.15, "Ki": 0.01, "Kd": 0.03, "rho_star": 0.6},
        {"Kp": 0.15, "Ki": 0.01, "Kd": 0.03, "gamma_beta": 0.5},
        {"Kp": 0.0, "Ki": 0.0, "Kd": 0.0},
        {"Kp": 0.3, "Ki": 0.05, "Kd": 0.1, "rho_star": 0.9, "gamma_beta": 0.7},
    ],

    # --- Group 12: Random gain samples (populated at runtime by matrix.py) ---
    "random_gain_samples": [],

    # --- Group 13: High-gain stress (boundary mapping) ---
    "high_gain_stress": [
        {"Kp": 0.4, "Ki": 0.01, "Kd": 0.03},
        {"Kp": 0.5, "Ki": 0.01, "Kd": 0.03},
        {"Kp": 0.15, "Ki": 0.1, "Kd": 0.03},
        {"Kp": 0.15, "Ki": 0.2, "Kd": 0.03},
        {"Kp": 0.15, "Ki": 0.01, "Kd": 0.3},
        {"Kp": 0.4, "Ki": 0.1, "Kd": 0.2},
    ],

    # --- Group 14: High-mu stress (sycophancy penalty boundary) ---
    "high_mu_stress": [
        {"mu": 3.0},
        {"mu": 5.0},
        {"mu": 10.0},
        {"mu": 5.0, "Kp": 0.3},
        {"mu": 10.0, "Ki": 0.05},
    ],

    # --- Group 15: High-rho_star stress (unreachable target) ---
    "high_rho_star_stress": [
        {"rho_star": 0.85},
        {"rho_star": 0.9},
        {"rho_star": 0.95},
        {"rho_star": 0.99},
        {"rho_star": 0.95, "Kp": 0.3, "Ki": 0.05},
    ],
}

# Named labels for groups that have fixed-order entries
INTERACTION_LABELS = [
    "aggressive_p", "aggressive_i", "aggressive_d",
    "high_bar", "low_bar", "fast_decay", "no_pid", "kitchen_sink",
]
PHASE_LABELS = ["all_phases", "critique_only", "revise_only", "propose_only"]
TICKER_LABELS = [
    "3-cross-sector", "5-non-tech", "10-broad", "15-large", "20-full",
]
AGENT_LABELS = [
    "2-agent", "3-agent", "4+adversarial", "5-agent", "5+adversarial",
]
HIGH_GAIN_STRESS_LABELS = [
    "Kp-0.4", "Kp-0.5", "Ki-0.1", "Ki-0.2", "Kd-0.3", "all-high",
]
HIGH_MU_STRESS_LABELS = [
    "mu-3", "mu-5", "mu-10", "mu-5_Kp-0.3", "mu-10_Ki-0.05",
]
HIGH_RHO_STRESS_LABELS = [
    "rho-0.85", "rho-0.9", "rho-0.95", "rho-0.99", "rho-0.95_aggressive",
]


# =============================================================================
# SCENARIO OBSERVATIONS
# =============================================================================

REFERENCE_PRICES: dict[str, float] = {
    "AAPL": 185.50, "NVDA": 420.00, "MSFT": 390.00,
    "GOOG": 142.30, "META": 310.00, "NFLX": 450.00,
    "AMZN": 175.00, "TSLA": 240.00,
    "COST": 580.00, "WMT": 165.00,
    "JPM": 195.00, "GS": 380.00, "BAC": 37.50,
    "UNH": 520.00, "LLY": 750.00, "JNJ": 155.00,
    "XOM": 105.00, "CAT": 340.00, "DAL": 48.00, "AMT": 210.00,
}


def _build_observation(
    tickers: list[str],
    prices: dict[str, float],
    returns: dict[str, float],
    volatility: dict[str, float],
    text_context: str,
) -> Observation:
    """Build a synthetic Observation for ablation testing."""
    return Observation(
        timestamp="2025-03-15T10:00:00Z",
        universe=tickers,
        market_state=MarketState(
            prices={t: prices.get(t, 150.0) for t in tickers},
            returns={t: returns.get(t, 0.0) for t in tickers},
            volatility={t: volatility.get(t, 0.25) for t in tickers},
        ),
        text_context=text_context,
        portfolio_state=PortfolioState(cash=100000.0, positions={}),
        constraints=Constraints(max_leverage=2.0, max_position_size=500),
    )


def build_scenario_observations(tickers: list[str]) -> dict[str, Observation]:
    """Build scenario observations for the given ticker universe.

    Returns dict mapping scenario name to Observation.
    Includes 4 scenarios: bullish, neutral, riskoff, conflicted.
    """
    bullish_returns = {t: 0.02 for t in tickers}
    bullish_vol = {t: 0.20 for t in tickers}

    neutral_returns = {t: (0.005 if i % 2 == 0 else -0.003)
                       for i, t in enumerate(tickers)}
    neutral_vol = {t: 0.25 for t in tickers}

    riskoff_returns = {t: -0.03 for t in tickers}
    riskoff_vol = {t: 0.35 for t in tickers}

    # Conflicted scenario: high dispersion, contradictory signals
    # Designed to induce genuine disagreement and quadrant transitions
    conflicted_returns: dict[str, float] = {}
    conflicted_vol: dict[str, float] = {}
    for i, t in enumerate(tickers):
        if i % 4 == 0:
            conflicted_returns[t] = 0.04   # Strong momentum
            conflicted_vol[t] = 0.30
        elif i % 4 == 1:
            conflicted_returns[t] = -0.05  # Sharp decline
            conflicted_vol[t] = 0.40
        elif i % 4 == 2:
            conflicted_returns[t] = 0.001  # Flat
            conflicted_vol[t] = 0.15
        else:
            conflicted_returns[t] = -0.02  # Mild decline
            conflicted_vol[t] = 0.35

    return {
        "bullish": _build_observation(
            tickers, REFERENCE_PRICES, bullish_returns, bullish_vol,
            "Fed signals rate cuts. Earnings season strong. Market breadth improving. "
            "Risk appetite high across sectors.",
        ),
        "neutral": _build_observation(
            tickers, REFERENCE_PRICES, neutral_returns, neutral_vol,
            "Mixed economic signals. Inflation data in line. Tech sector consolidating. "
            "Bond yields stable. Sideways market expected.",
        ),
        "riskoff": _build_observation(
            tickers, REFERENCE_PRICES, riskoff_returns, riskoff_vol,
            "Geopolitical tensions escalate. VIX spikes to 28. Broad selloff underway. "
            "Flight to safety. Treasury yields dropping sharply.",
        ),
        "conflicted": _build_observation(
            tickers, REFERENCE_PRICES, conflicted_returns, conflicted_vol,
            "Contradictory signals across sectors. Tech names showing strong momentum "
            "despite negative earnings surprises in NVDA. Energy collapsing on OPEC "
            "production increase while financials rally on yield curve steepening. "
            "Fed rhetoric hawkish but market pricing in cuts. High cross-sector "
            "dispersion. Analyst consensus fractured.",
        ),
    }


# =============================================================================
# ALL AVAILABLE GROUP NAMES
# =============================================================================

ALL_GROUPS = list(SWEEP_GROUPS.keys())
