"""Tests for scripts/ablation/config.py — constants, baseline, and scenario builders."""

from __future__ import annotations

import pytest

from scripts.ablation.config import (
    ALL_GROUPS,
    BASELINE,
    CONVERGENCE_WINDOW,
    DEFAULT_DATASET_PATH,
    DEFAULT_MEMO_FORMAT,
    DEFAULT_REPLICATES,
    EPSILON_BAND,
    HIGH_CORRECTION_THRESHOLD,
    INVEST_QUARTERS,
    NUM_RANDOM_GAIN_SAMPLES,
    OSCILLATION_K,
    REFERENCE_PRICES,
    STOCHASTIC_REGIME_THRESHOLD,
    SWEEP_GROUPS,
    build_scenario_observations,
)


# ===========================================================================
# Constants
# ===========================================================================


class TestConstants:
    def test_default_replicates_positive(self):
        assert DEFAULT_REPLICATES >= 1

    def test_convergence_window_positive(self):
        assert CONVERGENCE_WINDOW >= 1

    def test_oscillation_k_positive(self):
        assert OSCILLATION_K >= 1

    def test_epsilon_band_positive(self):
        assert EPSILON_BAND > 0

    def test_high_correction_threshold_range(self):
        assert 0 < HIGH_CORRECTION_THRESHOLD <= 1.0

    def test_stochastic_regime_threshold_range(self):
        assert 0 < STOCHASTIC_REGIME_THRESHOLD <= 1.0

    def test_num_random_gain_samples_positive(self):
        assert NUM_RANDOM_GAIN_SAMPLES >= 1

    def test_invest_quarters_four(self):
        assert len(INVEST_QUARTERS) == 4
        for q in INVEST_QUARTERS:
            assert q.startswith("2025Q")

    def test_default_dataset_path_is_string(self):
        assert isinstance(DEFAULT_DATASET_PATH, str)

    def test_default_memo_format_valid(self):
        assert DEFAULT_MEMO_FORMAT in ("text", "json")


# ===========================================================================
# Baseline
# ===========================================================================


class TestBaseline:
    def test_baseline_has_pid_gains(self):
        for key in ("Kp", "Ki", "Kd"):
            assert key in BASELINE
            assert isinstance(BASELINE[key], (int, float))

    def test_baseline_has_required_fields(self):
        required = [
            "rho_star", "gamma_beta", "delta_js", "delta_s", "delta_beta",
            "epsilon", "mu", "initial_beta", "max_rounds", "temperature",
            "model_name", "tickers", "roles", "agreeableness",
            "pid_propose", "pid_critique", "pid_revise", "enable_adversarial",
        ]
        for key in required:
            assert key in BASELINE, f"Missing baseline key: {key}"

    def test_baseline_gains_in_valid_range(self):
        assert 0 <= BASELINE["Kp"] <= 1.0
        assert 0 <= BASELINE["Ki"] <= 1.0
        assert 0 <= BASELINE["Kd"] <= 1.0

    def test_baseline_rho_star_in_range(self):
        assert 0 < BASELINE["rho_star"] <= 1.0

    def test_baseline_gamma_beta_in_range(self):
        assert 0 < BASELINE["gamma_beta"] < 1.0

    def test_baseline_initial_beta_in_range(self):
        assert 0 <= BASELINE["initial_beta"] <= 1.0

    def test_baseline_tickers_nonempty(self):
        assert len(BASELINE["tickers"]) >= 1

    def test_baseline_roles_nonempty(self):
        assert len(BASELINE["roles"]) >= 2


# ===========================================================================
# Sweep Groups
# ===========================================================================


class TestSweepGroups:
    def test_all_groups_match_sweep_groups_keys(self):
        assert set(ALL_GROUPS) == set(SWEEP_GROUPS.keys())

    def test_gains_group_has_entries(self):
        assert len(SWEEP_GROUPS["gains"]) == 15  # 5 Kp + 5 Ki + 5 Kd

    def test_quality_group_has_entries(self):
        assert len(SWEEP_GROUPS["quality"]) == 4

    def test_dynamics_group_has_entries(self):
        assert len(SWEEP_GROUPS["dynamics"]) == 10

    def test_thresholds_group_has_entries(self):
        assert len(SWEEP_GROUPS["thresholds"]) == 15

    def test_sycophancy_group_has_entries(self):
        assert len(SWEEP_GROUPS["sycophancy"]) == 10

    def test_phases_group_has_entries(self):
        assert len(SWEEP_GROUPS["phases"]) == 4

    def test_models_group_has_entries(self):
        assert len(SWEEP_GROUPS["models"]) >= 3

    def test_interactions_group_has_no_pid_entry(self):
        no_pid = {"Kp": 0.0, "Ki": 0.0, "Kd": 0.0}
        entries = SWEEP_GROUPS["interactions"]
        assert any(
            e.get("Kp") == 0.0 and e.get("Ki") == 0.0 and e.get("Kd") == 0.0
            for e in entries
        )

    def test_random_gain_samples_starts_empty(self):
        # Populated at runtime by matrix.py
        assert SWEEP_GROUPS["random_gain_samples"] == []

    def test_stress_groups_exist(self):
        assert "high_gain_stress" in SWEEP_GROUPS
        assert "high_mu_stress" in SWEEP_GROUPS
        assert "high_rho_star_stress" in SWEEP_GROUPS

    def test_each_group_entry_is_dict(self):
        for name, entries in SWEEP_GROUPS.items():
            for entry in entries:
                assert isinstance(entry, dict), f"Non-dict entry in {name}"


# ===========================================================================
# Reference Prices
# ===========================================================================


class TestReferencePrices:
    def test_all_baseline_tickers_have_prices(self):
        for ticker in BASELINE["tickers"]:
            assert ticker in REFERENCE_PRICES

    def test_prices_are_positive(self):
        for ticker, price in REFERENCE_PRICES.items():
            assert price > 0, f"Non-positive price for {ticker}"


# ===========================================================================
# Scenario Observations
# ===========================================================================


class TestScenarioObservations:
    def test_four_scenarios_returned(self):
        tickers = ["AAPL", "NVDA"]
        obs = build_scenario_observations(tickers)
        assert set(obs.keys()) == {"bullish", "neutral", "riskoff", "conflicted"}

    def test_observation_has_correct_universe(self):
        tickers = ["AAPL", "JPM", "XOM"]
        obs = build_scenario_observations(tickers)
        for name, o in obs.items():
            assert o.universe == tickers, f"Wrong universe for {name}"

    def test_observation_has_market_state(self):
        tickers = ["AAPL"]
        obs = build_scenario_observations(tickers)
        for name, o in obs.items():
            assert o.market_state is not None
            assert "AAPL" in o.market_state.prices

    def test_observation_has_portfolio_state(self):
        tickers = ["AAPL"]
        obs = build_scenario_observations(tickers)
        for name, o in obs.items():
            assert o.portfolio_state is not None
            assert o.portfolio_state.cash == 100000.0

    def test_observation_has_constraints(self):
        tickers = ["AAPL"]
        obs = build_scenario_observations(tickers)
        for name, o in obs.items():
            assert o.constraints is not None
            assert o.constraints.max_leverage == 2.0

    def test_bullish_has_positive_returns(self):
        tickers = ["AAPL", "NVDA"]
        obs = build_scenario_observations(tickers)
        for t in tickers:
            assert obs["bullish"].market_state.returns[t] > 0

    def test_riskoff_has_negative_returns(self):
        tickers = ["AAPL", "NVDA"]
        obs = build_scenario_observations(tickers)
        for t in tickers:
            assert obs["riskoff"].market_state.returns[t] < 0

    def test_riskoff_has_high_volatility(self):
        tickers = ["AAPL"]
        obs = build_scenario_observations(tickers)
        assert obs["riskoff"].market_state.volatility["AAPL"] > 0.30

    def test_conflicted_has_mixed_returns(self):
        tickers = ["AAPL", "NVDA", "MSFT", "GOOG"]
        obs = build_scenario_observations(tickers)
        returns = obs["conflicted"].market_state.returns
        vals = list(returns.values())
        # Should have both positive and negative returns
        assert any(v > 0 for v in vals)
        assert any(v < 0 for v in vals)

    def test_text_context_nonempty(self):
        tickers = ["AAPL"]
        obs = build_scenario_observations(tickers)
        for name, o in obs.items():
            assert o.text_context and len(o.text_context) > 10

    def test_single_ticker_works(self):
        obs = build_scenario_observations(["AAPL"])
        assert len(obs) == 4

    def test_large_ticker_set_works(self):
        tickers = list(REFERENCE_PRICES.keys())
        obs = build_scenario_observations(tickers)
        for name, o in obs.items():
            assert len(o.universe) == len(tickers)
