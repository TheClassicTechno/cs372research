"""
Unit tests for analysis/stratified_analysis.py

Each test class covers one logical section of the module:
  TestRegimeManifest       — loading and validating the YAML manifest
  TestLookupRegime         — pattern matching and quarter fallback
  TestNormaliseQuarter     — invest_quarter string normalisation
  TestExtractScenarioName  — pulling scenario name from manifest dict
  TestLoadCritScores       — reading CRIT data from fixture directories
  TestLoadFinancialMetrics — reading _dashboard/financial_metrics.json
  TestLoadRunAsAgentRows   — end-to-end row extraction from one run dir
  TestCorrPair             — correlation computation helper
  TestCorrByRole           — per-role correlation
  TestCorrByRegime         — per-regime correlation
  TestCorrGrid             — role × regime cross-stratification
  TestFormatting           — output formatting (smoke tests)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from analysis.stratified_analysis import (
    PILLARS,
    MIN_OBS_SKIP,
    _corr_pair,
    _fmt_corr,
    _normalise_crit_block,
    _normalise_crit_block_from_scores_file,
    _normalise_quarter,
    corr_by_regime,
    corr_by_role,
    corr_grid,
    extract_scenario_name,
    format_grid_table,
    format_regime_table,
    format_role_table,
    load_all_agent_rows,
    load_financial_metrics,
    load_regime_manifest,
    load_run_as_agent_rows,
    lookup_regime,
    pillar_corr_by_role,
    write_grid_csv,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_MANIFEST_DATA = {
    "patterns": [
        {"pattern": "inflation_shock", "regime": "inflation_shock"},
        {"pattern": "recession",       "regime": "recession"},
        {"pattern": "ai_boom",         "regime": "tech_rally"},
        {"pattern": "mock",            "regime": "neutral"},
    ],
    "quarter_fallback": {
        "2022-03-31": "inflation_shock",
        "2022-06-30": "recession",
        "2023-12-31": "tech_rally",
    },
}

SAMPLE_CRIT_SCORES_FILE = {
    "round": 1,
    "rho_bar": 0.80,
    "agent_scores": {
        "macro": {
            "rho_i": 0.85,
            "pillar_scores": {"LV": 0.80, "ES": 0.88, "AC": 0.90, "CA": 0.82},
            "diagnostics": {},
            "explanations": {},
        },
        "risk": {
            "rho_i": 0.75,
            "pillar_scores": {"LV": 0.70, "ES": 0.78, "AC": 0.80, "CA": 0.72},
            "diagnostics": {},
            "explanations": {},
        },
        "technical": {
            "rho_i": 0.80,
            "pillar_scores": {"LV": 0.82, "ES": 0.79, "AC": 0.81, "CA": 0.78},
            "diagnostics": {},
            "explanations": {},
        },
    },
}

SAMPLE_PID_CRIT_ALL_ROUNDS = [
    {
        "type": "pid_round",
        "round": 1,
        "beta_in": 0.4,
        "crit": {
            "rho_bar": 0.80,
            "rho_i": {"macro": 0.85, "risk": 0.75, "technical": 0.80},
            "agents": {
                "macro":    {"rho_i": 0.85, "pillars": {"LV": 0.80, "ES": 0.88, "AC": 0.90, "CA": 0.82}},
                "risk":     {"rho_i": 0.75, "pillars": {"LV": 0.70, "ES": 0.78, "AC": 0.80, "CA": 0.72}},
                "technical":{"rho_i": 0.80, "pillars": {"LV": 0.82, "ES": 0.79, "AC": 0.81, "CA": 0.78}},
            },
        },
    }
]

SAMPLE_FINANCIAL_METRICS = {
    "daily_metrics_annualized_sharpe": 0.75,
    "daily_metrics_total_return_pct": 8.5,
    "daily_metrics_max_drawdown_pct": 5.2,
}

SAMPLE_MANIFEST_JSON = {
    "experiment_name": "test_ablation",
    "run_id": "run_2022-01-01_00-00-00",
    "config_paths": [
        "configs/debate.yaml",
        "configs/scenarios/2022Q1_inflation_shock.yaml",
    ],
    "invest_quarter": "2022-03-31",
    "roles": ["macro", "risk", "technical"],
}


def _write_json(path: Path, data) -> None:
    """Write JSON to path, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _build_run_dir(
    base: Path,
    experiment: str = "test_exp",
    run_id: str = "run_2022-01-01_00-00-00",
    manifest: dict | None = None,
    crit_scores: dict | None = None,
    pid_crit_all: list | None = None,
    financial: dict | None = None,
) -> Path:
    """Build a synthetic run directory for use in tests.

    Creates the standard directory structure. Pass None for any optional
    argument to omit that file (tests can verify graceful handling).
    """
    run_dir = base / experiment / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if manifest is not None:
        _write_json(run_dir / "manifest.json", manifest)

    if crit_scores is not None:
        _write_json(
            run_dir / "rounds" / "round_001" / "metrics" / "crit_scores.json",
            crit_scores,
        )

    if pid_crit_all is not None:
        _write_json(run_dir / "final" / "pid_crit_all_rounds.json", pid_crit_all)

    if financial is not None:
        _write_json(run_dir / "_dashboard" / "financial_metrics.json", financial)

    return run_dir


# ---------------------------------------------------------------------------
# TestRegimeManifest
# ---------------------------------------------------------------------------

class TestRegimeManifest:
    """Tests for load_regime_manifest()."""

    def test_loads_real_manifest(self):
        # The actual config/regime_manifest.yaml must exist and be valid YAML
        manifest_data = load_regime_manifest()
        assert "patterns" in manifest_data
        assert isinstance(manifest_data["patterns"], list)
        assert len(manifest_data["patterns"]) > 0

    def test_patterns_have_required_keys(self):
        manifest_data = load_regime_manifest()
        for entry in manifest_data["patterns"]:
            assert "pattern" in entry, f"Entry missing 'pattern': {entry}"
            assert "regime" in entry, f"Entry missing 'regime': {entry}"

    def test_quarter_fallback_present(self):
        manifest_data = load_regime_manifest()
        assert "quarter_fallback" in manifest_data
        assert isinstance(manifest_data["quarter_fallback"], dict)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_regime_manifest(tmp_path / "nonexistent.yaml")

    def test_loads_from_custom_path(self, tmp_path):
        # Write a minimal manifest to a temp path and load it
        custom = tmp_path / "custom_manifest.yaml"
        custom.write_text(yaml.dump(SAMPLE_MANIFEST_DATA))
        result = load_regime_manifest(custom)
        assert result["patterns"][0]["regime"] == "inflation_shock"


# ---------------------------------------------------------------------------
# TestLookupRegime
# ---------------------------------------------------------------------------

class TestLookupRegime:
    """Tests for lookup_regime()."""

    def test_pattern_match_first_wins(self):
        # "inflation_shock" pattern should match before quarter fallback
        regime = lookup_regime(
            "2022Q1_inflation_shock",
            "2022-03-31",
            SAMPLE_MANIFEST_DATA,
        )
        assert regime == "inflation_shock"

    def test_pattern_match_case_insensitive(self):
        regime = lookup_regime("2022Q1_INFLATION_SHOCK", "2022-03-31", SAMPLE_MANIFEST_DATA)
        assert regime == "inflation_shock"

    def test_falls_back_to_quarter(self):
        # No pattern matches "mystery_scenario" — use quarter fallback
        regime = lookup_regime("mystery_scenario", "2022-06-30", SAMPLE_MANIFEST_DATA)
        assert regime == "recession"

    def test_unknown_when_no_match(self):
        regime = lookup_regime("completely_unknown", "1999-01-01", SAMPLE_MANIFEST_DATA)
        assert regime == "unknown"

    def test_quarter_in_compact_form(self):
        # "2022Q1" compact form should be normalised and matched
        regime = lookup_regime("mystery", "2022Q1", SAMPLE_MANIFEST_DATA)
        assert regime == "inflation_shock"

    def test_ai_boom_pattern(self):
        regime = lookup_regime("2023Q4_AI_boom", "2023-12-31", SAMPLE_MANIFEST_DATA)
        assert regime == "tech_rally"

    def test_mock_pattern(self):
        regime = lookup_regime("debate_mock_001", "2025-01-01", SAMPLE_MANIFEST_DATA)
        assert regime == "neutral"


# ---------------------------------------------------------------------------
# TestNormaliseQuarter
# ---------------------------------------------------------------------------

class TestNormaliseQuarter:
    """Tests for _normalise_quarter()."""

    def test_iso_date_unchanged(self):
        assert _normalise_quarter("2022-06-30") == "2022-06-30"

    def test_compact_q1(self):
        assert _normalise_quarter("2022Q1") == "2022-03-31"

    def test_compact_q2(self):
        assert _normalise_quarter("2022Q2") == "2022-06-30"

    def test_compact_q3(self):
        assert _normalise_quarter("2022Q3") == "2022-09-30"

    def test_compact_q4(self):
        assert _normalise_quarter("2022Q4") == "2022-12-31"

    def test_empty_string(self):
        assert _normalise_quarter("") == ""

    def test_whitespace_stripped(self):
        assert _normalise_quarter("  2022Q2  ") == "2022-06-30"


# ---------------------------------------------------------------------------
# TestExtractScenarioName
# ---------------------------------------------------------------------------

class TestExtractScenarioName:
    """Tests for extract_scenario_name()."""

    def test_extracts_from_config_paths(self):
        manifest = {
            "config_paths": [
                "configs/debate.yaml",
                "/path/to/scenarios/2022Q1_inflation_shock.yaml",
            ]
        }
        assert extract_scenario_name(manifest) == "2022Q1_inflation_shock"

    def test_extracts_basename_without_extension(self):
        manifest = {
            "config_paths": ["debate.yaml", "scenario_2022Q4_tech.yaml"]
        }
        assert extract_scenario_name(manifest) == "scenario_2022Q4_tech"

    def test_falls_back_to_invest_quarter(self):
        # Only one config_path (no scenario path at index 1)
        manifest = {"config_paths": ["debate.yaml"], "invest_quarter": "2022Q2"}
        name = extract_scenario_name(manifest)
        assert name == "2022-06-30"

    def test_missing_config_paths(self):
        manifest = {"invest_quarter": "2022Q3"}
        name = extract_scenario_name(manifest)
        assert name == "2022-09-30"

    def test_empty_manifest(self):
        name = extract_scenario_name({})
        assert name == "unknown_scenario"


# ---------------------------------------------------------------------------
# TestLoadCritScores  (internal normalisation helpers)
# ---------------------------------------------------------------------------

class TestLoadCritScores:
    """Tests for _normalise_crit_block and _normalise_crit_block_from_scores_file."""

    def test_normalise_from_pid_crit_block(self):
        # Simulates the 'crit' subdict from pid_crit_all_rounds.json
        crit_block = SAMPLE_PID_CRIT_ALL_ROUNDS[0]["crit"]
        result = _normalise_crit_block(crit_block)
        assert result["rho_bar"] == 0.80
        assert "macro" in result["agent_scores"]
        assert result["agent_scores"]["macro"]["rho_i"] == 0.85
        # Pillars stored as "pillars" key in aggregated file
        assert result["agent_scores"]["macro"]["pillar_scores"]["LV"] == 0.80

    def test_normalise_from_scores_file(self):
        result = _normalise_crit_block_from_scores_file(SAMPLE_CRIT_SCORES_FILE)
        assert result["rho_bar"] == 0.80
        assert "risk" in result["agent_scores"]
        assert result["agent_scores"]["risk"]["pillar_scores"]["ES"] == 0.78

    def test_three_roles_present(self):
        result = _normalise_crit_block_from_scores_file(SAMPLE_CRIT_SCORES_FILE)
        assert set(result["agent_scores"].keys()) == {"macro", "risk", "technical"}

    def test_all_pillars_present(self):
        result = _normalise_crit_block_from_scores_file(SAMPLE_CRIT_SCORES_FILE)
        for role in ("macro", "risk", "technical"):
            ps = result["agent_scores"][role]["pillar_scores"]
            for pillar in PILLARS:
                assert pillar in ps, f"Missing pillar {pillar} for role {role}"


# ---------------------------------------------------------------------------
# TestLoadFinancialMetrics
# ---------------------------------------------------------------------------

class TestLoadFinancialMetrics:
    """Tests for load_financial_metrics()."""

    def test_loads_valid_file(self, tmp_path):
        run_dir = tmp_path / "run_001"
        _write_json(run_dir / "_dashboard" / "financial_metrics.json", SAMPLE_FINANCIAL_METRICS)
        result = load_financial_metrics(run_dir)
        assert result is not None
        assert result["daily_metrics_annualized_sharpe"] == 0.75
        assert result["daily_metrics_total_return_pct"] == 8.5

    def test_returns_none_when_file_missing(self, tmp_path):
        run_dir = tmp_path / "run_001"
        run_dir.mkdir()
        assert load_financial_metrics(run_dir) is None

    def test_returns_none_when_sharpe_missing(self, tmp_path):
        run_dir = tmp_path / "run_001"
        # File exists but sharpe key is absent
        _write_json(
            run_dir / "_dashboard" / "financial_metrics.json",
            {"daily_metrics_total_return_pct": 5.0},
        )
        assert load_financial_metrics(run_dir) is None

    def test_returns_none_when_total_return_missing(self, tmp_path):
        run_dir = tmp_path / "run_001"
        _write_json(
            run_dir / "_dashboard" / "financial_metrics.json",
            {"daily_metrics_annualized_sharpe": 0.5},
        )
        assert load_financial_metrics(run_dir) is None

    def test_returns_none_on_corrupt_json(self, tmp_path):
        run_dir = tmp_path / "run_001"
        path = run_dir / "_dashboard" / "financial_metrics.json"
        path.parent.mkdir(parents=True)
        path.write_text("not json {{{{")
        assert load_financial_metrics(run_dir) is None


# ---------------------------------------------------------------------------
# TestLoadRunAsAgentRows
# ---------------------------------------------------------------------------

class TestLoadRunAsAgentRows:
    """Tests for load_run_as_agent_rows() — one run dir → list of agent rows."""

    def test_returns_one_row_per_role(self, tmp_path):
        run_dir = _build_run_dir(
            tmp_path,
            manifest=SAMPLE_MANIFEST_JSON,
            pid_crit_all=SAMPLE_PID_CRIT_ALL_ROUNDS,
            financial=SAMPLE_FINANCIAL_METRICS,
        )
        rows = load_run_as_agent_rows(run_dir, SAMPLE_MANIFEST_DATA)
        assert len(rows) == 3
        roles_found = {r["role"] for r in rows}
        assert roles_found == {"macro", "risk", "technical"}

    def test_row_has_correct_values(self, tmp_path):
        run_dir = _build_run_dir(
            tmp_path,
            manifest=SAMPLE_MANIFEST_JSON,
            pid_crit_all=SAMPLE_PID_CRIT_ALL_ROUNDS,
            financial=SAMPLE_FINANCIAL_METRICS,
        )
        rows = load_run_as_agent_rows(run_dir, SAMPLE_MANIFEST_DATA)
        macro_row = next(r for r in rows if r["role"] == "macro")
        assert macro_row["rho_i"] == 0.85
        assert macro_row["LV"] == 0.80
        assert macro_row["sharpe"] == 0.75
        assert macro_row["total_return"] == 8.5

    def test_regime_resolved_from_scenario(self, tmp_path):
        run_dir = _build_run_dir(
            tmp_path,
            manifest=SAMPLE_MANIFEST_JSON,   # config_paths[1] = inflation_shock scenario
            pid_crit_all=SAMPLE_PID_CRIT_ALL_ROUNDS,
            financial=SAMPLE_FINANCIAL_METRICS,
        )
        rows = load_run_as_agent_rows(run_dir, SAMPLE_MANIFEST_DATA)
        assert all(r["regime"] == "inflation_shock" for r in rows)

    def test_falls_back_to_per_round_crit(self, tmp_path):
        # No pid_crit_all_rounds.json — should fall back to crit_scores.json
        run_dir = _build_run_dir(
            tmp_path,
            manifest=SAMPLE_MANIFEST_JSON,
            crit_scores=SAMPLE_CRIT_SCORES_FILE,   # per-round file
            financial=SAMPLE_FINANCIAL_METRICS,
        )
        rows = load_run_as_agent_rows(run_dir, SAMPLE_MANIFEST_DATA)
        assert len(rows) == 3

    def test_returns_empty_when_no_manifest(self, tmp_path):
        run_dir = tmp_path / "test_exp" / "run_empty"
        run_dir.mkdir(parents=True)
        rows = load_run_as_agent_rows(run_dir, SAMPLE_MANIFEST_DATA)
        assert rows == []

    def test_returns_empty_when_no_financial(self, tmp_path):
        run_dir = _build_run_dir(
            tmp_path,
            manifest=SAMPLE_MANIFEST_JSON,
            pid_crit_all=SAMPLE_PID_CRIT_ALL_ROUNDS,
            financial=None,   # deliberately omit
        )
        rows = load_run_as_agent_rows(run_dir, SAMPLE_MANIFEST_DATA)
        assert rows == []

    def test_returns_empty_when_no_crit(self, tmp_path):
        run_dir = _build_run_dir(
            tmp_path,
            manifest=SAMPLE_MANIFEST_JSON,
            financial=SAMPLE_FINANCIAL_METRICS,
            # no crit_scores or pid_crit_all
        )
        rows = load_run_as_agent_rows(run_dir, SAMPLE_MANIFEST_DATA)
        assert rows == []

    def test_experiment_and_run_id_fields(self, tmp_path):
        run_dir = _build_run_dir(
            tmp_path,
            experiment="my_exp",
            run_id="run_2022-03-01_10-00-00",
            manifest=SAMPLE_MANIFEST_JSON,
            pid_crit_all=SAMPLE_PID_CRIT_ALL_ROUNDS,
            financial=SAMPLE_FINANCIAL_METRICS,
        )
        rows = load_run_as_agent_rows(run_dir, SAMPLE_MANIFEST_DATA)
        assert all(r["experiment"] == "my_exp" for r in rows)
        assert all(r["run_id"] == "run_2022-03-01_10-00-00" for r in rows)


# ---------------------------------------------------------------------------
# TestCorrPair
# ---------------------------------------------------------------------------

class TestCorrPair:
    """Tests for _corr_pair() — the core correlation primitive."""

    def test_perfect_positive_correlation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = x * 2.0
        result = _corr_pair(x, y)
        assert result["pearson_r"] == pytest.approx(1.0, abs=1e-4)
        assert result["pearson_p"] < 0.001
        assert result["n"] == 5

    def test_perfect_negative_correlation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = -x
        result = _corr_pair(x, y)
        assert result["pearson_r"] == pytest.approx(-1.0, abs=1e-4)

    def test_no_correlation(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(50)
        y = rng.standard_normal(50)
        result = _corr_pair(x, y)
        assert result["pearson_r"] is not None
        assert abs(result["pearson_r"]) < 0.5   # expected to be low

    def test_insufficient_data_returns_none_values(self):
        # MIN_OBS_SKIP == 3 — fewer than that should skip
        x = np.array([1.0, 2.0])
        y = np.array([2.0, 4.0])
        result = _corr_pair(x, y)
        assert result["pearson_r"] is None
        assert result["insufficient"] is True

    def test_nan_values_are_masked(self):
        x = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, np.nan, 8.0, 10.0])
        result = _corr_pair(x, y)
        # Only 3 clean pairs: (1,2), (4,8), (5,10)
        assert result["n"] == 3

    def test_all_nan_returns_skip(self):
        x = np.array([np.nan, np.nan, np.nan])
        y = np.array([np.nan, np.nan, np.nan])
        result = _corr_pair(x, y)
        assert result["pearson_r"] is None

    def test_warn_flag_set_when_n_below_min_obs_warn(self):
        # n=4 is below MIN_OBS_WARN=5 but above MIN_OBS_SKIP=3
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([2.0, 4.0, 6.0, 8.0])
        result = _corr_pair(x, y)
        assert result["insufficient"] is True
        assert result["pearson_r"] is not None   # computed anyway


# ---------------------------------------------------------------------------
# TestCorrByRole
# ---------------------------------------------------------------------------

class TestCorrByRole:
    """Tests for corr_by_role()."""

    def _make_df(self, n_per_role: int = 10) -> pd.DataFrame:
        """Build a synthetic agent-row DataFrame with a known correlation."""
        rng = np.random.default_rng(0)
        rows = []
        regimes = ["inflation_shock", "tech_rally"]
        for i, role in enumerate(["macro", "risk", "technical"]):
            rho_i = rng.uniform(0.5, 0.9, n_per_role)
            # macro has strong positive relationship; others noisy
            if role == "macro":
                sharpe = rho_i * 2 + rng.normal(0, 0.05, n_per_role)
            else:
                sharpe = rng.normal(0.5, 0.5, n_per_role)
            for j in range(n_per_role):
                rows.append({
                    "run_id": f"run_{role}_{j}",
                    "role": role,
                    "regime": regimes[j % 2],
                    "rho_i": rho_i[j],
                    "sharpe": sharpe[j],
                    "total_return": sharpe[j] * 10,
                    "rho_bar": rho_i[j],
                    **{p: rho_i[j] + rng.normal(0, 0.05) for p in ("LV", "ES", "AC", "CA")},
                })
        return pd.DataFrame(rows)

    def test_returns_entry_for_each_role(self):
        df = self._make_df()
        result = corr_by_role(df, "sharpe")
        assert set(result.keys()) == {"macro", "risk", "technical"}

    def test_macro_has_higher_correlation(self):
        df = self._make_df(n_per_role=20)
        result = corr_by_role(df, "sharpe")
        assert abs(result["macro"]["pearson_r"]) > 0.5
        # macro should dominate because of the construction above

    def test_insufficient_data_role_handled(self):
        df = self._make_df(n_per_role=2)
        result = corr_by_role(df, "sharpe")
        # n=2 < MIN_OBS_SKIP — should return None for pearson_r
        for role in result:
            assert result[role]["pearson_r"] is None

    def test_accepts_total_return_outcome(self):
        df = self._make_df()
        result = corr_by_role(df, "total_return")
        assert "macro" in result


# ---------------------------------------------------------------------------
# TestCorrByRegime
# ---------------------------------------------------------------------------

class TestCorrByRegime:
    """Tests for corr_by_regime()."""

    def _make_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        rows = []
        for regime in ["inflation_shock", "tech_rally", "recession"]:
            n = 12
            rho_bar = rng.uniform(0.5, 0.9, n)
            if regime == "inflation_shock":
                sharpe = rho_bar + rng.normal(0, 0.05, n)
            else:
                sharpe = rng.normal(0, 1.0, n)
            for j in range(n):
                rows.append({
                    "run_id": f"run_{regime}_{j}",
                    "role": "macro",
                    "regime": regime,
                    "rho_bar": rho_bar[j],
                    "sharpe": sharpe[j],
                    "total_return": sharpe[j] * 10,
                    "rho_i": rho_bar[j],
                    **{p: rho_bar[j] for p in ("LV", "ES", "AC", "CA")},
                })
        return pd.DataFrame(rows)

    def test_returns_entry_for_each_regime(self):
        df = self._make_df()
        result = corr_by_regime(df, "sharpe")
        assert set(result.keys()) == {"inflation_shock", "tech_rally", "recession"}

    def test_deduplicates_by_run_id(self):
        # Two rows with same run_id but different roles — regime should dedup
        df = pd.DataFrame([
            {"run_id": "r1", "role": "macro",   "regime": "recession",
             "rho_bar": 0.8, "rho_i": 0.8, "sharpe": 1.0, "total_return": 10.0,
             "LV": 0.8, "ES": 0.8, "AC": 0.8, "CA": 0.8},
            {"run_id": "r1", "role": "risk",    "regime": "recession",
             "rho_bar": 0.8, "rho_i": 0.7, "sharpe": 1.0, "total_return": 10.0,
             "LV": 0.7, "ES": 0.7, "AC": 0.7, "CA": 0.7},
            {"run_id": "r2", "role": "macro",   "regime": "recession",
             "rho_bar": 0.6, "rho_i": 0.6, "sharpe": 0.5, "total_return": 5.0,
             "LV": 0.6, "ES": 0.6, "AC": 0.6, "CA": 0.6},
            {"run_id": "r2", "role": "risk",    "regime": "recession",
             "rho_bar": 0.6, "rho_i": 0.5, "sharpe": 0.5, "total_return": 5.0,
             "LV": 0.5, "ES": 0.5, "AC": 0.5, "CA": 0.5},
        ])
        # After dedup: n should be 2 runs, not 4 rows
        result = corr_by_regime(df, "sharpe")
        assert result["recession"]["n"] == 2


# ---------------------------------------------------------------------------
# TestCorrGrid
# ---------------------------------------------------------------------------

class TestCorrGrid:
    """Tests for corr_grid()."""

    def _make_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(99)
        rows = []
        for role in ["macro", "risk"]:
            for regime in ["inflation_shock", "tech_rally"]:
                for j in range(8):
                    rho_i = rng.uniform(0.5, 0.9)
                    rows.append({
                        "run_id": f"r_{role}_{regime}_{j}",
                        "role": role, "regime": regime,
                        "rho_i": rho_i, "rho_bar": rho_i,
                        "sharpe": rho_i + rng.normal(0, 0.1),
                        "total_return": rho_i * 10,
                        "LV": rho_i, "ES": rho_i, "AC": rho_i, "CA": rho_i,
                    })
        return pd.DataFrame(rows)

    def test_grid_shape_correct(self):
        df = self._make_df()
        grid = corr_grid(df, "sharpe")
        assert set(grid.keys()) == {"macro", "risk"}
        for role in grid:
            assert set(grid[role].keys()) == {"inflation_shock", "tech_rally"}

    def test_each_cell_has_result_dict(self):
        df = self._make_df()
        grid = corr_grid(df, "sharpe")
        for role in grid:
            for regime in grid[role]:
                cell = grid[role][regime]
                assert "n" in cell
                assert "pearson_r" in cell

    def test_cell_n_matches_stratum_size(self):
        df = self._make_df()
        grid = corr_grid(df, "sharpe")
        # Each stratum has 8 rows in our synthetic data
        assert grid["macro"]["inflation_shock"]["n"] == 8

    def test_grid_with_empty_stratum(self):
        df = self._make_df()
        # Remove all "risk" + "tech_rally" rows
        df = df[~((df["role"] == "risk") & (df["regime"] == "tech_rally"))]
        grid = corr_grid(df, "sharpe")
        cell = grid["risk"]["tech_rally"]
        # Should return skip result, not raise
        assert cell["pearson_r"] is None


# ---------------------------------------------------------------------------
# TestFormatting  (smoke tests — check for crash-free output)
# ---------------------------------------------------------------------------

class TestFormatting:
    """Smoke tests for format_role_table, format_regime_table, format_grid_table."""

    def _make_minimal_df(self) -> pd.DataFrame:
        rows = []
        rng = np.random.default_rng(1)
        for role in ["macro", "risk"]:
            for regime in ["inflation_shock", "tech_rally"]:
                for j in range(6):
                    v = rng.uniform(0.5, 0.9)
                    rows.append({
                        "run_id": f"r_{role}_{regime}_{j}",
                        "role": role, "regime": regime,
                        "rho_i": v, "rho_bar": v,
                        "sharpe": v + rng.normal(0, 0.1),
                        "total_return": v * 10,
                        "LV": v, "ES": v, "AC": v, "CA": v,
                    })
        return pd.DataFrame(rows)

    def test_format_role_table_runs(self):
        df = self._make_minimal_df()
        role_corr = corr_by_role(df, "sharpe")
        pillar_corr = pillar_corr_by_role(df, "sharpe")
        text = format_role_table(role_corr, pillar_corr, "sharpe")
        assert "macro" in text
        assert "risk" in text
        assert "LV" in text

    def test_format_regime_table_runs(self):
        df = self._make_minimal_df()
        regime_corr = corr_by_regime(df, "sharpe")
        text = format_regime_table(regime_corr, "sharpe")
        assert "inflation_shock" in text
        assert "tech_rally" in text

    def test_format_grid_table_runs(self):
        df = self._make_minimal_df()
        grid = corr_grid(df, "sharpe")
        text = format_grid_table(grid, "sharpe")
        assert "macro" in text
        assert "inflation_shock" in text

    def test_fmt_corr_significant_shows_star(self):
        result = {"pearson_r": 0.72, "pearson_p": 0.01, "n": 12, "insufficient": False}
        text = _fmt_corr(result)
        assert "*" in text

    def test_fmt_corr_insignificant_no_star(self):
        result = {"pearson_r": 0.15, "pearson_p": 0.40, "n": 10, "insufficient": False}
        text = _fmt_corr(result)
        assert "*" not in text

    def test_fmt_corr_skip(self):
        result = {"pearson_r": None, "n": 2, "insufficient": True}
        text = _fmt_corr(result)
        assert "skip" in text

    def test_write_grid_csv(self, tmp_path):
        df = self._make_minimal_df()
        grid = corr_grid(df, "sharpe")
        csv_path = tmp_path / "grid.csv"
        write_grid_csv(grid, "sharpe", csv_path)
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "outcome" in content
        assert "role" in content
        assert "regime" in content


# ---------------------------------------------------------------------------
# TestLoadAllAgentRows  (integration-style, uses temp dirs)
# ---------------------------------------------------------------------------

class TestLoadAllAgentRows:
    """Tests for load_all_agent_rows() — walks multiple run dirs."""

    def test_loads_multiple_runs(self, tmp_path):
        for i in range(3):
            manifest = SAMPLE_MANIFEST_JSON.copy()
            manifest["run_id"] = f"run_000{i}"
            _build_run_dir(
                tmp_path,
                experiment="exp_a",
                run_id=f"run_000{i}",
                manifest=manifest,
                pid_crit_all=SAMPLE_PID_CRIT_ALL_ROUNDS,
                financial=SAMPLE_FINANCIAL_METRICS,
            )
        df = load_all_agent_rows(tmp_path, manifest_data=SAMPLE_MANIFEST_DATA)
        # 3 runs × 3 agents each = 9 rows
        assert len(df) == 9

    def test_returns_empty_df_for_nonexistent_root(self, tmp_path):
        df = load_all_agent_rows(
            tmp_path / "nonexistent",
            manifest_data=SAMPLE_MANIFEST_DATA,
        )
        assert df.empty

    def test_skips_runs_without_financial_data(self, tmp_path):
        # Run with financial data
        _build_run_dir(
            tmp_path, experiment="exp_a", run_id="run_good",
            manifest=SAMPLE_MANIFEST_JSON,
            pid_crit_all=SAMPLE_PID_CRIT_ALL_ROUNDS,
            financial=SAMPLE_FINANCIAL_METRICS,
        )
        # Run without financial data — should be silently skipped
        _build_run_dir(
            tmp_path, experiment="exp_a", run_id="run_no_fin",
            manifest=SAMPLE_MANIFEST_JSON,
            pid_crit_all=SAMPLE_PID_CRIT_ALL_ROUNDS,
            financial=None,
        )
        df = load_all_agent_rows(tmp_path, manifest_data=SAMPLE_MANIFEST_DATA)
        assert df["run_id"].nunique() == 1
        assert "run_good" in df["run_id"].values

    def test_filters_by_experiment_list(self, tmp_path):
        for exp in ("exp_a", "exp_b", "exp_c"):
            _build_run_dir(
                tmp_path, experiment=exp, run_id="run_001",
                manifest=SAMPLE_MANIFEST_JSON,
                pid_crit_all=SAMPLE_PID_CRIT_ALL_ROUNDS,
                financial=SAMPLE_FINANCIAL_METRICS,
            )
        df = load_all_agent_rows(
            tmp_path,
            experiments=["exp_a", "exp_b"],
            manifest_data=SAMPLE_MANIFEST_DATA,
        )
        assert set(df["experiment"].unique()) == {"exp_a", "exp_b"}

    def test_numeric_columns_cast_correctly(self, tmp_path):
        _build_run_dir(
            tmp_path, experiment="exp_a", run_id="run_001",
            manifest=SAMPLE_MANIFEST_JSON,
            pid_crit_all=SAMPLE_PID_CRIT_ALL_ROUNDS,
            financial=SAMPLE_FINANCIAL_METRICS,
        )
        df = load_all_agent_rows(tmp_path, manifest_data=SAMPLE_MANIFEST_DATA)
        for col in ("rho_i", "LV", "ES", "AC", "CA", "sharpe", "total_return"):
            assert pd.api.types.is_float_dtype(df[col]), f"{col} should be float"
