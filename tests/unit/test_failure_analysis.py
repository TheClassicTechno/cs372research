"""
Unit tests for analysis/failure_analysis.py

Test classes:
  TestBuildRunLevelDf          — deduplication of agent rows to run-level
  TestSelectQuadrantCases      — high-CRIT/low-return quadrant selection logic
  TestSelectInterventionCases  — intervention-helped vs hurt detection
  TestTruncateText             — text truncation helper
  TestLoadRoundTrace           — loading one round's artifacts from disk
  TestLoadRunFullTrace         — loading an entire run's trace
  TestFormatCritScores         — CRIT score formatting
  TestFormatPortfolio          — portfolio formatting
  TestFormatReasoningExcerpt   — reasoning text extraction
  TestFormatCaseStudy          — end-to-end case study format (smoke)
  TestBuildReport              — full report assembly (smoke + structure)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.failure_analysis import (
    EXCERPT_MAX_CHARS,
    QUARTILE_HIGH,
    QUARTILE_LOW,
    _find_run_dir,
    _load_json_safe,
    build_report,
    build_run_level_df,
    format_case_study,
    format_crit_scores,
    format_portfolio,
    format_reasoning_excerpt,
    load_round_trace,
    load_run_full_trace,
    select_intervention_cases,
    select_quadrant_cases,
    truncate_text,
)


# ---------------------------------------------------------------------------
# Shared fixture data
# ---------------------------------------------------------------------------

SAMPLE_MANIFEST_DATA = {
    "patterns": [{"pattern": "inflation_shock", "regime": "inflation_shock"}],
    "quarter_fallback": {"2022-03-31": "inflation_shock"},
}

SAMPLE_CRIT_SCORES_JSON = {
    "round": 1,
    "rho_bar": 0.82,
    "agent_scores": {
        "macro": {
            "rho_i": 0.85,
            "pillar_scores": {"LV": 0.80, "ES": 0.88, "AC": 0.90, "CA": 0.82},
            "diagnostics": {"contradictions": False},
            "explanations": {
                "logical_validity": "Reasoning was internally consistent.",
                "evidential_support": "All claims had cited evidence.",
            },
        },
        "risk": {
            "rho_i": 0.79,
            "pillar_scores": {"LV": 0.75, "ES": 0.80, "AC": 0.82, "CA": 0.79},
            "diagnostics": {},
            "explanations": {"logical_validity": "Minor contradiction detected."},
        },
    },
}

SAMPLE_PID_CRIT_ALL = [
    {
        "type": "pid_round",
        "round": 1,
        "crit": {
            "rho_bar": 0.82,
            "agents": {
                "macro": {"rho_i": 0.85, "pillars": {"LV": 0.80, "ES": 0.88, "AC": 0.90, "CA": 0.82}},
                "risk":  {"rho_i": 0.79, "pillars": {"LV": 0.75, "ES": 0.80, "AC": 0.82, "CA": 0.79}},
            },
        },
    }
]

SAMPLE_FINANCIAL = {
    "daily_metrics_annualized_sharpe": 0.65,
    "daily_metrics_total_return_pct": 7.2,
    "daily_metrics_max_drawdown_pct": 4.8,
}

SAMPLE_MANIFEST_JSON = {
    "experiment_name": "test_exp",
    "run_id": "run_2022-01-01_00-00-00",
    "config_paths": ["debate.yaml", "scenarios/2022Q1_inflation_shock.yaml"],
    "invest_quarter": "2022-03-31",
    "roles": ["macro", "risk"],
    "intervention_config": {"enabled": False},
}

SAMPLE_MANIFEST_WITH_INTERVENTION = {
    **SAMPLE_MANIFEST_JSON,
    "intervention_config": {"enabled": True},
}


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _make_agent_df(n_runs: int = 10, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic agent-row DataFrame for testing."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_runs):
        rho_bar = rng.uniform(0.5, 0.95)
        sharpe = rng.uniform(-1.0, 2.0)
        for role in ("macro", "risk"):
            rows.append({
                "run_id": f"run_{i:03d}",
                "experiment": "test_exp",
                "scenario": "2022Q1_inflation_shock",
                "regime": "inflation_shock",
                "role": role,
                "rho_i": rho_bar + rng.normal(0, 0.05),
                "rho_bar": rho_bar,
                "sharpe": sharpe,
                "total_return": sharpe * 10,
                "drawdown": rng.uniform(2, 15),
                "LV": rho_bar, "ES": rho_bar, "AC": rho_bar, "CA": rho_bar,
            })
    return pd.DataFrame(rows)


def _build_run_dir(
    base: Path,
    run_id: str = "run_001",
    experiment: str = "test_exp",
    manifest: dict | None = None,
    financial: dict | None = None,
    with_rounds: bool = True,
    with_intervention: bool = False,
) -> Path:
    """Build a complete synthetic run directory."""
    run_dir = base / experiment / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    man = (manifest or SAMPLE_MANIFEST_JSON).copy()
    if with_intervention:
        man["intervention_config"] = {"enabled": True}
    _write_json(run_dir / "manifest.json", man)

    if financial is not None:
        _write_json(run_dir / "_dashboard" / "financial_metrics.json", financial)

    _write_json(run_dir / "final" / "final_portfolio.json", {"AAPL": 0.4, "MSFT": 0.6})
    _write_json(run_dir / "final" / "pid_crit_all_rounds.json", SAMPLE_PID_CRIT_ALL)

    if with_rounds:
        rd = run_dir / "rounds" / "round_001"
        # Proposals
        _write_text(rd / "proposals" / "macro" / "response.txt",
                    '{"allocation": {"AAPL": 0.5, "MSFT": 0.5}, "claims": []}')
        _write_json(rd / "proposals" / "macro" / "portfolio.json", {"AAPL": 0.5, "MSFT": 0.5})
        # Critiques
        _write_json(rd / "critiques" / "macro" / "response.json", {
            "critiques": [{
                "target_role": "risk",
                "target_claim": "C1",
                "objection": "Risk overstates JPM earnings durability.",
                "counter_evidence": ["[JPM-EC4]"],
                "portfolio_implication": "Reduce JPM sizing.",
                "suggested_adjustment": "Cut JPM by 30%.",
            }]
        })
        # Revisions
        _write_text(rd / "revisions" / "macro" / "response.txt",
                    "After reviewing critiques, we maintain our energy thesis but reduce financials.")
        _write_json(rd / "revisions" / "macro" / "portfolio.json", {"AAPL": 0.4, "MSFT": 0.6})
        # Metrics
        _write_json(rd / "metrics" / "crit_scores.json", SAMPLE_CRIT_SCORES_JSON)
        _write_json(rd / "metrics" / "js_divergence.json", {"round": 1, "js_divergence": 0.45})

    return run_dir


# ---------------------------------------------------------------------------
# TestBuildRunLevelDf
# ---------------------------------------------------------------------------

class TestBuildRunLevelDf:
    """Tests for build_run_level_df() — dedup agent rows to run-level."""

    def test_deduplicates_to_one_row_per_run(self):
        df = _make_agent_df(n_runs=5)
        run_df = build_run_level_df(df)
        assert len(run_df) == 5

    def test_preserves_financial_columns(self):
        df = _make_agent_df(n_runs=3)
        run_df = build_run_level_df(df)
        assert "sharpe" in run_df.columns
        assert "total_return" in run_df.columns

    def test_preserves_scenario_and_regime(self):
        df = _make_agent_df(n_runs=3)
        run_df = build_run_level_df(df)
        assert "scenario" in run_df.columns
        assert "regime" in run_df.columns

    def test_numeric_columns_are_float(self):
        df = _make_agent_df(n_runs=5)
        run_df = build_run_level_df(df)
        assert pd.api.types.is_float_dtype(run_df["sharpe"])
        assert pd.api.types.is_float_dtype(run_df["rho_bar"])

    def test_empty_input_returns_empty(self):
        run_df = build_run_level_df(pd.DataFrame())
        assert run_df.empty


# ---------------------------------------------------------------------------
# TestSelectQuadrantCases
# ---------------------------------------------------------------------------

class TestSelectQuadrantCases:
    """Tests for select_quadrant_cases() — quadrant membership logic."""

    def _make_polarised_df(self) -> pd.DataFrame:
        """Create a DataFrame with clear high-CRIT/low-return and low-CRIT/high-return cases."""
        rows = [
            # High CRIT, low return — expected in high_crit_low_return
            {"run_id": "hc_lr_1", "rho_bar": 0.95, "sharpe": -1.5, "total_return": -15.0, "drawdown": 20.0},
            {"run_id": "hc_lr_2", "rho_bar": 0.92, "sharpe": -1.2, "total_return": -12.0, "drawdown": 18.0},
            # Low CRIT, high return — expected in low_crit_high_return
            {"run_id": "lc_hr_1", "rho_bar": 0.52, "sharpe":  2.1, "total_return":  21.0, "drawdown":  3.0},
            {"run_id": "lc_hr_2", "rho_bar": 0.55, "sharpe":  1.8, "total_return":  18.0, "drawdown":  4.0},
            # High CRIT, high return — expected in high_crit_high_return
            {"run_id": "hc_hr_1", "rho_bar": 0.93, "sharpe":  2.0, "total_return":  20.0, "drawdown":  2.0},
            # Middle — expected in no quadrant
            {"run_id": "mid_1",   "rho_bar": 0.72, "sharpe":  0.3, "total_return":   3.0, "drawdown":  8.0},
            {"run_id": "mid_2",   "rho_bar": 0.68, "sharpe":  0.1, "total_return":   1.0, "drawdown":  9.0},
            {"run_id": "mid_3",   "rho_bar": 0.74, "sharpe":  0.5, "total_return":   5.0, "drawdown":  7.0},
        ]
        return pd.DataFrame(rows)

    def test_high_crit_low_return_identified(self):
        df = self._make_polarised_df()
        result = select_quadrant_cases(df, n=3)
        assert "hc_lr_1" in result["high_crit_low_return"]

    def test_low_crit_high_return_identified(self):
        df = self._make_polarised_df()
        result = select_quadrant_cases(df, n=3)
        assert "lc_hr_1" in result["low_crit_high_return"]

    def test_high_crit_high_return_identified(self):
        df = self._make_polarised_df()
        result = select_quadrant_cases(df, n=3)
        assert "hc_hr_1" in result["high_crit_high_return"]

    def test_respects_n_limit(self):
        df = self._make_polarised_df()
        result = select_quadrant_cases(df, n=1)
        for cat in result:
            assert len(result[cat]) <= 1

    def test_returns_empty_lists_on_empty_df(self):
        result = select_quadrant_cases(pd.DataFrame(), n=3)
        for cat in result:
            assert result[cat] == []

    def test_high_crit_low_return_sorted_worst_sharpe_first(self):
        df = self._make_polarised_df()
        result = select_quadrant_cases(df, n=5)
        ids = result["high_crit_low_return"]
        if len(ids) > 1:
            sharpes = df.set_index("run_id").loc[ids, "sharpe"].values
            assert sharpes[0] <= sharpes[-1]   # ascending (worst first)

    def test_handles_nan_rows_gracefully(self):
        df = self._make_polarised_df()
        df.loc[0, "sharpe"] = float("nan")
        result = select_quadrant_cases(df, n=3)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# TestSelectInterventionCases
# ---------------------------------------------------------------------------

class TestSelectInterventionCases:
    """Tests for select_intervention_cases()."""

    def test_returns_correct_keys(self, tmp_path):
        run_df = pd.DataFrame([{
            "run_id": "run_001", "scenario": "test", "sharpe": 0.5
        }])
        result = select_intervention_cases(tmp_path, run_df, n=2)
        assert "intervention_helped" in result
        assert "intervention_hurt" in result

    def test_no_intervention_runs_returns_empty(self, tmp_path):
        # Build a run without intervention
        _build_run_dir(tmp_path, run_id="run_001", with_intervention=False, financial=SAMPLE_FINANCIAL)
        run_df = pd.DataFrame([{
            "run_id": "run_001", "scenario": "test", "sharpe": 1.0
        }])
        result = select_intervention_cases(tmp_path, run_df, n=2)
        assert result["intervention_helped"] == []
        assert result["intervention_hurt"] == []

    def test_empty_run_df_returns_empty(self, tmp_path):
        result = select_intervention_cases(tmp_path, pd.DataFrame(), n=2)
        assert result["intervention_helped"] == []
        assert result["intervention_hurt"] == []


# ---------------------------------------------------------------------------
# TestTruncateText
# ---------------------------------------------------------------------------

class TestTruncateText:
    """Tests for truncate_text()."""

    def test_short_text_unchanged(self):
        text = "Hello world."
        assert truncate_text(text, 100) == text

    def test_long_text_truncated(self):
        text = "A " * 500
        result = truncate_text(text, 50)
        assert len(result) <= 50 + len("...[truncated]")
        assert "truncated" in result

    def test_cuts_at_sentence_boundary(self):
        text = "First sentence. Second sentence. Third sentence. " * 20
        result = truncate_text(text, 60)
        assert result.endswith("...[truncated]") or result.endswith(".")

    def test_empty_string_unchanged(self):
        assert truncate_text("", 100) == ""

    def test_exact_length_unchanged(self):
        text = "x" * EXCERPT_MAX_CHARS
        assert truncate_text(text) == text


# ---------------------------------------------------------------------------
# TestLoadRoundTrace
# ---------------------------------------------------------------------------

class TestLoadRoundTrace:
    """Tests for load_round_trace()."""

    def test_loads_proposals_text(self, tmp_path):
        run_dir = _build_run_dir(tmp_path)
        trace = load_round_trace(run_dir, 1)
        assert "macro" in trace["proposals"]
        assert trace["proposals"]["macro"]["text"] != ""

    def test_loads_proposal_portfolio(self, tmp_path):
        run_dir = _build_run_dir(tmp_path)
        trace = load_round_trace(run_dir, 1)
        assert trace["proposals"]["macro"]["portfolio"] is not None

    def test_loads_critiques(self, tmp_path):
        run_dir = _build_run_dir(tmp_path)
        trace = load_round_trace(run_dir, 1)
        critiques = trace["critiques"].get("macro", [])
        assert len(critiques) >= 1
        assert "objection" in critiques[0]

    def test_loads_revisions(self, tmp_path):
        run_dir = _build_run_dir(tmp_path)
        trace = load_round_trace(run_dir, 1)
        assert "macro" in trace["revisions"]
        assert trace["revisions"]["macro"]["text"] != ""

    def test_loads_crit_scores(self, tmp_path):
        run_dir = _build_run_dir(tmp_path)
        trace = load_round_trace(run_dir, 1)
        assert trace["crit_scores"].get("rho_bar") == 0.82

    def test_loads_js_divergence(self, tmp_path):
        run_dir = _build_run_dir(tmp_path)
        trace = load_round_trace(run_dir, 1)
        assert trace["js_divergence"] == 0.45

    def test_nonexistent_round_returns_empty(self, tmp_path):
        run_dir = _build_run_dir(tmp_path)
        trace = load_round_trace(run_dir, 99)
        assert trace == {}

    def test_missing_files_handled_gracefully(self, tmp_path):
        # Round dir exists but has no subdirs
        run_dir = tmp_path / "test_exp" / "run_001"
        (run_dir / "rounds" / "round_001").mkdir(parents=True)
        trace = load_round_trace(run_dir, 1)
        assert trace["proposals"] == {}
        assert trace["critiques"] == {}
        assert trace["revisions"] == {}


# ---------------------------------------------------------------------------
# TestLoadRunFullTrace
# ---------------------------------------------------------------------------

class TestLoadRunFullTrace:
    """Tests for load_run_full_trace()."""

    def test_loads_manifest(self, tmp_path):
        run_dir = _build_run_dir(tmp_path, financial=SAMPLE_FINANCIAL)
        trace = load_run_full_trace(run_dir)
        assert trace["manifest"].get("experiment_name") == "test_exp"

    def test_loads_financial(self, tmp_path):
        run_dir = _build_run_dir(tmp_path, financial=SAMPLE_FINANCIAL)
        trace = load_run_full_trace(run_dir)
        assert trace["financial"]["daily_metrics_annualized_sharpe"] == 0.65

    def test_loads_final_portfolio(self, tmp_path):
        run_dir = _build_run_dir(tmp_path, financial=SAMPLE_FINANCIAL)
        trace = load_run_full_trace(run_dir)
        assert "AAPL" in trace["final_portfolio"]

    def test_loads_pid_crit_all(self, tmp_path):
        run_dir = _build_run_dir(tmp_path, financial=SAMPLE_FINANCIAL)
        trace = load_run_full_trace(run_dir)
        assert len(trace["pid_crit_all"]) == 1

    def test_loads_round_traces(self, tmp_path):
        run_dir = _build_run_dir(tmp_path, financial=SAMPLE_FINANCIAL)
        trace = load_run_full_trace(run_dir)
        assert len(trace["rounds"]) == 1
        assert trace["rounds"][0]["round"] == 1

    def test_missing_financial_returns_empty_dict(self, tmp_path):
        run_dir = _build_run_dir(tmp_path, financial=None)
        trace = load_run_full_trace(run_dir)
        assert trace["financial"] == {}

    def test_no_rounds_dir_returns_empty_list(self, tmp_path):
        run_dir = _build_run_dir(tmp_path, with_rounds=False, financial=SAMPLE_FINANCIAL)
        trace = load_run_full_trace(run_dir)
        assert trace["rounds"] == []


# ---------------------------------------------------------------------------
# TestFormatCritScores
# ---------------------------------------------------------------------------

class TestFormatCritScores:
    """Tests for format_crit_scores()."""

    def test_includes_rho_bar(self):
        text = format_crit_scores(SAMPLE_CRIT_SCORES_JSON)
        assert "0.82" in text

    def test_includes_all_roles(self):
        text = format_crit_scores(SAMPLE_CRIT_SCORES_JSON)
        assert "macro" in text
        assert "risk" in text

    def test_includes_pillar_scores(self):
        text = format_crit_scores(SAMPLE_CRIT_SCORES_JSON)
        assert "LV" in text
        assert "ES" in text

    def test_includes_explanations(self):
        text = format_crit_scores(SAMPLE_CRIT_SCORES_JSON)
        assert "internally consistent" in text

    def test_handles_empty_crit(self):
        text = format_crit_scores({})
        assert isinstance(text, str)

    def test_explanations_truncated_to_reasonable_length(self):
        long_crit = {
            "rho_bar": 0.8,
            "agent_scores": {
                "macro": {
                    "rho_i": 0.8,
                    "pillar_scores": {},
                    "explanations": {"logical_validity": "X" * 2000},
                }
            },
        }
        text = format_crit_scores(long_crit)
        # Explanation should be truncated — the full 2000 chars should not appear
        assert "X" * 300 not in text


# ---------------------------------------------------------------------------
# TestFormatPortfolio
# ---------------------------------------------------------------------------

class TestFormatPortfolio:
    """Tests for format_portfolio()."""

    def test_formats_non_zero_tickers(self):
        p = {"AAPL": 0.4, "MSFT": 0.0, "NVDA": 0.6}
        text = format_portfolio(p)
        assert "AAPL" in text
        assert "NVDA" in text
        assert "MSFT" not in text   # zero weight excluded

    def test_sorted_by_weight_descending(self):
        p = {"AAPL": 0.1, "MSFT": 0.5, "NVDA": 0.4}
        text = format_portfolio(p)
        msft_pos = text.index("MSFT")
        nvda_pos = text.index("NVDA")
        aapl_pos = text.index("AAPL")
        assert msft_pos < nvda_pos < aapl_pos

    def test_empty_portfolio(self):
        text = format_portfolio({})
        assert "no portfolio" in text.lower()

    def test_percentage_formatting(self):
        text = format_portfolio({"AAPL": 0.4})
        assert "40.0%" in text


# ---------------------------------------------------------------------------
# TestFormatReasoningExcerpt
# ---------------------------------------------------------------------------

class TestFormatReasoningExcerpt:
    """Tests for format_reasoning_excerpt()."""

    def _sample_trace(self) -> dict:
        return {
            "round": 1,
            "proposals": {
                "macro": {"text": "We propose overweighting energy.", "portfolio": {}}
            },
            "critiques": {
                "macro": [{
                    "target_role": "risk",
                    "target_claim": "C1",
                    "objection": "Risk overstates financials.",
                    "counter_evidence": ["[JPM-EC4]"],
                    "portfolio_implication": "Reduce JPM.",
                    "suggested_adjustment": "Cut JPM by 30%.",
                }]
            },
            "revisions": {
                "macro": {"text": "After critique, we reduce financials and increase energy.", "portfolio": {}}
            },
        }

    def test_extracts_revision_text(self):
        trace = self._sample_trace()
        text = format_reasoning_excerpt(trace, "macro", "revision")
        assert "reduce financials" in text

    def test_extracts_proposal_text(self):
        trace = self._sample_trace()
        text = format_reasoning_excerpt(trace, "macro", "proposal")
        assert "overweighting energy" in text

    def test_extracts_critique(self):
        trace = self._sample_trace()
        text = format_reasoning_excerpt(trace, "macro", "critique")
        assert "overstates financials" in text
        assert "JPM-EC4" in text

    def test_missing_role_returns_note(self):
        trace = self._sample_trace()
        text = format_reasoning_excerpt(trace, "nonexistent_role", "revision")
        assert "not found" in text.lower() or "no" in text.lower()

    def test_empty_trace_returns_note(self):
        text = format_reasoning_excerpt({}, "macro", "revision")
        assert isinstance(text, str)
        assert len(text) > 0


# ---------------------------------------------------------------------------
# TestFormatCaseStudy  (smoke — check for crash-free output with right sections)
# ---------------------------------------------------------------------------

class TestFormatCaseStudy:
    """Smoke tests for format_case_study()."""

    def _make_run_row(self) -> pd.Series:
        return pd.Series({
            "run_id": "run_001",
            "experiment": "test_exp",
            "scenario": "2022Q1_inflation_shock",
            "regime": "inflation_shock",
            "rho_bar": 0.92,
            "sharpe": -0.8,
            "total_return": -8.0,
            "drawdown": 12.0,
        })

    def _make_full_trace(self, tmp_path) -> dict:
        run_dir = _build_run_dir(tmp_path, financial=SAMPLE_FINANCIAL)
        return load_run_full_trace(run_dir)

    def test_high_crit_low_return_runs(self, tmp_path):
        row = self._make_run_row()
        trace = self._make_full_trace(tmp_path)
        text = format_case_study("high_crit_low_return", "run_001", row, trace)
        assert "HIGH CRIT" in text
        assert "LOW RETURN" in text

    def test_low_crit_high_return_runs(self, tmp_path):
        row = self._make_run_row()
        trace = self._make_full_trace(tmp_path)
        text = format_case_study("low_crit_high_return", "run_001", row, trace)
        assert "LOW CRIT" in text

    def test_intervention_helped_runs(self, tmp_path):
        row = self._make_run_row()
        trace = self._make_full_trace(tmp_path)
        text = format_case_study("intervention_helped", "run_001", row, trace)
        assert "INTERVENTION" in text

    def test_contains_crit_section(self, tmp_path):
        row = self._make_run_row()
        trace = self._make_full_trace(tmp_path)
        text = format_case_study("high_crit_low_return", "run_001", row, trace)
        assert "CRIT REASONING QUALITY" in text

    def test_contains_financial_section(self, tmp_path):
        row = self._make_run_row()
        trace = self._make_full_trace(tmp_path)
        text = format_case_study("high_crit_low_return", "run_001", row, trace)
        assert "FINANCIAL OUTCOME" in text

    def test_contains_portfolio_section(self, tmp_path):
        row = self._make_run_row()
        trace = self._make_full_trace(tmp_path)
        text = format_case_study("high_crit_low_return", "run_001", row, trace)
        assert "FINAL PORTFOLIO" in text

    def test_contains_reasoning_section(self, tmp_path):
        row = self._make_run_row()
        trace = self._make_full_trace(tmp_path)
        text = format_case_study("high_crit_low_return", "run_001", row, trace)
        assert "REASONING EXCERPTS" in text

    def test_contains_human_analysis_marker(self, tmp_path):
        row = self._make_run_row()
        trace = self._make_full_trace(tmp_path)
        text = format_case_study("high_crit_low_return", "run_001", row, trace)
        assert "HUMAN ANALYSIS NEEDED" in text

    def test_no_crash_on_empty_trace(self, tmp_path):
        row = self._make_run_row()
        text = format_case_study("high_crit_low_return", "run_001", row, {})
        assert isinstance(text, str)
        assert len(text) > 0


# ---------------------------------------------------------------------------
# TestBuildReport  (smoke — structure and content checks)
# ---------------------------------------------------------------------------

class TestBuildReport:
    """Smoke tests for build_report() — the top-level report assembler."""

    def _setup(self, tmp_path):
        """Build two run dirs and a matching DataFrame."""
        run_dir_1 = _build_run_dir(
            tmp_path, run_id="run_001", financial=SAMPLE_FINANCIAL
        )
        run_dir_2 = _build_run_dir(
            tmp_path, run_id="run_002",
            financial={**SAMPLE_FINANCIAL, "daily_metrics_annualized_sharpe": -0.9},
        )
        agent_df = pd.DataFrame([
            {"run_id": "run_001", "experiment": "test_exp", "scenario": "2022Q1_inflation_shock",
             "regime": "inflation_shock", "role": "macro", "rho_bar": 0.92, "rho_i": 0.92,
             "sharpe": 0.65, "total_return": 7.2, "drawdown": 4.8,
             "LV": 0.8, "ES": 0.8, "AC": 0.8, "CA": 0.8},
            {"run_id": "run_002", "experiment": "test_exp", "scenario": "2022Q1_inflation_shock",
             "regime": "inflation_shock", "role": "macro", "rho_bar": 0.55, "rho_i": 0.55,
             "sharpe": -0.9, "total_return": -9.0, "drawdown": 15.0,
             "LV": 0.5, "ES": 0.5, "AC": 0.5, "CA": 0.5},
        ])
        run_df = build_run_level_df(agent_df)
        return run_df, agent_df

    def test_report_is_list_of_strings(self, tmp_path):
        run_df, agent_df = self._setup(tmp_path)
        quadrant = select_quadrant_cases(run_df, n=1)
        report = build_report(quadrant, {}, tmp_path, run_df, agent_df)
        assert isinstance(report, list)
        assert all(isinstance(line, str) for line in report)

    def test_report_contains_header(self, tmp_path):
        run_df, agent_df = self._setup(tmp_path)
        quadrant = select_quadrant_cases(run_df, n=1)
        report = build_report(quadrant, {}, tmp_path, run_df, agent_df)
        full_text = "\n".join(report)
        assert "QUALITATIVE FAILURE ANALYSIS" in full_text

    def test_report_mentions_run_count(self, tmp_path):
        run_df, agent_df = self._setup(tmp_path)
        quadrant = select_quadrant_cases(run_df, n=1)
        report = build_report(quadrant, {}, tmp_path, run_df, agent_df)
        full_text = "\n".join(report)
        assert "2" in full_text   # 2 total runs

    def test_empty_cases_shows_not_found_message(self, tmp_path):
        run_df, agent_df = self._setup(tmp_path)
        report = build_report(
            {"high_crit_low_return": [], "low_crit_high_return": [], "high_crit_high_return": []},
            {},
            tmp_path, run_df, agent_df
        )
        full_text = "\n".join(report)
        assert "no cases found" in full_text.lower()
