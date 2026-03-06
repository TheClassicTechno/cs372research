"""L4 integration tests: debate artifact schema validation.

Runs a 1-round mock debate with DebateLogger writing to tmp_path,
then validates the on-disk artifact tree:
- manifest.json has all required keys
- portfolio.json values are in [0, 1] and sum to ~1.0
- Directory structure matches expected layout (rounds/round_001/ etc.)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from multi_agent.config import AgentRole, DebateConfig
from multi_agent.debate_logger import DebateLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "NVDA"]
ROLES = [AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK]


def _make_config(tmp_path: Path) -> DebateConfig:
    """Create a DebateConfig with logging_mode='standard' for artifact tests."""
    return DebateConfig(
        mock=True,
        roles=list(ROLES),
        max_rounds=1,
        parallel_agents=False,
        console_display=False,
        logging_mode="standard",
        experiment_name="test_artifact_schema",
        trace_dir=str(tmp_path / "traces"),
    )


def _make_observation_dict() -> dict:
    """Minimal observation dict (not a pydantic model -- logger expects dict)."""
    return {
        "universe": TICKERS,
        "timestamp": "2025-01-15T00:00:00Z",
    }


def _make_mock_proposals() -> list[dict]:
    """Fabricate mock proposal dicts matching the structure DebateLogger expects."""
    eq = round(1.0 / len(TICKERS), 4)
    proposals = []
    for role in ROLES:
        proposals.append({
            "role": role.value,
            "raw_response": f"[{role.value} mock] Equal-weight proposal reasoning text.",
            "action_dict": {
                "allocation": {t: eq for t in TICKERS},
                "justification": f"[{role.value}] Equal-weight allocation",
                "confidence": 0.5,
            },
        })
    return proposals


def _make_mock_critiques() -> list[dict]:
    """Fabricate mock critique dicts."""
    critiques = []
    for role in ROLES:
        critiques.append({
            "role": role.value,
            "critiques": [
                {
                    "target_role": ROLES[(ROLES.index(role) + 1) % len(ROLES)].value,
                    "objection": f"[mock] {role.value} challenges allocation assumption",
                }
            ],
            "self_critique": f"[mock] {role.value} may be overweighting.",
        })
    return critiques


def _make_mock_revisions() -> list[dict]:
    """Fabricate mock revision dicts."""
    eq = round(1.0 / len(TICKERS), 4)
    revisions = []
    for role in ROLES:
        revisions.append({
            "role": role.value,
            "raw_response": f"[{role.value} mock] Revised reasoning.",
            "action_dict": {
                "allocation": {t: eq for t in TICKERS},
                "justification": f"[{role.value}] Revised allocation",
                "confidence": 0.4,
                "revision_notes": f"[mock] {role.value} lowered confidence",
            },
        })
    return revisions


def _run_logger_lifecycle(tmp_path: Path) -> Path:
    """Run the full DebateLogger lifecycle and return the run directory."""
    config = _make_config(tmp_path)
    logger = DebateLogger(config, experiment_name="test_artifact_schema")

    # Override run_dir to use tmp_path
    logger._run_dir = tmp_path / "run_test"
    obs = _make_observation_dict()

    # Init
    logger.init_run("debate-test-001", obs, "Mock enriched context for testing.")

    # Round 1
    logger.start_round(1, beta=0.5)
    proposals = _make_mock_proposals()
    logger.write_proposals(proposals)
    logger.write_critiques(_make_mock_critiques())
    logger.write_revisions(_make_mock_revisions())

    return logger._run_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestManifestSchema:
    """manifest.json must contain all required keys."""

    REQUIRED_KEYS = [
        "experiment_name",
        "run_id",
        "debate_id",
        "started_at",
        "model_name",
        "roles",
        "max_rounds",
        "logging_mode",
    ]

    def test_manifest_exists(self, tmp_path):
        run_dir = _run_logger_lifecycle(tmp_path)
        manifest_path = run_dir / "manifest.json"
        assert manifest_path.exists(), f"manifest.json not found at {manifest_path}"

    def test_manifest_has_required_keys(self, tmp_path):
        run_dir = _run_logger_lifecycle(tmp_path)
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())

        missing = [k for k in self.REQUIRED_KEYS if k not in manifest]
        assert not missing, f"manifest.json missing keys: {missing}"

    def test_manifest_experiment_name(self, tmp_path):
        run_dir = _run_logger_lifecycle(tmp_path)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["experiment_name"] == "test_artifact_schema"

    def test_manifest_roles_match_config(self, tmp_path):
        run_dir = _run_logger_lifecycle(tmp_path)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        expected_roles = sorted(r.value for r in ROLES)
        actual_roles = sorted(manifest["roles"])
        assert actual_roles == expected_roles, (
            f"Manifest roles {actual_roles} != config roles {expected_roles}"
        )


@pytest.mark.integration
class TestPortfolioArtifacts:
    """Portfolio.json values must be in [0, 1] and sum to ~1.0."""

    def test_proposal_portfolios_valid(self, tmp_path):
        run_dir = _run_logger_lifecycle(tmp_path)
        round_dir = run_dir / "rounds" / "round_001"

        for role in ROLES:
            portfolio_path = round_dir / "proposals" / role.value / "portfolio.json"
            assert portfolio_path.exists(), f"Missing portfolio: {portfolio_path}"

            portfolio = json.loads(portfolio_path.read_text())
            assert len(portfolio) > 0, "Portfolio is empty"

            total = 0.0
            for ticker, weight in portfolio.items():
                assert 0.0 <= weight <= 1.0, (
                    f"{role.value} proposal: {ticker} weight {weight} not in [0, 1]"
                )
                total += weight

            assert abs(total - 1.0) < 0.02, (
                f"{role.value} proposal portfolio sums to {total:.4f}, expected ~1.0"
            )

    def test_revision_portfolios_valid(self, tmp_path):
        run_dir = _run_logger_lifecycle(tmp_path)
        round_dir = run_dir / "rounds" / "round_001"

        for role in ROLES:
            portfolio_path = round_dir / "revisions" / role.value / "portfolio.json"
            assert portfolio_path.exists(), f"Missing portfolio: {portfolio_path}"

            portfolio = json.loads(portfolio_path.read_text())
            total = sum(portfolio.values())
            assert abs(total - 1.0) < 0.02, (
                f"{role.value} revision portfolio sums to {total:.4f}, expected ~1.0"
            )


@pytest.mark.integration
class TestDirectoryStructure:
    """Artifact directory tree must match expected layout."""

    def test_round_001_exists(self, tmp_path):
        run_dir = _run_logger_lifecycle(tmp_path)
        round_dir = run_dir / "rounds" / "round_001"
        assert round_dir.is_dir(), f"round_001 not found at {round_dir}"

    def test_round_001_subdirs_exist(self, tmp_path):
        run_dir = _run_logger_lifecycle(tmp_path)
        round_dir = run_dir / "rounds" / "round_001"

        expected_subdirs = ["proposals", "critiques", "revisions", "CRIT", "metrics"]
        for subdir in expected_subdirs:
            assert (round_dir / subdir).is_dir(), (
                f"Missing subdirectory: {round_dir / subdir}"
            )

    def test_shared_context_exists(self, tmp_path):
        run_dir = _run_logger_lifecycle(tmp_path)
        assert (run_dir / "shared_context").is_dir()
        memo_path = run_dir / "shared_context" / "memo.txt"
        assert memo_path.exists(), "memo.txt not found in shared_context/"
        assert len(memo_path.read_text()) > 0, "memo.txt is empty"

    def test_final_directory_exists(self, tmp_path):
        run_dir = _run_logger_lifecycle(tmp_path)
        assert (run_dir / "final").is_dir()

    def test_critique_response_files_exist(self, tmp_path):
        run_dir = _run_logger_lifecycle(tmp_path)
        round_dir = run_dir / "rounds" / "round_001"

        for role in ROLES:
            response_path = round_dir / "critiques" / role.value / "response.json"
            assert response_path.exists(), f"Missing: {response_path}"

            data = json.loads(response_path.read_text())
            assert "critiques" in data or "self_critique" in data
