"""
Replay-based pipeline integrity integration test.

Validates that a real run artifact logged to disk is complete, well-formed,
and internally consistent across all pipeline stages. No LLM calls — pure
artifact validation.

Usage:
    pytest tests/integration/test_run_artifact_integrity.py -v
    DEBUG_ARTIFACT_TEST=1 pytest tests/integration/test_run_artifact_integrity.py -v -s
    RUN_ARTIFACT_DIR=/path/to/run pytest tests/integration/test_run_artifact_integrity.py -v
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# RunArtifactLoader — Pure IO layer
# ---------------------------------------------------------------------------

_DEFAULT_RUN = "logging/runs/vskarich_ablation_4/run_2026-03-10_19-01-01"


class RunArtifactLoader:
    """Reads files and parses JSON/text. No validation.

    Raises ``FileNotFoundError`` for missing files and wraps JSON/Unicode
    errors in ``AssertionError("Corrupt artifact file: {path}: {error}")``
    for clear diagnostics.
    """

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir

    # --- internal helpers ---------------------------------------------------

    def _read_json(self, path: Path) -> dict | list:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as err:
            raise AssertionError(f"Corrupt artifact file: {path}: {err}") from err

    def _read_text(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError as err:
            raise AssertionError(f"Corrupt artifact file: {path}: {err}") from err

    # --- top-level ----------------------------------------------------------

    def manifest(self) -> dict:
        return self._read_json(self.run_dir / "manifest.json")

    def prompt_manifest(self) -> dict:
        return self._read_json(self.run_dir / "prompt_manifest.json")

    def shared_memo(self) -> str:
        return self._read_text(self.run_dir / "shared_context" / "memo.txt")

    # --- dynamic discovery --------------------------------------------------

    def roles(self) -> list[str]:
        return self.manifest()["roles"]

    def rounds(self) -> list[int]:
        m = self.manifest()
        return list(range(1, m["actual_rounds"] + 1))

    def ticker_universe(self) -> list[str]:
        return self.manifest()["ticker_universe"]

    def round_dir(self, r: int) -> Path:
        return self.run_dir / "rounds" / f"round_{r:03d}"

    def retry_indices(self, r: int) -> list[int]:
        rd = self.round_dir(r)
        indices = []
        for d in sorted(rd.iterdir()):
            if d.is_dir() and d.name.startswith("revisions_retry_"):
                try:
                    indices.append(int(d.name.split("_")[-1]))
                except ValueError:
                    pass
        return sorted(indices)

    def intervention_files(self, r: int) -> list[Path]:
        idir = self.round_dir(r) / "interventions"
        if not idir.exists():
            return []
        return sorted(idir.glob("intervention_*.json"))

    def phase_role_dirs(self, r: int, phase: str) -> list[str]:
        pdir = self.round_dir(r) / phase
        if not pdir.exists():
            return []
        return sorted(d.name for d in pdir.iterdir() if d.is_dir())

    # --- per-round, per-phase, per-role -------------------------------------

    def proposal_portfolio(self, r: int, role: str) -> dict:
        return self._read_json(
            self.round_dir(r) / "proposals" / role / "portfolio.json"
        )

    def proposal_prompt(self, r: int, role: str) -> str:
        return self._read_text(
            self.round_dir(r) / "proposals" / role / "prompt.txt"
        )

    def proposal_response(self, r: int, role: str) -> str:
        return self._read_text(
            self.round_dir(r) / "proposals" / role / "response.txt"
        )

    def critique_response(self, r: int, role: str) -> dict:
        return self._read_json(
            self.round_dir(r) / "critiques" / role / "response.json"
        )

    def critique_prompt(self, r: int, role: str) -> str:
        return self._read_text(
            self.round_dir(r) / "critiques" / role / "prompt.txt"
        )

    def revision_portfolio(self, r: int, role: str) -> dict:
        return self._read_json(
            self.round_dir(r) / "revisions" / role / "portfolio.json"
        )

    def revision_prompt(self, r: int, role: str) -> str:
        return self._read_text(
            self.round_dir(r) / "revisions" / role / "prompt.txt"
        )

    def revision_response(self, r: int, role: str) -> str:
        return self._read_text(
            self.round_dir(r) / "revisions" / role / "response.txt"
        )

    def revision_retry_portfolio(self, r: int, retry: int, role: str) -> dict:
        return self._read_json(
            self.round_dir(r)
            / f"revisions_retry_{retry:03d}"
            / role
            / "portfolio.json"
        )

    def revision_retry_prompt(self, r: int, retry: int, role: str) -> str:
        return self._read_text(
            self.round_dir(r)
            / f"revisions_retry_{retry:03d}"
            / role
            / "prompt.txt"
        )

    def revision_retry_response(self, r: int, retry: int, role: str) -> str:
        return self._read_text(
            self.round_dir(r)
            / f"revisions_retry_{retry:03d}"
            / role
            / "response.txt"
        )

    # --- CRIT ---------------------------------------------------------------

    def crit_prompt(self, r: int, role: str) -> str:
        return self._read_text(
            self.round_dir(r) / "CRIT" / role / "prompt.txt"
        )

    def crit_response(self, r: int, role: str) -> str:
        return self._read_text(
            self.round_dir(r) / "CRIT" / role / "response.txt"
        )

    def crit_scores(self, r: int) -> dict:
        return self._read_json(
            self.round_dir(r) / "metrics" / "crit_scores.json"
        )

    # --- metrics ------------------------------------------------------------

    def round_state(self, r: int) -> dict:
        return self._read_json(self.round_dir(r) / "round_state.json")

    def js_divergence(self, r: int, suffix: str = "") -> dict:
        fname = f"js_divergence{suffix}.json"
        return self._read_json(self.round_dir(r) / "metrics" / fname)

    def evidence_overlap(self, r: int, suffix: str = "") -> dict:
        fname = f"evidence_overlap{suffix}.json"
        return self._read_json(self.round_dir(r) / "metrics" / fname)

    def pid_state(self, r: int) -> dict:
        return self._read_json(self.round_dir(r) / "metrics" / "pid_state.json")

    # --- interventions ------------------------------------------------------

    def intervention(self, r: int, index: int) -> dict:
        return self._read_json(
            self.round_dir(r) / "interventions" / f"intervention_{index:03d}.json"
        )

    def interventions_all_rounds(self) -> list[dict]:
        return self._read_json(
            self.run_dir / "final" / "interventions_all_rounds.json"
        )

    # --- final --------------------------------------------------------------

    def final_portfolio(self) -> dict:
        return self._read_json(self.run_dir / "final" / "final_portfolio.json")

    def judge_prompt(self) -> str:
        return self._read_text(self.run_dir / "final" / "judge_prompt.txt")

    def judge_response(self) -> str:
        return self._read_text(self.run_dir / "final" / "judge_response.txt")

    def pid_crit_all_rounds(self) -> list[dict]:
        return self._read_json(
            self.run_dir / "final" / "pid_crit_all_rounds.json"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_run_dir() -> Path:
    env = os.environ.get("RUN_ARTIFACT_DIR")
    if env:
        return Path(env)
    # Walk up to find the cs372research package root
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / _DEFAULT_RUN
        if candidate.exists():
            return candidate
    # Fallback: relative to cwd
    return Path(_DEFAULT_RUN)


def _validate_portfolio(
    portfolio: dict,
    tickers: list[str],
    *,
    context: str,
) -> None:
    """Assert portfolio weights are valid."""
    assert isinstance(portfolio, dict), (
        f"Portfolio is not a dict {context}: {type(portfolio)}"
    )
    assert len(portfolio) > 0, f"Portfolio is empty {context}"

    for ticker, weight in portfolio.items():
        assert isinstance(weight, (int, float)), (
            f"Non-numeric weight for {ticker} {context}: {weight}"
        )
        assert weight >= 0, (
            f"Negative weight for {ticker} {context}: {weight}"
        )

    seen = list(portfolio.keys())
    assert len(seen) == len(set(seen)), (
        f"Duplicate tickers in portfolio {context}: {seen}"
    )

    universe = set(tickers)
    extra = set(seen) - universe
    assert not extra, (
        f"Tickers outside universe {context}: {sorted(extra)}"
    )

    total = sum(portfolio.values())
    assert abs(total - 1.0) < 0.02, (
        f"Portfolio weights sum to {total:.4f} (expected ~1.0) {context}"
    )


def _extract_system_prompt(prompt_text: str) -> str:
    """Extract the system prompt section (text before === USER PROMPT ===)."""
    marker = "=== USER PROMPT ==="
    idx = prompt_text.find(marker)
    if idx == -1:
        return prompt_text
    return prompt_text[:idx]


def _extract_evidence_tokens(text: str) -> set[str]:
    """Extract bracketed evidence IDs like [L0-FOO], [L1-BAR], [TICKER-X]."""
    return set(re.findall(r"\[([A-Z][A-Z0-9]+-[A-Z0-9_-]+)\]", text))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def loader():
    run_dir = _resolve_run_dir()
    if not run_dir.exists():
        pytest.skip(f"Run artifact not found: {run_dir}")
    return RunArtifactLoader(run_dir)


@pytest.fixture(scope="module")
def manifest(loader):
    return loader.manifest()


@pytest.fixture(scope="module")
def roles(manifest):
    return manifest["roles"]


@pytest.fixture(scope="module")
def rounds(manifest):
    return list(range(1, manifest["actual_rounds"] + 1))


@pytest.fixture(scope="module")
def tickers(manifest):
    return manifest["ticker_universe"]


@pytest.fixture(scope="module")
def memo(loader):
    return loader.shared_memo()


def _debug_banner(manifest, loader):
    """Print run summary if DEBUG_ARTIFACT_TEST=1."""
    if not os.environ.get("DEBUG_ARTIFACT_TEST"):
        return
    print("\n" + "=" * 60)
    print("RUN ARTIFACT INTEGRITY TEST — DEBUG BANNER")
    print("=" * 60)
    print(f"  experiment : {manifest.get('experiment_name')}")
    print(f"  run_id     : {manifest.get('run_id')}")
    print(f"  debate_id  : {manifest.get('debate_id')}")
    print(f"  model      : {manifest.get('model_name')}")
    print(f"  roles      : {manifest.get('roles')}")
    print(f"  rounds     : {manifest.get('actual_rounds')}/{manifest.get('max_rounds')}")
    print(f"  tickers    : {len(manifest.get('ticker_universe', []))} stocks")
    print(f"  final_js   : {manifest.get('final_js')}")
    print(f"  started    : {manifest.get('started_at')}")
    print(f"  completed  : {manifest.get('completed_at')}")
    print(f"  termination: {manifest.get('termination_reason')}")
    if "artifact_version" in manifest:
        print(f"  artifact_v : {manifest['artifact_version']}")
    print(f"  run_dir    : {loader.run_dir}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# 3A. TestRunArtifactStructure
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestRunArtifactStructure:
    """Validates manifest, directory layout, and structural invariants."""

    def test_manifest_required_keys(self, manifest):
        required = {
            "roles", "max_rounds", "actual_rounds", "model_name",
            "temperature", "intervention_config", "experiment_name",
            "run_id", "debate_id", "started_at", "completed_at",
            "ticker_universe", "final_js",
        }
        missing = required - set(manifest.keys())
        assert not missing, f"Manifest missing keys: {sorted(missing)}"

    def test_manifest_intervention_config(self, manifest):
        ic = manifest["intervention_config"]
        assert ic["enabled"] is True, "intervention_config.enabled must be True"
        assert isinstance(ic["rules"], dict), "intervention_config.rules must be dict"
        assert len(ic["rules"]) > 0, "intervention_config.rules must be non-empty"

    def test_intervention_nudge_prompts_per_role(self, manifest, roles):
        ic = manifest["intervention_config"]
        js_nudges = ic["rules"]["js_collapse"]["nudge_prompts"]
        for role in roles:
            assert role in js_nudges, (
                f"js_collapse.nudge_prompts missing role {role}"
            )

    def test_prompt_manifest_exists(self, loader):
        pm = loader.prompt_manifest()
        assert len(pm) > 0, "prompt_manifest.json is empty"
        assert "phase_templates" in pm, "prompt_manifest missing 'phase_templates'"
        assert "role_files" in pm, "prompt_manifest missing 'role_files'"

    def test_shared_memo_exists(self, memo):
        assert len(memo) > 100, (
            f"shared_context/memo.txt too short: {len(memo)} chars"
        )

    def test_debug_banner(self, manifest, loader):
        _debug_banner(manifest, loader)

    def test_round_subdirectories(self, loader, rounds, roles):
        required_phases = ["proposals", "critiques", "revisions", "CRIT", "metrics"]
        for r in rounds:
            rd = loader.round_dir(r)
            assert rd.exists(), f"Round directory missing: {rd}"
            for phase in required_phases:
                pdir = rd / phase
                assert pdir.exists(), (
                    f"Missing {phase}/ in round {r}: {pdir}"
                )

    def test_phase_role_coverage(self, loader, rounds, roles):
        """critiques/, revisions/, CRIT/ each contain one sub-dir per role."""
        role_phases = ["critiques", "revisions", "CRIT"]
        for r in rounds:
            for phase in role_phases:
                phase_roles = loader.phase_role_dirs(r, phase)
                assert set(phase_roles) == set(roles), (
                    f"Role mismatch in round {r} {phase}/: "
                    f"found {sorted(phase_roles)}, expected {sorted(roles)}"
                )

    def test_proposals_round1_has_files(self, loader, roles):
        """proposals/ has role sub-dirs with files only in round 1."""
        for role in roles:
            pdir = loader.round_dir(1) / "proposals" / role
            assert pdir.exists(), (
                f"Missing proposals/{role}/ in round 1: {pdir}"
            )
            files = list(pdir.iterdir())
            assert len(files) > 0, (
                f"Empty proposals/{role}/ in round 1: {pdir}"
            )

    def test_proposals_after_round1_empty_ok(self, loader, rounds):
        """proposals/ in round > 1 may be empty."""
        for r in rounds:
            if r == 1:
                continue
            pdir = loader.round_dir(r) / "proposals"
            # Must exist but may be empty
            assert pdir.exists(), (
                f"Missing proposals/ in round {r}: {pdir}"
            )

    def test_retry_dirs_sequential(self, loader, rounds):
        for r in rounds:
            indices = loader.retry_indices(r)
            for i, idx in enumerate(indices):
                expected = i + 1
                assert idx == expected, (
                    f"Non-sequential retry dirs in round {r}: "
                    f"found index {idx} but expected {expected}. "
                    f"All indices: {indices}"
                )

    def test_intervention_files_sequential(self, loader, rounds):
        for r in rounds:
            files = loader.intervention_files(r)
            for i, f in enumerate(files):
                expected_name = f"intervention_{i:03d}.json"
                assert f.name == expected_name, (
                    f"Non-sequential intervention file in round {r}: "
                    f"found {f.name}, expected {expected_name}"
                )

    def test_final_directory(self, loader):
        final = loader.run_dir / "final"
        assert final.exists(), f"Missing final/ directory: {final}"
        for fname in ["final_portfolio.json", "judge_response.txt",
                       "judge_prompt.txt"]:
            fpath = final / fname
            assert fpath.exists(), f"Missing final file: {fpath}"

    def test_round_count_consistency(self, manifest, loader):
        expected = manifest["actual_rounds"]
        rounds_dir = loader.run_dir / "rounds"
        actual_dirs = sorted(
            d for d in rounds_dir.iterdir()
            if d.is_dir() and d.name.startswith("round_")
        )
        assert len(actual_dirs) == expected, (
            f"manifest.actual_rounds={expected} but found "
            f"{len(actual_dirs)} round directories: "
            f"{[d.name for d in actual_dirs]}"
        )

    def test_termination_reason(self, manifest):
        reason = manifest.get("termination_reason")
        assert reason, "manifest.termination_reason missing or empty"

    def test_round_index_consistency(self, loader, rounds):
        """JSON files with a 'round' field must match directory path int."""
        for r in rounds:
            # round_state.json
            rs = loader.round_state(r)
            assert rs["round"] == r, (
                f"round_state.json round={rs['round']} != dir round {r}"
            )
            # crit_scores.json
            cs = loader.crit_scores(r)
            assert cs["round"] == r, (
                f"crit_scores.json round={cs['round']} != dir round {r}"
            )
            # js_divergence.json
            js = loader.js_divergence(r)
            assert js["round"] == r, (
                f"js_divergence.json round={js['round']} != dir round {r}"
            )
            # evidence_overlap.json
            eo = loader.evidence_overlap(r)
            assert eo["round"] == r, (
                f"evidence_overlap.json round={eo['round']} != dir round {r}"
            )
            # pid_state.json
            ps = loader.pid_state(r)
            assert ps["round"] == r, (
                f"pid_state.json round={ps['round']} != dir round {r}"
            )
            # intervention files
            for ifile in loader.intervention_files(r):
                data = loader._read_json(ifile)
                assert data["round"] == r, (
                    f"{ifile.name} round={data['round']} != dir round {r}"
                )

    def test_role_stability(self, loader, rounds, roles):
        """Same set of roles in every phase directory of every round."""
        for r in rounds:
            for phase in ["critiques", "revisions", "CRIT"]:
                phase_roles = loader.phase_role_dirs(r, phase)
                assert set(phase_roles) == set(roles), (
                    f"Role mismatch in round {r} {phase}/: "
                    f"found {sorted(phase_roles)}, expected {sorted(roles)}"
                )

    def test_artifact_version_logged(self, manifest):
        """If manifest has artifact_version, log it (future-proofing)."""
        if "artifact_version" in manifest:
            print(f"  artifact_version: {manifest['artifact_version']}")

    def test_non_empty_artifact(self, manifest, rounds, roles, loader):
        assert len(rounds) > 0, "No rounds in artifact"
        assert len(roles) > 0, "No roles in artifact"
        # At least one prompt file exists
        prompt_path = (
            loader.round_dir(1) / "proposals" / roles[0] / "prompt.txt"
        )
        assert prompt_path.exists(), (
            f"No prompt files found; expected at least: {prompt_path}"
        )


# ---------------------------------------------------------------------------
# 3B. TestDebatePhases
# ---------------------------------------------------------------------------

def _round_role_params(loader):
    """Generate (round, role) pairs for parametrization."""
    try:
        m = loader.manifest()
        roles = m["roles"]
        rnds = list(range(1, m["actual_rounds"] + 1))
        return [(r, role) for r in rnds for role in roles]
    except Exception:
        return []


def _get_loader():
    """Get a loader instance for parametrize-time discovery."""
    run_dir = _resolve_run_dir()
    if not run_dir.exists():
        return None
    return RunArtifactLoader(run_dir)


_LOADER_FOR_PARAMS = _get_loader()
_ROUND_ROLE_PARAMS = (
    _round_role_params(_LOADER_FOR_PARAMS) if _LOADER_FOR_PARAMS else []
)
_ROUND_PARAMS = (
    _LOADER_FOR_PARAMS.rounds() if _LOADER_FOR_PARAMS else []
)


@pytest.mark.integration
class TestDebatePhases:
    """Validates proposal, critique, and revision artifacts per (round, role)."""

    # --- Proposal phase (round 1 only) --------------------------------------

    @pytest.mark.parametrize("role", [p[1] for p in _ROUND_ROLE_PARAMS if p[0] == 1] or ["skip"])
    def test_proposal_prompt(self, loader, role):
        if role == "skip":
            pytest.skip("No artifact available")
        prompt = loader.proposal_prompt(1, role)
        assert len(prompt) > 100, (
            f"Proposal prompt too short for {role} in round 1: {len(prompt)} chars"
        )

    @pytest.mark.parametrize("role", [p[1] for p in _ROUND_ROLE_PARAMS if p[0] == 1] or ["skip"])
    def test_proposal_response_structure(self, loader, tickers, role):
        if role == "skip":
            pytest.skip("No artifact available")
        raw = loader.proposal_response(1, role)
        assert len(raw) > 50, (
            f"Proposal response too short for {role}: {len(raw)} chars"
        )
        data = json.loads(raw)

        # Required top-level keys
        assert "allocation" in data and isinstance(data["allocation"], dict), (
            f"Proposal response missing/invalid 'allocation' for {role}"
        )
        assert "claims" in data and len(data["claims"]) > 0, (
            f"Proposal response missing/empty 'claims' for {role}"
        )
        assert "position_rationale" in data and isinstance(data["position_rationale"], list), (
            f"Proposal response missing 'position_rationale' for {role}"
        )
        assert "confidence" in data, (
            f"Proposal response missing 'confidence' for {role}"
        )
        conf = data["confidence"]
        assert isinstance(conf, (int, float)) and 0 <= conf <= 1, (
            f"Proposal confidence out of range for {role}: {conf}"
        )

        # Claim structure
        for i, claim in enumerate(data["claims"]):
            ctx = f"claim[{i}] for {role}"
            assert "claim_id" in claim and isinstance(claim["claim_id"], str), (
                f"Missing/invalid claim_id in {ctx}"
            )
            assert "claim_text" in claim and len(claim["claim_text"]) > 0, (
                f"Empty claim_text in {ctx}"
            )
            assert "reasoning_type" in claim and isinstance(claim["reasoning_type"], str), (
                f"Missing reasoning_type in {ctx}"
            )
            for key in ["evidence", "assumptions", "falsifiers", "impacts_positions"]:
                assert key in claim and isinstance(claim[key], list), (
                    f"Missing/non-list '{key}' in {ctx}"
                )
            assert "confidence" in claim and isinstance(claim["confidence"], (int, float)), (
                f"Missing/non-numeric confidence in {ctx}"
            )

    @pytest.mark.parametrize("role", [p[1] for p in _ROUND_ROLE_PARAMS if p[0] == 1] or ["skip"])
    def test_proposal_portfolio(self, loader, tickers, role):
        if role == "skip":
            pytest.skip("No artifact available")
        portfolio = loader.proposal_portfolio(1, role)
        _validate_portfolio(
            portfolio, tickers,
            context=f"for {role} in round 1, phase proposals",
        )

    # --- Critique phase (every round) ---------------------------------------

    @pytest.mark.parametrize("r,role", _ROUND_ROLE_PARAMS or [pytest.param(0, "skip", marks=pytest.mark.skip)])
    def test_critique_prompt(self, loader, r, role):
        prompt = loader.critique_prompt(r, role)
        assert len(prompt) > 100, (
            f"Critique prompt too short for {role} in round {r}: {len(prompt)} chars"
        )

    @pytest.mark.parametrize("r,role", _ROUND_ROLE_PARAMS or [pytest.param(0, "skip", marks=pytest.mark.skip)])
    def test_critique_response_structure(self, loader, r, role):
        data = loader.critique_response(r, role)

        assert "critiques" in data and len(data["critiques"]) > 0, (
            f"Critique response missing/empty 'critiques' for {role} in round {r}"
        )
        assert "self_critique" in data and isinstance(data["self_critique"], dict), (
            f"Critique response missing 'self_critique' for {role} in round {r}"
        )
        sc = data["self_critique"]
        assert "weakest_claim" in sc, (
            f"self_critique missing 'weakest_claim' for {role} in round {r}"
        )
        assert "explanation" in sc, (
            f"self_critique missing 'explanation' for {role} in round {r}"
        )

        for i, crit in enumerate(data["critiques"]):
            ctx = f"critique[{i}] for {role} in round {r}"
            assert "target_role" in crit and isinstance(crit["target_role"], str), (
                f"Missing target_role in {ctx}"
            )
            assert "objection" in crit and len(crit["objection"]) > 0, (
                f"Empty objection in {ctx}"
            )
            assert "counter_evidence" in crit and isinstance(crit["counter_evidence"], list), (
                f"Missing counter_evidence in {ctx}"
            )
            assert "objection_confidence" in crit and isinstance(
                crit["objection_confidence"], (int, float)
            ), f"Missing/non-numeric objection_confidence in {ctx}"

    # --- Revision phase (every round) ---------------------------------------

    @pytest.mark.parametrize("r,role", _ROUND_ROLE_PARAMS or [pytest.param(0, "skip", marks=pytest.mark.skip)])
    def test_revision_prompt(self, loader, r, role):
        prompt = loader.revision_prompt(r, role)
        assert len(prompt) > 100, (
            f"Revision prompt too short for {role} in round {r}: {len(prompt)} chars"
        )

    @pytest.mark.parametrize("r,role", _ROUND_ROLE_PARAMS or [pytest.param(0, "skip", marks=pytest.mark.skip)])
    def test_revision_response_structure(self, loader, r, role):
        raw = loader.revision_response(r, role)
        assert len(raw) > 50, (
            f"Revision response too short for {role} in round {r}: {len(raw)} chars"
        )
        data = json.loads(raw)
        assert "allocation" in data and isinstance(data["allocation"], dict), (
            f"Revision response missing 'allocation' for {role} in round {r}"
        )
        assert "claims" in data and isinstance(data["claims"], list), (
            f"Revision response missing 'claims' for {role} in round {r}"
        )

    @pytest.mark.parametrize("r,role", _ROUND_ROLE_PARAMS or [pytest.param(0, "skip", marks=pytest.mark.skip)])
    def test_revision_portfolio(self, loader, tickers, r, role):
        portfolio = loader.revision_portfolio(r, role)
        _validate_portfolio(
            portfolio, tickers,
            context=f"for {role} in round {r}, phase revisions",
        )

    # --- Retry revisions ----------------------------------------------------

    @pytest.mark.parametrize("r,role", _ROUND_ROLE_PARAMS or [pytest.param(0, "skip", marks=pytest.mark.skip)])
    def test_retry_revision_files(self, loader, tickers, r, role):
        """Each retry has prompt, response, portfolio per role."""
        for retry in loader.retry_indices(r):
            ctx = f"for {role} in round {r}, retry {retry}"

            prompt = loader.revision_retry_prompt(r, retry, role)
            assert len(prompt) > 100, (
                f"Retry revision prompt too short {ctx}: {len(prompt)} chars"
            )

            raw = loader.revision_retry_response(r, retry, role)
            assert len(raw) > 50, (
                f"Retry revision response too short {ctx}: {len(raw)} chars"
            )
            data = json.loads(raw)
            assert "allocation" in data and isinstance(data["allocation"], dict), (
                f"Retry revision response missing 'allocation' {ctx}"
            )

            portfolio = loader.revision_retry_portfolio(r, retry, role)
            _validate_portfolio(
                portfolio, tickers,
                context=ctx,
            )

    # --- Round-state ↔ artifact sync ----------------------------------------

    def test_round_state_proposal_sync(self, loader, roles, tickers):
        """round_state.proposals[role].allocation == proposal portfolio (R1)."""
        rs = loader.round_state(1)
        for role in roles:
            rs_alloc = rs["proposals"][role]["allocation"]
            disk_alloc = loader.proposal_portfolio(1, role)
            assert rs_alloc == disk_alloc, (
                f"round_state.proposals[{role}].allocation != "
                f"proposals/{role}/portfolio.json in round 1"
            )
            conf = rs["proposals"][role]["confidence"]
            assert isinstance(conf, (int, float)) and 0 <= conf <= 1, (
                f"round_state.proposals[{role}].confidence "
                f"out of range: {conf}"
            )

    @pytest.mark.parametrize("r", _ROUND_PARAMS or [pytest.param(0, marks=pytest.mark.skip)])
    def test_round_state_revision_sync(self, loader, roles, r):
        """round_state.revisions matches one of the revision portfolios on disk."""
        rs = loader.round_state(r)
        retries = loader.retry_indices(r)

        for role in roles:
            # Collect all candidate portfolios: base revision + all retries
            candidates = {"base": loader.revision_portfolio(r, role)}
            for retry in retries:
                candidates[f"retry_{retry}"] = loader.revision_retry_portfolio(
                    r, retry, role
                )

            rs_alloc = rs["revisions"][role]["allocation"]
            matched = any(rs_alloc == c for c in candidates.values())
            assert matched, (
                f"round_state.revisions[{role}].allocation in round {r} "
                f"does not match any revision portfolio on disk. "
                f"Candidates: {list(candidates.keys())}"
            )
            conf = rs["revisions"][role]["confidence"]
            assert isinstance(conf, (int, float)) and 0 <= conf <= 1, (
                f"round_state.revisions[{role}].confidence "
                f"out of range in round {r}: {conf}"
            )

    # --- Cross-round proposal stability -------------------------------------

    def test_cross_round_proposal_stability(self, loader, rounds):
        """For all rounds r > 1: proposals identical to round 1."""
        if len(rounds) < 2:
            pytest.skip("Only one round — no cross-round comparison possible")
        r1_proposals = loader.round_state(1)["proposals"]
        for r in rounds[1:]:
            rn_proposals = loader.round_state(r)["proposals"]
            assert rn_proposals == r1_proposals, (
                f"Proposals changed between round 1 and round {r}: "
                f"proposals are supposed to be immutable after round 1"
            )

    # --- Prompt integrity (system prompt stability) -------------------------

    def test_prompt_system_section_stability(self, loader, rounds, roles):
        """System prompts for the same phase are identical across rounds."""
        if len(rounds) < 2:
            pytest.skip("Only one round — no cross-round comparison possible")

        # Critique prompts
        for role in roles:
            r1_sys = _extract_system_prompt(loader.critique_prompt(1, role))
            for r in rounds[1:]:
                rn_sys = _extract_system_prompt(loader.critique_prompt(r, role))
                assert rn_sys == r1_sys, (
                    f"Critique system prompt changed for {role} "
                    f"between round 1 and round {r}"
                )

        # Revision prompts
        for role in roles:
            r1_sys = _extract_system_prompt(loader.revision_prompt(1, role))
            for r in rounds[1:]:
                rn_sys = _extract_system_prompt(loader.revision_prompt(r, role))
                assert rn_sys == r1_sys, (
                    f"Revision system prompt changed for {role} "
                    f"between round 1 and round {r}"
                )

    def test_within_round_system_prompt_sections_exist(self, loader, roles):
        """Each proposal prompt in round 1 has a system prompt section."""
        for role in roles:
            prompt = loader.proposal_prompt(1, role)
            sys_section = _extract_system_prompt(prompt)
            assert len(sys_section) > 100, (
                f"Proposal system prompt section too short for {role} "
                f"in round 1: {len(sys_section)} chars"
            )


# ---------------------------------------------------------------------------
# 3C. TestMetricsIntegrity
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestMetricsIntegrity:
    """Validates CRIT, JS divergence, evidence overlap, and PID state."""

    # --- CRIT outputs -------------------------------------------------------

    @pytest.mark.parametrize("r", _ROUND_PARAMS or [pytest.param(0, marks=pytest.mark.skip)])
    def test_crit_prompt_response_exist(self, loader, roles, r):
        for role in roles:
            prompt = loader.crit_prompt(r, role)
            assert len(prompt) > 100, (
                f"CRIT prompt too short for {role} in round {r}: {len(prompt)}"
            )
            response = loader.crit_response(r, role)
            assert len(response) > 50, (
                f"CRIT response too short for {role} in round {r}: {len(response)}"
            )

    @pytest.mark.parametrize("r", _ROUND_PARAMS or [pytest.param(0, marks=pytest.mark.skip)])
    def test_crit_scores_structure(self, loader, roles, r):
        cs = loader.crit_scores(r)
        assert "round" in cs, f"crit_scores missing 'round' in round {r}"
        assert "rho_bar" in cs, f"crit_scores missing 'rho_bar' in round {r}"
        rho_bar = cs["rho_bar"]
        assert isinstance(rho_bar, (int, float)) and 0 <= rho_bar <= 1, (
            f"crit_scores.rho_bar out of range in round {r}: {rho_bar}"
        )
        assert "agent_scores" in cs, f"crit_scores missing 'agent_scores' in round {r}"

        for role in roles:
            assert role in cs["agent_scores"], (
                f"crit_scores.agent_scores missing role {role} in round {r}"
            )
            agent = cs["agent_scores"][role]

            # rho_i
            assert "rho_i" in agent and isinstance(agent["rho_i"], (int, float)), (
                f"Missing/invalid rho_i for {role} in round {r}"
            )
            assert 0 <= agent["rho_i"] <= 1, (
                f"rho_i out of range for {role} in round {r}: {agent['rho_i']}"
            )

            # pillar_scores
            assert "pillar_scores" in agent, (
                f"Missing pillar_scores for {role} in round {r}"
            )
            for pillar in ["LV", "ES", "AC", "CA"]:
                assert pillar in agent["pillar_scores"], (
                    f"Missing pillar {pillar} for {role} in round {r}"
                )
                val = agent["pillar_scores"][pillar]
                assert isinstance(val, (int, float)) and 0 <= val <= 1, (
                    f"Pillar {pillar} out of range for {role} in round {r}: {val}"
                )

            # diagnostics
            assert "diagnostics" in agent, (
                f"Missing diagnostics for {role} in round {r}"
            )
            diag_keys = {
                "contradictions", "unsupported_claims", "ignored_critiques",
                "premature_certainty", "causal_overreach", "conclusion_drift",
            }
            for dk in diag_keys:
                assert dk in agent["diagnostics"], (
                    f"Missing diagnostic '{dk}' for {role} in round {r}"
                )
                assert isinstance(agent["diagnostics"][dk], bool), (
                    f"Diagnostic '{dk}' not bool for {role} in round {r}"
                )

            # explanations
            assert "explanations" in agent, (
                f"Missing explanations for {role} in round {r}"
            )
            expl_keys = {
                "logical_validity", "evidential_support",
                "alternative_consideration", "causal_alignment",
            }
            for ek in expl_keys:
                assert ek in agent["explanations"], (
                    f"Missing explanation '{ek}' for {role} in round {r}"
                )
                assert len(agent["explanations"][ek]) > 0, (
                    f"Empty explanation '{ek}' for {role} in round {r}"
                )

    # --- CRIT input integrity -----------------------------------------------

    @pytest.mark.parametrize("r", _ROUND_PARAMS or [pytest.param(0, marks=pytest.mark.skip)])
    def test_crit_prompt_content(self, loader, roles, tickers, r):
        """CRIT prompt contains role name, ticker, revised marker, evidence ID."""
        for role in roles:
            prompt = loader.crit_prompt(r, role)

            assert role in prompt.lower(), (
                f"CRIT prompt for {role} in round {r} does not contain role name"
            )

            has_ticker = any(t in prompt for t in tickers)
            assert has_ticker, (
                f"CRIT prompt for {role} in round {r} contains no ticker "
                f"from universe (proves portfolio data was injected)"
            )

            assert "revised" in prompt.lower(), (
                f"CRIT prompt for {role} in round {r} does not contain "
                f"'revised' (proves revised argument was included)"
            )

            has_evidence = bool(
                re.search(r"\[L[01]-|\[[A-Z]+-", prompt)
            )
            assert has_evidence, (
                f"CRIT prompt for {role} in round {r} contains no evidence "
                f"ID bracket pattern (proves evidence was injected)"
            )

    # --- Round-state ↔ CRIT sync --------------------------------------------

    @pytest.mark.parametrize("r", _ROUND_PARAMS or [pytest.param(0, marks=pytest.mark.skip)])
    def test_round_state_crit_sync(self, loader, roles, r):
        rs = loader.round_state(r)
        cs = loader.crit_scores(r)

        assert rs["crit"]["rho_bar"] == cs["rho_bar"], (
            f"round_state.crit.rho_bar ({rs['crit']['rho_bar']}) != "
            f"crit_scores.rho_bar ({cs['rho_bar']}) in round {r}"
        )
        assert rs["metrics"]["rho_bar"] == cs["rho_bar"], (
            f"round_state.metrics.rho_bar ({rs['metrics']['rho_bar']}) != "
            f"crit_scores.rho_bar ({cs['rho_bar']}) in round {r}"
        )

        for role in roles:
            rs_rho = rs["crit"][role]["rho_i"]
            cs_rho = cs["agent_scores"][role]["rho_i"]
            assert rs_rho == cs_rho, (
                f"round_state.crit[{role}].rho_i ({rs_rho}) != "
                f"crit_scores.agent_scores[{role}].rho_i ({cs_rho}) "
                f"in round {r}"
            )

    # --- JS Divergence ------------------------------------------------------

    @pytest.mark.parametrize("r", _ROUND_PARAMS or [pytest.param(0, marks=pytest.mark.skip)])
    def test_js_divergence(self, loader, r):
        js = loader.js_divergence(r)
        assert "round" in js, f"js_divergence missing 'round' in round {r}"
        assert "phase" in js, f"js_divergence missing 'phase' in round {r}"
        assert "js_divergence" in js, (
            f"js_divergence missing 'js_divergence' in round {r}"
        )
        val = js["js_divergence"]
        assert isinstance(val, (int, float)) and val >= 0, (
            f"js_divergence value invalid in round {r}: {val}"
        )

    @pytest.mark.parametrize("r", _ROUND_PARAMS or [pytest.param(0, marks=pytest.mark.skip)])
    def test_js_divergence_propose_exists(self, loader, r):
        js = loader.js_divergence(r, suffix="_propose")
        assert "round" in js, (
            f"js_divergence_propose missing 'round' in round {r}"
        )

    @pytest.mark.parametrize("r", _ROUND_PARAMS or [pytest.param(0, marks=pytest.mark.skip)])
    def test_js_divergence_retries(self, loader, r):
        for retry in loader.retry_indices(r):
            js = loader.js_divergence(r, suffix=f"_retry_{retry:03d}")
            assert "round" in js, (
                f"js_divergence_retry_{retry:03d} missing 'round' in round {r}"
            )

    @pytest.mark.parametrize("r", _ROUND_PARAMS or [pytest.param(0, marks=pytest.mark.skip)])
    def test_round_state_js_divergence(self, loader, r):
        rs = loader.round_state(r)
        val = rs["metrics"]["js_divergence"]
        assert isinstance(val, (int, float)) and val >= 0, (
            f"round_state.metrics.js_divergence invalid in round {r}: {val}"
        )

    # --- Evidence Overlap ---------------------------------------------------

    @pytest.mark.parametrize("r", _ROUND_PARAMS or [pytest.param(0, marks=pytest.mark.skip)])
    def test_evidence_overlap(self, loader, r):
        eo = loader.evidence_overlap(r)
        assert "round" in eo, f"evidence_overlap missing 'round' in round {r}"
        # The float value may be under "mean_overlap" or "evidence_overlap"
        val = eo.get("mean_overlap", eo.get("evidence_overlap"))
        assert val is not None, (
            f"evidence_overlap missing float value in round {r}"
        )
        assert isinstance(val, (int, float)), (
            f"evidence_overlap value not numeric in round {r}: {val}"
        )

    @pytest.mark.parametrize("r", _ROUND_PARAMS or [pytest.param(0, marks=pytest.mark.skip)])
    def test_evidence_overlap_propose_exists(self, loader, r):
        eo = loader.evidence_overlap(r, suffix="_propose")
        assert "round" in eo, (
            f"evidence_overlap_propose missing 'round' in round {r}"
        )

    @pytest.mark.parametrize("r", _ROUND_PARAMS or [pytest.param(0, marks=pytest.mark.skip)])
    def test_evidence_overlap_retries(self, loader, r):
        for retry in loader.retry_indices(r):
            eo = loader.evidence_overlap(r, suffix=f"_retry_{retry:03d}")
            assert "round" in eo, (
                f"evidence_overlap_retry_{retry:03d} missing 'round' "
                f"in round {r}"
            )

    @pytest.mark.parametrize("r", _ROUND_PARAMS or [pytest.param(0, marks=pytest.mark.skip)])
    def test_round_state_evidence_overlap(self, loader, r):
        rs = loader.round_state(r)
        val = rs["metrics"]["evidence_overlap"]
        assert isinstance(val, (int, float)) and val >= 0, (
            f"round_state.metrics.evidence_overlap invalid in round {r}: {val}"
        )

    # --- PID State ----------------------------------------------------------

    @pytest.mark.parametrize("r", _ROUND_PARAMS or [pytest.param(0, marks=pytest.mark.skip)])
    def test_pid_state_exists(self, loader, r):
        ps = loader.pid_state(r)
        assert "round" in ps, f"pid_state missing 'round' in round {r}"


# ---------------------------------------------------------------------------
# 3D. TestInterventionIntegrity
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestInterventionIntegrity:
    """Validates intervention schema, nudge injection, and retry causality."""

    def _all_interventions(self, loader, rounds):
        """Yield (round, index, data) for all intervention files."""
        for r in rounds:
            for i, fpath in enumerate(loader.intervention_files(r)):
                yield r, i, loader._read_json(fpath)

    def test_intervention_schema(self, loader, rounds):
        """Per-intervention required keys and types."""
        required = {
            "type", "round", "stage", "rule", "action",
            "severity", "retry", "metrics", "nudge_text",
        }
        for r, idx, data in self._all_interventions(loader, rounds):
            ctx = f"intervention_{idx:03d} in round {r}"
            missing = required - set(data.keys())
            assert not missing, f"Missing keys in {ctx}: {sorted(missing)}"

            assert data["type"] == "intervention", (
                f"type != 'intervention' in {ctx}: {data['type']}"
            )
            assert data["round"] == r, (
                f"round mismatch in {ctx}: {data['round']} != {r}"
            )
            assert data["stage"] in {"post_revision", "post_crit"}, (
                f"Invalid stage in {ctx}: {data['stage']}"
            )
            assert data["rule"] in {"js_collapse", "reasoning_quality"}, (
                f"Invalid rule in {ctx}: {data['rule']}"
            )

    def test_js_collapse_metrics(self, loader, rounds):
        """JS collapse interventions have correct metrics structure."""
        for r, idx, data in self._all_interventions(loader, rounds):
            if data["rule"] != "js_collapse":
                continue
            ctx = f"js_collapse intervention_{idx:03d} in round {r}"
            m = data["metrics"]
            for key in ["js_proposal", "js_revision", "collapse_ratio",
                        "threshold", "min_js_proposal"]:
                assert key in m, f"Missing metric '{key}' in {ctx}"
                assert isinstance(m[key], (int, float)), (
                    f"Non-numeric metric '{key}' in {ctx}: {m[key]}"
                )

    def test_reasoning_quality_metrics(self, loader, rounds):
        """Reasoning quality interventions have correct metrics structure."""
        for r, idx, data in self._all_interventions(loader, rounds):
            if data["rule"] != "reasoning_quality":
                continue
            ctx = f"reasoning_quality intervention_{idx:03d} in round {r}"
            m = data["metrics"]
            assert "weak_agents" in m and len(m["weak_agents"]) > 0, (
                f"Missing/empty weak_agents in {ctx}"
            )
            assert "rho_threshold" in m and isinstance(
                m["rho_threshold"], (int, float)
            ), f"Missing/invalid rho_threshold in {ctx}"

            for wa in m["weak_agents"]:
                assert "role" in wa, f"weak_agent missing 'role' in {ctx}"
                assert "rho_i" in wa, f"weak_agent missing 'rho_i' in {ctx}"
                assert "weak_pillars" in wa, (
                    f"weak_agent missing 'weak_pillars' in {ctx}"
                )
                assert "pillar_explanations" in wa, (
                    f"weak_agent missing 'pillar_explanations' in {ctx}"
                )

    def test_nudge_text_structure(self, loader, rounds):
        """nudge_text is dict with >= 1 role key, each nudge len > 20."""
        for r, idx, data in self._all_interventions(loader, rounds):
            ctx = f"intervention_{idx:03d} in round {r}"
            nt = data["nudge_text"]
            assert isinstance(nt, dict) and len(nt) >= 1, (
                f"nudge_text must be non-empty dict in {ctx}"
            )
            for role, nudge in nt.items():
                assert len(nudge) > 20, (
                    f"Nudge too short for {role} in {ctx}: {len(nudge)} chars"
                )

    def test_js_collapse_nudge_markers(self, loader, rounds):
        """JS collapse nudges contain protocol marker and role-specific marker."""
        role_markers = {
            "macro": "MACRO INTERVENTION REMINDER",
            "technical": "TECHNICAL INTERVENTION REMINDER",
        }
        for r, idx, data in self._all_interventions(loader, rounds):
            if data["rule"] != "js_collapse":
                continue
            ctx = f"js_collapse intervention_{idx:03d} in round {r}"
            for role, nudge in data["nudge_text"].items():
                assert "DEBATE DIVERSITY PROTOCOL ACTIVATED" in nudge, (
                    f"Missing protocol marker in {ctx} nudge for {role}"
                )
                if role in role_markers:
                    assert role_markers[role] in nudge, (
                        f"Missing role marker '{role_markers[role]}' "
                        f"in {ctx} nudge for {role}"
                    )

    def test_reasoning_quality_nudge_markers(self, loader, rounds):
        """Reasoning quality nudges contain audit marker."""
        for r, idx, data in self._all_interventions(loader, rounds):
            if data["rule"] != "reasoning_quality":
                continue
            ctx = f"reasoning_quality intervention_{idx:03d} in round {r}"
            for role, nudge in data["nudge_text"].items():
                assert "reasoning quality audit" in nudge.lower(), (
                    f"Missing 'reasoning quality audit' marker in {ctx} "
                    f"nudge for {role}"
                )

    def test_intervention_retry_causality(self, loader, rounds):
        """Each retry-action intervention has matching revisions_retry_NNN/ dir."""
        for r, idx, data in self._all_interventions(loader, rounds):
            if data["action"] != "retry_revision":
                continue
            retry = data["retry"]
            retry_dir = loader.round_dir(r) / f"revisions_retry_{retry:03d}"
            assert retry_dir.exists(), (
                f"Intervention {idx} (rule={data['rule']}, retry={retry}) "
                f"exists but revisions_retry_{retry:03d}/ missing in round {r}"
            )

            # Retry dir must contain sub-dirs for targeted roles
            nudge_roles = set(data["nudge_text"].keys())
            retry_role_dirs = {
                d.name for d in retry_dir.iterdir() if d.is_dir()
            }
            missing_roles = nudge_roles - retry_role_dirs
            assert not missing_roles, (
                f"Intervention {idx} (retry={retry}) targets roles "
                f"{sorted(nudge_roles)} but retry dir missing: "
                f"{sorted(missing_roles)} in round {r}"
            )

    def test_nudge_injection_into_retry_prompts(self, loader, rounds):
        """Each retry prompt contains at least one intervention nudge marker.

        Multiple interventions can target the same retry; the pipeline may
        inject only a subset of nudges. We verify that each retry prompt
        contains at least one recognizable marker from the interventions
        targeting that retry.
        """
        markers = {
            "js_collapse": "DEBATE DIVERSITY PROTOCOL ACTIVATED",
            "reasoning_quality": "reasoning quality audit",
        }

        for r in rounds:
            # Group interventions by retry number
            retries_interventions: dict[int, list[dict]] = {}
            for _r, idx, data in self._all_interventions(loader, [r]):
                if data["action"] != "retry_revision":
                    continue
                retries_interventions.setdefault(data["retry"], []).append(data)

            for retry, interventions in retries_interventions.items():
                # Collect all roles targeted by these interventions
                targeted_roles: set[str] = set()
                for intv in interventions:
                    targeted_roles.update(intv["nudge_text"].keys())

                for role in targeted_roles:
                    prompt_path = (
                        loader.round_dir(r)
                        / f"revisions_retry_{retry:03d}"
                        / role
                        / "prompt.txt"
                    )
                    if not prompt_path.exists():
                        continue

                    prompt_text = loader._read_text(prompt_path)
                    prompt_lower = prompt_text.lower()

                    # Check that at least one marker appears
                    found = any(
                        (markers[intv["rule"]] in prompt_text
                         or markers[intv["rule"]].lower() in prompt_lower)
                        for intv in interventions
                        if role in intv["nudge_text"]
                        and intv["rule"] in markers
                    )
                    assert found, (
                        f"No nudge marker found in retry {retry} prompt "
                        f"for {role} in round {r}. "
                        f"Expected one of: {[markers[i['rule']] for i in interventions if i['rule'] in markers]}"
                    )

    def test_consolidated_interventions(self, loader, rounds):
        """final/interventions_all_rounds.json count matches per-round sum."""
        consolidated = loader.interventions_all_rounds()
        assert isinstance(consolidated, list), (
            "interventions_all_rounds.json is not a list"
        )

        per_round_total = sum(
            len(loader.intervention_files(r)) for r in rounds
        )
        assert len(consolidated) == per_round_total, (
            f"interventions_all_rounds has {len(consolidated)} entries "
            f"but per-round files total {per_round_total}"
        )


# ---------------------------------------------------------------------------
# 3E. TestFinalOutputs
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestFinalOutputs:
    """Validates final portfolio, judge, and evidence grounding."""

    def test_final_portfolio(self, loader, tickers):
        portfolio = loader.final_portfolio()
        _validate_portfolio(
            portfolio, tickers,
            context="for final portfolio",
        )

    def test_judge_response(self, loader):
        resp = loader.judge_response()
        assert len(resp) > 100, (
            f"judge_response.txt too short: {len(resp)} chars"
        )

    def test_judge_prompt(self, loader, tickers):
        prompt = loader.judge_prompt()
        assert len(prompt) > 100, (
            f"judge_prompt.txt too short: {len(prompt)} chars"
        )
        has_ticker = any(t in prompt for t in tickers)
        assert has_ticker, (
            "judge_prompt.txt contains no ticker from universe "
            "(proves portfolio context was injected)"
        )

    def test_pid_crit_all_rounds(self, loader, rounds):
        data = loader.pid_crit_all_rounds()
        assert isinstance(data, list), "pid_crit_all_rounds.json is not a list"
        # CRIT evaluations run multiple times per round (once per
        # retry cycle), so there are >= num_rounds entries.
        assert len(data) >= len(rounds), (
            f"pid_crit_all_rounds has {len(data)} entries "
            f"but expected at least {len(rounds)} (one per round)"
        )
        # Every round should be represented
        logged_rounds = {entry["round"] for entry in data if "round" in entry}
        for r in rounds:
            assert r in logged_rounds, (
                f"Round {r} missing from pid_crit_all_rounds"
            )

    def test_interventions_all_rounds_exists(self, loader):
        data = loader.interventions_all_rounds()
        assert isinstance(data, list), (
            "interventions_all_rounds.json is not a list"
        )

    def test_evidence_grounding(self, loader, rounds, roles, memo):
        """Bracketed evidence tokens in responses exist in memo."""
        all_tokens: set[str] = set()

        # Proposals (round 1)
        for role in roles:
            try:
                raw = loader.proposal_response(1, role)
                all_tokens |= _extract_evidence_tokens(raw)
            except FileNotFoundError:
                pass

        # Revisions (all rounds + retries)
        for r in rounds:
            for role in roles:
                try:
                    raw = loader.revision_response(r, role)
                    all_tokens |= _extract_evidence_tokens(raw)
                except FileNotFoundError:
                    pass

                for retry in loader.retry_indices(r):
                    try:
                        raw = loader.revision_retry_response(r, retry, role)
                        all_tokens |= _extract_evidence_tokens(raw)
                    except FileNotFoundError:
                        pass

        # Check each token exists in memo
        ungrounded = {t for t in all_tokens if t not in memo}
        assert not ungrounded, (
            f"Evidence tokens referenced in responses but not found in "
            f"shared_context/memo.txt ({len(ungrounded)} tokens): "
            f"{sorted(ungrounded)[:10]}{'...' if len(ungrounded) > 10 else ''}"
        )
