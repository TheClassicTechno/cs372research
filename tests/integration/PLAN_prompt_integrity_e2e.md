# Prompt-Integrity End-to-End Integration Test System

## Design Plan (Revised)

---

## 1. Test Architecture

### Goal
Exercise the **full debate pipeline** (`propose → critique → revise → CRIT → PID`) with intercepted LLM calls, verifying that:
- Prompts are assembled correctly at every phase
- Agent JSON responses parse correctly through normalization
- State transitions between phases preserve structural fields
- CRIT receives well-formed reasoning bundles
- Cross-round state propagation works (proposals carry into round 2 critique)
- **Exact LLM call counts match expectations (no silent phase skips)**
- **Critique routing is correct (agents only see critiques targeting them)**
- **Rendered prompts match golden snapshots (detect wording/ordering regressions)**

### Approach: Seam-Based Testing
Rather than mocking at the graph-node level (which the existing `test_graph_state_propagation.py` already does with `mock=True`), we **intercept at the `_call_llm` boundary** and inject realistic enriched-format responses. This lets us:

1. Exercise the real prompt assembly code (`build_proposal_user_prompt`, `build_critique_prompt`, `build_revision_prompt`)
2. Exercise the real normalization pipeline (`_normalize_claims`, `_normalize_position_rationale`, `build_reasoning_bundle`)
3. Capture and inspect the exact prompts sent to the LLM
4. Validate the exact JSON responses flowing through the pipeline

### Test Levels
- **Level 1: Call Count** — Assert exact number of LLM calls per phase
- **Level 2: Prompt Capture** — Run pipeline, capture all prompts, validate content
- **Level 3: Golden Snapshots** — Compare rendered prompts against canonical snapshots
- **Level 4: Response Validation** — Inject realistic JSON, verify it parses through normalization
- **Level 5: State Transition** — Verify phase outputs appear in next phase's prompts
- **Level 6: Critique Routing** — Verify each agent's revision prompt contains only its own critiques
- **Level 7: Cross-Round** — Verify round 1 revisions appear in round 2 critique prompts
- **Level 8: CRIT Bundle** — Verify reasoning bundles sent to CRIT are well-formed
- **Level 9: Canonical Replay** — Replay real run logs and verify prompt compatibility

### Test File
Single new file: `tests/integration/test_prompt_integrity_e2e.py`

Depends on: `multi_agent.runner.MultiAgentRunner`, `multi_agent.config.DebateConfig`, the enriched agent profile system.

---

## 2. Canonical Run Replay

### Two Test Modes

#### Mode 1 — Synthetic Fixture Mode
Build fixture factories that produce well-formed enriched-format JSON responses. Used for the majority of integration tests (flexible, maintainable, independent of memo content).

Each factory produces complete, well-formed JSON with:
- `allocation` dict (all tickers, sums to 1.0)
- `claims` array (3 claims with C1/C2/C3 IDs, evidence, assumptions, falsifiers, impacts_positions)
- `position_rationale` array (one per positive-weight ticker with `supported_by_claims`)
- `portfolio_rationale` string
- `confidence` float
- `risks_or_falsifiers` array

#### Mode 2 — Canonical Replay Mode (NEW)
Load real prompts and responses from a canonical run directory and replay them through the pipeline.

**Canonical run source:**
```
logging/runs/test/run_2026-03-07_19-50-06/
  manifest.json                         # run metadata (roles, tickers, rounds)
  rounds/round_001/
    proposals/{value,risk,technical}/
      prompt.txt                        # rendered prompt (system + user combined)
      response.txt                      # raw LLM JSON response
      portfolio.json                    # parsed allocation
    critiques/{value,risk,technical}/
      prompt.txt
      response.json
    revisions/{value,risk,technical}/
      prompt.txt
      response.txt
      portfolio.json
    CRIT/{value,risk,technical}/
      prompt.txt
      response.txt
  rounds/round_002/                     # same structure
  shared_context/memo.txt               # the full memo used in this run
```

This run has:
- 3 agents: value, risk, technical
- 2 rounds with complete propose/critique/revise/CRIT
- 9 tickers: AMD, AMZN, BAC, CAT, CVX, PG, RTX, SLB, _CASH_
- Enriched templates, PID enabled, gpt-5-mini + gpt-5 CRIT

**Canonical replay loader:**
```python
CANONICAL_RUN = Path("logging/runs/test/run_2026-03-07_19-50-06")

def _load_canonical_responses(run_dir: Path, round_num: int) -> dict:
    """Load all LLM responses from a canonical run directory.

    Returns: {(phase, role): response_text}
    """
    responses = {}
    round_dir = run_dir / f"rounds/round_{round_num:03d}"
    for phase in ("proposals", "critiques", "revisions"):
        phase_dir = round_dir / phase
        if not phase_dir.exists():
            continue
        for agent_dir in sorted(phase_dir.iterdir()):
            role = agent_dir.name
            resp_file = agent_dir / "response.txt"
            if not resp_file.exists():
                resp_file = agent_dir / "response.json"
            if resp_file.exists():
                responses[(phase.rstrip("s"), role)] = resp_file.read_text()
    # CRIT responses
    crit_dir = round_dir / "CRIT"
    if crit_dir.exists():
        for agent_dir in sorted(crit_dir.iterdir()):
            responses[("crit", agent_dir.name)] = (agent_dir / "response.txt").read_text()
    return responses
```

**Canonical replay safety guard** — before returning a response, assert the (phase, role) key exists:
```python
class CanonicalReplayLLM:
    """Replays stored responses from a canonical run, with safety assertions."""

    def __init__(self, canonical_responses: dict):
        self._responses = canonical_responses
        self.calls: list[dict] = []

    def __call__(self, config, system_prompt, user_prompt,
                 role=None, phase=None, round_num=0) -> str:
        self.calls.append(...)
        # Map phase names: nodes use "propose"/"critique"/"revise",
        # but log dirs use "proposal"/"critique"/"revision"
        phase_key = _phase_to_log_key(phase)
        key = (phase_key, role)
        if key not in self._responses:
            raise KeyError(
                f"Missing canonical response for phase={phase!r} "
                f"role={role!r} round={round_num}. "
                f"Available keys: {sorted(self._responses.keys())}"
            )
        return self._responses[key]
```

This guard ensures the replay system cannot silently misalign responses if execution ordering changes (e.g., due to parallelism refactoring).

**Canonical replay test:**
```python
class TestCanonicalReplay:
    """Replay real run logs through the pipeline and verify prompt compatibility."""

    def test_canonical_responses_parse_through_normalization(self):
        """Every response from the canonical run parses into valid action_dicts."""
        ...

    def test_canonical_proposals_produce_valid_crit_bundles(self):
        """build_reasoning_bundle() succeeds on canonical run state."""
        ...

    def test_rendered_prompts_match_canonical(self):
        """Prompts generated by current code ≈ prompts stored in canonical run."""
        ...
```

---

## 3. Mock LLM Design

### Injection Point
Monkeypatch `_call_llm` at both consumer sites with a capturing mock that:
1. Records every `(system_prompt, user_prompt, role, phase, round_num)` call
2. Returns the appropriate fixture response based on the **`phase` parameter** (not prompt text inspection)

**Verified import chain:**
- `nodes.py` line 117: `from .llm import _call_llm` → creates `multi_agent.graph.nodes._call_llm`
- `runner.py` line 128: `from .graph import _call_llm` → creates `multi_agent.runner._call_llm`

Both are module-level name bindings. The `_call_llm_with_lifecycle` wrapper in nodes.py and the CRIT scorer lambda in runner.py both resolve `_call_llm` at call time from their respective module namespaces.

```python
class CapturingLLM:
    """Records all LLM calls and returns phase-appropriate fixture responses."""

    def __init__(self, tickers, roles):
        self.calls: list[dict] = []
        self._tickers = tickers
        self._roles = roles

    def __call__(self, config, system_prompt, user_prompt,
                 role=None, phase=None, round_num=0) -> str:
        self.calls.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "role": role,
            "phase": phase,
            "round_num": round_num,
        })
        # Route strictly by phase parameter — never inspect prompt text
        if phase == "crit":
            return _make_crit_response()
        if phase == "judge":
            return _make_judge_response(self._tickers)
        if phase == "propose":
            return _make_enriched_proposal(role, self._tickers)
        if phase == "critique":
            return _make_enriched_critique(role, self._roles)
        if phase == "revise":
            return _make_enriched_revision(role, self._tickers)
        raise ValueError(f"Unexpected phase={phase!r} role={role!r}")

    def calls_for(self, phase=None, role=None) -> list[dict]:
        """Filter captured calls by phase and/or role."""
        result = self.calls
        if phase is not None:
            result = [c for c in result if c["phase"] == phase]
        if role is not None:
            result = [c for c in result if c["role"] == role]
        return result
```

### Monkeypatch Strategy (Verified)
```python
@pytest.fixture
def capturing_llm(monkeypatch):
    llm = CapturingLLM(TICKERS, ROLES)
    # Patch at both consumer sites
    monkeypatch.setattr("multi_agent.graph.nodes._call_llm", llm)
    monkeypatch.setattr("multi_agent.runner._call_llm", llm)
    return llm
```

### Why Not `mock=True`?
The existing `mock=True` path in `mocks.py` produces **base-format** responses (minimal claims, no position_rationale, no evidence IDs). It also bypasses prompt assembly entirely in graph nodes (line 327-329 of nodes.py: `if is_mock: result = _mock_proposal(...)` returns before `_call_llm` is reached). We need the **real prompt assembly** to exercise the enriched template pipeline.

---

## 4. Prompt Integrity Verification

### Layer 1 — Structural Checks (per-phase assertions)

**Propose Prompts** — for each enriched agent (value, risk, technical):
- System prompt contains the role specialization text
- System prompt contains the causal contract
- User prompt contains the memo/context data
- User prompt contains the allocation output instructions (enumerated version)
- User prompt contains `claim_type` with all 5 values: `macro | sector | firm | risk | technical`
- User prompt contains `reasoning_type` with all 4 values
- User prompt contains evidence citation rules
- No unresolved `{{ }}` template variables remain

**Critique Prompts** — for each agent:
- User prompt contains `{{ my_proposal_v2 }}` rendered content (structural fields from own proposal)
- User prompt contains `{{ others_text_v2 }}` rendered content (other agents' proposals)
- Enriched structural fields (claims, position_rationale) appear in rendered text
- No unresolved `{{ }}` template variables remain

**Revision Prompts** — for each agent:
- User prompt contains `{{ my_proposal_v2 }}` rendered content
- User prompt contains `{{ critiques_text_v2 }}` rendered content (critiques targeting this agent)
- **Revision prompt for agent X does NOT contain critiques targeting other agents** (see Section 7b)
- Allocation output instructions are present
- No unresolved `{{ }}` template variables remain

**CRIT Prompts** — for each agent:
- System prompt is rendered from `crit_system_enumerated.jinja` with correct `agent_role`
- User prompt contains the proposal bundle
- User prompt contains critiques_received
- User prompt contains the revised_argument bundle
- All claims in the bundle have `evidence_ids` (populated by normalizer)
- Position rationale has `supporting_claims` (populated by normalizer)

### Layer 2 — Golden Prompt Snapshots (NEW)

Structural checks catch presence/absence but miss regressions like:
- Instruction wording changes
- Missing prompt sections
- Truncated memo injection
- Role prompt not being inserted
- Section ordering changes

Golden snapshots catch **all** of these.

**Snapshot directory:**
```
tests/integration/golden_prompts/
  propose_value_system.txt
  propose_value_user.txt
  propose_risk_system.txt
  propose_risk_user.txt
  propose_technical_system.txt
  propose_technical_user.txt
  critique_value_system.txt
  critique_value_user.txt
  critique_risk_system.txt
  critique_risk_user.txt
  critique_technical_system.txt
  critique_technical_user.txt
  revise_value_system.txt
  revise_value_user.txt
  revise_risk_system.txt
  revise_risk_user.txt
  revise_technical_system.txt
  revise_technical_user.txt
```

**Normalization function** — removes unstable content before comparison:
```python
def _normalize_prompt_for_snapshot(prompt_text: str) -> str:
    """Remove unstable elements from a prompt for snapshot comparison.

    Strips:
    - timestamps / dates (ISO format, quarter dates)
    - UUIDs
    - floating-point numbers (allocation weights, confidence scores, PID values)
    - excessive whitespace (collapse to single spaces / newlines)
    """
    text = prompt_text
    # Normalize UUIDs
    text = re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "<UUID>", text)
    # Normalize ISO timestamps
    text = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[^\s\"]*", "<TIMESTAMP>", text)
    # Normalize floating-point numbers (e.g., 0.10, 0.100, 0.0999999, 3.14)
    # This prevents noisy failures from rounding differences or PID-driven weight changes
    text = re.sub(r"\b\d+\.\d+\b", "<FLOAT>", text)
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
```

**Snapshot update mechanism:**
```python
# conftest.py
def pytest_addoption(parser):
    parser.addoption("--update-snapshots", action="store_true", default=False)

# In tests:
def _assert_snapshot(prompt_text, snapshot_name, request):
    """Compare prompt against golden snapshot, or update if --update-snapshots."""
    snapshot_path = GOLDEN_DIR / snapshot_name
    normalized = _normalize_prompt_for_snapshot(prompt_text)

    if request.config.getoption("--update-snapshots"):
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(normalized)
        return

    assert snapshot_path.exists(), f"Golden snapshot missing: {snapshot_path}"
    expected = snapshot_path.read_text()
    assert normalized == expected, (
        f"Prompt regression detected in {snapshot_name}. "
        f"Run with --update-snapshots to regenerate."
    )
```

**First run**: execute with `--update-snapshots` to generate golden files.
**Subsequent runs**: compare against golden files, fail on any change.

---

## 5. LLM Call Count Assertions (NEW)

### Why This Matters
Without explicit call count checks, a pipeline bug could silently skip a phase (e.g., critique is bypassed but revise still runs). The test suite would still pass if it only checks output format.

### Expected Call Counts

**Single round, 3 agents (value, risk, technical):**
```
propose:  3 calls (1 per agent)
critique: 3 calls (1 per agent)
revise:   3 calls (1 per agent)
crit:     3 calls (1 per agent, for CRIT scoring)
judge:    1 call  (final aggregation)
─────────────────────────
total:   13 calls
```

**Two rounds, 3 agents:**
```
Round 1: propose(3) + critique(3) + revise(3) + crit(3) = 12
Round 2: propose(0, idempotent) + critique(3) + revise(3) + crit(3) = 9
Judge:   1
─────────────────────────
total:   22 calls
```

Note: Round 2 propose calls 0 because `propose_node` has an idempotency guard (returns no-op if proposals already exist). This is a critical invariant to verify.

### Test Implementation
```python
class TestLLMCallCounts:
    """Verify exact number of LLM calls per phase."""

    def test_single_round_total_calls(self, single_round_state, capturing_llm):
        assert len(capturing_llm.calls) == 13

    def test_single_round_propose_calls(self, single_round_state, capturing_llm):
        assert len(capturing_llm.calls_for(phase="propose")) == 3

    def test_single_round_critique_calls(self, single_round_state, capturing_llm):
        assert len(capturing_llm.calls_for(phase="critique")) == 3

    def test_single_round_revise_calls(self, single_round_state, capturing_llm):
        assert len(capturing_llm.calls_for(phase="revise")) == 3

    def test_single_round_crit_calls(self, single_round_state, capturing_llm):
        assert len(capturing_llm.calls_for(phase="crit")) == 3

    def test_single_round_judge_calls(self, single_round_state, capturing_llm):
        assert len(capturing_llm.calls_for(phase="judge")) == 1

    def test_two_round_total_calls(self, two_round_state, capturing_llm_2rd):
        assert len(capturing_llm_2rd.calls) == 22

    def test_round2_propose_is_noop(self, two_round_state, capturing_llm_2rd):
        """Propose is idempotent — no LLM calls in round 2."""
        r2_proposes = [c for c in capturing_llm_2rd.calls_for(phase="propose")
                       if c["round_num"] == 2]
        assert len(r2_proposes) == 0

    def test_each_role_called_once_per_phase(self, single_round_state, capturing_llm):
        """No agent is called twice in the same phase."""
        for phase in ("propose", "critique", "revise"):
            roles_called = [c["role"] for c in capturing_llm.calls_for(phase=phase)]
            assert sorted(roles_called) == sorted(ROLES)
```

---

## 6. JSON Response Validation

### Agent Response Validation
After each phase, validate the captured LLM response against the expected JSON schema:

```python
ENRICHED_PROPOSAL_SCHEMA = {
    "required": ["allocation", "claims", "position_rationale",
                  "portfolio_rationale", "confidence", "risks_or_falsifiers"],
    "claims_required": ["claim_id", "claim_text", "claim_type", "reasoning_type",
                        "evidence", "assumptions", "falsifiers", "impacts_positions", "confidence"],
    "position_rationale_required": ["ticker", "weight", "supported_by_claims", "explanation"],
}

ENRICHED_CRITIQUE_SCHEMA = {
    "required": ["critiques"],
    "critique_required": ["target_role", "objection"],
}

ENRICHED_REVISION_SCHEMA = {
    "required": ["allocation", "claims", "position_rationale",
                  "portfolio_rationale", "confidence", "risks_or_falsifiers", "revision_notes"],
}
```

### Normalization Validation
After normalization runs (in `build_reasoning_bundle`), verify:
- Every claim has `evidence_ids` (computed from `evidence`)
- Every claim has `claim_type` (defaulted to "unknown" if missing)
- Position rationale has `supporting_claims` (mapped from `supported_by_claims`)
- Allocations sum to ~1.0

### CRIT Response Validation
Validate CRIT responses match the `CritResult` schema:
- `pillar_scores`: 4 floats in [0, 1]
- `diagnostics`: 6 booleans + 5 counts
- `explanations`: 4 strings

---

## 7. Debate State Transition Verification

### 7a. Phase Transition Checks

**Propose → Critique:**
- `state["proposals"]` has one entry per agent
- Each proposal has `role`, `action_dict`, `action` (JSON string), `raw_response`
- Critique prompt for agent X contains rendered proposal from agent X
- Critique prompt for agent X contains rendered proposals from agents Y, Z

**Critique → Revise:**
- `state["critiques"]` has one entry per agent
- Each critique entry has `role`, `critiques` (list of critique objects)
- Critique objects have `target_role` matching another agent
- Revise prompt for agent X contains only critiques where `target_role == X`
- Revise prompt for agent X contains rendered proposal from agent X

**Revise → CRIT:**
- `state["revisions"]` has one entry per agent
- Each revision has `action_dict` with complete allocation + claims
- `build_reasoning_bundle()` produces a well-formed bundle with:
  - `proposal` (from state["proposals"])
  - `critiques_received` (filtered from state["critiques"])
  - `revised_argument` (from state["revisions"])
  - All evidence_ids populated

**CRIT → PID:**
- `RoundCritResult` has one `CritResult` per agent
- `rho_bar` is mean of per-agent rho values
- PID controller receives valid `(rho_bar, js, ov)` inputs

### 7b. Critique Routing Validation (NEW — Critical)

This is one of the most important tests in the suite. The critique-to-revision routing must be correct: each agent's revision prompt must contain **only** critiques targeting that agent, and must **not** contain critiques targeting other agents.

**Bug this catches:**
```
value's critique targeting risk
appears in macro's revision prompt
→ macro responds to irrelevant critiques
→ reasoning quality degrades silently
```

**Test implementation:**
```python
class TestCritiqueRouting:
    """Verify critique-to-revision routing is correct."""

    def test_revision_prompt_contains_only_own_critiques(self, capturing_llm):
        """Each agent's revision prompt contains only critiques targeting them."""
        for role in ROLES:
            revise_calls = capturing_llm.calls_for(phase="revise", role=role)
            assert len(revise_calls) == 1
            user_prompt = revise_calls[0]["user_prompt"]

            # Must contain critiques targeting this role
            # (Our fixtures produce critiques with target_role matching other agents)
            # Verify by checking for the role name in critique context

            for other_role in ROLES:
                if other_role == role:
                    continue
                # The rendered critiques_text_v2 includes "targeting: ROLE"
                # or "[FROM_ROLE]: objection about ROLE"
                # We check that critiques about OTHER roles don't appear
                # by checking for the specific fixture objection text

    def test_revision_prompt_excludes_other_agents_critiques(self, capturing_llm):
        """Revision prompt for agent X does NOT contain critiques targeting Y."""
        for role in ROLES:
            revise_calls = capturing_llm.calls_for(phase="revise", role=role)
            user_prompt = revise_calls[0]["user_prompt"]

            # Our fixture critiques contain unique marker text per target.
            # Verify that marker text for OTHER targets does NOT appear.
            for other_role in ROLES:
                if other_role == role:
                    continue
                # Fixture critique targeting `other_role` contains
                # f"challenges {other_role}'s" — should NOT appear in
                # this agent's revision prompt
                marker = f"challenges {other_role}'s"
                assert marker not in user_prompt.lower(), (
                    f"{role}'s revision prompt contains critique targeting {other_role}"
                )

    def test_critique_target_roles_match_other_agents(self, single_round_state):
        """Every critique targets an agent that exists in the debate."""
        for critic in single_round_state["critiques"]:
            for crit in critic["critiques"]:
                target = crit.get("target_role", "").lower().split()[0]
                assert target in [r.lower() for r in ROLES], (
                    f"Critique targets unknown role: {target}"
                )

    def test_critique_routing_via_structured_state(self, single_round_state, capturing_llm):
        """Cross-verify critique routing using structured state data, not just prompt text.

        For each agent, collect expected critiques from state["critiques"]
        where target_role matches, then verify those critiques appear in
        the revision prompt. This is robust against prompt formatting changes.
        """
        for role in ROLES:
            # Collect expected critiques from structured state
            expected_objections = []
            for critic in single_round_state["critiques"]:
                for crit in critic["critiques"]:
                    raw_target = crit.get("target_role", "")
                    target = raw_target.lower().split()[0] if raw_target else ""
                    if target == role.lower():
                        expected_objections.append(crit["objection"])

            # Verify expected critiques appear in the revision prompt
            revise_calls = capturing_llm.calls_for(phase="revise", role=role)
            user_prompt = revise_calls[0]["user_prompt"]
            for objection_text in expected_objections:
                # Check a distinctive substring of the objection
                snippet = objection_text[:60]
                assert snippet in user_prompt, (
                    f"Expected critique for {role} not found in revision prompt: "
                    f"{snippet!r}"
                )
```

---

## 8. Cross-Round State Propagation

### What Changes Between Rounds
- Round 1: `proposals` are generated fresh, `revisions` replace proposals
- Round 2+: `proposals` are unchanged (idempotent guard), critique operates on previous round's `revisions`
- `critiques` list is reset each round (`state["critiques"] = []` at line 727 of runner.py)
- `revisions` are NOT reset (critique nodes need them)
- PID updates `beta` between rounds, affecting tone in system prompts

### Tests for 2-Round Debate
```python
class TestCrossRoundPropagation:
    """Run 2-round pipeline, verify cross-round state."""

    def test_round2_critique_sees_round1_revisions(self, capturing_llm_2rd):
        """Round 2 critique prompts contain round 1 revised allocations."""
        ...

    def test_round2_revision_sees_round2_critiques(self, capturing_llm_2rd):
        """Round 2 revision prompts contain round 2 critique content."""
        ...

    def test_proposals_unchanged_in_round2(self, two_round_state):
        """Proposals are set once in round 1 and not regenerated."""
        ...

    def test_beta_changes_between_rounds(self, capturing_llm_2rd):
        """PID beta update changes tone in round 2 system prompts."""
        ...

    def test_crit_bundles_use_latest_revisions(self, two_round_state):
        """Round 2 CRIT bundles reference round 2 revisions, not round 1."""
        ...

    def test_round2_critique_routing_correct(self, capturing_llm_2rd):
        """Critique routing is still correct in round 2."""
        ...
```

---

## 9. Test File Structure

```
tests/integration/
  test_prompt_integrity_e2e.py          # NEW - main e2e test file
  golden_prompts/                       # NEW - golden prompt snapshots
    propose_value_system.txt
    propose_value_user.txt
    propose_risk_system.txt
    propose_risk_user.txt
    propose_technical_system.txt
    propose_technical_user.txt
    critique_value_system.txt
    critique_value_user.txt
    ...
    revise_value_system.txt
    revise_value_user.txt
    ...
  prompt_hashes.json                    # NEW - sha256 prompt hashes (optional)
  conftest.py                           # shared fixtures + --update-snapshots option
  test_agent_crit_contract.py           # EXISTING - static contract tests
  test_graph_state_propagation.py       # EXISTING - mock-mode state tests
  test_crit_in_loop.py                  # EXISTING - CRIT scorer unit tests
```

### File Organization within `test_prompt_integrity_e2e.py`
```python
# --- Constants & Config ---
TICKERS = ["AAPL", "MSFT", "NVDA"]
ROLES = ["value", "risk", "technical"]
GOLDEN_DIR = Path(__file__).parent / "golden_prompts"
CANONICAL_RUN = Path(__file__).resolve().parent.parent.parent / (
    "logging/runs/test/run_2026-03-07_19-50-06"
)

# --- Fixture Factories ---
def _make_enriched_proposal(role, tickers) -> str: ...
def _make_enriched_critique(role, targets) -> str: ...
def _make_enriched_revision(role, tickers) -> str: ...
def _make_crit_response() -> str: ...
def _make_judge_response(tickers) -> str: ...

# --- Prompt Normalization ---
def _normalize_prompt_for_snapshot(text) -> str: ...
def _assert_snapshot(text, name, request): ...

# --- Canonical Run Loader ---
def _load_canonical_responses(run_dir, round_num) -> dict: ...

# --- Capturing LLM Mock ---
class CapturingLLM: ...

# --- Pytest Fixtures ---
@pytest.fixture
def capturing_llm(monkeypatch): ...

@pytest.fixture
def single_round_state(capturing_llm): ...

@pytest.fixture
def two_round_state(monkeypatch): ...

# --- Group 1: LLM Call Count Verification ---
class TestLLMCallCounts: ...

# --- Group 2: Prompt Content Verification (Structural) ---
class TestProposePromptContent: ...
class TestCritiquePromptContent: ...
class TestRevisionPromptContent: ...

# --- Group 3: Golden Prompt Snapshots ---
class TestGoldenSnapshots: ...

# --- Group 4: Response Parsing & Normalization ---
class TestResponseParsing: ...
class TestNormalizationIntegrity: ...

# --- Group 5: Phase Transition Verification ---
class TestProposeOutputFormat: ...
class TestCritiqueRoutingCorrectness: ...
class TestRevisionInputIntegrity: ...

# --- Group 6: Critique Routing Validation ---
class TestCritiqueRouting: ...

# --- Group 7: CRIT Bundle Verification ---
class TestCritBundleCompleteness: ...
class TestCritPromptContent: ...

# --- Group 8: Cross-Round Propagation ---
class TestCrossRoundPropagation: ...

# --- Group 9: Canonical Replay ---
class TestCanonicalReplay: ...

# --- Group 10: Prompt Structure Invariants ---
class TestPromptStructuralInvariants: ...

# --- Group 11: Prompt Hash Stability (Optional) ---
class TestPromptHashStability: ...
```

---

## 10. Test Execution Flow

### Single-Round Test Flow
```
1. Configure DebateConfig with enriched agent profiles, mock=False, PID enabled
2. Monkeypatch _call_llm at both import sites (nodes._call_llm + runner._call_llm)
3. Run MultiAgentRunner.run_returning_state(observation)
4. CapturingLLM records all (system_prompt, user_prompt, role, phase, round_num) calls
5. CapturingLLM returns realistic enriched-format JSON, routed by phase parameter
6. After pipeline completes, verify:
   a. Call counts — 13 total (3 propose + 3 critique + 3 revise + 3 crit + 1 judge)
   b. Prompt content — structural checks + golden snapshot comparison
   c. Critique routing — each agent only sees its own critiques
   d. State transitions — phase outputs appear in next phase's inputs
   e. CRIT bundles — well-formed with normalized fields
```

### Two-Round Test Flow
Same as above but with `max_rounds=2`. Additional verifications:
- 22 total calls (12 round1 + 9 round2 + 1 judge)
- Round 2 propose is a no-op (0 LLM calls)
- Round 2 critique prompts reference round 1 revisions
- Round 2 critique routing still correct
- PID beta changes between rounds

### Canonical Replay Flow
```
1. Load responses from logging/runs/test/run_2026-03-07_19-50-06/
2. Build CanonicalReplayLLM that returns stored responses by (phase, role)
3. Run pipeline with same config as canonical run
4. Verify rendered prompts ≈ stored prompts (after normalization)
5. Verify responses parse through normalization without errors
```

### Running the Tests
```bash
# Run just the e2e tests
pytest tests/integration/test_prompt_integrity_e2e.py -v

# Generate golden snapshots (first time or after intentional prompt changes)
pytest tests/integration/test_prompt_integrity_e2e.py -v --update-snapshots

# Run with existing integration tests
pytest tests/integration/ -v

# Run the full test suite
pytest -v
```

### Pytest Markers
```python
pytestmark = [pytest.mark.integration, pytest.mark.no_external_api]
```

All tests carry both `integration` and `no_external_api` markers. No real API calls are made (monkeypatched LLM). Tests should complete in <10 seconds. The `no_external_api` marker makes it easy to include these in CI without worrying about API key availability:

```bash
# Run only tests that need no external APIs
pytest -m no_external_api -v

# Run full integration suite (some may need API keys)
pytest -m integration -v
```

### Optional: Prompt Hash Stability Test

For an additional layer of regression detection, maintain sha256 hashes of normalized prompts in a JSON sidecar file. This is faster than full-text snapshot comparison and useful for quick CI smoke tests.

**Hash file:** `tests/integration/prompt_hashes.json`
```json
{
  "propose_value_system": "a1b2c3...",
  "propose_value_user": "d4e5f6...",
  "critique_risk_system": "...",
  ...
}
```

**Implementation:**
```python
import hashlib, json

HASH_FILE = Path(__file__).parent / "prompt_hashes.json"

def _compute_prompt_hash(prompt_text: str) -> str:
    normalized = _normalize_prompt_for_snapshot(prompt_text)
    return hashlib.sha256(normalized.encode()).hexdigest()

class TestPromptHashStability:
    """Quick sha256 check that prompts haven't changed."""

    def test_prompt_hashes_match(self, capturing_llm, request):
        if request.config.getoption("--update-snapshots"):
            hashes = {}
            for call in capturing_llm.calls:
                if call["phase"] in ("propose", "critique", "revise"):
                    key_sys = f"{call['phase']}_{call['role']}_system"
                    key_usr = f"{call['phase']}_{call['role']}_user"
                    hashes[key_sys] = _compute_prompt_hash(call["system_prompt"])
                    hashes[key_usr] = _compute_prompt_hash(call["user_prompt"])
            HASH_FILE.write_text(json.dumps(hashes, indent=2, sort_keys=True))
            pytest.skip("Updated prompt hashes")

        assert HASH_FILE.exists(), "Run with --update-snapshots to generate prompt hashes"
        expected = json.loads(HASH_FILE.read_text())
        for call in capturing_llm.calls:
            if call["phase"] in ("propose", "critique", "revise"):
                for suffix, field in [("system", "system_prompt"), ("user", "user_prompt")]:
                    key = f"{call['phase']}_{call['role']}_{suffix}"
                    actual = _compute_prompt_hash(call[field])
                    assert actual == expected.get(key), (
                        f"Prompt hash mismatch for {key}. "
                        f"Run with --update-snapshots to regenerate."
                    )
```

This test is **optional** — the golden snapshot tests provide the same regression coverage. But the hash test is faster to run and easier to diff in code review.

---

## Key Design Decisions

1. **Monkeypatch `_call_llm` instead of `mock=True`**: The `mock=True` path in `mocks.py` bypasses prompt assembly entirely. We need real prompt assembly to test prompt integrity.

2. **Phase-based routing, not prompt text inspection**: `CapturingLLM` routes responses using the `phase` parameter passed to `_call_llm`, never by inspecting prompt content. This is robust against prompt wording changes.

3. **Two-layer prompt validation**: Structural checks (key phrases, section headers, no unresolved vars) + golden snapshots (full normalized text comparison). Together these catch both missing content and subtle regressions.

4. **Two test modes**: Synthetic fixtures for flexible integration tests + canonical replay for historical regression tests. Both exercise the same pipeline seam.

5. **Explicit call count assertions**: Prevents silent phase skips. The exact number of LLM calls is a critical pipeline invariant.

6. **Critique routing validation**: Dedicated test group verifying each agent's revision prompt contains only its own critiques. This is a common debate pipeline bug with high impact on reasoning quality.

7. **State inspection via `run_returning_state()`**: This method (already used by `test_graph_state_propagation.py`) returns the raw LangGraph state dict without disk I/O, ideal for testing.

8. **Sequential execution**: Tests use `parallel_agents=False` for deterministic call ordering (easier to verify which call corresponds to which agent/phase).

---

## Dependencies

- No new pip dependencies required
- Uses existing: `pytest`, `multi_agent.runner`, `multi_agent.config`, `eval.crit`
- Agent profiles: must load from `config/agent_profiles/` (existing enriched profiles)
- Canonical run: `logging/runs/test/run_2026-03-07_19-50-06/` (checked into repo)

---

## Resolved Questions

1. **Import path for monkeypatch**: Verified. `nodes.py` does `from .llm import _call_llm`, creating `multi_agent.graph.nodes._call_llm`. `runner.py` does `from .graph import _call_llm`, creating `multi_agent.runner._call_llm`. Both must be patched.

2. **Pipeline graph**: Phase 1 (news → data → build_context) runs through `pipeline_graph.invoke()`. With `mock=False` but no API keys needed (pipeline graph builds enriched_context from observation data, which we provide via `text_context`).

3. **PID config**: Constructed as a minimal `PIDConfig` in test fixtures with default gains.

4. **Phase routing**: `_call_llm` receives explicit `phase=` parameter from all call sites in nodes.py (`"propose"`, `"critique"`, `"revise"`, `"judge"`) and runner.py (`"crit"`). No prompt text inspection needed.
