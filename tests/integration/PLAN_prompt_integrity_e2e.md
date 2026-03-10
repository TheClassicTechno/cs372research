# Prompt-Integrity End-to-End Integration Test System

## Design Plan

---

## 1. Test Architecture

### Goal
Exercise the **full debate pipeline** (`propose → critique → revise → CRIT → PID`) with intercepted LLM calls, verifying that:
- Prompts are assembled correctly at every phase
- Agent JSON responses parse correctly through normalization
- State transitions between phases preserve structural fields
- CRIT receives well-formed reasoning bundles
- Cross-round state propagation works (proposals carry into round 2 critique)

### Approach: Seam-Based Testing
Rather than mocking at the graph-node level (which the existing `test_graph_state_propagation.py` already does with `mock=True`), we **intercept at the `_call_llm` boundary** and inject realistic enriched-format responses. This lets us:

1. Exercise the real prompt assembly code (`build_proposal_user_prompt`, `build_critique_prompt`, `build_revision_prompt`)
2. Exercise the real normalization pipeline (`_normalize_claims`, `_normalize_position_rationale`, `build_reasoning_bundle`)
3. Capture and inspect the exact prompts sent to the LLM
4. Validate the exact JSON responses flowing through the pipeline

### Test Levels
- **Level 1: Prompt Capture** — Run pipeline, capture all prompts, validate content
- **Level 2: Response Validation** — Inject realistic JSON, verify it parses through normalization
- **Level 3: State Transition** — Verify phase outputs appear in next phase's prompts
- **Level 4: Cross-Round** — Verify round 1 revisions appear in round 2 critique prompts
- **Level 5: CRIT Bundle** — Verify reasoning bundles sent to CRIT are well-formed

### Test File
Single new file: `tests/integration/test_prompt_integrity_e2e.py`

Depends on: `multi_agent.runner.MultiAgentRunner`, `multi_agent.config.DebateConfig`, the enriched agent profile system.

---

## 2. Canonical Run Replay

### Fixture Source
Use the existing canonical run logs in `logging/runs/` as a reference for what realistic LLM responses look like. Specifically:

```
logging/runs/enriched_no_macro_1rd_pid/run_2026-03-07_18-31-00/
  rounds/round_001/
    proposals/{value,risk,technical}/prompt.txt + response.txt + portfolio.json
    critiques/{value,risk,technical}/prompt.txt + response.json
    revisions/{value,risk,technical}/prompt.txt
  shared_context/memo.txt
  manifest.json
```

### Approach
**Do NOT replay the canonical run directly** (it requires the real memo and real LLM responses). Instead:

1. **Extract response schemas** from canonical logs to understand the realistic JSON structure
2. **Build fixture factories** that produce responses matching this structure
3. **Use `DebateConfig(mock=False)` with a monkeypatched `_call_llm`** so all real prompt assembly runs, but LLM calls return our fixtures

### Fixture Factories (defined in test file)
```python
def _make_enriched_proposal(role, tickers) -> str:
    """Return JSON string matching enriched proposal format."""

def _make_enriched_critique(role, targets) -> str:
    """Return JSON string matching enriched critique format."""

def _make_enriched_revision(role, tickers) -> str:
    """Return JSON string matching enriched revision format."""

def _make_crit_response() -> str:
    """Return JSON string matching CRIT scorer response format."""

def _make_judge_response(tickers) -> str:
    """Return JSON string matching judge response format."""
```

Each factory produces complete, well-formed JSON with:
- `allocation` dict (all tickers, sums to 1.0)
- `claims` array (3 claims with C1/C2/C3 IDs, evidence, assumptions, falsifiers, impacts_positions)
- `position_rationale` array (one per positive-weight ticker with `supported_by_claims`)
- `portfolio_rationale` string
- `confidence` float
- `risks_or_falsifiers` array

---

## 3. Mock LLM Design

### Injection Point
Monkeypatch `multi_agent.graph.llm._call_llm` with a capturing mock that:
1. Records every `(system_prompt, user_prompt, role, phase, round_num)` call
2. Returns the appropriate fixture response based on `phase` and `role`

```python
class CapturingLLM:
    """Records all LLM calls and returns phase-appropriate fixture responses."""

    def __init__(self, tickers, roles):
        self.calls: list[dict] = []  # captured calls
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
        # Route to appropriate fixture factory
        if phase == "crit":
            return _make_crit_response()
        if "judge" in (phase or "") or role == "judge":
            return _make_judge_response(self._tickers)
        # Debate phases: detect from user_prompt content
        if phase == "propose" or "initial proposal" in user_prompt.lower():
            return _make_enriched_proposal(role, self._tickers)
        if phase == "critique" or "critique" in user_prompt.lower():
            return _make_enriched_critique(role, self._roles)
        if phase == "revise" or "revision" in user_prompt.lower():
            return _make_enriched_revision(role, self._tickers)
        return _make_enriched_proposal(role, self._tickers)  # fallback

    def calls_for(self, phase=None, role=None) -> list[dict]:
        """Filter captured calls by phase and/or role."""
        ...
```

### Why Not `mock=True`?
The existing `mock=True` path in `mocks.py` produces **base-format** responses (no claims array, no position_rationale, no evidence IDs). It also bypasses prompt assembly entirely in graph nodes (the mock shortcut returns before `_call_llm` is reached). We need the **real prompt assembly** to exercise the enriched template pipeline.

### Monkeypatch Strategy
```python
@pytest.fixture
def capturing_llm(monkeypatch, tickers, roles):
    llm = CapturingLLM(tickers, roles)
    monkeypatch.setattr("multi_agent.graph.llm._call_llm", llm)
    # Also patch the CRIT scorer's LLM path
    monkeypatch.setattr("multi_agent.graph._call_llm", llm)
    return llm
```

**Critical**: The nodes in `nodes.py` call `_call_llm` from `multi_agent.graph` (imported via `from .graph import _call_llm`). The monkeypatch must target the name as imported by the consumer module, not just the definition module. We must patch:
- `multi_agent.graph.llm._call_llm` (definition site)
- `multi_agent.graph._call_llm` (re-exported in `graph/__init__.py`)

Alternatively, since `nodes.py` imports `_call_llm` at the module top level, we need to check the import chain and patch accordingly. The safest approach is to patch `multi_agent.graph.nodes._call_llm` directly if it's imported as a name binding, or to use the mock mode differently.

**Alternative approach**: Instead of monkeypatching `_call_llm`, we can configure a custom `_call_llm` wrapper via the `_prompt_capture` config callback + a custom LLM function. But the cleanest approach is:

1. Set `mock=False` so nodes don't take the mock shortcut
2. Monkeypatch the `_call_llm` function at all import sites

We'll verify the correct import sites during implementation.

---

## 4. Prompt Integrity Verification

### What to Verify in Propose Prompts
For each enriched agent (value, risk, technical):
- System prompt contains the role specialization text
- System prompt contains the causal contract
- User prompt contains the memo/context data
- User prompt contains the allocation output instructions (enumerated version)
- User prompt contains `claim_type` with all 5 values: `macro | sector | firm | risk | technical`
- User prompt contains `reasoning_type` with all 4 values
- User prompt contains evidence citation rules

### What to Verify in Critique Prompts
For each agent:
- User prompt contains `{{ my_proposal_v2 }}` rendered content (structural fields from own proposal)
- User prompt contains `{{ others_text_v2 }}` rendered content (other agents' proposals)
- For enriched templates: structural fields (claims, position_rationale) appear in rendered text
- For base templates: raw JSON appears (xfail expected)

### What to Verify in Revision Prompts
For each agent:
- User prompt contains `{{ my_proposal_v2 }}` rendered content
- User prompt contains `{{ critiques_text_v2 }}` rendered content (critiques targeting this agent)
- Critique `target_role` routing is correct (each agent only sees critiques targeting them)
- Allocation output instructions are present

### What to Verify in CRIT Prompts
For each agent:
- System prompt is rendered from `crit_system_enumerated.jinja` with correct `agent_role`
- User prompt contains the proposal bundle
- User prompt contains critiques_received
- User prompt contains the revised_argument bundle
- All claims in the bundle have `evidence_ids` (populated by normalizer)
- Position rationale has `supporting_claims` (populated by normalizer)

---

## 5. Snapshot Prompt Testing

### Approach
Capture the full rendered prompt text for each (agent, phase) combination and compare against golden snapshots.

### Snapshot Storage
```
tests/integration/snapshots/
  propose_value.txt
  propose_risk.txt
  propose_technical.txt
  critique_value.txt
  critique_risk.txt
  critique_technical.txt
  revise_value.txt
  revise_risk.txt
  revise_technical.txt
```

### Update Mechanism
```python
@pytest.fixture
def update_snapshots(request):
    """Set --update-snapshots flag to regenerate golden files."""
    return request.config.getoption("--update-snapshots", default=False)
```

### Comparison Strategy
- **Do NOT compare exact text** (memo content, timestamps, UUIDs change)
- Instead, extract **structural markers** from the prompt and compare:
  1. Section headers present (e.g., `## Your Task`, `## Output Format`)
  2. Template variable placeholders are all resolved (no `{{ }}` in output)
  3. Key instruction phrases present (e.g., "3–5 claims maximum")
  4. Evidence ID patterns present (e.g., `[L1-*]`, `[TICKER-*]`)

### Implementation
```python
def _extract_prompt_structure(prompt_text: str) -> dict:
    """Extract structural markers from a rendered prompt."""
    return {
        "section_headers": re.findall(r"^##\s+(.+)$", prompt_text, re.MULTILINE),
        "unresolved_vars": re.findall(r"\{\{.*?\}\}", prompt_text),
        "evidence_patterns": bool(re.search(r"\[[\w-]+\]", prompt_text)),
        "claim_type_enum": "macro | sector | firm | risk | technical" in prompt_text,
        "has_allocation_rules": "Weights MUST" in prompt_text or "sum to 1.0" in prompt_text,
    }
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

### Phase Transition Checks

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

### Test Implementation
```python
class TestPhaseTransitions:
    """Run 1-round pipeline, verify state at each phase boundary."""

    def test_proposals_populated_after_propose(self, pipeline_state):
        ...

    def test_critiques_target_correct_agents(self, pipeline_state):
        ...

    def test_revision_prompts_contain_relevant_critiques(self, captured_prompts):
        ...

    def test_crit_bundles_have_all_fields(self, pipeline_state):
        ...
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

    def test_round2_critique_sees_round1_revisions(self, captured_prompts):
        """Round 2 critique prompts contain round 1 revised allocations."""
        ...

    def test_round2_revision_sees_round2_critiques(self, captured_prompts):
        """Round 2 revision prompts contain round 2 critique content."""
        ...

    def test_proposals_unchanged_in_round2(self, pipeline_state):
        """Proposals are set once in round 1 and not regenerated."""
        ...

    def test_beta_changes_between_rounds(self, captured_prompts):
        """PID beta update changes tone in round 2 system prompts."""
        ...

    def test_crit_bundles_use_latest_revisions(self, pipeline_state):
        """Round 2 CRIT bundles reference round 2 revisions, not round 1."""
        ...
```

---

## 9. Test File Structure

```
tests/integration/
  test_prompt_integrity_e2e.py      # NEW - main e2e test file
  conftest.py                       # shared fixtures (if not already present)
  test_agent_crit_contract.py       # EXISTING - static contract tests
  test_graph_state_propagation.py   # EXISTING - mock-mode state tests
  test_crit_in_loop.py              # EXISTING - CRIT scorer unit tests
```

### File Organization within `test_prompt_integrity_e2e.py`
```python
# --- Constants & Config ---
TICKERS = [...]
ROLES = [...]
ENRICHED_PROFILES = {...}  # agent profile dicts

# --- Fixture Factories ---
def _make_enriched_proposal(role, tickers) -> str: ...
def _make_enriched_critique(role, targets) -> str: ...
def _make_enriched_revision(role, tickers) -> str: ...
def _make_crit_response() -> str: ...
def _make_judge_response(tickers) -> str: ...

# --- Capturing LLM Mock ---
class CapturingLLM: ...

# --- Pytest Fixtures ---
@pytest.fixture
def capturing_llm(monkeypatch): ...

@pytest.fixture
def single_round_state(capturing_llm): ...

@pytest.fixture
def two_round_state(capturing_llm): ...

# --- Group 1: Prompt Content Verification ---
class TestProposePromptContent: ...
class TestCritiquePromptContent: ...
class TestRevisionPromptContent: ...

# --- Group 2: Response Parsing & Normalization ---
class TestResponseParsing: ...
class TestNormalizationIntegrity: ...

# --- Group 3: Phase Transition Verification ---
class TestProposeOutputFormat: ...
class TestCritiqueRoutingCorrectness: ...
class TestRevisionInputIntegrity: ...

# --- Group 4: CRIT Bundle Verification ---
class TestCritBundleCompleteness: ...
class TestCritPromptContent: ...

# --- Group 5: Cross-Round Propagation ---
class TestCrossRoundPropagation: ...

# --- Group 6: Prompt Structure Invariants ---
class TestPromptStructuralInvariants: ...
```

---

## 10. Test Execution Flow

### Single-Round Test Flow
```
1. Configure DebateConfig with enriched agent profiles, mock=False, PID enabled
2. Monkeypatch _call_llm with CapturingLLM
3. Run MultiAgentRunner.run_returning_state(observation)
4. CapturingLLM records all (system_prompt, user_prompt, role, phase) calls
5. CapturingLLM returns realistic enriched-format JSON for each phase
6. After pipeline completes, inspect:
   a. captured_prompts — verify prompt content at each phase
   b. returned state — verify state transitions and data flow
   c. CRIT bundles — verify well-formed reasoning bundles
```

### Two-Round Test Flow
Same as above but with `max_rounds=2`. Additional verifications:
- Round 2 critique prompts reference round 1 revisions
- Proposals are not regenerated in round 2
- PID beta changes between rounds

### Running the Tests
```bash
# Run just the e2e tests
pytest tests/integration/test_prompt_integrity_e2e.py -v

# Run with existing integration tests
pytest tests/integration/ -v

# Run the full test suite
pytest -v
```

### Pytest Markers
```python
pytestmark = pytest.mark.integration
```

All tests should be marked `@pytest.mark.integration`. No real API calls are made (monkeypatched LLM). Tests should complete in <5 seconds.

---

## Key Design Decisions

1. **Monkeypatch `_call_llm` instead of `mock=True`**: The `mock=True` path in `mocks.py` bypasses prompt assembly entirely. We need real prompt assembly to test prompt integrity.

2. **Enriched profiles only**: These tests focus on the enriched agent profile family (which produces structured claims, position_rationale, etc.). Base profiles are already covered by xfail tests in `test_agent_crit_contract.py`.

3. **Fixture factories over canonical replay**: Building responses from factories is more maintainable than replaying captured logs (which are tied to specific memo content and may change).

4. **Structural comparison over exact matching**: Prompts contain dynamic content (memo data, timestamps). We verify structural invariants rather than exact text.

5. **State inspection via `run_returning_state()`**: This method (already used by `test_graph_state_propagation.py`) returns the raw LangGraph state dict without disk I/O, ideal for testing.

---

## Dependencies

- No new pip dependencies required
- Uses existing: `pytest`, `multi_agent.runner`, `multi_agent.config`, `eval.crit`
- Agent profiles: must load from `config/agent_profiles/` (existing enriched profiles)

---

## Risk & Open Questions

1. **Import path for monkeypatch**: Need to verify exact import chain for `_call_llm` in `nodes.py`. If it's imported as `from ..graph import _call_llm`, we must patch `multi_agent.graph.nodes` module-level binding. Will verify during implementation.

2. **Pipeline graph nodes**: The `compile_pipeline_graph` (Phase 1: news → data → build_context) also calls functions that may need mocking. In mock mode this is handled; without mock mode, we need to ensure the pipeline phase works with minimal observation data. May need to set `config["skip_pipeline"] = True` or provide sufficient observation data.

3. **PID config requirement**: For CRIT + PID to run, `pid_config` must be set. We'll construct a minimal `PIDConfig` in the test fixture.

4. **Parallel vs sequential**: Tests should use `parallel_agents=False` for deterministic call ordering (easier to verify which call corresponds to which agent/phase).
