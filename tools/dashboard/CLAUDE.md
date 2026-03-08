# Dashboard Development Rules

**MANDATORY — You MUST read and abide by ALL of the following rule documents before making ANY dashboard change. This is non-negotiable. Do not skip this step. Do not rely on the summary below as a substitute. Read the full documents:**
- `rules/ARCHITECTURE.md`
- `rules/CLAUDE_DASHBOARD_RULES.md`
- `rules/LLM_WORKFLOW.md`
- `rules/TESTING_GUIDE.md`

**These documents are the authoritative, binding specification for all dashboard development. Every rule in them must be followed.** The summary below highlights the most critical constraints but is NOT a replacement for reading the full docs.

## Architecture Layers

```
app.js → router.js → views/* → components/* → utils/*
                              → api/*
```

Lower layers MUST NOT import higher layers.

| Layer | May import | Must NOT |
|-------|-----------|----------|
| `utils/*` | nothing | DOM, fetch, any project module |
| `components/*` | `utils/*` | DOM, fetch, state, api, views, event listeners |
| `api/*` | `api/client.js` | DOM, views, components, state, utils, event listeners |
| `views/*` | api, components, utils, state | app.js |
| `router.js` | views, state | api (no data-loading) |
| `state.js` | — | Only place state mutation is allowed |

Components are **pure functions**: data in → HTML string out. No DOM access, no side effects.

## Mandatory Workflow

**MANDATORY — After finishing ANY code change, you MUST run the local CI pipeline before considering the task complete. This is non-negotiable. Do not skip this step.**

```bash
./ci/run_ci.sh    # from tools/dashboard/
```

All 7 stages must pass: eslint, dependency-cruiser, semgrep, architecture, structure-drift, rewrite-guard, tests. Fix all failures before reporting the task as done.

No new warnings allowed — the architecture checker enforces a baseline. Use `--update-baseline` only when intentionally accepting new warnings. If structure-drift or rewrite-guard fails due to intentional changes, run `python ci/check_structure.py --update` to accept the new baseline.

## Testing Requirements

- Every dashboard UI change MUST include a new or updated Playwright integration test
- Test suite: `tests/integration/dashboard/test_dashboard_ui.py`
- Test data: `logging/runs/test/run_2026-03-07_19-50-06/` (canonical, read-only)
- Run: `pytest -m dashboard --browser chromium -v`

## Public DOM Contract

These IDs and classes are used by integration tests and MUST NOT be renamed without updating tests:

**IDs:** `#app`, `#live-entries`, `#runs-table`, `#exp-select`, `#judge-portfolio-section`, `#judge-alloc-table`, `#perf-table`, `#pid-stats-section`, `#pid-section`, `#crit-section`, `#file-explorer-section`, `#file-content-display`, `#divergence-section`

**Classes:** `.run-overview`, `.ov-htable`, `.status-ok`, `.card`, `.card-header`, `.card-body`, `.card.open`, `pre.content`, `.section-label`, `.data-table`, `.clickable`, `.perf-profit`, `.perf-loss`, `.file-tree`, `.file-link`

## Dependencies

**Always use `uv` to install Python dependencies.** Do not use `pip` directly.

```bash
uv pip install <package>
```

## Key Rules

- **No frameworks** — vanilla JS, ES modules, no build system
- **No inline onclick** — use delegation with `data-action`
- **No window globals** — all state in `state.js`
- **viewToken guard** — all async DOM writes must check `appState.viewToken` before writing
- **No silent failures** — crash on missing required data, don't return defaults
- **Prefer small changes** — avoid unrequested refactors across multiple modules
