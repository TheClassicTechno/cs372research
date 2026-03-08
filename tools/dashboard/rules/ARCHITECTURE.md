# Debate Dashboard Architecture

This document defines the **non-negotiable architectural rules of the Debate Dashboard**.

Any change that violates these rules **must be explicitly justified and documented in the PR description**.

The dashboard is a **developer debugging and inspection tool**, not a consumer-facing product.
Therefore the system prioritizes:

* strict architectural discipline
* fail-fast error handling
* predictable rendering
* reproducible integration testing
* deterministic logging and file-based data pipelines

The architecture is intentionally **simple, inspectable, and AI-safe** so that both humans and LLM assistants can modify the system without introducing instability.

---

# Core Architectural Philosophy

The dashboard follows five core engineering principles:

### 1. Fail Fast

The system should **crash immediately** when invariants are violated.

Silent failures, defensive defaults, and hidden fallbacks are forbidden.

---

### 2. Determinism

Given the same inputs, the system should produce the **same outputs**.

Sources of non-determinism (implicit globals, hidden state mutation, implicit template variables) are prohibited.

---

### 3. File-Based Observability

The dashboard operates on **structured artifacts written to disk**.

All run data should be inspectable through files rather than hidden inside processes or memory.

This ensures:

* reproducibility
* debuggability
* compatibility with offline analysis tools

---

### 4. Architectural Discipline

Strict module boundaries prevent architectural drift.

Lower layers must never import higher layers.

---

### 5. AI-Safe Development

The repository is designed to be modified by **AI coding assistants**.

Strict rules prevent accidental large-scale refactors or architectural violations.

---

# Layer Diagram

```
app.js → router.js → views/* → components/* → utils/*
                               → api/*
```

Lower layers must **never import higher layers**.

---

# Import Direction Rule

Imports must follow the architectural direction:

```
app → router → views → components → utils
                     → api
```

Lower layers must **never import higher layers**.

Examples:

* `components/*` must not import `views/*`
* `utils/*` must not import anything from this project
* `api/*` must not import `components` or `views`

This prevents circular dependencies and architectural drift.

---

# Dependency Rules

## `utils/*` (leaf layer)

* MUST be dependency-free (no imports from this project)
* MUST NOT import views, router, state, api, or components
* Contains only **pure helper functions**
* MUST NOT access the DOM
* MUST NOT perform network requests
* MUST NOT mutate external state

---

## `components/*` (pure render layer)

* MAY import `utils/*`
* MUST NOT import `api/*`
* MUST NOT import `state.js`
* MUST NOT call `fetch`
* MUST NOT access the DOM (`document.*`, `querySelector`, etc.)
* MUST NOT attach event listeners
* MUST NOT call `innerHTML`
* MUST NOT mutate state
* Receives all data as function arguments
* Returns **HTML strings only**

Components are **pure rendering functions**.

---

## `api/*` (network layer)

* MAY import `api/client.js`
* MUST NOT import views, components, state, or utils
* Wraps network calls only
* Responsible for constructing API requests and parsing responses

---

## `views/*` (orchestration layer)

* MAY import `api/*`, `components/*`, `utils/*`, `state.js`
* Responsible for orchestration and DOM updates
* MUST NOT construct data-driven HTML directly — all HTML via `components/*`
* MAY access the DOM
* MUST check `appState.viewToken` before DOM writes in async operations

---

## `router.js` (dispatch layer)

* Owns route dispatch and view lifecycle
* Responsible for `init` / `teardown` of views
* MUST NOT become a data-loading layer
* MUST NOT contain UI logic

---

## `state.js` (state layer)

* Exports mutable state objects
* Exports setter/reset functions
* No global variables allowed elsewhere
* State mutation must occur **only here**

---

# Module Responsibility Rule

Each module must have **a single clear responsibility**.

| Module     | Responsibility         |
| ---------- | ---------------------- |
| views      | orchestrate UI and DOM |
| components | generate HTML          |
| api        | network requests       |
| utils      | pure helper logic      |

Modules must **not mix responsibilities**.

---

# File-Based Data Architecture

The dashboard is designed to read **structured artifacts written to disk**.

The system must prefer **file-based data pipelines** over in-memory coupling.

### Advantages

* reproducible experiments
* easy debugging
* versionable artifacts
* compatibility with analysis tools

---

# Logging Directory Structure

Run artifacts must follow a **stable and predictable layout**.

Example:

```
logging/
  runs/
    <experiment_name>/
      <run_id>/
        manifest.json
        pid_config.json
        rounds/
          round_001/
            round_state.json
            crit_scores.json
            allocations.json
          round_002/
            ...
```

Rules:

* directories must contain **only structured data**
* file names must be deterministic
* files must use **JSON or text formats**
* artifacts must be **append-only when possible**

---

# File Naming Rules

Files must follow predictable naming conventions.

Example:

```
round_001.json
round_002.json
```

Avoid dynamic or ambiguous names such as:

```
latest.json
temp_output.json
```

---

# Public DOM Contract

These selectors MUST NOT be renamed, removed, or restructured.

Changes require **updating integration tests**.

---

## IDs

```
#nav
#app
#live-entries
#live-clear-btn
#live-status
#runs-table
#exp-select
#runs-search
#runs-status
#judge-portfolio-section
#judge-portfolio-layout
#judge-alloc-table
#judge-alloc-wrap
#perf-table
#perf-metrics
#detail-sections
#pid-stats-section
#pid-section
#crit-section
#portfolio-section
#file-explorer-section
#file-content-display
```

---

## Classes

```
.run-overview
.ov-htable
.ov-title
.ov-subtitle
.ov-warn
.status-ok
.status-incomplete
.status-partial
.card
.card-header
.card-body
.card.open
pre.content
.section-label
.perf-profit
.perf-loss
.data-table
.clickable
.best-run
.flag-collapse
.controls
.status-text
.back-link
.loading
.file-tree
.dir-toggle
.file-link
.chart-container
.arrow
```

---

## Attributes

```
data-eid
data-view
```

---

# Component Purity Contract

`components/*` are the **SOLE source of HTML generation**.

Views must **never construct HTML strings** except trivial scaffolding.

---

# Event Handling Contract

Event handling must follow **root event delegation**.

Rules:

* single delegation listener on `#app`
* view actions use `data-action`
* dispatch to `handleAction()`
* no inline `onclick`
* no global event listeners

---

# View Lifecycle Contract

Each view may export:

```
teardown()
```

Router responsibilities:

* call teardown before switching views
* ensure no leaked timers

---

# Stale Async Fetch Prevention

`state.js` exports:

```
viewToken
```

Async DOM writes must check:

```
if (appState.viewToken !== token) return
```

---

# Data Contract Enforcement

API responses are **strict contracts**.

Example:

```
if (!data.events) {
 throw new Error("API response missing required field: events")
}
```

---

# Avoid Implicit Type Coercion

Implicit coercion is forbidden.

Bad:

```
if (value)
```

Preferred:

```
if (value === undefined)
```

---

# Code Quality Standards

## Function Size

Functions should remain **≤ 30 lines**.

---

## Function Naming

Names must be descriptive.

Good:

```
renderRunOverview()
fetchPortfolioTrajectory()
buildAgentAllocationTable()
```

---

## Function Documentation

Every function must include a documentation comment.

---

## File Documentation

Every module must include a **top-level comment describing its purpose**.

---

# State Management Philosophy

Prefer **functional updates**.

Bad:

```
state.portfolio.push(update)
```

Good:

```
const newState = updatePortfolio(state, update)
```

---

# DOM Interaction Rules

Only views may interact with DOM APIs.

Forbidden in:

```
components/*
utils/*
api/*
```

---

# Performance Guidelines

Avoid excessive DOM writes.

Good:

```
const html = items.map(renderItem).join("")
container.innerHTML = html
```

---

# Integration Testing Contract

All UI features must be testable with **Playwright integration tests**.

Selectors must use:

```
data-testid
```

Tests must use:

```
page.get_by_test_id(...)
```

---

# Refactor Safety Rule

Large-scale refactors must **never occur implicitly**.

If more than **three modules** change:

1. explain why the refactor is needed
2. list modified files
3. confirm behavior preservation
4. confirm tests pass

---

# No Silent Failures

Silent failures are prohibited.

Bad:

```
if (!data) return
```

Correct:

```
if (!data) {
 throw new Error("Expected data but received undefined")
}
```

---

# Exception Handling Rules

Exceptions must not be caught unless recovery is possible.

Allowed cases:

* UI error boundary
* retry logic
* API error translation

---

# Single Source of Truth

Core logic must exist **in exactly one place**.

Examples:

* formatting helpers
* API request builders
* rendering helpers

---

# Avoid Copy-Paste Programming

Shared logic must be extracted into:

```
utils/*
components/*
```

---

# Template Rendering Rules

Templates must be deterministic.

Rules:

* no business logic
* no implicit variables
* missing variables must crash

Example configuration:

```
Environment(undefined=StrictUndefined)
```

---

# Do Not Introduce Frameworks

The dashboard intentionally uses:

* vanilla JavaScript
* ES modules
* no build systems

Frameworks such as React or Vue must **not be introduced**.

---

# Final Principles

The Debate Dashboard follows three core principles:

1. **Architectural discipline**
2. **Fail-fast reliability**
3. **Deterministic UI behavior**

These rules ensure the system remains **stable, debuggable, and safe for AI-assisted development**.

---

# Static Analysis Stack

Architecture constraints are enforced by four complementary tools.

Run all checks with:

```
npm run check-all
```

Or without semgrep:

```
npm run check-architecture
```

---

## ESLint

**Config:** `eslint.config.mjs`

Enforces general JavaScript quality and layer-specific restrictions:

- Safety: `no-eval`, `no-new-func`, `no-implicit-globals`
- Quality: `no-unused-vars`, `no-var`, `eqeqeq`
- Size limits: `max-lines` (400), `max-lines-per-function` (60)
- Layer rules via `no-restricted-globals`: utils cannot access DOM, components cannot fetch, api cannot access DOM

```
npm run lint
```

---

## dependency-cruiser

**Config:** `.dependency-cruiser.cjs`

Enforces the import direction contract at the module graph level:

- `utils/*` must not import any project modules
- `components/*` must not import `api/`, `state.js`, or `views/`
- `api/*` must not import `components/`, `views/`, or `state.js`
- `views/*` must not import `app.js`
- No circular dependencies allowed

```
npm run depcheck
```

---

## Semgrep

**Config:** `semgrep/dashboard_rules.yml`

Pattern-based static analysis for architectural invariants:

- No `fetch()` in components
- No DOM access in utils or api
- No `window` access in utils
- No inline `onclick` handlers
- No `eval()` or `new Function()`
- No `innerHTML` or `addEventListener` in components
- No `window` global assignments

```
npm run semgrep
```

Requires `semgrep` installed via pip (`pip install semgrep`).

---

## Python Architecture Checker

**Script:** `rules/check_dashboard_architecture.py`

Project-specific heuristic checks beyond what generic tools catch:

- Import direction validation
- File and function size limits
- Documentation comment requirements
- viewToken guard detection in async views
- Duplicate function name detection
- Framework introduction detection
- Silent failure pattern detection

```
python3 rules/check_dashboard_architecture.py
```
