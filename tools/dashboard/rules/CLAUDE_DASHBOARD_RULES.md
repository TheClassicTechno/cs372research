# Debate Dashboard Architecture

This document defines the **non-negotiable architectural rules of the Debate Dashboard**.

Any change that violates these rules **must be explicitly justified and documented in the PR description**.

The dashboard is a **developer debugging and inspection tool**, not a consumer-facing product.
Therefore the system prioritizes:

* strict architectural discipline
* fail-fast error handling
* predictable rendering
* reproducible integration testing

---

# Layer Diagram

```
app.js → router.js → views/* → components/* → utils/*
                               → api/*
```

Lower layers must never import higher layers.

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
* `utils/*` must not import anything from the project
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

Examples:

| Module     | Responsibility         |
| ---------- | ---------------------- |
| views      | orchestrate UI and DOM |
| components | generate HTML          |
| api        | network requests       |
| utils      | pure helper logic      |

Modules must **not mix responsibilities**.

---

# Public DOM Contract

These selectors MUST NOT be renamed, removed, or restructured.

Changes require **updating integration tests**.

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

## Behaviors

* Card expand/collapse via `.open`
* Navigation text: **Live Debate**, **Runs**
* `#live-entries .card` structure for tests
* `#runs-table .data-table tr.clickable`
* `pre.content` inside `.card-body`

---

# Component Purity Contract

`components/*` are the **SOLE source of HTML generation**.

Views must **never construct HTML strings** except for trivial container scaffolding.

All data-driven HTML must come from components.

---

# Event Handling Contract

Event handling must follow **root event delegation**.

Rules:

* Single delegation listener on `#app`
* Card toggle handled in `app.js`
* View-specific actions use `data-action`
* Actions dispatched to active view `handleAction()`
* Zero `window` globals
* Zero inline `onclick`
* Event listeners must **never be attached inside components or utils**

---

# View Lifecycle Contract

Each view may export:

```
teardown()
```

Router responsibilities:

* call teardown before switching views
* prevent leaking timers
* ensure only one active polling loop

Example:

* Live view teardown clears polling interval

---

# Stale Async Fetch Prevention

`state.js` exports:

```
viewToken
```

Each route change sets a new `Symbol()` token.

All async DOM writes must check:

```
if (appState.viewToken !== token) return
```

This prevents stale fetch responses from mutating the UI.

---

# Data Contract Enforcement

API responses are **strict contracts**.

Views must validate required fields before use.

Example:

```
if (!data.events) {
  throw new Error("API response missing required field: events")
}
```

Do not assume optional structures exist.

---

# Avoid Implicit Type Coercion

JavaScript implicit coercion is forbidden.

Bad:

```
if (value)
```

Preferred:

```
if (value === undefined)
if (value === null)
if (Array.isArray(items))
```

Explicit checks improve reliability.

---

# Test Selector Contract

The dashboard uses **data-testid attributes**.

Tests must select elements using:

```
page.get_by_test_id("...")
```

Tests must **not rely on**:

* CSS classes
* DOM hierarchy
* text selectors

Example:

Correct:

```
page.get_by_test_id("run-row")
page.get_by_test_id("section-crit")
```

Incorrect:

```
page.locator("#runs-table tr")
page.locator(".card-header")
page.locator("text=Runs")
```

`data-testid` attributes are part of the **public testing contract**.

---

# Code Quality Standards

## Function Size

Functions should remain small.

Rules:

* Preferred maximum: **≤ 30 lines**
* Larger functions must be split into helpers
* Long functions usually indicate mixed responsibilities

---

## Function Naming

Names must be descriptive.

Good:

```
renderRunOverview()
fetchPortfolioTrajectory()
buildAgentAllocationTable()
```

Bad:

```
handleData()
doThing()
processStuff()
```

---

## Function Documentation

Each function must include a documentation comment.

Example:

```javascript
/**
 * Renders portfolio trajectory table.
 * Accepts trajectory data and returns HTML markup.
 */
```

Documentation must describe:

* purpose
* inputs
* outputs

---

## File-Level Documentation

Each file must start with a module description.

Example:

```javascript
/**
 * runDetail/index.js
 *
 * Orchestrates Run Detail page.
 */
```

---

# State Management Philosophy

## Prefer Functional Programming

Prefer:

```
const newState = updatePortfolio(state, update)
```

Avoid:

```
state.portfolio.push(update)
```

---

## State Mutation Rules

State mutation allowed **only in `state.js`**.

Other modules must treat state as **read-only**.

---

## Avoid Hidden State

Forbidden:

```
let cache = {}
window.someGlobal = ...
```

All persistent state belongs in `state.js`.

---

# DOM Interaction Rules

## Views Own the DOM

Only views may interact with DOM APIs.

Allowed:

```
document.getElementById
querySelector
innerHTML
insertAdjacentHTML
```

Forbidden in:

```
components/*
utils/*
api/*
```

---

## Avoid Fragile DOM Selectors

Avoid:

```
document.querySelector("#runs-table tr td:nth-child(3)")
```

Prefer:

```
IDs
data-testid
```

---

# Async and Network Rules

Network requests allowed only in:

```
api/*
views/*
```

Never in:

```
components/*
utils/*
```

---

# Performance Guidelines

## Avoid Excessive DOM Writes

Bad:

```
for (...) {
 container.innerHTML += ...
}
```

Good:

```
const html = items.map(renderItem).join("")
container.innerHTML = html
```

---

## Avoid Repeated DOM Queries

Cache DOM elements.

Bad:

```
document.getElementById("runs-table")
```

inside loops.

Good:

```
const table = document.getElementById("runs-table")
```

---

## Avoid Layout Thrashing

Avoid reading layout metrics while writing DOM styles repeatedly.

Bad:

```
element.offsetHeight
element.style.height = ...
```

inside loops.

Batch DOM reads and writes.

---

# Console Logging Rules

Console logging may be used for debugging but must follow rules:

* Do not rely on logs instead of exceptions
* Avoid excessive debug logs
* Do not suppress errors with logs

---

# File Size Guidelines

Modules should remain under **~400 lines**.

Large files should be split.

---

# Integration Testing Contract

All UI behavior must be testable with **Playwright integration tests**.

Interactive elements must expose:

```
data-testid
```

Tests must use:

```
page.get_by_test_id(...)
```

---

# Code Change Philosophy

## Prefer Small Changes

Favor incremental edits.

Avoid unrelated refactors.

---

## Preserve Behavior

The dashboard is a **debugging tool**.

Changes must remain **behavior-preserving unless intentional**.

---

# Refactor Safety Rule

Large-scale refactors must **never occur implicitly**.

If a change modifies **more than three modules**, the PR must include:

1. explanation of the refactor
2. list of files changed
3. confirmation behavior is preserved
4. confirmation tests pass

---

# No Silent Failures

Silent failures are prohibited.

Bad:

```javascript
const result = obj?.field ?? 0
```

Bad:

```
if (!data) return
```

---

# Required Behavior

Crash immediately when invariants fail.

```
if (!data) {
 throw new Error("Expected data but received undefined")
}
```

---

# Exception Handling Rules

Exceptions must not be caught unless recovery is possible.

Allowed cases:

1. UI error boundary
2. network retry logic
3. API error translation

---

# Single Source of Truth

Core logic must exist **in one place only**.

Examples:

* formatting helpers
* rendering helpers
* API construction

---

# Avoid Copy-Paste Programming

Duplicate implementations must be refactored into shared helpers.

---

# Template Rendering Rules

Templates must be deterministic.

Templates must **not contain business logic**.

Templates must only interpolate provided variables.

Missing variables must **crash immediately**.

Example configuration:

```
Environment(undefined=StrictUndefined)
```

Templates must document expected variables.

---

# Do Not Introduce Frameworks

The dashboard intentionally uses:

* vanilla JavaScript
* ES modules
* no build systems

Do **not introduce React, Vue, or other frameworks**.

---

# Final Principles

The Debate Dashboard follows three core principles:

1. **Architectural discipline**
2. **Fail-fast reliability**
3. **Deterministic UI behavior**

These rules ensure the dashboard remains stable, debuggable, and safe to modify using AI-assisted workflows.
