# Dashboard Testing Guide

This repository follows a **behavior-driven testing philosophy**.

The goal of tests is to verify **user-visible behavior**, not internal implementation details.

Tests must prioritize **integration tests over unit tests**.

---

# Test Hierarchy

Tests fall into three categories:

1. **Integration tests (Playwright + pytest)**
   Primary safety net for the dashboard.

2. **Logic tests (pure unit tests)**
   Used only for math or deterministic algorithms.

3. **Architecture checks**
   Enforced by `check_dashboard_architecture.py`.

Approximate ratios:

Integration tests: ~75%
Logic tests: ~20%
Architecture checks: ~5%

---

# Integration Tests (Primary)

Integration tests verify the entire system working together:

```
browser
UI
server
API
filesystem
state
```

They simulate real user behavior.

Integration tests must be written whenever a change affects:

* UI behavior
* navigation
* dashboard rendering
* API responses
* filesystem interaction
* data visualization

Integration tests use:

* **pytest**
* **Playwright browser automation**

Example:

```python
def test_run_detail_page(page, dashboard_server):
    page.goto(dashboard_server.url)

    page.click("text=Runs")
    page.wait_for_selector("#runs-table")

    page.click("#runs-table tr.clickable")

    page.wait_for_selector(".run-overview")

    assert page.locator("#pid-section").is_visible()
```

Tests should verify **visible outcomes**, not internal state.

---

# Logic Tests (Unit Tests)

Unit tests should exist only for:

* mathematical computations
* deterministic algorithms
* formatting functions
* PID controller logic
* portfolio calculations

Example:

```
tests/logic/test_pid_controller.py
tests/logic/test_portfolio_math.py
tests/logic/test_format_utils.py
```

Do not write unit tests for UI glue.

---

# Test Fixtures

Integration tests must run against **stable test fixtures**.

Fixtures live in:

```
tests/fixtures/
```

Example:

```
tests/fixtures/sample_runs/
   experiment_A/
      run_001/
         manifest.json
         rounds/
         pid.json
         crit.json
```

Tests must **not depend on live experimental runs**.

---

# Writing Tests for New Features

Whenever adding a new feature:

1. Identify user-visible behavior.
2. Add an integration test that exercises the feature.
3. Use real fixtures instead of mocks.
4. Verify DOM structure or UI state.

Examples:

Feature → Test

```
new dashboard section → integration test
new chart → integration test
new navigation flow → integration test
new API endpoint → integration test
```

---

# What Not To Do

Do not write:

* snapshot tests
* mock-heavy tests
* tests for private functions
* tests tied to internal implementation details

Tests should remain stable even if internal code is refactored.

---

# Running Tests

Integration tests:

```
pytest -m dashboard --browser chromium
```

Logic tests:

```
pytest tests/logic
```

Architecture validation:

```
python tools/prompt_viewer/check_dashboard_architecture.py
```
