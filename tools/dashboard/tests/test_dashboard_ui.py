"""Playwright integration tests for the Debate Dashboard.

Uses a real FastAPI server started by the ``dashboard_url`` session fixture
(see conftest.py) against a **copied** test dataset so source data is never
mutated.

Run:
    pytest -m dashboard --browser chromium -v
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page

pytestmark = pytest.mark.dashboard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _goto_run_detail(page: Page, url: str) -> None:
    """Navigate to the test run detail page."""
    page.goto(f"{url}/#run/test/run_2026-03-07_19-50-06")
    page.wait_for_selector(".run-overview", timeout=5000)


# ---------------------------------------------------------------------------
# TEST 1 — Runs page loads and shows the test run
# ---------------------------------------------------------------------------

class TestRunsPageLoads:
    def test_runs_table_shows_test_run(self, page: Page, dashboard_url: str):
        """Navigate to dashboard, click Runs tab, verify the test run row."""
        page.goto(dashboard_url)
        page.click("text=Runs")

        page.wait_for_selector("#runs-table .data-table tr.clickable", timeout=5000)

        table_text = page.text_content("#runs-table")
        assert "run_2026-03-07_19-50-06" in table_text, (
            "Test run ID not found in runs table"
        )


# ---------------------------------------------------------------------------
# TEST 2 — Run overview renders with correct fields and metrics
# ---------------------------------------------------------------------------

class TestRunOverviewRenders:
    def test_overview_fields_and_values(self, page: Page, dashboard_url: str):
        """Navigate directly to run detail and verify overview panel."""
        _goto_run_detail(page, dashboard_url)

        overview_text = page.text_content(".run-overview")

        for label in ("Run ID", "Experiment", "Model", "CRIT Model",
                       "Config", "Status"):
            assert label in overview_text, f"Missing label: {label}"

        assert "run_2026-03-07_19-50-06" in overview_text
        assert "gpt-5-mini" in overview_text
        assert "gpt-5" in overview_text
        assert "complete" in overview_text

    def test_overview_metrics_present(self, page: Page, dashboard_url: str):
        """Run overview shows Final beta, Final rho, and rounds count."""
        _goto_run_detail(page, dashboard_url)

        overview_text = page.text_content(".run-overview")

        assert "Final" in overview_text, "No 'Final' metric label found"
        assert "Rounds" in overview_text, "No 'Rounds' label found"
        assert "2 / 2" in overview_text, "Rounds count '2 / 2' not found"
        assert "0.32" in overview_text, "Final beta value not rendered"


# ---------------------------------------------------------------------------
# TEST 3 — Agent names come from config (exact strings, no .yaml suffix)
# ---------------------------------------------------------------------------

class TestAgentNamesFromConfig:
    def test_overview_shows_enriched_agent_names(
        self, page: Page, dashboard_url: str,
    ):
        """Agents cell displays exact config-derived names without .yaml."""
        _goto_run_detail(page, dashboard_url)

        overview_text = page.text_content(".run-overview")

        for name in ("value_enriched", "risk_enriched",
                      "technical_enriched"):
            assert name in overview_text, (
                f"Expected agent name '{name}' not found in overview"
            )

        # Must NOT have .yaml suffix
        assert ".yaml" not in overview_text, (
            "Agent names should not include .yaml suffix"
        )


# ---------------------------------------------------------------------------
# TEST 4 — Judge portfolio layout (vertical alloc table + metrics table)
# ---------------------------------------------------------------------------

class TestJudgePortfolioLayout:
    def test_portfolio_section_has_two_tables(
        self, page: Page, dashboard_url: str,
    ):
        """Judge portfolio section contains a vertical ticker table and a
        performance metrics table side by side."""
        _goto_run_detail(page, dashboard_url)

        section = page.wait_for_selector("#judge-portfolio-section", timeout=5000)
        assert section is not None

        # Wait for the allocation table to render
        page.wait_for_selector("#judge-alloc-table", timeout=5000)

        # Multi-agent allocation table: agent columns + JUDGE column
        alloc_table = page.query_selector("#judge-alloc-table")
        assert alloc_table is not None, "Allocation table not found"
        alloc_text = alloc_table.text_content()
        assert "JUDGE" in alloc_text, "Allocation table missing 'JUDGE' column"

        # Should have header row + at least a few ticker rows
        rows = alloc_table.query_selector_all("tr")
        assert len(rows) >= 3, (
            f"Allocation table should have header + ticker rows, got {len(rows)}"
        )

        # Header row should have agent columns
        header_cells = rows[0].query_selector_all("th")
        assert len(header_cells) >= 3, (
            f"Expected at least 3 columns (agent(s) + JUDGE), got {len(header_cells)}"
        )

        # Layout container uses flexbox for side-by-side
        layout = page.query_selector("#judge-portfolio-layout")
        assert layout is not None, "Side-by-side layout container not found"

    def test_portfolio_shows_tickers(self, page: Page, dashboard_url: str):
        """Allocation table lists the actual portfolio tickers."""
        _goto_run_detail(page, dashboard_url)
        page.wait_for_selector("#judge-alloc-table", timeout=5000)

        alloc_text = page.text_content("#judge-alloc-table")
        # Test run has these tickers in final portfolio
        for ticker in ("PG", "CVX", "RTX", "CAT"):
            assert ticker in alloc_text, (
                f"Ticker '{ticker}' missing from allocation table"
            )

    def test_alloc_table_has_agent_columns(
        self, page: Page, dashboard_url: str,
    ):
        """Allocation table header includes per-agent columns from config."""
        _goto_run_detail(page, dashboard_url)
        page.wait_for_selector("#judge-alloc-table", timeout=5000)

        alloc_text = page.text_content("#judge-alloc-table")
        # Agent names are uppercased in headers
        for name in ("VALUE_ENRICHED", "RISK_ENRICHED", "TECHNICAL_ENRICHED"):
            assert name in alloc_text, (
                f"Agent column '{name}' not found in allocation table"
            )

    def test_perf_metrics_table_structure(
        self, page: Page, dashboard_url: str,
    ):
        """Performance metrics table has the correct rows (no SPY)."""
        _goto_run_detail(page, dashboard_url)
        # Wait for the async performance fetch to complete
        page.wait_for_selector("#perf-table", timeout=10000)

        perf_text = page.text_content("#perf-table")
        for label in ("Initial Capital", "Final Value", "Profit/Loss", "Return"):
            assert label in perf_text, (
                f"Metrics table missing '{label}'"
            )

        # SPY Return should NOT be present
        assert "SPY" not in perf_text, (
            "SPY Return should not appear in performance metrics"
        )


# ---------------------------------------------------------------------------
# TEST 8 — Return calculation correctness
# ---------------------------------------------------------------------------

class TestReturnCalculation:
    def test_return_equals_profit_over_capital(
        self, page: Page, dashboard_url: str,
    ):
        """Return % must equal (final_value - initial_capital) / initial_capital.

        The backend sends return_pct already as a percentage.  The frontend
        must display it directly without multiplying by 100 again.
        """
        _goto_run_detail(page, dashboard_url)
        page.wait_for_selector("#perf-table", timeout=10000)

        rows = page.query_selector_all("#perf-table tr")
        values = {}
        for row in rows:
            cells = row.query_selector_all("td")
            if len(cells) == 2:
                label = cells[0].text_content().strip()
                val = cells[1].text_content().strip()
                values[label] = val

        # Parse dollar amounts
        def parse_dollar(s):
            return float(s.replace("$", "").replace(",", "").replace("+", ""))

        initial = parse_dollar(values["Initial Capital"])
        final_val = parse_dollar(values["Final Value"])
        profit = parse_dollar(values["Profit/Loss"])

        # Parse return percentage
        ret_str = values["Return"].replace("%", "").replace("+", "")
        ret_pct = float(ret_str)

        # Verify: return = (final - initial) / initial * 100
        expected_return = (final_val - initial) / initial * 100
        assert abs(ret_pct - expected_return) < 0.1, (
            f"Return {ret_pct}% != expected "
            f"({final_val} - {initial}) / {initial} * 100 = {expected_return:.2f}%"
        )

        # Also verify profit = final - initial
        assert abs(profit - (final_val - initial)) < 0.1, (
            f"Profit ${profit} != Final ${final_val} - Initial ${initial}"
        )


# ---------------------------------------------------------------------------
# TEST 9 — Color logic (profit/loss and status)
# ---------------------------------------------------------------------------

class TestColorLogic:
    def test_status_complete_is_green(self, page: Page, dashboard_url: str):
        """Status 'complete' renders with the green .status-ok class."""
        _goto_run_detail(page, dashboard_url)

        # Find the Status cell — it's in the first ov-htable
        status_cell = page.query_selector(".ov-htable .status-ok")
        assert status_cell is not None, (
            "No element with .status-ok class found in overview"
        )
        assert status_cell.text_content().strip() == "complete", (
            f"Status cell text is '{status_cell.text_content().strip()}', "
            "expected 'complete'"
        )

    def test_profit_values_colored_green(self, page: Page, dashboard_url: str):
        """When profit > 0, Final Value and Profit/Loss use .perf-profit."""
        _goto_run_detail(page, dashboard_url)
        page.wait_for_selector("#perf-table", timeout=10000)

        # The test run has a small positive profit, so perf-profit should apply
        profit_cells = page.query_selector_all("#perf-table .perf-profit")
        assert len(profit_cells) >= 2, (
            f"Expected at least 2 cells with .perf-profit (Final Value, "
            f"Profit/Loss), found {len(profit_cells)}"
        )

        # No perf-loss cells should exist for a profitable run
        loss_cells = page.query_selector_all("#perf-table .perf-loss")
        assert len(loss_cells) == 0, (
            f"Found {len(loss_cells)} .perf-loss cells in a profitable run"
        )

    def test_final_value_has_color_class(self, page: Page, dashboard_url: str):
        """Final Value cell has a perf-profit or perf-loss class."""
        _goto_run_detail(page, dashboard_url)
        page.wait_for_selector("#perf-table", timeout=10000)

        rows = page.query_selector_all("#perf-table tr")
        for row in rows:
            cells = row.query_selector_all("td")
            if len(cells) == 2 and cells[0].text_content().strip() == "Final Value":
                cls = cells[1].get_attribute("class") or ""
                assert "perf-profit" in cls or "perf-loss" in cls, (
                    f"Final Value cell has no profit/loss class: '{cls}'"
                )
                return
        pytest.fail("Final Value row not found in perf table")


# ---------------------------------------------------------------------------
# TEST 10 — Divergence table axes: phases as rows, metrics as columns
# ---------------------------------------------------------------------------

class TestDivergenceTableLayout:
    def test_divergence_table_has_phase_rows(
        self, page: Page, dashboard_url: str,
    ):
        """Divergence table has Phase/JS Divergence/Evidence Overlap columns
        with Proposed and Revised as rows."""
        _goto_run_detail(page, dashboard_url)

        section = page.wait_for_selector("#divergence-section", timeout=5000)
        assert section is not None

        table = page.query_selector("#divergence-section .data-table")
        assert table is not None, "Divergence table not found"

        # Header row should have Phase, JS Divergence, Evidence Overlap
        headers = table.query_selector_all("tr:first-child th")
        header_texts = [h.text_content().strip() for h in headers]
        assert header_texts == ["Phase", "JS Divergence", "Evidence Overlap"], (
            f"Expected ['Phase', 'JS Divergence', 'Evidence Overlap'], "
            f"got {header_texts}"
        )

        # Data rows should be Proposed and Revised
        rows = table.query_selector_all("tr:not(:first-child)")
        row_labels = [
            r.query_selector("td").text_content().strip() for r in rows
        ]
        assert "Proposed" in row_labels, "Missing 'Proposed' row"
        assert "Revised" in row_labels, "Missing 'Revised' row"


# ---------------------------------------------------------------------------
# TEST 11 — Ablation metrics use side-by-side row layout
# ---------------------------------------------------------------------------

class TestAblationSideBySideLayout:
    def test_quality_metrics_row_has_grid_display(
        self, page: Page, dashboard_url: str,
    ):
        """Quality metrics row (rho, pillars, JS, evidence) uses grid layout."""
        page.goto(f"{dashboard_url}/#ablation")
        page.wait_for_selector(
            "[data-testid='ablation-experiment']", timeout=5000,
        )
        header = page.query_selector(
            "[data-testid='ablation-experiment'] .card-header",
        )
        header.click()

        row = page.wait_for_selector(
            "[data-testid='metrics-row-quality']", timeout=3000,
        )
        display = page.evaluate(
            "(el) => window.getComputedStyle(el).display", row,
        )
        assert display == "grid", (
            f"Expected metrics-row display:grid, got '{display}'"
        )

    def test_breakdowns_row_side_by_side(
        self, page: Page, dashboard_url: str,
    ):
        """Per Scenario and Per Agent Config tables are in same grid row."""
        page.goto(f"{dashboard_url}/#ablation")
        page.wait_for_selector(
            "[data-testid='ablation-experiment']", timeout=5000,
        )
        header = page.query_selector(
            "[data-testid='ablation-experiment'] .card-header",
        )
        header.click()

        row = page.wait_for_selector(
            "[data-testid='metrics-row-breakdowns']", timeout=3000,
        )
        display = page.evaluate(
            "(el) => window.getComputedStyle(el).display", row,
        )
        assert display == "grid", (
            f"Expected breakdowns row display:grid, got '{display}'"
        )
        text = row.text_content()
        assert "Per Scenario" in text, "Missing 'Per Scenario' in breakdowns row"
        assert "Per Agent Config" in text, (
            "Missing 'Per Agent Config' in breakdowns row"
        )

    def test_metrics_col_has_card_styling(
        self, page: Page, dashboard_url: str,
    ):
        """Quality metrics columns have card-like border and padding."""
        page.goto(f"{dashboard_url}/#ablation")
        page.wait_for_selector(
            "[data-testid='ablation-experiment']", timeout=5000,
        )
        header = page.query_selector(
            "[data-testid='ablation-experiment'] .card-header",
        )
        header.click()

        col = page.wait_for_selector(
            "[data-testid='metrics-row-quality'] .metrics-col", timeout=3000,
        )
        border = page.evaluate(
            "(el) => window.getComputedStyle(el).borderStyle", col,
        )
        assert border == "solid", (
            f"Expected metrics-col border-style:solid, got '{border}'"
        )

    def test_data_tables_full_width(
        self, page: Page, dashboard_url: str,
    ):
        """Data tables inside metrics columns fill their container width."""
        page.goto(f"{dashboard_url}/#ablation")
        page.wait_for_selector(
            "[data-testid='ablation-experiment']", timeout=5000,
        )
        header = page.query_selector(
            "[data-testid='ablation-experiment'] .card-header",
        )
        header.click()

        page.wait_for_selector(
            "[data-testid='metrics-row-quality'] .data-table", timeout=3000,
        )
        tables = page.query_selector_all(
            "[data-testid='metrics-row-quality'] .data-table",
        )
        assert len(tables) > 0, "No data tables found in quality metrics row"
        for table in tables:
            parent_width = page.evaluate(
                "(el) => el.parentElement.clientWidth", table,
            )
            table_width = page.evaluate(
                "(el) => el.offsetWidth", table,
            )
            # Table should fill at least 90% of parent width
            assert table_width >= parent_width * 0.9, (
                f"Table width {table_width} too narrow for parent {parent_width}"
            )


# ---------------------------------------------------------------------------
# TEST 12 — Duration column in runs table
# ---------------------------------------------------------------------------

class TestRunsDurationColumn:
    def test_runs_table_has_duration_header(
        self, page: Page, dashboard_url: str,
    ):
        """Runs table includes a 'duration' column header."""
        page.goto(dashboard_url)
        page.click("text=Runs")
        page.wait_for_selector(
            "#runs-table .data-table tr.clickable", timeout=5000,
        )
        header_text = page.text_content(
            "#runs-table .data-table tr:first-child",
        )
        assert "duration" in header_text, (
            "Duration column header not found in runs table"
        )

    def test_runs_table_duration_cell_has_value(
        self, page: Page, dashboard_url: str,
    ):
        """Duration cell shows a formatted time value (e.g. '8m 9s')."""
        page.goto(f"{dashboard_url}/#runs")
        row = page.wait_for_selector(
            "#runs-table .data-table tr.clickable", timeout=5000,
        )
        cells = row.query_selector_all("td")
        # Duration is the 11th column (0-indexed: 10)
        duration_cell = cells[10]
        text = duration_cell.text_content().strip()
        assert "s" in text or "m" in text, (
            f"Duration cell should show formatted time, got '{text}'"
        )


# ---------------------------------------------------------------------------
# TEST 13 — Per-round allocation and performance tables
# ---------------------------------------------------------------------------

class TestPerRoundAllocations:
    def test_round_1_proposals_table_renders(
        self, page: Page, dashboard_url: str,
    ):
        """Round 1 PROPOSALS allocation table renders with agent columns."""
        _goto_run_detail(page, dashboard_url)
        page.wait_for_selector(
            "[data-testid='round-1-proposals-alloc']", timeout=10000,
        )
        table = page.query_selector("[data-testid='round-1-proposals-alloc']")
        assert table is not None, "Round 1 proposals alloc table not found"
        headers = table.query_selector_all("tr:first-child th")
        assert len(headers) >= 3, (
            f"Expected at least 3 columns in round 1 proposals, got {len(headers)}"
        )

    def test_round_1_revisions_table_renders(
        self, page: Page, dashboard_url: str,
    ):
        """Round 1 REVISIONS allocation table renders with agent columns."""
        _goto_run_detail(page, dashboard_url)
        page.wait_for_selector(
            "[data-testid='round-1-revisions-alloc']", timeout=10000,
        )
        table = page.query_selector("[data-testid='round-1-revisions-alloc']")
        assert table is not None, "Round 1 revisions alloc table not found"
        rows = table.query_selector_all("tr")
        assert len(rows) >= 3, (
            f"Expected header + ticker rows, got {len(rows)}"
        )

    def test_round_2_proposals_table_renders(
        self, page: Page, dashboard_url: str,
    ):
        """Round 2 PROPOSALS allocation table renders."""
        _goto_run_detail(page, dashboard_url)
        page.wait_for_selector(
            "[data-testid='round-2-proposals-alloc']", timeout=10000,
        )
        table = page.query_selector("[data-testid='round-2-proposals-alloc']")
        assert table is not None, "Round 2 proposals alloc table not found"

    def test_round_section_titles_present(
        self, page: Page, dashboard_url: str,
    ):
        """Section titles for round proposals/revisions are present."""
        _goto_run_detail(page, dashboard_url)
        page.wait_for_selector(
            "[data-testid='round-1-proposals-title']", timeout=10000,
        )
        r1p = page.text_content("[data-testid='round-1-proposals-title']")
        assert "ROUND 1" in r1p and "PROPOSALS" in r1p, (
            f"Expected 'ROUND 1 — PROPOSALS' title, got '{r1p}'"
        )
        r1r = page.text_content("[data-testid='round-1-revisions-title']")
        assert "ROUND 1" in r1r and "REVISIONS" in r1r, (
            f"Expected 'ROUND 1 — REVISIONS' title, got '{r1r}'"
        )

    def test_round_perf_tables_appear(
        self, page: Page, dashboard_url: str,
    ):
        """Performance tables appear alongside round allocation tables."""
        _goto_run_detail(page, dashboard_url)
        page.wait_for_selector(
            "[data-testid='round-1-proposals-alloc']", timeout=10000,
        )
        # The per-round-sections div should contain perf tables
        wrap = page.query_selector("#per-round-sections")
        assert wrap is not None
        text = wrap.text_content()
        assert "Initial Capital" in text, (
            "Performance tables should show 'Initial Capital'"
        )
        assert "Return" in text, (
            "Performance tables should show 'Return'"
        )


# ---------------------------------------------------------------------------
# TEST 14 — Debate impact section
# ---------------------------------------------------------------------------

class TestDebateImpact:
    def test_debate_impact_agents_table_renders(
        self, page: Page, dashboard_url: str,
    ):
        """Debate impact table shows per-agent R1 proposal vs final revision."""
        _goto_run_detail(page, dashboard_url)
        page.wait_for_selector(
            "[data-testid='debate-impact-agents']", timeout=10000,
        )
        table = page.query_selector("[data-testid='debate-impact-agents']")
        assert table is not None, "Debate impact agents table not found"
        text = table.text_content()
        assert "R1 Proposal" in text, "Missing 'R1 Proposal' column header"
        assert "R1 Revision" in text, "Missing 'R1 Revision' column header"

    def test_debate_impact_has_agent_rows(
        self, page: Page, dashboard_url: str,
    ):
        """Debate impact table has rows for each agent."""
        _goto_run_detail(page, dashboard_url)
        page.wait_for_selector(
            "[data-testid='debate-impact-agents']", timeout=10000,
        )
        table = page.query_selector("[data-testid='debate-impact-agents']")
        rows = table.query_selector_all("tr")
        # Header + at least 3 agent rows (risk, technical, value)
        assert len(rows) >= 4, (
            f"Expected header + 3 agent rows, got {len(rows)}"
        )

    def test_debate_impact_mean_table_renders(
        self, page: Page, dashboard_url: str,
    ):
        """Mean portfolio comparison table shows R1 proposals vs R1 revisions."""
        _goto_run_detail(page, dashboard_url)
        page.wait_for_selector(
            "[data-testid='debate-impact-mean']", timeout=10000,
        )
        table = page.query_selector("[data-testid='debate-impact-mean']")
        assert table is not None, "Mean portfolio table not found"
        text = table.text_content()
        assert "R1 Mean Portfolio" in text, "Missing 'R1 Mean Portfolio' header"
        assert "Proposals" in text, "Missing 'Proposals' label"
        assert "Revisions" in text, "Missing 'Revisions' label"
        assert "Critique Impact" in text, "Missing 'Critique Impact' row"

    def test_debate_impact_r2_mean_table_renders(
        self, page: Page, dashboard_url: str,
    ):
        """R2 mean portfolio comparison table renders when round 2 data exists."""
        _goto_run_detail(page, dashboard_url)
        page.wait_for_selector(
            "[data-testid='debate-impact-mean-r2']", timeout=10000,
        )
        table = page.query_selector("[data-testid='debate-impact-mean-r2']")
        assert table is not None, "R2 mean portfolio table not found"
        text = table.text_content()
        assert "R2 Mean Portfolio" in text, "Missing 'R2 Mean Portfolio' header"
        assert "Critique Impact" in text, "Missing 'Critique Impact' row"

    def test_debate_impact_section_title(
        self, page: Page, dashboard_url: str,
    ):
        """DEBATE IMPACT title appears above the section."""
        _goto_run_detail(page, dashboard_url)
        page.wait_for_selector(
            "[data-testid='debate-impact-agents']", timeout=10000,
        )
        section = page.query_selector("#judge-portfolio-section")
        text = section.text_content()
        assert "DEBATE IMPACT" in text, "Missing 'DEBATE IMPACT' title"


# ---------------------------------------------------------------------------
# TEST 15 — Ablation debate impact (per-agent-config breakdown)
# ---------------------------------------------------------------------------

class TestAblationDebateImpact:
    def test_debate_impact_section_renders(
        self, page: Page, dashboard_url: str,
    ):
        """Debate impact section renders inside the experiment card."""
        page.goto(f"{dashboard_url}/#ablation")
        page.wait_for_selector(
            "[data-testid='ablation-experiment']", timeout=5000,
        )
        header = page.query_selector(
            "[data-testid='ablation-experiment'] .card-header",
        )
        header.click()

        section = page.wait_for_selector(
            "[data-testid='debate-impact-section']", timeout=5000,
        )
        assert section is not None, "Debate impact section not found"

    def test_debate_impact_shows_config_with_run_count(
        self, page: Page, dashboard_url: str,
    ):
        """Per-config subsection shows config name and run count."""
        page.goto(f"{dashboard_url}/#ablation")
        page.wait_for_selector(
            "[data-testid='ablation-experiment']", timeout=5000,
        )
        header = page.query_selector(
            "[data-testid='ablation-experiment'] .card-header",
        )
        header.click()

        section = page.wait_for_selector(
            "[data-testid='debate-impact-section']", timeout=5000,
        )
        text = section.text_content()
        assert "runs)" in text, "Missing run count in config label"

    def test_debate_impact_has_agent_deltas(
        self, page: Page, dashboard_url: str,
    ):
        """Debate impact section contains per-agent deltas table."""
        page.goto(f"{dashboard_url}/#ablation")
        page.wait_for_selector(
            "[data-testid='ablation-experiment']", timeout=5000,
        )
        header = page.query_selector(
            "[data-testid='ablation-experiment'] .card-header",
        )
        header.click()

        section = page.wait_for_selector(
            "[data-testid='debate-impact-section']", timeout=5000,
        )
        text = section.text_content()
        assert "Debate Impact" in text, "Missing 'Debate Impact' label"
        assert "R1 Proposal" in text, "Missing 'R1 Proposal' column"

    def test_debate_impact_has_mean_portfolio(
        self, page: Page, dashboard_url: str,
    ):
        """Debate impact section contains mean portfolio critique summary."""
        page.goto(f"{dashboard_url}/#ablation")
        page.wait_for_selector(
            "[data-testid='ablation-experiment']", timeout=5000,
        )
        header = page.query_selector(
            "[data-testid='ablation-experiment'] .card-header",
        )
        header.click()

        section = page.wait_for_selector(
            "[data-testid='debate-impact-section']", timeout=5000,
        )
        text = section.text_content()
        assert "Mean Portfolio" in text, "Missing 'Mean Portfolio' label"
        assert "Critique" in text, "Missing 'Critique' delta row"


# ---------------------------------------------------------------------------
# TEST 16 — Run overview config parameter grids
# ---------------------------------------------------------------------------

class TestRunOverviewConfigPanel:
    def test_debate_config_grid_renders(self, page: Page, dashboard_url: str):
        """Debate config grid appears with config parameters."""
        _goto_run_detail(page, dashboard_url)
        grid = page.query_selector("[data-testid='debate-config-grid']")
        assert grid is not None, "Debate config grid not found"
        text = grid.text_content()
        assert "pid_kp" in text, "Missing pid_kp in debate config grid"

    def test_scenario_config_grid_renders(self, page: Page, dashboard_url: str):
        """Scenario config grid appears with config parameters."""
        _goto_run_detail(page, dashboard_url)
        grid = page.query_selector("[data-testid='scenario-config-grid']")
        assert grid is not None, "Scenario config grid not found"
        text = grid.text_content()
        assert "invest_quarter" in text, "Missing invest_quarter in scenario config grid"

    def test_run_id_shown(self, page: Page, dashboard_url: str):
        """Run ID is displayed in the overview."""
        _goto_run_detail(page, dashboard_url)
        overview_text = page.text_content(".run-overview")
        assert "run_2026-03-07_19-50-06" in overview_text, (
            "Run ID should be visible in overview"
        )

    def test_config_grid_has_monospace_values(self, page: Page, dashboard_url: str):
        """Config values use monospace font."""
        _goto_run_detail(page, dashboard_url)
        cell = page.query_selector("[data-testid='debate-config-grid'] .ov-config-val")
        assert cell is not None, "No config value cell found"
        font = cell.evaluate("el => getComputedStyle(el).fontFamily")
        assert "monospace" in font.lower() or "courier" in font.lower(), (
            f"Config values should use monospace font, got: {font}"
        )

    def test_temperature_displayed(self, page: Page, dashboard_url: str):
        """Temperature is visible in the debate config card."""
        _goto_run_detail(page, dashboard_url)
        grid = page.query_selector("[data-testid='debate-config-grid']")
        assert grid is not None, "Debate config grid not found"
        text = grid.text_content()
        assert "temperature" in text.lower(), "Temperature not found in debate config"


# ---------------------------------------------------------------------------
# TEST 17 — Extended debate impact (PV columns, Judge, Sharpe)
# ---------------------------------------------------------------------------

class TestExtendedDebateImpact:
    def test_debate_impact_has_pv_columns(self, page: Page, dashboard_url: str):
        """Extended table has PV and delta sub-columns."""
        _goto_run_detail(page, dashboard_url)
        page.wait_for_selector("[data-testid='debate-impact-agents']", timeout=10000)
        table = page.query_selector("[data-testid='debate-impact-agents']")
        text = table.text_content()
        assert "PV" in text, "Missing PV column header"

    def test_debate_impact_has_judge_column(self, page: Page, dashboard_url: str):
        """Extended table has Judge column."""
        _goto_run_detail(page, dashboard_url)
        page.wait_for_selector("[data-testid='debate-impact-agents']", timeout=10000)
        table = page.query_selector("[data-testid='debate-impact-agents']")
        text = table.text_content()
        assert "Judge" in text, "Missing Judge column header"

    def test_sharpe_table_renders(self, page: Page, dashboard_url: str):
        """Sharpe ratios table renders with phase rows."""
        _goto_run_detail(page, dashboard_url)
        page.wait_for_selector("[data-testid='debate-impact-sharpe']", timeout=10000)
        table = page.query_selector("[data-testid='debate-impact-sharpe']")
        assert table is not None, "Sharpe table not found"
        text = table.text_content()
        assert "Sharpe" in text, "Missing Sharpe header"
        assert "R1 Proposal" in text, "Missing R1 Proposal row"
        assert "R1 Revision" in text, "Missing R1 Revision row"

    def test_sharpe_table_has_numeric_values(self, page: Page, dashboard_url: str):
        """Sharpe table has numeric values (not all dashes)."""
        _goto_run_detail(page, dashboard_url)
        page.wait_for_selector("[data-testid='debate-impact-sharpe']", timeout=10000)
        table = page.query_selector("[data-testid='debate-impact-sharpe']")
        cells = table.query_selector_all("td")
        numeric_count = 0
        for cell in cells:
            txt = cell.text_content().strip()
            try:
                float(txt)
                numeric_count += 1
            except ValueError:
                pass
        assert numeric_count >= 2, (
            f"Expected at least 2 numeric Sharpe values, got {numeric_count}"
        )


# ---------------------------------------------------------------------------
# TEST 18 — Debate summary panel
# ---------------------------------------------------------------------------

class TestDebateSummaryPanel:
    def test_summary_panel_renders(self, page: Page, dashboard_url: str):
        """Debate summary panel appears with key metrics."""
        _goto_run_detail(page, dashboard_url)
        page.wait_for_selector("[data-testid='debate-summary-panel']", timeout=10000)
        panel = page.query_selector("[data-testid='debate-summary-panel']")
        assert panel is not None
        text = panel.text_content()
        assert "Debate Alpha" in text

    def test_summary_debate_alpha_is_numeric(self, page: Page, dashboard_url: str):
        """Debate Alpha shows a numeric percentage value."""
        import re
        _goto_run_detail(page, dashboard_url)
        page.wait_for_selector("[data-testid='debate-summary-panel']", timeout=10000)
        panel = page.query_selector("[data-testid='debate-summary-panel']")
        text = panel.text_content()
        assert re.search(r'[+-]?\d+\.\d+%', text), "No numeric % found in summary"

    def test_summary_has_judge_return(self, page: Page, dashboard_url: str):
        """Summary shows Judge Return metric."""
        _goto_run_detail(page, dashboard_url)
        page.wait_for_selector("[data-testid='debate-summary-panel']", timeout=10000)
        panel = page.query_selector("[data-testid='debate-summary-panel']")
        text = panel.text_content()
        assert "Judge Return" in text


# ---------------------------------------------------------------------------
# TEST 19 — Collapse diagnostics
# ---------------------------------------------------------------------------

class TestCollapseDiagnostics:
    def test_collapse_section_renders(self, page: Page, dashboard_url: str):
        """Collapse diagnostics section renders with content."""
        _goto_run_detail(page, dashboard_url)
        page.wait_for_selector("#collapse-section", timeout=10000)
        section = page.query_selector("#collapse-section")
        text = section.text_content()
        assert len(text.strip()) > 0, "Collapse section is empty"

    def test_collapse_has_movement_column(self, page: Page, dashboard_url: str):
        """Collapse table has Movement column."""
        _goto_run_detail(page, dashboard_url)
        page.wait_for_selector("#collapse-section", timeout=10000)
        text = page.text_content("#collapse-section")
        assert "Movement" in text

    def test_collapse_identifies_leader(self, page: Page, dashboard_url: str):
        """Collapse diagnostics show a collapse leader."""
        _goto_run_detail(page, dashboard_url)
        page.wait_for_selector("#collapse-section", timeout=10000)
        text = page.text_content("#collapse-section")
        assert "Leader" in text or "leader" in text

    def test_collapse_definitions_grid_layout(self, page: Page, dashboard_url: str):
        """Collapse definitions use a grid layout with term/description pairs."""
        _goto_run_detail(page, dashboard_url)
        page.wait_for_selector("#collapse-section .collapse-definitions", timeout=10000)
        defs = page.query_selector("#collapse-section .collapse-definitions")
        display = page.evaluate(
            "(el) => window.getComputedStyle(el).display", defs,
        )
        assert display == "grid", (
            f"Expected collapse-definitions display:grid, got '{display}'"
        )
        terms = page.query_selector_all("#collapse-section .collapse-def-term")
        assert len(terms) == 4, f"Expected 4 definition terms, got {len(terms)}"
        term_texts = [t.text_content().strip() for t in terms]
        assert "Movement" in term_texts
        assert "Dissent" in term_texts

    def test_collapse_definitions_typography(self, page: Page, dashboard_url: str):
        """Collapse definition terms use readable font size (>= 15px)."""
        _goto_run_detail(page, dashboard_url)
        page.wait_for_selector(
            "#collapse-section .collapse-def-term", timeout=10000,
        )
        term = page.query_selector("#collapse-section .collapse-def-term")
        font_size = page.evaluate(
            "(el) => window.getComputedStyle(el).fontSize", term,
        )
        size_px = float(font_size.replace("px", ""))
        assert size_px >= 15, (
            f"Expected term font-size >= 15px, got {font_size}"
        )

    def test_collapse_definitions_row_spacing(self, page: Page, dashboard_url: str):
        """Collapse definition rows have visible padding for readability."""
        _goto_run_detail(page, dashboard_url)
        page.wait_for_selector(
            "#collapse-section .collapse-def-desc", timeout=10000,
        )
        desc = page.query_selector("#collapse-section .collapse-def-desc")
        padding = page.evaluate(
            "(el) => window.getComputedStyle(el).paddingTop", desc,
        )
        pad_px = float(padding.replace("px", ""))
        assert pad_px >= 8, (
            f"Expected description padding-top >= 8px, got {padding}"
        )


# ---------------------------------------------------------------------------
# TEST 20 — Ticker performance table
# ---------------------------------------------------------------------------

class TestTickerPerformance:
    def test_ticker_perf_table_renders(self, page: Page, dashboard_url: str):
        """Ticker performance table renders in overview."""
        _goto_run_detail(page, dashboard_url)
        table = page.query_selector("[data-testid='ticker-perf-table']")
        assert table is not None
        text = table.text_content()
        assert "AMD" in text, "Missing AMD in ticker perf table"

    def test_ticker_perf_has_delta_column(self, page: Page, dashboard_url: str):
        """Ticker performance shows percentage values."""
        _goto_run_detail(page, dashboard_url)
        table = page.query_selector("[data-testid='ticker-perf-table']")
        text = table.text_content()
        assert "%" in text

    def test_ticker_perf_color_coding(self, page: Page, dashboard_url: str):
        """Positive ticker returns use perf-profit class."""
        _goto_run_detail(page, dashboard_url)
        table = page.query_selector("[data-testid='ticker-perf-table']")
        profit_cells = table.query_selector_all(".perf-profit")
        loss_cells = table.query_selector_all(".perf-loss")
        assert len(profit_cells) + len(loss_cells) > 0, "No color-coded cells"


# ---------------------------------------------------------------------------
# TEST 21 — Color formatting
# ---------------------------------------------------------------------------

class TestColorFormatting:
    def test_positive_values_green(self, page: Page, dashboard_url: str):
        """Positive percentage values use perf-profit (green) class."""
        _goto_run_detail(page, dashboard_url)
        page.wait_for_selector("[data-testid='debate-summary-panel']", timeout=10000)
        profit_spans = page.query_selector_all(".perf-profit")
        loss_spans = page.query_selector_all(".perf-loss")
        assert len(profit_spans) + len(loss_spans) > 0, "No color-coded values found"


# ---------------------------------------------------------------------------
# TEST 22 — Backward compatibility (no JS intervention data)
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_dashboard_works_without_js_interventions(
        self, page: Page, dashboard_url: str,
    ):
        """Dashboard renders correctly when JS intervention data is missing."""
        _goto_run_detail(page, dashboard_url)
        page.wait_for_selector("[data-testid='debate-impact-agents']", timeout=10000)
        table = page.query_selector("[data-testid='debate-impact-agents']")
        assert table is not None


# ---------------------------------------------------------------------------
# TEST 23 — No truncation in ov-htable cells
# ---------------------------------------------------------------------------

class TestNoTruncation:
    def test_ov_htable_cells_not_truncated(self, page: Page, dashboard_url: str):
        """ov-htable cells should not use text-overflow: ellipsis."""
        _goto_run_detail(page, dashboard_url)
        cells = page.query_selector_all(".ov-htable td")
        assert len(cells) > 0, "No ov-htable cells found"
        for cell in cells[:5]:
            overflow = cell.evaluate("el => getComputedStyle(el).textOverflow")
            assert overflow != "ellipsis", (
                f"ov-htable cell should not truncate, got text-overflow: {overflow}"
            )


# ---------------------------------------------------------------------------
# TEST — Table labels loaded from JSON config
# ---------------------------------------------------------------------------

class TestTableLabelsLoaded:
    def test_table_labels_loaded(self, page: Page, dashboard_url: str):
        """Dashboard loads and uses table_labels.json for display strings."""
        _goto_run_detail(page, dashboard_url)
        page.wait_for_selector(".run-overview", timeout=10000)
        text = page.text_content(".run-overview")
        assert "Run ID" in text
        assert "Experiment" in text

    def test_divergence_labels_from_config(self, page: Page, dashboard_url: str):
        """Divergence card title comes from table_labels.json."""
        _goto_run_detail(page, dashboard_url)
        section = page.wait_for_selector("#divergence-section", timeout=5000)
        assert section is not None
        text = section.text_content()
        assert "Divergence Overview" in text, (
            "Card title 'Divergence Overview' not found — labels may not be loading"
        )


# ---------------------------------------------------------------------------
# Financial Tests — Redesigned three-part layout
# ---------------------------------------------------------------------------


def _open_ablation_experiment(page: Page, url: str) -> None:
    """Navigate to ablation tab and open the first experiment card."""
    page.goto(f"{url}/#ablation")
    page.wait_for_selector(
        "[data-testid='ablation-experiment']", timeout=5000,
    )
    header = page.query_selector(
        "[data-testid='ablation-experiment'] .card-header",
    )
    header.click()
    page.wait_for_selector(
        "[data-testid='ablation-experiment'].open .card-body",
        timeout=3000,
    )


class TestFinancialTestsSection:
    def test_financial_tests_section_renders(
        self, page: Page, dashboard_url: str,
    ):
        """Financial tests section appears inside the experiment card."""
        _open_ablation_experiment(page, dashboard_url)
        section = page.wait_for_selector(
            "[data-testid='financial-tests-section']", timeout=8000,
        )
        assert section is not None, "Financial tests section did not render"

    def test_financial_section_has_title(
        self, page: Page, dashboard_url: str,
    ):
        """Section title includes 'Financial Performance Impact'."""
        _open_ablation_experiment(page, dashboard_url)
        section = page.wait_for_selector(
            "[data-testid='financial-tests-section']", timeout=8000,
        )
        text = section.text_content()
        assert "Financial Performance Impact" in text, (
            f"Expected 'Financial Performance Impact' in section, got: {text[:200]}"
        )

    def test_financial_section_has_subtitle(
        self, page: Page, dashboard_url: str,
    ):
        """Subtitle mentions effect of intervention vs baseline."""
        _open_ablation_experiment(page, dashboard_url)
        section = page.wait_for_selector(
            "[data-testid='financial-tests-section']", timeout=8000,
        )
        subtitle = section.query_selector(".fin-subtitle")
        assert subtitle is not None, "Subtitle element not found"
        text = subtitle.text_content()
        assert "intervention" in text.lower() or "baseline" in text.lower(), (
            f"Subtitle should mention intervention/baseline: {text}"
        )

    def test_financial_table_has_correct_columns(
        self, page: Page, dashboard_url: str,
    ):
        """Table header has Metric, Baseline, Intervention, Delta, p-value."""
        _open_ablation_experiment(page, dashboard_url)
        table = page.wait_for_selector(
            "[data-testid='financial-tests-table']", timeout=8000,
        )
        headers = table.query_selector_all("thead th")
        header_texts = [h.text_content().strip() for h in headers]
        assert "Metric" in header_texts, f"Missing 'Metric' header: {header_texts}"
        assert "Baseline" in header_texts, f"Missing 'Baseline' header: {header_texts}"
        assert "Intervention" in header_texts, (
            f"Missing 'Intervention' header: {header_texts}"
        )
        assert "p-value" in header_texts, f"Missing 'p-value' header: {header_texts}"

    def test_financial_table_has_group_headers(
        self, page: Page, dashboard_url: str,
    ):
        """Table contains group subheader rows (Performance, Risk, etc.)."""
        _open_ablation_experiment(page, dashboard_url)
        table = page.wait_for_selector(
            "[data-testid='financial-tests-table']", timeout=8000,
        )
        groups = table.query_selector_all("tr.fin-group-header")
        group_texts = [g.text_content().strip() for g in groups]
        assert len(group_texts) >= 2, (
            f"Expected at least 2 group headers, got {len(group_texts)}: {group_texts}"
        )
        assert "Performance" in group_texts, (
            f"Missing 'Performance' group header: {group_texts}"
        )

    def test_financial_table_numeric_cells_right_aligned(
        self, page: Page, dashboard_url: str,
    ):
        """Numeric cells (num-cell) are right-aligned."""
        _open_ablation_experiment(page, dashboard_url)
        table = page.wait_for_selector(
            "[data-testid='financial-tests-table']", timeout=8000,
        )
        cells = table.query_selector_all("td.num-cell")
        assert len(cells) > 0, "No numeric cells found"
        alignment = page.evaluate(
            "(el) => window.getComputedStyle(el).textAlign", cells[0],
        )
        assert alignment == "right", (
            f"Expected numeric cells right-aligned, got '{alignment}'"
        )

    def test_ci_hint_present(
        self, page: Page, dashboard_url: str,
    ):
        """CI hint text appears below the table."""
        _open_ablation_experiment(page, dashboard_url)
        page.wait_for_selector(
            "[data-testid='financial-tests-section']", timeout=8000,
        )
        hint = page.query_selector(".fin-ci-hint")
        assert hint is not None, "CI hint not found"
        assert "confidence interval" in hint.text_content().lower(), (
            "CI hint should mention confidence intervals"
        )

    def test_delta_cells_have_ci_tooltip(
        self, page: Page, dashboard_url: str,
    ):
        """Delta cells have a title attribute with CI info."""
        _open_ablation_experiment(page, dashboard_url)
        table = page.wait_for_selector(
            "[data-testid='financial-tests-table']", timeout=8000,
        )
        delta_cells = table.query_selector_all("td.num-cell[title]")
        assert len(delta_cells) > 0, "No delta cells with title tooltip found"
        title = delta_cells[0].get_attribute("title")
        assert "95% CI" in title, f"Expected '95% CI' in tooltip, got: {title}"


class TestFinancialSignificanceEndpoint:
    def test_financial_significance_endpoint_returns_data(
        self, page: Page, dashboard_url: str,
    ):
        """The /api/ablation/financial-significance endpoint returns valid data."""
        import json
        import urllib.request

        resp = urllib.request.urlopen(
            f"{dashboard_url}/api/ablation/financial-significance",
        )
        data = json.loads(resp.read())
        assert "experiments" in data, "Missing 'experiments' key"
        assert "metrics" in data, "Missing 'metrics' key"
        assert len(data["experiments"]) >= 1, "Expected at least 1 experiment"
        assert len(data["metrics"]) >= 1, "Expected at least 1 metric"

    def test_financial_significance_metric_has_results(
        self, page: Page, dashboard_url: str,
    ):
        """Each metric entry contains results keyed by experiment."""
        import json
        import urllib.request

        resp = urllib.request.urlopen(
            f"{dashboard_url}/api/ablation/financial-significance",
        )
        data = json.loads(resp.read())
        for m in data["metrics"]:
            assert "metric" in m, "Missing 'metric' key in entry"
            assert "results" in m, "Missing 'results' key in entry"
            for exp in data["experiments"]:
                r = m["results"].get(exp)
                if r is not None:
                    assert "mean_diff" in r, f"Missing mean_diff for {exp}"
                    assert "p_value" in r, f"Missing p_value for {exp}"
