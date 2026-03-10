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
    def test_quality_metrics_row_has_flex_display(
        self, page: Page, dashboard_url: str,
    ):
        """Quality metrics row (rho, pillars, JS, evidence) uses flex layout."""
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
        assert display == "flex", (
            f"Expected metrics-row display:flex, got '{display}'"
        )

    def test_breakdowns_row_side_by_side(
        self, page: Page, dashboard_url: str,
    ):
        """Per Scenario and Per Agent Config tables are in same flex row."""
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
        assert display == "flex", (
            f"Expected breakdowns row display:flex, got '{display}'"
        )
        text = row.text_content()
        assert "Per Scenario" in text, "Missing 'Per Scenario' in breakdowns row"
        assert "Per Agent Config" in text, (
            "Missing 'Per Agent Config' in breakdowns row"
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
        assert "Final Revision" in text, "Missing 'Final Revision' column header"

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
