"""Playwright integration tests for the Debate Dashboard.

Uses a real FastAPI server started by the ``dashboard_url`` session fixture
(see conftest.py) against a **copied** test dataset so source data is never
mutated.

Run:
    pytest -m dashboard --browser chromium -v
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page

pytestmark = pytest.mark.dashboard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _goto_live(page: Page, url: str) -> None:
    """Navigate to the Live Debate view and wait for event cards."""
    page.goto(f"{url}/#live")
    page.wait_for_selector("#live-entries", timeout=5000)
    page.wait_for_selector("#live-entries .card", timeout=10000)


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
# TEST 4 — Live debate panel renders events
# ---------------------------------------------------------------------------

class TestLiveDebatePanel:
    def test_events_appear(self, page: Page, dashboard_url: str):
        """Live Debate tab loads and shows event cards."""
        _goto_live(page, dashboard_url)
        cards = page.query_selector_all("#live-entries .card")
        assert len(cards) > 0, "Live debate panel rendered no events"

    def test_phase_labels_present(self, page: Page, dashboard_url: str):
        """Event cards contain phase labels (PROPOSAL / CRITIQUE / REVISION)."""
        _goto_live(page, dashboard_url)

        entries_text = page.text_content("#live-entries")
        phases_found = [
            p for p in ("PROPOSAL", "CRITIQUE", "REVISION")
            if p in entries_text
        ]
        assert len(phases_found) >= 1, (
            f"No phase labels found. Text sample: {entries_text[:300]}"
        )

    def test_agent_names_in_event_labels(
        self, page: Page, dashboard_url: str,
    ):
        """Event card headers contain agent names from the run."""
        _goto_live(page, dashboard_url)

        entries_text = page.text_content("#live-entries")
        agents_found = [
            a for a in ("risk", "technical", "value")
            if a in entries_text
        ]
        assert len(agents_found) >= 1, (
            "No agent names found in event labels"
        )


# ---------------------------------------------------------------------------
# TEST 5 — Event ordering (newest first)
# ---------------------------------------------------------------------------

class TestEventOrdering:
    def test_events_newest_first(self, page: Page, dashboard_url: str):
        """Events are rendered newest first (highest round at top)."""
        _goto_live(page, dashboard_url)

        headers = page.query_selector_all("#live-entries .card-header")
        assert len(headers) >= 2, "Need at least 2 events to test ordering"

        rounds = []
        for h in headers:
            text = h.text_content()
            match = re.search(r"\[ROUND\s+(\d+)\]", text)
            if match:
                rounds.append(int(match.group(1)))

        assert len(rounds) >= 2, (
            f"Could not parse round numbers from headers: "
            f"{[h.text_content() for h in headers[:5]]}"
        )

        assert rounds[0] == max(rounds), (
            f"First event round ({rounds[0]}) is not the highest "
            f"({max(rounds)})"
        )

        for i in range(len(rounds) - 1):
            assert rounds[i] >= rounds[i + 1], (
                f"Events not in descending round order at index {i}: "
                f"{rounds[i]} < {rounds[i + 1]}. Full: {rounds}"
            )


# ---------------------------------------------------------------------------
# TEST 6 — Expand event payload and verify content
# ---------------------------------------------------------------------------

class TestExpandEventPayload:
    def test_click_expands_card(self, page: Page, dashboard_url: str):
        """Clicking an event card header reveals the card body."""
        _goto_live(page, dashboard_url)

        first_card = page.query_selector("#live-entries .card")
        assert first_card is not None, "No event card found"

        classes_before = first_card.get_attribute("class") or ""
        assert "open" not in classes_before, (
            "Card unexpectedly open before click"
        )

        header = first_card.query_selector(".card-header")
        header.click()

        classes_after = first_card.get_attribute("class") or ""
        assert "open" in classes_after, (
            "Card did not get 'open' class after click"
        )

        body = first_card.query_selector(".card-body")
        assert body is not None, "Card body element not found"
        assert body.is_visible(), "Card body not visible after expand"

        pre = body.query_selector("pre")
        assert pre is not None, "No <pre> element in card body"
        content = pre.text_content().strip()
        assert len(content) > 0, "Event payload is empty"

    def test_payload_contains_structured_data(
        self, page: Page, dashboard_url: str,
    ):
        """Expanded PROPOSAL payload contains expected keys."""
        _goto_live(page, dashboard_url)

        cards = page.query_selector_all("#live-entries .card")
        proposal_card = None
        for card in cards:
            header_text = card.query_selector(".card-header").text_content()
            if "PROPOSAL" in header_text:
                proposal_card = card
                break

        if proposal_card is None:
            pytest.skip("No PROPOSAL event card found to test payload")

        proposal_card.query_selector(".card-header").click()
        body = proposal_card.query_selector(".card-body")
        pre = body.query_selector("pre.content")
        assert pre is not None, "No <pre class='content'> in proposal card"

        content = pre.text_content().strip()
        assert len(content) > 10, "Payload too short to be meaningful"

        assert "allocation" in content.lower() or "claim" in content.lower(), (
            "Proposal payload doesn't contain 'allocation' or 'claim'"
        )


# ---------------------------------------------------------------------------
# TEST 7 — Judge portfolio layout (vertical alloc table + metrics table)
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
