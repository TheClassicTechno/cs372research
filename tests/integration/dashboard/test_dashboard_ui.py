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
# TEST 7 — Divergence table axes: phases as rows, metrics as columns
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
# TEST 8 — File explorer shows full path from logging root
# ---------------------------------------------------------------------------

class TestFileExplorerFullPath:
    def test_file_label_shows_full_path(self, page: Page, dashboard_url: str):
        """Clicking a file in the explorer shows the full path from logging/."""
        _goto_run_detail(page, dashboard_url)

        # Open the File Explorer card
        page.wait_for_selector("#file-explorer-section .card", timeout=5000)
        header = page.query_selector("#file-explorer-section .card-header")
        header.click()

        # Click the first file link in the tree
        file_link = page.wait_for_selector(
            "#file-explorer-section .file-tree .file-link", timeout=5000,
        )
        file_link.click()

        # Wait for file content to load
        label = page.wait_for_selector(
            "#file-content-display .section-label", timeout=5000,
        )
        label_text = label.text_content().strip()

        assert label_text.startswith("logging/runs/"), (
            f"File label should start with 'logging/runs/', got: '{label_text}'"
        )
        assert "run_2026-03-07_19-50-06" in label_text, (
            f"File label should contain run ID, got: '{label_text}'"
        )


# ---------------------------------------------------------------------------
# TEST 9 — PID Stats formatting: score classes, 3 decimals, delta, rho emphasis
# ---------------------------------------------------------------------------

class TestPIDStatsFormatting:
    def test_scores_use_3_decimals(self, page: Page, dashboard_url: str):
        """PID Stats table renders scores with 3 decimal places."""
        _goto_run_detail(page, dashboard_url)

        section = page.wait_for_selector("#pid-stats-section", timeout=5000)
        if not section or not section.text_content().strip():
            pytest.skip("PID Stats section not populated for test run")

        cells = page.query_selector_all(
            "#pid-stats-section td.num-cell",
        )
        assert len(cells) > 0, "No numeric cells found in PID Stats"

        for cell in cells:
            text = cell.text_content().strip()
            if text == "\u2014":
                continue
            parts = text.split(".")
            assert len(parts) == 2 and len(parts[1]) == 3, (
                f"Expected 3 decimal places, got '{text}'"
            )

    def test_rho_column_has_emphasis(self, page: Page, dashboard_url: str):
        """The rho_i column cells have the rho-col class for emphasis."""
        _goto_run_detail(page, dashboard_url)

        section = page.wait_for_selector("#pid-stats-section", timeout=5000)
        if not section or not section.text_content().strip():
            pytest.skip("PID Stats section not populated for test run")

        rho_cells = page.query_selector_all(
            "#pid-stats-section td.rho-col",
        )
        assert len(rho_cells) >= 1, "No rho-col cells found"

    def test_score_cells_have_shading_class(self, page: Page, dashboard_url: str):
        """Numeric cells get a score-high/mid/low/bad class."""
        _goto_run_detail(page, dashboard_url)

        section = page.wait_for_selector("#pid-stats-section", timeout=5000)
        if not section or not section.text_content().strip():
            pytest.skip("PID Stats section not populated for test run")

        cells = page.query_selector_all(
            "#pid-stats-section td.num-cell",
        )
        score_classes = {"score-high", "score-mid", "score-low", "score-bad"}
        tagged = 0
        for cell in cells:
            cls = cell.get_attribute("class") or ""
            if any(sc in cls for sc in score_classes):
                tagged += 1
        assert tagged >= 1, "No score shading classes found on numeric cells"

    def test_round_delta_shown(self, page: Page, dashboard_url: str):
        """Round 2+ headers show a delta indicator."""
        _goto_run_detail(page, dashboard_url)

        section = page.wait_for_selector("#pid-stats-section", timeout=5000)
        if not section or not section.text_content().strip():
            pytest.skip("PID Stats section not populated for test run")

        labels = page.query_selector_all(
            "#pid-stats-section .section-label",
        )
        if len(labels) < 2:
            pytest.skip("Need at least 2 rounds to test delta")

        second_label = labels[1].text_content()
        assert "(" in second_label and ")" in second_label, (
            f"Round 2 label missing delta indicator: '{second_label}'"
        )


# ---------------------------------------------------------------------------
# TEST 10 — Global bold text (font-weight: 600 on body)
# ---------------------------------------------------------------------------

class TestGlobalBoldText:
    def test_body_font_weight(self, page: Page, dashboard_url: str):
        """Body element has font-weight 600 for readability."""
        _goto_run_detail(page, dashboard_url)

        fw = page.evaluate(
            "window.getComputedStyle(document.body).fontWeight"
        )
        assert fw == "600", f"Expected body font-weight 600, got '{fw}'"


# ---------------------------------------------------------------------------
# TEST 11 — No truncation on overview table cells
# ---------------------------------------------------------------------------

class TestNoTruncation:
    def test_ov_htable_cells_not_truncated(self, page: Page, dashboard_url: str):
        """Overview table cells use word-wrap, not text-overflow: ellipsis."""
        _goto_run_detail(page, dashboard_url)

        cells = page.query_selector_all("table.ov-htable td")
        assert len(cells) > 0, "No ov-htable cells found"

        for cell in cells:
            overflow = page.evaluate(
                "(el) => window.getComputedStyle(el).textOverflow", cell,
            )
            assert overflow != "ellipsis", (
                f"ov-htable td still has text-overflow: ellipsis"
            )


# ---------------------------------------------------------------------------
# TEST 12 — Divergence section shows numeric values (not dashes)
# ---------------------------------------------------------------------------

class TestDivergenceValues:
    def test_revised_js_divergence_not_dash(
        self, page: Page, dashboard_url: str,
    ):
        """Revised JS Divergence cell shows a number, not a dash."""
        _goto_run_detail(page, dashboard_url)

        section = page.wait_for_selector("#divergence-section", timeout=5000)
        assert section is not None

        table = page.query_selector("#divergence-section .data-table")
        assert table is not None, "Divergence table not found"

        rows = table.query_selector_all("tr:not(:first-child)")
        for row in rows:
            tds = row.query_selector_all("td")
            label = tds[0].text_content().strip() if tds else ""
            if label == "Revised":
                js_val = tds[1].text_content().strip()
                assert js_val != "\u2014", (
                    "Revised JS Divergence shows '\u2014' — key mismatch bug"
                )
                assert re.match(r"^\d+\.\d+$", js_val), (
                    f"Expected numeric value, got '{js_val}'"
                )
                break
        else:
            pytest.skip("No 'Revised' row in divergence table")

    def test_proposed_js_divergence_present(
        self, page: Page, dashboard_url: str,
    ):
        """Proposed JS Divergence cell shows a number from test data."""
        _goto_run_detail(page, dashboard_url)

        section = page.wait_for_selector("#divergence-section", timeout=5000)
        assert section is not None

        table = page.query_selector("#divergence-section .data-table")
        assert table is not None, "Divergence table not found"

        rows = table.query_selector_all("tr:not(:first-child)")
        for row in rows:
            tds = row.query_selector_all("td")
            label = tds[0].text_content().strip() if tds else ""
            if label == "Proposed":
                js_val = tds[1].text_content().strip()
                assert js_val != "\u2014", (
                    "Proposed JS Divergence shows '\u2014'"
                )
                break


# ---------------------------------------------------------------------------
# TEST 13 — Allocation table: bold + right-aligned agent cells
# ---------------------------------------------------------------------------

class TestAllocationTableStyling:
    def test_agent_cells_bold_and_right_aligned(
        self, page: Page, dashboard_url: str,
    ):
        """Agent allocation cells in the JUDGE table are bold and right-aligned."""
        _goto_run_detail(page, dashboard_url)

        table = page.wait_for_selector("#judge-alloc-table", timeout=10000)
        assert table is not None, "Judge allocation table not found"

        data_cells = table.query_selector_all("tr:not(:first-child) td")
        # Skip the first cell in each row (ticker name column)
        numeric_cells = [
            c for i, c in enumerate(data_cells) if i % (len(data_cells) // 2 + 1) != 0
        ]
        if not numeric_cells:
            pytest.skip("No numeric cells in allocation table")

        # Check at least one non-ticker cell is bold + right-aligned
        rows = table.query_selector_all("tr:not(:first-child)")
        found_styled = False
        for row in rows:
            tds = row.query_selector_all("td")
            if len(tds) < 2:
                continue
            # Check agent cells (skip first = ticker)
            for td in tds[1:]:
                fw = page.evaluate(
                    "(el) => window.getComputedStyle(el).fontWeight", td,
                )
                ta = page.evaluate(
                    "(el) => window.getComputedStyle(el).textAlign", td,
                )
                if fw in ("600", "700", "bold") and ta == "right":
                    found_styled = True
                    break
            if found_styled:
                break

        assert found_styled, (
            "No allocation cells found with font-weight:600 and text-align:right"
        )

    def test_judge_column_has_border(self, page: Page, dashboard_url: str):
        """JUDGE column has a visible left border for visual separation."""
        _goto_run_detail(page, dashboard_url)

        table = page.wait_for_selector("#judge-alloc-table", timeout=10000)
        assert table is not None, "Judge allocation table not found"

        rows = table.query_selector_all("tr:not(:first-child)")
        if not rows:
            pytest.skip("No data rows in allocation table")

        # Last td in each row is the JUDGE column
        last_td = rows[0].query_selector("td:last-child")
        border = page.evaluate(
            "(el) => window.getComputedStyle(el).borderLeftWidth", last_td,
        )
        assert border != "0px", (
            f"JUDGE column missing left border, got borderLeftWidth={border}"
        )


# ---------------------------------------------------------------------------
# TEST 14 — Muted color palette on section labels and table headers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TEST — Ablation tab
# ---------------------------------------------------------------------------

class TestAblationTab:
    def test_ablation_tab_renders(self, page: Page, dashboard_url: str):
        """Navigate to #ablation, verify the content container appears."""
        page.goto(f"{dashboard_url}/#ablation")
        page.wait_for_selector(
            "[data-testid='ablation-content']", timeout=5000,
        )
        container = page.get_by_test_id("ablation-content")
        assert container.is_visible(), "Ablation content container not visible"

    def test_ablation_experiment_card_shows(self, page: Page, dashboard_url: str):
        """Ablation view shows at least one experiment card with name."""
        page.goto(f"{dashboard_url}/#ablation")
        page.wait_for_selector(
            "[data-testid='ablation-experiment']", timeout=5000,
        )
        card = page.get_by_test_id("ablation-experiment").first
        assert card.is_visible(), "Ablation experiment card not visible"
        text = card.text_content()
        assert "test" in text, "Experiment name 'test' not found in card"

    def test_ablation_regenerate_button_exists(
        self, page: Page, dashboard_url: str,
    ):
        """Regenerate button is visible on the ablation view."""
        page.goto(f"{dashboard_url}/#ablation")
        page.wait_for_selector(
            "[data-testid='regenerate-ablation']", timeout=5000,
        )
        btn = page.get_by_test_id("regenerate-ablation")
        assert btn.is_visible(), "Regenerate button not visible"

    def test_ablation_metrics_table(self, page: Page, dashboard_url: str):
        """Ablation content contains a data-table with rho/pillar values."""
        page.goto(f"{dashboard_url}/#ablation")
        page.wait_for_selector(
            "[data-testid='ablation-content'] .data-table", timeout=5000,
        )
        table_text = page.text_content(
            "[data-testid='ablation-content'] .data-table",
        )
        assert "0.7200" in table_text, (
            "Expected rho value 0.7200 in ablation table"
        )


class TestColorPalette:
    def test_section_labels_use_brown(self, page: Page, dashboard_url: str):
        """Section labels use the dark brown color (#5c4632)."""
        _goto_run_detail(page, dashboard_url)

        labels = page.query_selector_all(".section-label")
        if not labels:
            pytest.skip("No section labels found")

        color = page.evaluate(
            "(el) => window.getComputedStyle(el).color", labels[0],
        )
        # #5c4632 → rgb(92, 70, 50)
        assert "92" in color and "70" in color and "50" in color, (
            f"Section label color not dark brown, got '{color}'"
        )

    def test_ov_title_uses_brown(self, page: Page, dashboard_url: str):
        """Overview title uses the dark brown color."""
        _goto_run_detail(page, dashboard_url)

        title = page.query_selector(".ov-title")
        if not title:
            pytest.skip("No .ov-title found")

        color = page.evaluate(
            "(el) => window.getComputedStyle(el).color", title,
        )
        assert "92" in color and "70" in color and "50" in color, (
            f"Overview title color not dark brown, got '{color}'"
        )
