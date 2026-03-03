"""Tests for SEC EDGAR downloader (data-pipeline/EDGAR/get_sec_data.py)."""

import sys
from datetime import date
from pathlib import Path

import pytest

# data-pipeline has a hyphen so we can't use normal imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data-pipeline" / "EDGAR"))
import get_sec_data as sec


# ==========================
# Helpers
# ==========================

def _make_filings(**kwargs):
    """Build a minimal SEC filings dict for filter_filings."""
    n = len(kwargs.get("form", []))
    defaults = {
        "form": ["10-Q"] * n,
        "filingDate": ["2024-05-15"] * n,
        "accessionNumber": ["0001234567-24-000001"] * n,
        "primaryDocument": ["filing.htm"] * n,
    }
    defaults.update(kwargs)
    return {"filings": {"recent": defaults}}


# ==========================
# TestQuarterMapping
# ==========================

class TestQuarterMapping:
    @pytest.mark.parametrize("month,expected", [
        (1, "Q1"), (2, "Q1"), (3, "Q1"),
        (4, "Q2"), (5, "Q2"), (6, "Q2"),
        (7, "Q3"), (8, "Q3"), (9, "Q3"),
        (10, "Q4"), (11, "Q4"), (12, "Q4"),
    ])
    def test_quarter_from_month_all_months(self, month, expected):
        assert sec.quarter_from_month(month) == expected


# ==========================
# TestNextQuarter
# ==========================

class TestNextQuarter:
    def test_mid_year(self):
        assert sec.next_quarter(2024, "Q2") == (2024, "Q3")

    def test_year_boundary(self):
        assert sec.next_quarter(2024, "Q4") == (2025, "Q1")

    def test_q1_to_q2(self):
        assert sec.next_quarter(2024, "Q1") == (2024, "Q2")

    def test_q3_to_q4(self):
        assert sec.next_quarter(2024, "Q3") == (2024, "Q4")


# ==========================
# TestAmendmentFiltering
# ==========================

class TestAmendmentFiltering:
    def test_is_amendment_true(self):
        assert sec.is_amendment("10-K/A") is True
        assert sec.is_amendment("10-Q/A") is True

    def test_is_amendment_false(self):
        assert sec.is_amendment("10-K") is False
        assert sec.is_amendment("10-Q") is False
        assert sec.is_amendment("8-K") is False

    def test_amendments_excluded_by_default(self):
        filings = _make_filings(
            form=["10-K", "10-K/A"],
            filingDate=["2024-11-15", "2024-12-01"],
            accessionNumber=["acc1", "acc2"],
            primaryDocument=["d1.htm", "d2.htm"],
        )
        targets = [(2024, None)]
        result = sec.filter_filings(filings, targets, {"10-K"}, include_amendments=False)
        assert len(result) == 1
        assert result[0]["form"] == "10-K"

    def test_amendments_included_with_flag(self):
        filings = _make_filings(
            form=["10-K", "10-K/A"],
            filingDate=["2024-11-15", "2024-12-01"],
            accessionNumber=["acc1", "acc2"],
            primaryDocument=["d1.htm", "d2.htm"],
        )
        targets = [(2024, None)]
        result = sec.filter_filings(filings, targets, {"10-K"}, include_amendments=True)
        assert len(result) == 2
        forms = [r["form"] for r in result]
        assert "10-K" in forms
        assert "10-K/A" in forms

    def test_10k_amendment_handling(self):
        filings = _make_filings(
            form=["10-K/A"],
            filingDate=["2024-11-15"],
            accessionNumber=["acc1"],
            primaryDocument=["d1.htm"],
        )
        targets = [(2024, None)]
        result = sec.filter_filings(filings, targets, {"10-K"}, include_amendments=True)
        assert len(result) == 1
        assert result[0]["form"] == "10-K/A"


# ==========================
# TestComputeLastNQuarters
# ==========================

class TestComputeLastNQuarters:
    def test_n_equals_1(self):
        result = sec.compute_last_n_quarters(1, today=date(2024, 8, 15))
        assert result == [(2024, "Q2")]

    def test_n_equals_4(self):
        result = sec.compute_last_n_quarters(4, today=date(2024, 8, 15))
        assert result == [
            (2024, "Q2"), (2024, "Q1"), (2023, "Q4"), (2023, "Q3")
        ]

    def test_n_greater_than_4(self):
        result = sec.compute_last_n_quarters(6, today=date(2024, 8, 15))
        assert len(result) == 6
        assert result[-1] == (2023, "Q1")

    def test_year_boundary(self):
        result = sec.compute_last_n_quarters(2, today=date(2024, 2, 1))
        assert result == [(2023, "Q4"), (2023, "Q3")]

    def test_excludes_current_quarter(self):
        result = sec.compute_last_n_quarters(1, today=date(2024, 7, 1))
        # July = Q3, so last completed = Q2
        assert result == [(2024, "Q2")]

    def test_q1_wraps_to_q4(self):
        # In Q1 2024, last completed is Q4 2023
        result = sec.compute_last_n_quarters(1, today=date(2024, 1, 15))
        assert result == [(2023, "Q4")]

    def test_deterministic(self):
        fixed = date(2024, 6, 1)
        r1 = sec.compute_last_n_quarters(4, today=fixed)
        r2 = sec.compute_last_n_quarters(4, today=fixed)
        assert r1 == r2


# ==========================
# TestFilterFilings
# ==========================

class TestFilterFilings:
    def test_matches_10q_in_q2(self):
        filings = _make_filings(
            form=["10-Q"],
            filingDate=["2024-05-15"],
            accessionNumber=["acc1"],
            primaryDocument=["d.htm"],
        )
        targets = [(2024, "Q2")]
        result = sec.filter_filings(filings, targets, {"10-Q"})
        assert len(result) == 1

    def test_matches_10k_in_q4(self):
        filings = _make_filings(
            form=["10-K"],
            filingDate=["2024-11-15"],
            accessionNumber=["acc1"],
            primaryDocument=["d.htm"],
        )
        targets = [(2024, "Q4")]
        result = sec.filter_filings(filings, targets, {"10-K"})
        assert len(result) == 1

    def test_annual_matches_any_quarter(self):
        """ANNUAL target (quarter=None) matches filings in any quarter."""
        filings = _make_filings(
            form=["10-K", "10-K"],
            filingDate=["2024-02-15", "2024-11-15"],
            accessionNumber=["acc1", "acc2"],
            primaryDocument=["d1.htm", "d2.htm"],
        )
        targets = [(2024, None)]
        result = sec.filter_filings(filings, targets, {"10-K"})
        assert len(result) == 2

    def test_form_filter_independent_of_time(self):
        """All allowed forms in the time window should match."""
        filings = _make_filings(
            form=["10-Q", "8-K", "10-K"],
            filingDate=["2024-05-15", "2024-05-20", "2024-05-25"],
            accessionNumber=["acc1", "acc2", "acc3"],
            primaryDocument=["d1.htm", "d2.htm", "d3.htm"],
        )
        targets = [(2024, "Q2")]
        result = sec.filter_filings(filings, targets, {"10-Q", "8-K", "10-K"})
        assert len(result) == 3

    def test_form_not_in_allowed_excluded(self):
        filings = _make_filings(
            form=["SC 13G", "4"],
            filingDate=["2024-05-15", "2024-05-20"],
            accessionNumber=["acc1", "acc2"],
            primaryDocument=["d1.htm", "d2.htm"],
        )
        targets = [(2024, "Q2")]
        result = sec.filter_filings(filings, targets, sec.DEFAULT_FORMS)
        assert len(result) == 0

    def test_amendments_excluded_by_default(self):
        filings = _make_filings(
            form=["10-Q", "10-Q/A"],
            filingDate=["2024-05-15", "2024-06-01"],
            accessionNumber=["acc1", "acc2"],
            primaryDocument=["d1.htm", "d2.htm"],
        )
        targets = [(2024, "Q2")]
        result = sec.filter_filings(filings, targets, {"10-Q"}, include_amendments=False)
        assert len(result) == 1
        assert result[0]["form"] == "10-Q"

    def test_amendments_included(self):
        filings = _make_filings(
            form=["10-Q", "10-Q/A"],
            filingDate=["2024-05-15", "2024-06-01"],
            accessionNumber=["acc1", "acc2"],
            primaryDocument=["d1.htm", "d2.htm"],
        )
        targets = [(2024, "Q2")]
        result = sec.filter_filings(filings, targets, {"10-Q"}, include_amendments=True)
        assert len(result) == 2

    def test_multiple_targets(self):
        filings = _make_filings(
            form=["10-Q", "10-Q", "10-K"],
            filingDate=["2024-02-15", "2024-05-15", "2024-11-15"],
            accessionNumber=["acc1", "acc2", "acc3"],
            primaryDocument=["d1.htm", "d2.htm", "d3.htm"],
        )
        targets = [(2024, "Q1"), (2024, "Q4")]
        result = sec.filter_filings(filings, targets, {"10-Q", "10-K"})
        assert len(result) == 2

    def test_empty_filings(self):
        filings = {"filings": {"recent": {
            "form": [], "filingDate": [], "accessionNumber": [], "primaryDocument": [],
        }}}
        targets = [(2024, "Q2")]
        result = sec.filter_filings(filings, targets, sec.DEFAULT_FORMS)
        assert result == []

    def test_matched_quarter_field(self):
        filings = _make_filings(
            form=["10-Q"],
            filingDate=["2024-05-15"],
            accessionNumber=["acc1"],
            primaryDocument=["d.htm"],
        )
        targets = [(2024, "Q2")]
        result = sec.filter_filings(filings, targets, {"10-Q"})
        assert result[0]["matched_quarter"] == "Q2"
        assert result[0]["matched_year"] == 2024

    def test_custom_forms_override(self):
        """--forms override: only specified forms should match."""
        filings = _make_filings(
            form=["10-Q", "8-K", "DEF 14A"],
            filingDate=["2024-05-15", "2024-05-20", "2024-05-25"],
            accessionNumber=["acc1", "acc2", "acc3"],
            primaryDocument=["d1.htm", "d2.htm", "d3.htm"],
        )
        targets = [(2024, "Q2")]
        # Only allow DEF 14A
        result = sec.filter_filings(filings, targets, {"DEF 14A"})
        assert len(result) == 1
        assert result[0]["form"] == "DEF 14A"

    def test_report_date_used_for_quarter_mapping(self):
        """When reportDate is available, it should be used instead of filingDate."""
        filings = _make_filings(
            form=["10-K"],
            filingDate=["2025-02-26"],
            accessionNumber=["acc1"],
            primaryDocument=["d.htm"],
            reportDate=["2025-01-26"],
        )
        # reportDate is Jan -> Q1, filingDate is Feb -> Q1 too in this case
        # but let's test with a case where they differ
        targets = [(2025, "Q1")]
        result = sec.filter_filings(filings, targets, {"10-K"})
        assert len(result) == 1
        assert result[0]["matched_quarter"] == "Q1"

    def test_spillover_10k_to_prior_quarter(self):
        """10-K with reportDate in next quarter can spill over to the prior quarter."""
        filings = _make_filings(
            form=["10-K"],
            filingDate=["2025-02-26"],
            accessionNumber=["acc1"],
            primaryDocument=["d.htm"],
            reportDate=["2025-01-26"],
        )
        # reportDate Q1, but targeting Q4 2024 only — spillover should match
        targets = [(2024, "Q4")]
        result = sec.filter_filings(filings, targets, {"10-K"})
        assert len(result) == 1
        assert result[0]["matched_quarter"] == "Q4"
        assert result[0]["matched_year"] == 2024

    def test_direct_match_wins_over_spillover(self):
        """When both direct match and spillover are possible, direct match wins."""
        filings = _make_filings(
            form=["10-K"],
            filingDate=["2025-02-26"],
            accessionNumber=["acc1"],
            primaryDocument=["d.htm"],
            reportDate=["2025-01-26"],
        )
        # Both Q4 2024 and Q1 2025 are targets — direct match to Q1 should win
        targets = [(2024, "Q4"), (2025, "Q1")]
        result = sec.filter_filings(filings, targets, {"10-K"})
        assert len(result) == 1
        assert result[0]["matched_quarter"] == "Q1"
        assert result[0]["matched_year"] == 2025

    def test_dedup_by_accession(self):
        """Duplicate accession numbers should be deduplicated."""
        filings = _make_filings(
            form=["10-Q", "10-Q"],
            filingDate=["2024-05-15", "2024-05-15"],
            accessionNumber=["acc1", "acc1"],
            primaryDocument=["d.htm", "d.htm"],
        )
        targets = [(2024, "Q2")]
        result = sec.filter_filings(filings, targets, {"10-Q"})
        assert len(result) == 1


# ==========================
# TestBuildFilingUrl
# ==========================

class TestBuildFilingUrl:
    def test_basic(self):
        url = sec.build_filing_url("0000320193", "0000320193-24-000001", "filing.htm")
        assert "320193" in url
        assert "filing.htm" in url
        assert url.startswith("https://www.sec.gov/Archives/edgar/data/")

    def test_strips_leading_zeros_from_cik(self):
        url = sec.build_filing_url("0000320193", "0000320193-24-000001", "doc.htm")
        # CIK in URL path should not have leading zeros
        assert "/320193/" in url

    def test_removes_dashes_from_accession(self):
        url = sec.build_filing_url("0000320193", "0000320193-24-000001", "doc.htm")
        # Accession directory should have no dashes
        assert "000032019324000001" in url


# ==========================
# TestCleanHtmlToText
# ==========================

class TestCleanHtmlToText:
    def test_strips_tags(self):
        html = "<p>Hello <b>world</b></p>"
        result = sec.clean_html_to_text(html)
        assert "Hello" in result
        assert "world" in result
        assert "<p>" not in result
        assert "<b>" not in result

    def test_strips_script_and_style(self):
        html = "<div>Visible</div><script>var x=1;</script><style>.a{color:red}</style><p>Also visible</p>"
        result = sec.clean_html_to_text(html)
        assert "Visible" in result
        assert "Also visible" in result
        assert "var x" not in result
        assert "color:red" not in result

    def test_block_tags_become_newlines(self):
        html = "<p>First paragraph</p><p>Second paragraph</p>"
        result = sec.clean_html_to_text(html)
        assert "First paragraph" in result
        assert "Second paragraph" in result
        # Block tags should produce line separation
        lines = [l for l in result.splitlines() if l.strip()]
        assert len(lines) >= 2

    def test_normalizes_whitespace(self):
        html = "<p>  lots   of    spaces  </p>"
        result = sec.clean_html_to_text(html)
        assert "lots of spaces" in result

    def test_collapses_excessive_newlines(self):
        html = "<p>A</p><br><br><br><br><br><p>B</p>"
        result = sec.clean_html_to_text(html)
        assert "\n\n\n" not in result

    def test_html_entities_decoded(self):
        html = "<p>AT&amp;T &lt;10&gt;</p>"
        result = sec.clean_html_to_text(html)
        assert "AT&T" in result
        assert "<10>" in result

    def test_empty_html(self):
        assert sec.clean_html_to_text("") == ""

    def test_script_content_stripped(self):
        html = "<script>var x = 1;</script><p>visible</p>"
        result = sec.clean_html_to_text(html)
        assert "visible" in result
        assert "var x" not in result

    def test_noscript_stripped(self):
        html = "<noscript>Enable JS</noscript><p>Content</p>"
        result = sec.clean_html_to_text(html)
        assert "Enable JS" not in result
        assert "Content" in result


# ==========================
# TestDownloadFiling
# ==========================

class TestDownloadFiling:
    """Tests for the download_filing pipeline.
    Uses monkeypatch to mock download_html — no SEC API calls."""

    def _sample_filing(self):
        return {
            "form": "10-K",
            "filing_date": "2024-11-01",
            "accession": "0001234567-24-000001",
            "primary_doc": "filing.htm",
            "matched_year": 2024,
            "matched_quarter": "Q4",
        }

    @pytest.fixture(autouse=True)
    def _clear_failed_accessions(self):
        """Clear module-level _failed_accessions between tests."""
        sec._failed_accessions.clear()
        yield
        sec._failed_accessions.clear()

    def test_successful_extraction(self, tmp_path, monkeypatch):
        html = "<p>We face significant risks in our business.</p>"
        monkeypatch.setattr(sec, "download_html", lambda url: (html, None))

        filing = self._sample_filing()
        accession, success = sec.download_filing("AAPL", "0000320193", filing, tmp_path)
        assert success is True
        assert accession == filing["accession"]

        # Verify both HTML and TXT files were created
        html_files = list(tmp_path.rglob("*.html"))
        txt_files = list(tmp_path.rglob("*.txt"))
        assert len(html_files) == 1
        assert len(txt_files) == 1

        # Verify text content has metadata header and full text
        content = txt_files[0].read_text()
        assert "FORM: 10-K" in content
        assert "FILING_DATE: 2024-11-01" in content
        assert "ACCESSION: 0001234567-24-000001" in content
        assert "significant risks" in content

    def test_failed_download_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sec, "download_html", lambda url: (None, "HTTP 500"))

        filing = self._sample_filing()
        accession, success = sec.download_filing("AAPL", "0000320193", filing, tmp_path)
        assert success is False
        assert accession == filing["accession"]

    def test_output_path_structure(self, tmp_path, monkeypatch):
        html = "<p>Simple content</p>"
        monkeypatch.setattr(sec, "download_html", lambda url: (html, None))

        filing = self._sample_filing()
        sec.download_filing("AAPL", "0000320193", filing, tmp_path)

        expected_html = tmp_path / "raw_html" / "AAPL" / "AAPL_2024_Q4_10-K_2024-11-01.html"
        expected_txt = tmp_path / "clean_text" / "AAPL" / "AAPL_2024_Q4_10-K_2024-11-01.txt"
        assert expected_html.exists()
        assert expected_txt.exists()

    def test_html_saved_unchanged(self, tmp_path, monkeypatch):
        original_html = "<html><body><p>Original content</p></body></html>"
        monkeypatch.setattr(sec, "download_html", lambda url: (original_html, None))

        filing = self._sample_filing()
        sec.download_filing("AAPL", "0000320193", filing, tmp_path)

        html_path = tmp_path / "raw_html" / "AAPL" / "AAPL_2024_Q4_10-K_2024-11-01.html"
        assert html_path.read_text() == original_html

    def test_no_tmp_file_leftover(self, tmp_path, monkeypatch):
        html = "<p>Content</p>"
        monkeypatch.setattr(sec, "download_html", lambda url: (html, None))

        filing = self._sample_filing()
        sec.download_filing("AAPL", "0000320193", filing, tmp_path)

        tmp_files = list(tmp_path.rglob("*.tmp"))
        assert len(tmp_files) == 0

    def test_no_section_markers_in_output(self, tmp_path, monkeypatch):
        """Output should be full text, no ==== SECTION ==== markers."""
        html = "<p>Item 1A. Risk Factors</p><p>Risks here.</p>"
        monkeypatch.setattr(sec, "download_html", lambda url: (html, None))

        filing = self._sample_filing()
        sec.download_filing("AAPL", "0000320193", filing, tmp_path)

        txt_files = list(tmp_path.rglob("*.txt"))
        content = txt_files[0].read_text()
        assert "====" not in content
        assert "SECTION" not in content

    def test_skips_existing_files(self, tmp_path, monkeypatch):
        """If both HTML and TXT already exist, download_filing should skip."""
        filing = self._sample_filing()

        # Create the expected output files
        raw_dir = tmp_path / "raw_html" / "AAPL"
        txt_dir = tmp_path / "clean_text" / "AAPL"
        raw_dir.mkdir(parents=True)
        txt_dir.mkdir(parents=True)
        (raw_dir / "AAPL_2024_Q4_10-K_2024-11-01.html").write_text("existing")
        (txt_dir / "AAPL_2024_Q4_10-K_2024-11-01.txt").write_text("existing")

        # download_html should never be called
        def _should_not_call(url):
            raise AssertionError("Should not download when files exist")
        monkeypatch.setattr(sec, "download_html", _should_not_call)

        accession, success = sec.download_filing("AAPL", "0000320193", filing, tmp_path)
        assert success is True

    def test_failed_accession_skipped_on_retry(self, tmp_path, monkeypatch):
        """Once an accession fails, subsequent calls should skip it."""
        monkeypatch.setattr(sec, "download_html", lambda url: (None, "HTTP 500"))

        filing = self._sample_filing()

        # First call fails and adds to _failed_accessions
        _, success1 = sec.download_filing("AAPL", "0000320193", filing, tmp_path)
        assert success1 is False

        # Second call should also fail immediately (skipped via _failed_accessions)
        _, success2 = sec.download_filing("AAPL", "0000320193", filing, tmp_path)
        assert success2 is False

    def test_amendment_form_sanitized_in_filename(self, tmp_path, monkeypatch):
        """10-K/A form should have slash replaced with dash in filename."""
        html = "<p>Amendment content</p>"
        monkeypatch.setattr(sec, "download_html", lambda url: (html, None))

        filing = self._sample_filing()
        filing["form"] = "10-K/A"
        sec.download_filing("AAPL", "0000320193", filing, tmp_path)

        txt_files = list(tmp_path.rglob("*.txt"))
        assert len(txt_files) == 1
        assert "10-K-A" in txt_files[0].name
        assert "/" not in txt_files[0].name


# ==========================
# TestFilterFilingsNonStandardFYE
# ==========================

class TestFilterFilingsNonStandardFYE:
    """Unit tests for filter_filings() with non-standard fiscal year-end periods.

    Verifies that reportDate-based quarter mapping works correctly for
    companies whose fiscal years don't end on Dec 31:
      NVDA (Jan 26), WMT (Jan 31), AAPL (Sep 27), COST (Aug 31), MSFT (Jun 30)
    """

    def test_nvda_10k_jan_fye_maps_to_q1(self):
        filings = _make_filings(
            form=["10-K"],
            filingDate=["2025-02-26"],
            accessionNumber=["nvda-10k-fy25"],
            primaryDocument=["nvda-20250126.htm"],
            reportDate=["2025-01-26"],
        )
        results = sec.filter_filings(filings, [(2025, "Q1")], {"10-K"})
        assert len(results) == 1
        assert results[0]["matched_year"] == 2025
        assert results[0]["matched_quarter"] == "Q1"

    def test_wmt_10k_jan_fye_maps_to_q1(self):
        filings = _make_filings(
            form=["10-K"],
            filingDate=["2025-03-14"],
            accessionNumber=["wmt-10k-fy25"],
            primaryDocument=["wmt-20250131.htm"],
            reportDate=["2025-01-31"],
        )
        results = sec.filter_filings(filings, [(2025, "Q1")], {"10-K"})
        assert len(results) == 1
        assert results[0]["matched_year"] == 2025
        assert results[0]["matched_quarter"] == "Q1"

    def test_aapl_10k_sep_fye_maps_to_q3(self):
        filings = _make_filings(
            form=["10-K"],
            filingDate=["2024-11-01"],
            accessionNumber=["aapl-10k-fy24"],
            primaryDocument=["aapl-20240928.htm"],
            reportDate=["2024-09-28"],
        )
        results = sec.filter_filings(filings, [(2024, "Q3")], {"10-K"})
        assert len(results) == 1
        assert results[0]["matched_year"] == 2024
        assert results[0]["matched_quarter"] == "Q3"

    def test_cost_10k_aug_fye_maps_to_q3(self):
        filings = _make_filings(
            form=["10-K"],
            filingDate=["2024-10-09"],
            accessionNumber=["cost-10k-fy24"],
            primaryDocument=["cost-20240901.htm"],
            reportDate=["2024-09-01"],
        )
        results = sec.filter_filings(filings, [(2024, "Q3")], {"10-K"})
        assert len(results) == 1
        assert results[0]["matched_year"] == 2024
        assert results[0]["matched_quarter"] == "Q3"

    def test_msft_10k_jun_fye_maps_to_q2(self):
        filings = _make_filings(
            form=["10-K"],
            filingDate=["2025-07-30"],
            accessionNumber=["msft-10k-fy25"],
            primaryDocument=["msft-20250630.htm"],
            reportDate=["2025-06-30"],
        )
        results = sec.filter_filings(filings, [(2025, "Q2")], {"10-K"})
        assert len(results) == 1
        assert results[0]["matched_year"] == 2025
        assert results[0]["matched_quarter"] == "Q2"
