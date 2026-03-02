"""Tests for SEC EDGAR downloader (data-pipeline/EDGAR/get_sec_data.py)."""

import json
import sys
from datetime import date, datetime, timezone
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

    def test_quarter_from_month_invalid(self):
        with pytest.raises(ValueError):
            sec.quarter_from_month(0)
        with pytest.raises(ValueError):
            sec.quarter_from_month(13)


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
# TestBuildHtmlPath
# ==========================

class TestBuildHtmlPath:
    def test_basic(self):
        p = sec.build_html_path(Path("/out"), "AAPL", 2024, "Q2", "10-Q", "2024-05-02")
        assert p == Path("/out/raw_html/AAPL/AAPL_2024_Q2_10-Q_2024-05-02.html")

    def test_amendment_sanitized(self):
        p = sec.build_html_path(Path("/out"), "NVDA", 2024, "Q4", "10-K/A", "2024-11-15")
        assert "10-K-A" in p.name
        assert "/" not in p.name

    def test_returns_path(self):
        p = sec.build_html_path(Path("/base"), "T", 2024, "Q1", "8-K", "2024-03-01")
        assert isinstance(p, Path)


# ==========================
# TestBuildTextPath
# ==========================

class TestBuildTextPath:
    def test_basic(self):
        p = sec.build_text_path(Path("/out"), "AAPL", 2024, "Q2", "10-Q", "2024-05-02")
        assert p == Path("/out/clean_text/AAPL/AAPL_2024_Q2_10-Q_2024-05-02.txt")

    def test_different_ticker(self):
        p = sec.build_text_path(Path("/data"), "MSFT", 2023, "Q4", "10-K", "2023-11-01")
        assert p == Path("/data/clean_text/MSFT/MSFT_2023_Q4_10-K_2023-11-01.txt")

    def test_returns_path(self):
        p = sec.build_text_path(Path("/base"), "T", 2024, "Q1", "8-K", "2024-03-01")
        assert isinstance(p, Path)


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


# ==========================
# TestBuildTargetsFromArgs
# ==========================

class TestBuildTargetsFromArgs:
    def test_manual_cross_product(self):
        targets = sec.build_targets_from_args(years=[2023, 2024], quarters=["Q1", "Q2"])
        assert len(targets) == 4
        assert (2023, "Q1") in targets
        assert (2024, "Q2") in targets

    def test_annual_produces_none_quarter(self):
        targets = sec.build_targets_from_args(years=[2024], quarters=["ANNUAL"])
        assert targets == [(2024, None)]

    def test_last_n_delegation(self):
        targets = sec.build_targets_from_args(last_n=4)
        assert len(targets) == 4
        # All entries are (year, quarter) 2-tuples
        for entry in targets:
            assert len(entry) == 2

    def test_last_n_returns_year_quarter_pairs(self):
        targets = sec.build_targets_from_args(last_n=2)
        assert len(targets) == 2
        for y, q in targets:
            assert isinstance(y, int)
            assert q.startswith("Q")


# ==========================
# TestLoadIndex
# ==========================

class TestLoadIndex:
    def test_missing_file(self, tmp_path):
        result = sec.load_index(tmp_path / "_index.json")
        assert result == {"cik": "", "last_updated": "", "filings": {}}

    def test_valid_index(self, tmp_path):
        index_path = tmp_path / "_index.json"
        data = {
            "cik": "0000320193",
            "last_updated": "2024-01-01T00:00:00+00:00",
            "filings": {
                "acc1": {
                    "form": "10-K",
                    "filing_date": "2024-11-01",
                    "primary_doc": "d.htm",
                    "text_saved": True,
                    "last_downloaded": "2024-12-01T00:00:00+00:00",
                }
            },
        }
        index_path.write_text(json.dumps(data))
        result = sec.load_index(index_path)
        assert result["cik"] == "0000320193"
        assert "acc1" in result["filings"]
        assert result["filings"]["acc1"]["text_saved"] is True

    def test_corrupted_json(self, tmp_path):
        index_path = tmp_path / "_index.json"
        index_path.write_text("{{not valid json")
        result = sec.load_index(index_path)
        assert result == {"cik": "", "last_updated": "", "filings": {}}

    def test_missing_filings_key(self, tmp_path):
        index_path = tmp_path / "_index.json"
        index_path.write_text(json.dumps({"cik": "123"}))
        result = sec.load_index(index_path)
        assert result == {"cik": "", "last_updated": "", "filings": {}}

    def test_filings_not_dict(self, tmp_path):
        index_path = tmp_path / "_index.json"
        index_path.write_text(json.dumps({"filings": ["not", "a", "dict"]}))
        result = sec.load_index(index_path)
        assert result == {"cik": "", "last_updated": "", "filings": {}}


# ==========================
# TestSaveIndex
# ==========================

class TestSaveIndex:
    def test_round_trip(self, tmp_path):
        index_path = tmp_path / "_index.json"
        data = {
            "cik": "0000320193",
            "last_updated": "2024-01-01T00:00:00+00:00",
            "filings": {"acc1": {"text_saved": True}},
        }
        sec.save_index(index_path, data)
        assert index_path.exists()
        loaded = json.loads(index_path.read_text())
        assert loaded == data

    def test_atomic_write_no_tmp_leftover(self, tmp_path):
        index_path = tmp_path / "_index.json"
        sec.save_index(index_path, {"cik": "", "last_updated": "", "filings": {}})
        tmp_file = index_path.with_suffix(".json.tmp")
        assert not tmp_file.exists()

    def test_creates_parent_dirs(self, tmp_path):
        index_path = tmp_path / "sub" / "dir" / "_index.json"
        sec.save_index(index_path, {"cik": "", "last_updated": "", "filings": {}})
        assert index_path.exists()


# ==========================
# TestScheduleDownloads
# ==========================

class TestScheduleDownloads:
    def _sample_filings(self):
        return [
            {"accession": "acc1", "form": "10-K", "filing_date": "2024-11-01",
             "primary_doc": "d1.htm", "matched_year": 2024, "matched_quarter": "Q4"},
            {"accession": "acc2", "form": "10-Q", "filing_date": "2024-05-15",
             "primary_doc": "d2.htm", "matched_year": 2024, "matched_quarter": "Q2"},
            {"accession": "acc3", "form": "10-Q", "filing_date": "2024-08-15",
             "primary_doc": "d3.htm", "matched_year": 2024, "matched_quarter": "Q3"},
        ]

    def test_default_skips_indexed(self):
        filtered = self._sample_filings()
        index = {
            "cik": "123", "last_updated": "", "filings": {
                "acc1": {"text_saved": True},
            },
        }
        result = sec.schedule_downloads(filtered, index)
        assert len(result) == 2
        accessions = [f["accession"] for f in result]
        assert "acc1" not in accessions
        assert "acc2" in accessions
        assert "acc3" in accessions

    def test_default_downloads_not_downloaded(self):
        """Accession in index but downloaded=False should be re-scheduled."""
        filtered = self._sample_filings()[:1]
        index = {
            "cik": "123", "last_updated": "", "filings": {
                "acc1": {"text_saved": False},
            },
        }
        result = sec.schedule_downloads(filtered, index)
        assert len(result) == 1

    def test_force_refresh_downloads_all(self):
        filtered = self._sample_filings()
        index = {
            "cik": "123", "last_updated": "", "filings": {
                "acc1": {"text_saved": True},
                "acc2": {"text_saved": True},
                "acc3": {"text_saved": True},
            },
        }
        result = sec.schedule_downloads(filtered, index, force_refresh=True)
        assert len(result) == 3

    def test_no_cache_downloads_all(self):
        filtered = self._sample_filings()
        index = {
            "cik": "123", "last_updated": "", "filings": {
                "acc1": {"text_saved": True},
                "acc2": {"text_saved": True},
            },
        }
        result = sec.schedule_downloads(filtered, index, no_cache=True)
        assert len(result) == 3

    def test_empty_index_downloads_all(self):
        filtered = self._sample_filings()
        index = {"cik": "", "last_updated": "", "filings": {}}
        result = sec.schedule_downloads(filtered, index)
        assert len(result) == 3

    def test_empty_filtered_downloads_none(self):
        index = {"cik": "", "last_updated": "", "filings": {}}
        result = sec.schedule_downloads([], index)
        assert result == []

    def test_all_indexed_downloads_none(self):
        filtered = self._sample_filings()
        index = {
            "cik": "123", "last_updated": "", "filings": {
                "acc1": {"text_saved": True},
                "acc2": {"text_saved": True},
                "acc3": {"text_saved": True},
            },
        }
        result = sec.schedule_downloads(filtered, index)
        assert result == []


# ==========================
# TestSubmissionsCachePolicy
# ==========================

class TestSubmissionsCachePolicy:
    """Tests for get_company_filings_cached cache read/write logic.
    These test the caching layer only — no SEC API calls."""

    def _write_cache(self, path, data, age_seconds=0):
        fetched_at = datetime.now(timezone.utc)
        if age_seconds:
            from datetime import timedelta
            fetched_at = fetched_at - timedelta(seconds=age_seconds)
        wrapper = {"fetched_at": fetched_at.isoformat(), "data": data}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(wrapper))

    def test_fresh_cache_is_used(self, tmp_path, monkeypatch):
        cache_path = tmp_path / "_submissions.json"
        fake_data = {"filings": {"recent": {"form": []}}}
        self._write_cache(cache_path, fake_data, age_seconds=100)

        # Monkeypatch get_company_filings to fail if called
        def _should_not_call(cik):
            raise AssertionError("Should not fetch from SEC when cache is fresh")
        monkeypatch.setattr(sec, "get_company_filings", _should_not_call)

        result = sec.get_company_filings_cached("0000320193", cache_path)
        assert result == fake_data

    def test_stale_cache_triggers_fetch(self, tmp_path, monkeypatch):
        cache_path = tmp_path / "_submissions.json"
        stale_data = {"filings": {"recent": {"form": ["OLD"]}}}
        self._write_cache(cache_path, stale_data, age_seconds=90000)

        fresh_data = {"filings": {"recent": {"form": ["FRESH"]}}}
        monkeypatch.setattr(sec, "get_company_filings", lambda cik: fresh_data)

        result = sec.get_company_filings_cached("0000320193", cache_path)
        assert result == fresh_data

    def test_force_refresh_bypasses_cache(self, tmp_path, monkeypatch):
        cache_path = tmp_path / "_submissions.json"
        cached_data = {"filings": {"recent": {"form": ["CACHED"]}}}
        self._write_cache(cache_path, cached_data, age_seconds=10)

        fresh_data = {"filings": {"recent": {"form": ["FRESH"]}}}
        monkeypatch.setattr(sec, "get_company_filings", lambda cik: fresh_data)

        result = sec.get_company_filings_cached(
            "0000320193", cache_path, force_refresh=True,
        )
        assert result == fresh_data

    def test_force_refresh_writes_cache(self, tmp_path, monkeypatch):
        cache_path = tmp_path / "_submissions.json"
        fresh_data = {"filings": {"recent": {"form": ["NEW"]}}}
        monkeypatch.setattr(sec, "get_company_filings", lambda cik: fresh_data)

        sec.get_company_filings_cached(
            "0000320193", cache_path, force_refresh=True,
        )
        assert cache_path.exists()
        wrapper = json.loads(cache_path.read_text())
        assert wrapper["data"] == fresh_data

    def test_no_cache_bypasses_read(self, tmp_path, monkeypatch):
        cache_path = tmp_path / "_submissions.json"
        cached_data = {"filings": {"recent": {"form": ["CACHED"]}}}
        self._write_cache(cache_path, cached_data, age_seconds=10)

        fresh_data = {"filings": {"recent": {"form": ["FRESH"]}}}
        monkeypatch.setattr(sec, "get_company_filings", lambda cik: fresh_data)

        result = sec.get_company_filings_cached(
            "0000320193", cache_path, no_cache=True,
        )
        assert result == fresh_data

    def test_no_cache_does_not_write(self, tmp_path, monkeypatch):
        cache_path = tmp_path / "_submissions.json"
        monkeypatch.setattr(
            sec, "get_company_filings",
            lambda cik: {"filings": {"recent": {"form": []}}},
        )

        sec.get_company_filings_cached(
            "0000320193", cache_path, no_cache=True,
        )
        assert not cache_path.exists()

    def test_corrupted_cache_refetches(self, tmp_path, monkeypatch):
        cache_path = tmp_path / "_submissions.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("corrupted{{{")

        fresh_data = {"filings": {"recent": {"form": ["FRESH"]}}}
        monkeypatch.setattr(sec, "get_company_filings", lambda cik: fresh_data)

        result = sec.get_company_filings_cached("0000320193", cache_path)
        assert result == fresh_data


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

    def test_successful_extraction(self, tmp_path, monkeypatch):
        html = "<p>We face significant risks in our business.</p>"
        monkeypatch.setattr(sec, "download_html", lambda url: html)
        monkeypatch.setattr(sec, "rate_limit", lambda: None)

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
        monkeypatch.setattr(sec, "download_html", lambda url: None)
        monkeypatch.setattr(sec, "rate_limit", lambda: None)

        filing = self._sample_filing()
        accession, success = sec.download_filing("AAPL", "0000320193", filing, tmp_path)
        assert success is False
        assert accession == filing["accession"]

    def test_output_path_structure(self, tmp_path, monkeypatch):
        html = "<p>Simple content</p>"
        monkeypatch.setattr(sec, "download_html", lambda url: html)
        monkeypatch.setattr(sec, "rate_limit", lambda: None)

        filing = self._sample_filing()
        sec.download_filing("AAPL", "0000320193", filing, tmp_path)

        expected_html = tmp_path / "raw_html" / "AAPL" / "AAPL_2024_Q4_10-K_2024-11-01.html"
        expected_txt = tmp_path / "clean_text" / "AAPL" / "AAPL_2024_Q4_10-K_2024-11-01.txt"
        assert expected_html.exists()
        assert expected_txt.exists()

    def test_html_saved_unchanged(self, tmp_path, monkeypatch):
        original_html = "<html><body><p>Original content</p></body></html>"
        monkeypatch.setattr(sec, "download_html", lambda url: original_html)
        monkeypatch.setattr(sec, "rate_limit", lambda: None)

        filing = self._sample_filing()
        sec.download_filing("AAPL", "0000320193", filing, tmp_path)

        html_path = tmp_path / "raw_html" / "AAPL" / "AAPL_2024_Q4_10-K_2024-11-01.html"
        assert html_path.read_text() == original_html

    def test_no_tmp_file_leftover(self, tmp_path, monkeypatch):
        html = "<p>Content</p>"
        monkeypatch.setattr(sec, "download_html", lambda url: html)
        monkeypatch.setattr(sec, "rate_limit", lambda: None)

        filing = self._sample_filing()
        sec.download_filing("AAPL", "0000320193", filing, tmp_path)

        tmp_files = list(tmp_path.rglob("*.tmp"))
        assert len(tmp_files) == 0

    def test_no_section_markers_in_output(self, tmp_path, monkeypatch):
        """Output should be full text, no ==== SECTION ==== markers."""
        html = "<p>Item 1A. Risk Factors</p><p>Risks here.</p>"
        monkeypatch.setattr(sec, "download_html", lambda url: html)
        monkeypatch.setattr(sec, "rate_limit", lambda: None)

        filing = self._sample_filing()
        sec.download_filing("AAPL", "0000320193", filing, tmp_path)

        txt_files = list(tmp_path.rglob("*.txt"))
        content = txt_files[0].read_text()
        assert "====" not in content
        assert "SECTION" not in content
