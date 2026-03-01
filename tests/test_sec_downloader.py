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

    @pytest.mark.parametrize("quarter,expected", [
        ("Q1", "10-Q"), ("Q2", "10-Q"), ("Q3", "10-Q"), ("Q4", "10-K"),
    ])
    def test_form_type_for_quarter(self, quarter, expected):
        assert sec.form_type_for_quarter(quarter) == expected


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
        targets = [(2024, None, "10-K")]
        result = sec.filter_filings(filings, targets, include_amendments=False)
        assert len(result) == 1
        assert result[0]["form"] == "10-K"

    def test_amendments_included_with_flag(self):
        filings = _make_filings(
            form=["10-K", "10-K/A"],
            filingDate=["2024-11-15", "2024-12-01"],
            accessionNumber=["acc1", "acc2"],
            primaryDocument=["d1.htm", "d2.htm"],
        )
        targets = [(2024, None, "10-K")]
        result = sec.filter_filings(filings, targets, include_amendments=True)
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
        targets = [(2024, None, "10-K")]
        result = sec.filter_filings(filings, targets, include_amendments=True)
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
# TestBuildOutputPath
# ==========================

class TestBuildOutputPath:
    def test_basic_10q(self):
        p = sec.build_output_path("/out", "AAPL", 2024, "Q2", "10-Q", "2024-05-15")
        assert p == Path("/out/AAPL/2024/Q2/10-Q_2024-05-15")

    def test_basic_10k(self):
        p = sec.build_output_path("/out", "MSFT", 2023, "Q4", "10-K", "2023-11-01")
        assert p == Path("/out/MSFT/2023/Q4/10-K_2023-11-01")

    def test_amendment_form_sanitization(self):
        p = sec.build_output_path("/out", "GOOG", 2024, "Q4", "10-K/A", "2024-12-01")
        assert "10-K-A" in p.name
        assert "/" not in p.name

    def test_cross_platform_path(self):
        p = sec.build_output_path("/base", "T", 2024, "Q1", "10-Q", "2024-04-01")
        assert isinstance(p, Path)



# ==========================
# TestFilterFilings
# ==========================

class TestFilterFilings:
    def test_manual_q4_is_10q(self):
        """In manual mode, Q4 target uses 10-Q form type."""
        filings = _make_filings(
            form=["10-Q"],
            filingDate=["2024-11-15"],
            accessionNumber=["acc1"],
            primaryDocument=["d.htm"],
        )
        targets = [(2024, "Q4", "10-Q")]
        result = sec.filter_filings(filings, targets)
        assert len(result) == 1

    def test_last_n_q4_is_10k(self):
        """In --last-n mode, Q4 maps to 10-K."""
        filings = _make_filings(
            form=["10-K"],
            filingDate=["2024-11-15"],
            accessionNumber=["acc1"],
            primaryDocument=["d.htm"],
        )
        targets = [(2024, "Q4", "10-K")]
        result = sec.filter_filings(filings, targets)
        assert len(result) == 1

    def test_annual_matches_any_quarter(self):
        """ANNUAL target (quarter=None) matches 10-K filed in any quarter."""
        filings = _make_filings(
            form=["10-K", "10-K"],
            filingDate=["2024-02-15", "2024-11-15"],
            accessionNumber=["acc1", "acc2"],
            primaryDocument=["d1.htm", "d2.htm"],
        )
        targets = [(2024, None, "10-K")]
        result = sec.filter_filings(filings, targets)
        assert len(result) == 2

    def test_amendments_excluded_by_default(self):
        filings = _make_filings(
            form=["10-Q", "10-Q/A"],
            filingDate=["2024-05-15", "2024-06-01"],
            accessionNumber=["acc1", "acc2"],
            primaryDocument=["d1.htm", "d2.htm"],
        )
        targets = [(2024, "Q2", "10-Q")]
        result = sec.filter_filings(filings, targets, include_amendments=False)
        assert len(result) == 1
        assert result[0]["form"] == "10-Q"

    def test_amendments_included(self):
        filings = _make_filings(
            form=["10-Q", "10-Q/A"],
            filingDate=["2024-05-15", "2024-06-01"],
            accessionNumber=["acc1", "acc2"],
            primaryDocument=["d1.htm", "d2.htm"],
        )
        targets = [(2024, "Q2", "10-Q")]
        result = sec.filter_filings(filings, targets, include_amendments=True)
        assert len(result) == 2

    def test_multiple_targets(self):
        filings = _make_filings(
            form=["10-Q", "10-Q", "10-K"],
            filingDate=["2024-02-15", "2024-05-15", "2024-11-15"],
            accessionNumber=["acc1", "acc2", "acc3"],
            primaryDocument=["d1.htm", "d2.htm", "d3.htm"],
        )
        targets = [(2024, "Q1", "10-Q"), (2024, "Q4", "10-K")]
        result = sec.filter_filings(filings, targets)
        assert len(result) == 2

    def test_unrelated_forms_excluded(self):
        filings = _make_filings(
            form=["8-K", "SC 13G"],
            filingDate=["2024-05-15", "2024-05-20"],
            accessionNumber=["acc1", "acc2"],
            primaryDocument=["d1.htm", "d2.htm"],
        )
        targets = [(2024, "Q2", "10-Q")]
        result = sec.filter_filings(filings, targets)
        assert len(result) == 0

    def test_empty_filings(self):
        filings = {"filings": {"recent": {
            "form": [], "filingDate": [], "accessionNumber": [], "primaryDocument": [],
        }}}
        targets = [(2024, "Q2", "10-Q")]
        result = sec.filter_filings(filings, targets)
        assert result == []

    def test_matched_quarter_field(self):
        filings = _make_filings(
            form=["10-Q"],
            filingDate=["2024-05-15"],
            accessionNumber=["acc1"],
            primaryDocument=["d.htm"],
        )
        targets = [(2024, "Q2", "10-Q")]
        result = sec.filter_filings(filings, targets)
        assert result[0]["matched_quarter"] == "Q2"
        assert result[0]["matched_year"] == 2024


# ==========================
# TestBuildTargetsFromArgs
# ==========================

class TestBuildTargetsFromArgs:
    def test_manual_cross_product(self):
        targets = sec.build_targets_from_args(years=[2023, 2024], quarters=["Q1", "Q2"])
        assert len(targets) == 4
        assert (2023, "Q1", "10-Q") in targets
        assert (2024, "Q2", "10-Q") in targets

    def test_annual_produces_none_quarter(self):
        targets = sec.build_targets_from_args(years=[2024], quarters=["ANNUAL"])
        assert targets == [(2024, None, "10-K")]

    def test_last_n_delegation(self):
        targets = sec.build_targets_from_args(last_n=4)
        assert len(targets) == 4
        # Q4 entry should map to 10-K
        for y, q, f in targets:
            if q == "Q4":
                assert f == "10-K"
            else:
                assert f == "10-Q"

    def test_manual_with_form_types(self):
        targets = sec.build_targets_from_args(
            years=[2024], quarters=["Q1"], form_types=["10-Q", "8-K"],
        )
        assert len(targets) == 2
        assert (2024, "Q1", "10-Q") in targets
        assert (2024, "Q1", "8-K") in targets

    def test_annual_with_form_types(self):
        targets = sec.build_targets_from_args(
            years=[2024], quarters=["ANNUAL"], form_types=["10-K", "8-K"],
        )
        assert len(targets) == 2
        assert (2024, None, "10-K") in targets
        assert (2024, None, "8-K") in targets

    def test_last_n_with_form_types_ignores_quarter_mapping(self):
        targets = sec.build_targets_from_args(last_n=2, form_types=["8-K", "4"])
        assert len(targets) == 4  # 2 quarters x 2 forms
        forms = {f for _, _, f in targets}
        assert forms == {"8-K", "4"}

    def test_form_types_none_preserves_default(self):
        targets = sec.build_targets_from_args(
            years=[2024], quarters=["Q1"], form_types=None,
        )
        assert targets == [(2024, "Q1", "10-Q")]


# ==========================
# TestMatchesForm
# ==========================

class TestMatchesForm:
    def test_exact_match(self):
        assert sec.matches_form("10-K", ["10-K"]) is True

    def test_exact_mismatch(self):
        assert sec.matches_form("10-Q", ["10-K"]) is False

    def test_prefix_match(self):
        assert sec.matches_form("424B2", ["424B*"]) is True
        assert sec.matches_form("424B3", ["424B*"]) is True
        assert sec.matches_form("424B5", ["424B*"]) is True

    def test_prefix_no_match(self):
        assert sec.matches_form("10-K", ["424B*"]) is False

    def test_case_insensitive(self):
        assert sec.matches_form("10-k", ["10-K"]) is True
        assert sec.matches_form("sc 13d", ["SC 13D"]) is True

    def test_whitespace_trimmed(self):
        assert sec.matches_form("  10-K  ", ["10-K"]) is True
        assert sec.matches_form("10-K", ["  10-K  "]) is True

    def test_multiple_patterns(self):
        patterns = ["10-K", "10-Q", "8-K"]
        assert sec.matches_form("8-K", patterns) is True
        assert sec.matches_form("SC 13D", patterns) is False

    def test_prefix_with_multiple_patterns(self):
        patterns = ["10-K", "424B*"]
        assert sec.matches_form("424B2", patterns) is True
        assert sec.matches_form("10-K", patterns) is True
        assert sec.matches_form("8-K", patterns) is False


# ==========================
# TestResolveFormTypes
# ==========================

class TestResolveFormTypes:
    def test_forms_arg_takes_precedence(self):
        result = sec.resolve_form_types(forms_arg=["10-Q", "8-K"], bundle="core")
        assert result == ["10-Q", "8-K"]

    def test_bundle_core(self):
        result = sec.resolve_form_types(bundle="core")
        assert result == sec.CORE_FORMS

    def test_neither_returns_none(self):
        result = sec.resolve_form_types()
        assert result is None

    def test_core_forms_has_expected_entries(self):
        assert "10-K" in sec.CORE_FORMS
        assert "10-Q" in sec.CORE_FORMS
        assert "8-K" in sec.CORE_FORMS
        assert "424B*" in sec.CORE_FORMS

    def test_core_bundle_is_copy(self):
        """resolve_form_types returns a copy, not the original list."""
        result = sec.resolve_form_types(bundle="core")
        result.append("EXTRA")
        assert "EXTRA" not in sec.CORE_FORMS


# ==========================
# TestFilterWithFormMatching
# ==========================

class TestFilterWithFormMatching:
    def test_prefix_wildcard_in_target(self):
        """424B* target should match 424B2 filings."""
        filings = _make_filings(
            form=["424B2", "424B5", "10-K"],
            filingDate=["2024-05-15", "2024-06-01", "2024-11-15"],
            accessionNumber=["acc1", "acc2", "acc3"],
            primaryDocument=["d1.htm", "d2.htm", "d3.htm"],
        )
        targets = [(2024, "Q2", "424B*")]
        result = sec.filter_filings(filings, targets)
        assert len(result) == 2
        forms = [r["form"] for r in result]
        assert "424B2" in forms
        assert "424B5" in forms

    def test_case_insensitive_matching(self):
        filings = _make_filings(
            form=["sc 13d"],
            filingDate=["2024-05-15"],
            accessionNumber=["acc1"],
            primaryDocument=["d.htm"],
        )
        targets = [(2024, "Q2", "SC 13D")]
        result = sec.filter_filings(filings, targets)
        assert len(result) == 1


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
        html = "<p>AT&amp;T &lt;10&gt; &#x27;quoted&#x27;</p>"
        result = sec.clean_html_to_text(html)
        assert "AT&T" in result
        assert "<10>" in result

    def test_empty_html(self):
        assert sec.clean_html_to_text("") == ""

    def test_nested_skip_tags(self):
        """HTMLParser treats <script> as CDATA — inner tags aren't parsed.
        Only the first </script> closes the block, matching browser behavior."""
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
# TestExtractSections
# ==========================

class TestExtractSections:
    def test_10k_mda_extraction(self):
        text = (
            "Blah blah\n"
            "Item 7. Management's Discussion and Analysis\n"
            "This is the MD&A section content.\n"
            "Item 8. Financial Statements\n"
            "Numbers here."
        )
        sections = sec.extract_sections("10-K", text)
        assert "MDA" in sections
        assert sections["MDA"] is not None
        assert "MD&A section content" in sections["MDA"]
        assert "Numbers here" not in sections["MDA"]

    def test_10q_mda_uses_item_2(self):
        text = (
            "Item 2. Management's Discussion and Analysis\n"
            "Quarterly discussion content.\n"
            "Item 3. Quantitative Disclosures\n"
        )
        sections = sec.extract_sections("10-Q", text)
        assert sections["MDA"] is not None
        assert "Quarterly discussion" in sections["MDA"]

    def test_risk_factors_extraction(self):
        text = (
            "Item 1A. Risk Factors\n"
            "We face significant risks.\n"
            "Item 2. Properties\n"
        )
        sections = sec.extract_sections("10-K", text)
        assert "RISK_FACTORS" in sections
        assert sections["RISK_FACTORS"] is not None
        assert "significant risks" in sections["RISK_FACTORS"]

    def test_8k_returns_full_body(self):
        text = "Full 8-K filing body text here."
        sections = sec.extract_sections("8-K", text)
        assert "BODY" in sections
        assert sections["BODY"] == text

    def test_8k_amendment_returns_body(self):
        text = "Amended 8-K body."
        sections = sec.extract_sections("8-K/A", text)
        assert "BODY" in sections
        assert sections["BODY"] == text

    def test_missing_section_returns_none(self):
        text = "This filing has no item headings at all."
        sections = sec.extract_sections("10-K", text)
        assert sections["MDA"] is None
        assert sections["RISK_FACTORS"] is None

    def test_section_at_end_of_text(self):
        """Section at the very end (no following Item heading) should still be captured."""
        text = (
            "Item 1A. Risk Factors\n"
            "Risks discussed here without a following item heading."
        )
        sections = sec.extract_sections("10-K", text)
        assert sections["RISK_FACTORS"] is not None
        assert "Risks discussed here" in sections["RISK_FACTORS"]

    def test_case_insensitive_headings(self):
        text = (
            "ITEM 1A. RISK FACTORS\n"
            "Risks in uppercase heading.\n"
            "ITEM 2. PROPERTIES\n"
        )
        sections = sec.extract_sections("10-K", text)
        assert sections["RISK_FACTORS"] is not None


# ==========================
# TestFormatTextOutput
# ==========================

class TestFormatTextOutput:
    def test_basic_format(self):
        sections = {"MDA": "Discussion content.", "RISK_FACTORS": "Risk content."}
        result = sec.format_text_output("10-K", "2024-11-01", "acc123", sections)
        assert "FORM: 10-K" in result
        assert "FILING_DATE: 2024-11-01" in result
        assert "ACCESSION: acc123" in result
        assert "==== SECTION: MDA ====" in result
        assert "Discussion content." in result
        assert "==== SECTION: RISK_FACTORS ====" in result
        assert "Risk content." in result

    def test_missing_section_placeholder(self):
        sections = {"MDA": None, "RISK_FACTORS": "Has content."}
        result = sec.format_text_output("10-K", "2024-11-01", "acc123", sections)
        assert "[SECTION NOT FOUND]" in result
        assert "Has content." in result

    def test_8k_body_section(self):
        sections = {"BODY": "Full 8-K text."}
        result = sec.format_text_output("8-K", "2024-06-15", "acc456", sections)
        assert "FORM: 8-K" in result
        assert "==== SECTION: BODY ====" in result
        assert "Full 8-K text." in result

    def test_all_sections_missing(self):
        sections = {"MDA": None, "RISK_FACTORS": None}
        result = sec.format_text_output("10-K", "2024-11-01", "acc123", sections)
        assert result.count("[SECTION NOT FOUND]") == 2


# ==========================
# TestDownloadFiling
# ==========================

class TestDownloadFiling:
    """Tests for the download_filing text extraction pipeline.
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
        html = "<p>Item 1A. Risk Factors</p><p>We face risks.</p><p>Item 2. Properties</p>"
        monkeypatch.setattr(sec, "download_html", lambda url: html)
        monkeypatch.setattr(sec, "rate_limit", lambda: None)

        filing = self._sample_filing()
        accession, success = sec.download_filing("AAPL", "0000320193", filing, tmp_path)
        assert success is True
        assert accession == filing["accession"]

        # Verify .txt file was created
        txt_files = list(tmp_path.rglob("*.txt"))
        assert len(txt_files) == 1
        content = txt_files[0].read_text()
        assert "FORM: 10-K" in content
        assert "We face risks" in content

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

        expected = tmp_path / "AAPL" / "2024" / "Q4" / "10-K_2024-11-01.txt"
        assert expected.exists()

    def test_no_tmp_file_leftover(self, tmp_path, monkeypatch):
        html = "<p>Content</p>"
        monkeypatch.setattr(sec, "download_html", lambda url: html)
        monkeypatch.setattr(sec, "rate_limit", lambda: None)

        filing = self._sample_filing()
        sec.download_filing("AAPL", "0000320193", filing, tmp_path)

        tmp_files = list(tmp_path.rglob("*.tmp"))
        assert len(tmp_files) == 0
