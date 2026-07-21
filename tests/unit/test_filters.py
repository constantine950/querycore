"""
test_filters.py

Tests for src/search/filters.py

Run with:  python -m pytest tests/unit/test_filters.py -v
"""

import json
from datetime import date
from pathlib import Path

import pytest

from src.indexer.inverted_index import InvertedIndex
from src.search.filters import (
    CategoryFilter, DateRangeFilter, WordCountFilter, FilterSet
)
from src.search.query_parser import QueryParser
from src.search.ranking import Ranker, SearchResult
from src.search.retrieval import Retriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def corpus():
    return [
        {"id": "doc_001", "title": "Search Engine",        "body": "A search engine indexes documents for fast retrieval.",
            "category": "computer_science", "date": "2024-01-15", "word_count": 9},
        {"id": "doc_002", "title": "Machine Learning",     "body": "Machine learning algorithms improve classification of data.",
            "category": "computer_science", "date": "2023-06-01", "word_count": 8},
        {"id": "doc_003", "title": "Black Hole",           "body": "A black hole is a region of spacetime with extreme gravity.",
            "category": "science",          "date": "2024-03-20", "word_count": 13},
        {"id": "doc_004", "title": "Ancient Rome",         "body": "Ancient Rome was a civilization that emerged from Italy.",
            "category": "history",          "date": "2022-11-05", "word_count": 11},
        {"id": "doc_005", "title": "Inverted Index",       "body": "The inverted index maps terms to documents efficiently.",
            "category": "computer_science", "date": "2024-06-30", "word_count": 8},
        {"id": "doc_006", "title": "DNA Structure",        "body": "DNA is a polymer composed of two polynucleotide chains.",
            "category": "science",          "date": "2023-09-14", "word_count": 10},
        {"id": "doc_007", "title": "World War II",         "body": "World War II was a global conflict from 1939 to 1945.",
            "category": "history",          "date": "2021-07-04", "word_count": 12},
    ]


@pytest.fixture(scope="module")
def idx(corpus):
    index = InvertedIndex()
    index.build(corpus)
    return index


@pytest.fixture(scope="module")
def fs(idx):
    return FilterSet(idx)


@pytest.fixture
def fresh_fs(idx):
    """Fresh FilterSet per test so filters don't leak between tests."""
    return FilterSet(idx)


def make_result(doc_id, score, category, date_str, snippet="some text here and more"):
    return SearchResult(
        doc_id=doc_id, score=score, title=doc_id,
        snippet=snippet, category=category, date=date_str,
    )


ALL_IDS = {"doc_001", "doc_002", "doc_003",
           "doc_004", "doc_005", "doc_006", "doc_007"}


# ---------------------------------------------------------------------------
# CategoryFilter
# ---------------------------------------------------------------------------

class TestCategoryFilter:
    def test_matches_correct_category(self):
        f = CategoryFilter(categories={"computer_science"})
        assert f.matches_meta({"category": "computer_science"}) is True

    def test_rejects_wrong_category(self):
        f = CategoryFilter(categories={"computer_science"})
        assert f.matches_meta({"category": "science"}) is False

    def test_multiple_categories(self):
        f = CategoryFilter(categories={"science", "history"})
        assert f.matches_meta({"category": "science"}) is True
        assert f.matches_meta({"category": "history"}) is True
        assert f.matches_meta({"category": "computer_science"}) is False

    def test_empty_category_field(self):
        f = CategoryFilter(categories={"computer_science"})
        assert f.matches_meta({}) is False


# ---------------------------------------------------------------------------
# DateRangeFilter
# ---------------------------------------------------------------------------

class TestDateRangeFilter:
    def test_within_range(self):
        f = DateRangeFilter(start=date(2024, 1, 1), end=date(2024, 12, 31))
        assert f.matches_meta({"date": "2024-06-15"}) is True

    def test_before_start(self):
        f = DateRangeFilter(start=date(2024, 1, 1))
        assert f.matches_meta({"date": "2023-12-31"}) is False

    def test_after_end(self):
        f = DateRangeFilter(end=date(2023, 12, 31))
        assert f.matches_meta({"date": "2024-01-01"}) is False

    def test_on_start_boundary(self):
        f = DateRangeFilter(start=date(2024, 1, 15))
        assert f.matches_meta({"date": "2024-01-15"}) is True

    def test_on_end_boundary(self):
        f = DateRangeFilter(end=date(2024, 6, 30))
        assert f.matches_meta({"date": "2024-06-30"}) is True

    def test_no_bounds_matches_all(self):
        f = DateRangeFilter()
        assert f.matches_meta({"date": "2020-01-01"}) is True

    def test_only_start_bound(self):
        f = DateRangeFilter(start=date(2024, 1, 1))
        assert f.matches_meta({"date": "2025-01-01"}) is True
        assert f.matches_meta({"date": "2023-01-01"}) is False

    def test_only_end_bound(self):
        f = DateRangeFilter(end=date(2023, 12, 31))
        assert f.matches_meta({"date": "2023-01-01"}) is True
        assert f.matches_meta({"date": "2024-01-01"}) is False

    def test_missing_date_field(self):
        f = DateRangeFilter(start=date(2024, 1, 1))
        assert f.matches_meta({}) is False

    def test_invalid_date_string(self):
        f = DateRangeFilter(start=date(2024, 1, 1))
        assert f.matches_meta({"date": "not-a-date"}) is False


# ---------------------------------------------------------------------------
# WordCountFilter
# ---------------------------------------------------------------------------

class TestWordCountFilter:
    def test_within_range(self):
        f = WordCountFilter(min_words=5, max_words=15)
        assert f.matches_meta({"word_count": 10}) is True

    def test_below_min(self):
        f = WordCountFilter(min_words=10)
        assert f.matches_meta({"word_count": 5}) is False

    def test_above_max(self):
        f = WordCountFilter(max_words=10)
        assert f.matches_meta({"word_count": 15}) is False

    def test_on_boundaries(self):
        f = WordCountFilter(min_words=5, max_words=10)
        assert f.matches_meta({"word_count": 5}) is True
        assert f.matches_meta({"word_count": 10}) is True

    def test_no_bounds_matches_all(self):
        f = WordCountFilter()
        assert f.matches_meta({"word_count": 999}) is True

    def test_falls_back_to_snippet_length(self):
        f = WordCountFilter(min_words=3)
        # No word_count key — estimates from snippet
        assert f.matches_meta({"snippet": "one two three four five"}) is True
        assert f.matches_meta({"snippet": "one"}) is False


# ---------------------------------------------------------------------------
# FilterSet — apply_to_candidates()
# ---------------------------------------------------------------------------

class TestFilterSetCandidates:
    def test_no_filters_returns_all(self, fresh_fs):
        assert fresh_fs.apply_to_candidates(ALL_IDS) == ALL_IDS

    def test_category_filter_narrows(self, fresh_fs):
        fresh_fs.add_category("science")
        result = fresh_fs.apply_to_candidates(ALL_IDS)
        assert result == {"doc_003", "doc_006"}

    def test_date_range_filter(self, fresh_fs):
        fresh_fs.add_date_range("2024-01-01", "2024-12-31")
        result = fresh_fs.apply_to_candidates(ALL_IDS)
        assert "doc_001" in result   # 2024-01-15
        assert "doc_003" in result   # 2024-03-20
        assert "doc_005" in result   # 2024-06-30
        assert "doc_002" not in result  # 2023-06-01
        assert "doc_004" not in result  # 2022-11-05

    def test_combined_filters_and(self, fresh_fs):
        fresh_fs.add_category("computer_science").add_date_range("2024-01-01")
        result = fresh_fs.apply_to_candidates(ALL_IDS)
        # CS docs in 2024: doc_001 (2024-01-15) and doc_005 (2024-06-30)
        assert result == {"doc_001", "doc_005"}

    def test_empty_candidates(self, fresh_fs):
        fresh_fs.add_category("science")
        assert fresh_fs.apply_to_candidates(set()) == set()

    def test_active_property(self, fresh_fs):
        assert not fresh_fs.active
        fresh_fs.add_category("science")
        assert fresh_fs.active

    def test_clear_removes_filters(self, fresh_fs):
        fresh_fs.add_category("science")
        fresh_fs.clear()
        assert not fresh_fs.active
        assert fresh_fs.apply_to_candidates(ALL_IDS) == ALL_IDS

    def test_chaining(self, idx):
        result = (
            FilterSet(idx)
            .add_category("history")
            .add_date_range("2021-01-01", "2023-01-01")
            .apply_to_candidates(ALL_IDS)
        )
        # history docs: doc_004 (2022-11-05) ✓, doc_007 (2021-07-04) ✓
        assert "doc_004" in result
        assert "doc_007" in result
        assert "doc_003" not in result   # science

    def test_string_dates_accepted(self, idx):
        fs = FilterSet(idx).add_date_range("2024-01-01", "2024-12-31")
        result = fs.apply_to_candidates(ALL_IDS)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# FilterSet — apply_to_results()
# ---------------------------------------------------------------------------

class TestFilterSetResults:
    def test_no_filters_returns_all(self, fresh_fs):
        results = [make_result("doc_001", 1.0, "cs", "2024-01-01")]
        assert fresh_fs.apply_to_results(results) == results

    def test_category_filter_on_results(self, fresh_fs):
        fresh_fs.add_category("science")
        results = [
            make_result("doc_003", 0.9, "science", "2024-03-20"),
            make_result("doc_001", 0.5, "computer_science", "2024-01-15"),
        ]
        filtered = fresh_fs.apply_to_results(results)
        assert len(filtered) == 1
        assert filtered[0].doc_id == "doc_003"

    def test_order_preserved(self, fresh_fs):
        fresh_fs.add_category("computer_science")
        results = [
            make_result("doc_001", 0.9, "computer_science", "2024-01-15"),
            make_result("doc_005", 0.5, "computer_science", "2024-06-30"),
        ]
        filtered = fresh_fs.apply_to_results(results)
        assert [r.doc_id for r in filtered] == ["doc_001", "doc_005"]

    def test_empty_results(self, fresh_fs):
        fresh_fs.add_category("science")
        assert fresh_fs.apply_to_results([]) == []


# ---------------------------------------------------------------------------
# Integration: full pipeline with filters
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_filter_before_ranking(self, idx):
        qp = QueryParser()
        ret = Retriever(idx)
        rnk = Ranker(idx)

        pq = qp.parse("search index")
        candidates = ret.retrieve(pq)

        fs = FilterSet(idx).add_category("computer_science")
        narrowed = fs.apply_to_candidates(candidates)
        results = rnk.rank(pq, narrowed)

        assert all(r.category == "computer_science" for r in results)

    def test_full_dataset_category_filter(self):
        dataset = Path(__file__).parent.parent.parent / \
            "data" / "sample_documents.json"
        if not dataset.exists():
            pytest.skip("Dataset not present")
        with open(dataset) as f:
            docs = json.load(f)

        idx = InvertedIndex()
        idx.build(docs)
        all_ids = {d["id"] for d in docs}

        fs = FilterSet(idx).add_category("science")
        result = fs.apply_to_candidates(all_ids)
        science_ids = {d["id"] for d in docs if d["category"] == "science"}
        assert result == science_ids
