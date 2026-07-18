"""
test_paginator.py

Tests for src/search/paginator.py

Run with:  python -m pytest tests/unit/test_paginator.py -v
"""

import pytest
from src.search.ranking import SearchResult
from src.search.paginator import Paginator, Page, SortBy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_results(n: int) -> list[SearchResult]:
    """Create n SearchResult objects with descending scores and varying dates."""
    dates = ["2024-01-01", "2024-06-15",
             "2023-03-10", "2024-12-01", "2022-07-04"]
    titles = ["Zebra Doc", "Alpha Doc", "Mango Doc", "Beta Doc", "Omega Doc"]
    return [
        SearchResult(
            doc_id=f"doc_{i:03d}",
            score=float(n - i),   # doc_000 has highest score
            title=titles[i % len(titles)],
            snippet=f"snippet {i}",
            category="cs",
            date=dates[i % len(dates)],
        )
        for i in range(n)
    ]


@pytest.fixture
def results_25():
    return make_results(25)


@pytest.fixture
def paginator():
    return Paginator(page_size=10)


# ---------------------------------------------------------------------------
# Basic pagination
# ---------------------------------------------------------------------------

class TestBasicPagination:
    def test_first_page_has_10_results(self, paginator, results_25):
        page = paginator.paginate(results_25, page=1)
        assert len(page.results) == 10

    def test_second_page_has_10_results(self, paginator, results_25):
        page = paginator.paginate(results_25, page=2)
        assert len(page.results) == 10

    def test_third_page_has_5_results(self, paginator, results_25):
        page = paginator.paginate(results_25, page=3)
        assert len(page.results) == 5

    def test_total_is_correct(self, paginator, results_25):
        page = paginator.paginate(results_25, page=1)
        assert page.total == 25

    def test_total_pages_is_correct(self, paginator, results_25):
        page = paginator.paginate(results_25, page=1)
        assert page.total_pages == 3

    def test_page_number_stored(self, paginator, results_25):
        page = paginator.paginate(results_25, page=2)
        assert page.page == 2

    def test_empty_results_returns_empty_page(self, paginator):
        page = paginator.paginate([], page=1)
        assert page.results == []
        assert page.total == 0
        assert page.total_pages == 0
        assert not page.has_next
        assert not page.has_prev

    def test_single_result(self, paginator):
        page = paginator.paginate(make_results(1), page=1)
        assert len(page.results) == 1
        assert page.total_pages == 1


# ---------------------------------------------------------------------------
# Pagination metadata
# ---------------------------------------------------------------------------

class TestPaginationMetadata:
    def test_has_next_true_on_first_page(self, paginator, results_25):
        page = paginator.paginate(results_25, page=1)
        assert page.has_next is True

    def test_has_next_false_on_last_page(self, paginator, results_25):
        page = paginator.paginate(results_25, page=3)
        assert page.has_next is False

    def test_has_prev_false_on_first_page(self, paginator, results_25):
        page = paginator.paginate(results_25, page=1)
        assert page.has_prev is False

    def test_has_prev_true_on_second_page(self, paginator, results_25):
        page = paginator.paginate(results_25, page=2)
        assert page.has_prev is True

    def test_start_index_page_1(self, paginator, results_25):
        page = paginator.paginate(results_25, page=1)
        assert page.start == 1

    def test_end_index_page_1(self, paginator, results_25):
        page = paginator.paginate(results_25, page=1)
        assert page.end == 10

    def test_start_index_page_2(self, paginator, results_25):
        page = paginator.paginate(results_25, page=2)
        assert page.start == 11

    def test_end_index_page_3(self, paginator, results_25):
        page = paginator.paginate(results_25, page=3)
        assert page.end == 25

    def test_to_dict_has_all_keys(self, paginator, results_25):
        page = paginator.paginate(results_25, page=1)
        d = page.to_dict()
        for key in ("page", "page_size", "total", "total_pages",
                    "has_next", "has_prev", "start", "end", "sort_by", "results"):
            assert key in d

    def test_to_dict_results_are_dicts(self, paginator, results_25):
        page = paginator.paginate(results_25, page=1)
        for r in page.to_dict()["results"]:
            assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# Page clamping
# ---------------------------------------------------------------------------

class TestPageClamping:
    def test_page_0_clamped_to_1(self, paginator, results_25):
        page = paginator.paginate(results_25, page=0)
        assert page.page == 1

    def test_page_beyond_max_clamped(self, paginator, results_25):
        page = paginator.paginate(results_25, page=999)
        assert page.page == page.total_pages

    def test_negative_page_clamped_to_1(self, paginator, results_25):
        page = paginator.paginate(results_25, page=-5)
        assert page.page == 1


# ---------------------------------------------------------------------------
# Sort orders
# ---------------------------------------------------------------------------

class TestSortOrders:
    def test_sort_by_score_default(self, paginator, results_25):
        page = paginator.paginate(results_25, page=1, sort_by=SortBy.SCORE)
        scores = [r.score for r in page.results]
        assert scores == sorted(scores, reverse=True)

    def test_sort_by_date(self, paginator, results_25):
        page = paginator.paginate(results_25, page=1, sort_by=SortBy.DATE)
        dates = [r.date for r in page.results]
        assert dates == sorted(dates, reverse=True)

    def test_sort_by_title(self, paginator, results_25):
        page = paginator.paginate(results_25, page=1, sort_by=SortBy.TITLE)
        titles = [r.title.lower() for r in page.results]
        assert titles == sorted(titles)

    def test_sort_by_stored_in_page(self, paginator, results_25):
        page = paginator.paginate(results_25, page=1, sort_by=SortBy.DATE)
        assert page.sort_by == SortBy.DATE

    def test_sort_does_not_mutate_input(self, paginator, results_25):
        original_first = results_25[0].doc_id
        paginator.paginate(results_25, page=1, sort_by=SortBy.TITLE)
        assert results_25[0].doc_id == original_first


# ---------------------------------------------------------------------------
# Page size variants
# ---------------------------------------------------------------------------

class TestPageSizeVariants:
    def test_page_size_5(self):
        p = Paginator(page_size=5)
        page = p.paginate(make_results(12), page=1)
        assert len(page.results) == 5
        assert page.total_pages == 3

    def test_page_size_1(self):
        p = Paginator(page_size=1)
        page = p.paginate(make_results(3), page=2)
        assert len(page.results) == 1
        assert page.page == 2
        assert page.has_next is True
        assert page.has_prev is True

    def test_page_size_exceeds_max_clamped(self):
        p = Paginator(page_size=9999)
        assert p.page_size == Paginator.MAX_PAGE_SIZE

    def test_exact_page_boundary(self):
        p = Paginator(page_size=10)
        page = p.paginate(make_results(10), page=1)
        assert len(page.results) == 10
        assert page.total_pages == 1
        assert page.has_next is False


# ---------------------------------------------------------------------------
# Integration: with real ranking output
# ---------------------------------------------------------------------------

class TestIntegrationWithRanker:
    def test_paginate_real_results(self):
        import json
        from pathlib import Path
        from src.indexer.inverted_index import InvertedIndex
        from src.search.query_parser import QueryParser
        from src.search.retrieval import Retriever
        from src.search.ranking import Ranker

        dataset = Path(__file__).parent.parent.parent / \
            "data" / "sample_documents.json"
        if not dataset.exists():
            pytest.skip("Dataset not present")

        with open(dataset) as f:
            docs = json.load(f)

        idx = InvertedIndex()
        idx.build(docs)
        qp = QueryParser()
        ret = Retriever(idx)
        rnk = Ranker(idx)
        pag = Paginator(page_size=5)

        pq = qp.parse("search index algorithm")
        results = rnk.rank(pq, ret.retrieve(pq))
        page = pag.paginate(results, page=1)

        assert len(page.results) <= 5
        assert page.total == len(results)
        scores = [r.score for r in page.results]
        assert scores == sorted(scores, reverse=True)
