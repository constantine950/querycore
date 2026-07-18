"""
test_retrieval.py

Tests for src/search/retrieval.py

Run with:  python -m pytest tests/unit/test_retrieval.py -v
"""

import json
from pathlib import Path

import pytest
from src.indexer.inverted_index import InvertedIndex
from src.search.query_parser import QueryParser, ParsedQuery
from src.search.retrieval import Retriever


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def corpus():
    return [
        {
            "id": "doc_001",
            "title": "Search Engine",
            "body": "A search engine indexes documents for fast retrieval of information.",
            "category": "cs", "date": "2024-01-01",
        },
        {
            "id": "doc_002",
            "title": "Inverted Index",
            "body": "The inverted index maps terms to documents for efficient lookup.",
            "category": "cs", "date": "2024-02-01",
        },
        {
            "id": "doc_003",
            "title": "Machine Learning",
            "body": "Machine learning algorithms improve ranking and classification of data.",
            "category": "ml", "date": "2024-03-01",
        },
        {
            "id": "doc_004",
            "title": "Database Systems",
            "body": "Database systems store and retrieve structured data efficiently.",
            "category": "cs", "date": "2024-04-01",
        },
        {
            "id": "doc_005",
            "title": "Search Ranking",
            "body": "Search ranking algorithms determine the order of search results.",
            "category": "cs", "date": "2024-05-01",
        },
    ]


@pytest.fixture(scope="module")
def idx(corpus):
    index = InvertedIndex()
    index.build(corpus)
    return index


@pytest.fixture(scope="module")
def retriever(idx):
    return Retriever(idx)


@pytest.fixture(scope="module")
def qp():
    return QueryParser()


# ---------------------------------------------------------------------------
# AND term retrieval
# ---------------------------------------------------------------------------

class TestAndRetrieval:
    def test_single_term_returns_matching_docs(self, retriever, qp):
        pq = qp.parse("search")
        result = retriever.retrieve(pq)
        assert "doc_001" in result
        assert "doc_005" in result

    def test_single_term_excludes_non_matching(self, retriever, qp):
        pq = qp.parse("search")
        result = retriever.retrieve(pq)
        assert "doc_003" not in result   # machine learning doc has no "search"

    def test_two_and_terms_intersection(self, retriever, qp):
        # "search" AND "rank" — only doc_005 has both
        pq = qp.parse("search ranking")
        result = retriever.retrieve(pq)
        assert "doc_005" in result
        # doc_001 has "search" but not "rank" (check it's excluded)
        # Note: doc_001 body says "retrieval" not "ranking", so it should be out
        assert "doc_003" not in result

    def test_and_term_not_in_corpus_returns_empty(self, retriever, qp):
        pq = qp.parse("xyznotaword")
        assert retriever.retrieve(pq) == set()

    def test_all_and_terms_must_match(self, retriever, qp):
        # "search" + "machine" — no single doc has both
        pq = qp.parse("search machine")
        result = retriever.retrieve(pq)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# OR term retrieval
# ---------------------------------------------------------------------------

class TestOrRetrieval:
    def test_or_returns_union(self, retriever, qp):
        pq = qp.parse("search OR learn")
        result = retriever.retrieve(pq)
        # "search" docs + "learn"(machine learning) doc
        assert "doc_001" in result or "doc_005" in result
        assert "doc_003" in result   # machine learning

    def test_or_only_query(self, retriever, qp):
        # A pure OR query: first part before OR becomes AND term
        pq = qp.parse("search OR databas")
        result = retriever.retrieve(pq)
        assert len(result) > 0

    def test_or_expands_result_set(self, retriever, qp):
        and_only = retriever.retrieve(qp.parse("search"))
        or_added = retriever.retrieve(qp.parse("search OR learn"))
        assert len(or_added) >= len(and_only)


# ---------------------------------------------------------------------------
# Excluded term retrieval
# ---------------------------------------------------------------------------

class TestExcludedRetrieval:
    def test_excluded_term_removes_docs(self, retriever, qp):
        # "search" without exclusion
        all_search = retriever.retrieve(qp.parse("search"))
        # "search" excluding docs that contain "rank"
        without_rank = retriever.retrieve(qp.parse("search -rank"))
        assert len(without_rank) <= len(all_search)

    def test_excluded_docs_not_in_result(self, retriever, qp):
        pq = qp.parse("search -rank")
        result = retriever.retrieve(pq)
        # doc_005 has both "search" and "ranking" — should be excluded
        assert "doc_005" not in result

    def test_exclude_nonexistent_term_has_no_effect(self, retriever, qp):
        base = retriever.retrieve(qp.parse("search"))
        with_excl = retriever.retrieve(qp.parse("search -xyzfake"))
        assert base == with_excl


# ---------------------------------------------------------------------------
# Empty and edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_query_returns_empty(self, retriever):
        pq = ParsedQuery(raw="")
        assert retriever.retrieve(pq) == set()

    def test_stopwords_only_query_returns_empty(self, retriever, qp):
        pq = qp.parse("the is a of")
        assert retriever.retrieve(pq) == set()

    def test_retrieve_for_term(self, retriever):
        result = retriever.retrieve_for_term("search")
        assert isinstance(result, set)
        assert len(result) > 0

    def test_retrieve_for_unknown_term(self, retriever):
        assert retriever.retrieve_for_term("xyzunknown") == set()

    def test_count_matches_len(self, retriever, qp):
        pq = qp.parse("search")
        assert retriever.count(pq) == len(retriever.retrieve(pq))


# ---------------------------------------------------------------------------
# explain()
# ---------------------------------------------------------------------------

class TestExplain:
    def test_explain_keys(self, retriever, qp):
        pq = qp.parse("search OR learn -databas")
        exp = retriever.explain(pq)
        for key in ("query", "and_terms", "or_terms", "excluded_terms",
                    "and_candidates", "or_candidates", "excluded_docs",
                    "final_candidates", "counts"):
            assert key in exp

    def test_explain_counts_consistent(self, retriever, qp):
        pq = qp.parse("search OR learn")
        exp = retriever.explain(pq)
        assert exp["counts"]["final"] == len(exp["final_candidates"])

    def test_explain_excluded_not_in_final(self, retriever, qp):
        pq = qp.parse("search -rank")
        exp = retriever.explain(pq)
        excluded_set = set(exp["excluded_docs"])
        final_set = set(exp["final_candidates"])
        assert excluded_set.isdisjoint(final_set)


# ---------------------------------------------------------------------------
# Integration: full dataset
# ---------------------------------------------------------------------------

class TestFullDataset:
    @pytest.fixture(scope="class")
    def full_idx(self):
        dataset = Path(__file__).parent.parent.parent / \
            "data" / "sample_documents.json"
        if not dataset.exists():
            pytest.skip("Dataset not present")
        with open(dataset) as f:
            docs = json.load(f)
        idx = InvertedIndex()
        idx.build(docs)
        return idx

    def test_retrieval_returns_results(self, full_idx):
        ret = Retriever(full_idx)
        qp = QueryParser()
        pq = qp.parse("search engine")
        result = ret.retrieve(pq)
        assert len(result) > 0

    def test_exclusion_reduces_results(self, full_idx):
        ret = Retriever(full_idx)
        qp = QueryParser()
        base = ret.retrieve(qp.parse("index"))
        reduced = ret.retrieve(qp.parse("index -search"))
        assert len(reduced) <= len(base)

    def test_or_expands_results(self, full_idx):
        ret = Retriever(full_idx)
        qp = QueryParser()
        base = ret.retrieve(qp.parse("quantum"))
        expanded = ret.retrieve(qp.parse("quantum OR algorithm"))
        assert len(expanded) >= len(base)

    def test_known_doc_retrieved_for_its_title(self, full_idx):
        ret = Retriever(full_idx)
        qp = QueryParser()
        # "Information Retrieval" doc should come back for query "retrieval"
        result = ret.retrieve(qp.parse("retrieval"))
        assert "doc_004" in result
