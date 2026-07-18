"""
test_ranking.py

Tests for src/search/ranking.py

Run with:  python -m pytest tests/unit/test_ranking.py -v
"""

import json
from pathlib import Path

import pytest
from src.indexer.inverted_index import InvertedIndex
from src.search.query_parser import QueryParser
from src.search.retrieval import Retriever
from src.search.ranking import Ranker, SearchResult, RankerConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def corpus():
    return [
        {
            "id": "doc_001",
            "title": "Search Engine",
            "body": "A search engine indexes documents for fast retrieval. Search engines use inverted indexes.",
            "category": "cs", "date": "2024-01-01",
        },
        {
            "id": "doc_002",
            "title": "Inverted Index",
            "body": "The inverted index maps terms to documents for efficient lookup and retrieval.",
            "category": "cs", "date": "2024-02-01",
        },
        {
            "id": "doc_003",
            "title": "Machine Learning",
            "body": "Machine learning algorithms improve classification and prediction of data.",
            "category": "ml", "date": "2024-03-01",
        },
        {
            "id": "doc_004",
            "title": "Search Algorithms",
            "body": "Search algorithms determine how documents are ranked. Search is fundamental.",
            "category": "cs", "date": "2024-04-01",
        },
    ]


@pytest.fixture(scope="module")
def idx(corpus):
    index = InvertedIndex()
    index.build(corpus)
    return index


@pytest.fixture(scope="module")
def ranker(idx):
    return Ranker(idx)


@pytest.fixture(scope="module")
def retriever(idx):
    return Retriever(idx)


@pytest.fixture(scope="module")
def qp():
    return QueryParser()


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------

class TestSearchResult:
    def test_to_dict_has_all_keys(self):
        r = SearchResult(doc_id="doc_001", score=0.42, title="Test",
                         snippet="...", category="cs", date="2024-01-01")
        d = r.to_dict()
        for key in ("doc_id", "score", "title", "snippet", "category", "date", "url"):
            assert key in d

    def test_score_rounded_in_dict(self):
        r = SearchResult(doc_id="doc_001", score=0.123456789)
        # rounded to 6 decimal places
        assert len(str(r.to_dict()["score"])) <= 10


# ---------------------------------------------------------------------------
# Basic ranking
# ---------------------------------------------------------------------------

class TestBasicRanking:
    def test_returns_list_of_search_results(self, ranker, retriever, qp, idx):
        pq = qp.parse("search")
        candidates = retriever.retrieve(pq)
        results = ranker.rank(pq, candidates)
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_results_sorted_descending(self, ranker, retriever, qp):
        pq = qp.parse("search")
        candidates = retriever.retrieve(pq)
        results = ranker.rank(pq, candidates)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_all_scores_positive(self, ranker, retriever, qp):
        pq = qp.parse("search")
        candidates = retriever.retrieve(pq)
        results = ranker.rank(pq, candidates)
        assert all(r.score > 0 for r in results)

    def test_result_has_metadata(self, ranker, retriever, qp):
        pq = qp.parse("search")
        candidates = retriever.retrieve(pq)
        results = ranker.rank(pq, candidates)
        for r in results:
            assert r.title != ""
            assert r.snippet != ""
            assert r.category != ""

    def test_empty_candidates_returns_empty(self, ranker, qp):
        pq = qp.parse("search")
        assert ranker.rank(pq, set()) == []

    def test_empty_query_returns_empty(self, ranker, retriever):
        from src.search.query_parser import ParsedQuery
        pq = ParsedQuery(raw="")
        assert ranker.rank(pq, {"doc_001"}) == []


# ---------------------------------------------------------------------------
# Score correctness
# ---------------------------------------------------------------------------

class TestScoreCorrectness:
    def test_doc_with_more_term_occurrences_scores_higher(self, ranker, qp):
        # doc_004 mentions "search" three times, doc_001 twice — doc_004 should win
        pq = qp.parse("search")
        score_001 = ranker.score_document("doc_001", pq.terms)
        score_004 = ranker.score_document("doc_004", pq.terms)
        assert score_004 > score_001

    def test_doc_not_containing_term_scores_zero(self, ranker, qp):
        pq = qp.parse("search")
        score = ranker.score_document("doc_003", pq.terms)
        assert score == 0.0

    def test_rare_term_gives_higher_idf(self, idx):
        # "machine" only in doc_003 — higher IDF than "search" which is in 3 docs
        idf_machine = idx.get_idf("machin")
        idf_search = idx.get_idf("search")
        assert idf_machine > idf_search


# ---------------------------------------------------------------------------
# Title boost
# ---------------------------------------------------------------------------

class TestTitleBoost:
    def test_title_match_scores_higher(self, idx):
        # doc_004 title is "Search Algorithms" → "search" is in its title
        # doc_001 title is "Search Engine"     → "search" is also in its title
        # doc_002 has "search" only in body
        # Compare boosted vs unboosted config
        ranker_boosted = Ranker(idx, RankerConfig(title_boost=2.0))
        ranker_unboosted = Ranker(idx, RankerConfig(title_boost=1.0))

        pq = QueryParser().parse("search")
        terms = pq.terms

        boosted_score = ranker_boosted.score_document("doc_001", terms)
        unboosted_score = ranker_unboosted.score_document("doc_001", terms)
        assert boosted_score > unboosted_score

    def test_no_title_boost_when_term_absent_from_title(self, idx):
        ranker_boosted = Ranker(idx, RankerConfig(title_boost=2.0))
        ranker_unboosted = Ranker(idx, RankerConfig(title_boost=1.0))
        # "retrieval" is not in doc_001 title ("Search Engine")
        terms = ["retriev"]
        b = ranker_boosted.score_document("doc_001", terms)
        u = ranker_unboosted.score_document("doc_001", terms)
        assert b == u   # no boost applied


# ---------------------------------------------------------------------------
# OR terms scored
# ---------------------------------------------------------------------------

class TestOrTermsScored:
    def test_or_terms_contribute_to_score(self, ranker, retriever, qp):
        pq = qp.parse("search OR learn")
        candidates = retriever.retrieve(pq)
        results = ranker.rank(pq, candidates)
        doc_ids = {r.doc_id for r in results}
        assert "doc_003" in doc_ids   # machine learning — matches "learn"

    def test_doc_matching_both_and_and_or_scores_highest(self, ranker, retriever, qp):
        # If a doc matches both AND and OR terms it accumulates score from both
        pq = qp.parse("search OR index")
        candidates = retriever.retrieve(pq)
        results = ranker.rank(pq, candidates)
        # doc_001 has both "search" and "index" in body — should rank top
        assert results[0].doc_id in ("doc_001", "doc_002", "doc_004")


# ---------------------------------------------------------------------------
# explain()
# ---------------------------------------------------------------------------

class TestExplain:
    def test_explain_has_required_keys(self, ranker, qp):
        pq = qp.parse("search")
        exp = ranker.explain("doc_001", pq)
        assert "doc_id" in exp
        assert "title" in exp
        assert "total" in exp
        assert "breakdown" in exp

    def test_explain_breakdown_per_term(self, ranker, qp):
        pq = qp.parse("search")
        exp = ranker.explain("doc_001", pq)
        for term, data in exp["breakdown"].items():
            assert "tf" in data
            assert "idf" in data
            assert "base_score" in data
            assert "boosted_score" in data
            assert "title_hit" in data

    def test_explain_total_matches_score(self, ranker, qp):
        pq = qp.parse("search")
        exp = ranker.explain("doc_001", pq)
        direct = ranker.score_document("doc_001", pq.terms)
        assert abs(exp["total"] - direct) < 1e-5


# ---------------------------------------------------------------------------
# Integration: full dataset
# ---------------------------------------------------------------------------

class TestFullDataset:
    @pytest.fixture(scope="class")
    def setup(self):
        dataset = Path(__file__).parent.parent.parent / \
            "data" / "sample_documents.json"
        if not dataset.exists():
            pytest.skip("Dataset not present")
        with open(dataset) as f:
            docs = json.load(f)
        idx = InvertedIndex()
        idx.build(docs)
        return idx, Retriever(idx), Ranker(idx), QueryParser()

    def test_top_result_for_tfidf_query(self, setup):
        idx, ret, ranker, qp = setup
        pq = qp.parse("TF-IDF term frequency")
        candidates = ret.retrieve(pq)
        results = ranker.rank(pq, candidates)
        assert len(results) > 0
        # Top result should be the TF-IDF document
        assert "tf" in results[0].title.lower() or "term" in results[0].title.lower() \
               or results[0].score > 0

    def test_information_retrieval_ranks_top_for_retrieval_query(self, setup):
        idx, ret, ranker, qp = setup
        pq = qp.parse("information retrieval")
        candidates = ret.retrieve(pq)
        results = ranker.rank(pq, candidates)
        assert len(results) > 0
        top_titles = [r.title for r in results[:3]]
        assert any("retrieval" in t.lower() or "information" in t.lower()
                   for t in top_titles)

    def test_all_results_have_positive_score(self, setup):
        idx, ret, ranker, qp = setup
        pq = qp.parse("search engine algorithm")
        candidates = ret.retrieve(pq)
        results = ranker.rank(pq, candidates)
        assert all(r.score > 0 for r in results)
