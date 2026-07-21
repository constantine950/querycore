"""
test_phrase_match.py

Tests for src/search/phrase_match.py

Run with:  python -m pytest tests/unit/test_phrase_match.py -v
"""

import json
from pathlib import Path

import pytest
from src.indexer.inverted_index import InvertedIndex
from src.search.query_parser import QueryParser
from src.search.retrieval import Retriever
from src.search.phrase_match import PhraseFilter


# ---------------------------------------------------------------------------
# Fixtures
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
            "title": "Inverted Index Structure",
            "body": "The inverted index maps every term to a list of documents containing that term.",
            "category": "cs", "date": "2024-02-01",
        },
        {
            "id": "doc_003",
            "title": "Machine Learning",
            "body": "Machine learning algorithms improve search engine ranking results significantly.",
            "category": "ml", "date": "2024-03-01",
        },
        {
            "id": "doc_004",
            "title": "Information Retrieval",
            "body": "Information retrieval systems retrieve and rank documents based on query terms.",
            "category": "cs", "date": "2024-04-01",
        },
    ]


@pytest.fixture(scope="module")
def idx(corpus):
    index = InvertedIndex()
    index.build(corpus)
    return index


@pytest.fixture(scope="module")
def pf(idx):
    return PhraseFilter(idx)


@pytest.fixture(scope="module")
def qp():
    return QueryParser()


@pytest.fixture(scope="module")
def ret(idx):
    return Retriever(idx)


# ---------------------------------------------------------------------------
# PhraseFilter.matches() — single phrase, single doc
# ---------------------------------------------------------------------------

class TestMatches:
    def test_adjacent_terms_match(self, pf, qp):
        # "search engine" — these appear consecutively in doc_001 and doc_003
        phrase = qp.parse('"search engine"').phrases[0]
        assert pf.matches("doc_001", phrase) is True

    def test_non_adjacent_terms_do_not_match(self, pf):
        # "search retrieval" — both words appear in doc_001 but not consecutively
        # doc_001: "search engine indexes documents for fast retrieval"
        # "search" at pos 0, "retriev" (stemmed) at pos 5 — not adjacent
        assert pf.matches("doc_001", ["search", "retriev"]) is False

    def test_phrase_in_second_doc(self, pf, qp):
        # "search engine" also appears in doc_003 body
        phrase = qp.parse('"search engine"').phrases[0]
        assert pf.matches("doc_003", phrase) is True

    def test_phrase_absent_from_doc(self, pf, qp):
        phrase = qp.parse('"search engine"').phrases[0]
        # doc_002 (Inverted Index) doesn't contain "search engine"
        assert pf.matches("doc_002", phrase) is False

    def test_single_term_phrase_matches_if_present(self, pf):
        assert pf.matches("doc_001", ["search"]) is True

    def test_single_term_phrase_fails_if_absent(self, pf):
        assert pf.matches("doc_001", ["quantum"]) is False

    def test_empty_phrase_returns_false(self, pf):
        assert pf.matches("doc_001", []) is False

    def test_unknown_doc_returns_false(self, pf):
        assert pf.matches("doc_999", ["search"]) is False

    def test_three_term_phrase(self, pf, qp):
        # "machine learning algorithms" — three consecutive terms in doc_003
        phrase = qp.parse('"machine learning algorithms"').phrases[0]
        assert pf.matches("doc_003", phrase) is True

    def test_three_term_phrase_wrong_order(self, pf):
        # "learning machine algorithms" — wrong order, shouldn't match
        assert pf.matches("doc_003", ["learn", "machin", "algorithm"]) is False


# ---------------------------------------------------------------------------
# PhraseFilter.filter() — candidate set filtering
# ---------------------------------------------------------------------------

class TestFilter:
    def test_filter_reduces_candidate_set(self, pf, qp, ret):
        pq = qp.parse('"search engine"')
        all_candidates = ret.retrieve_for_term("search")
        filtered = pf.filter(all_candidates, pq.phrases)
        assert len(filtered) <= len(all_candidates)

    def test_filter_keeps_matching_docs(self, pf, qp):
        phrase = qp.parse('"search engine"').phrases[0]
        candidates = {"doc_001", "doc_002", "doc_003", "doc_004"}
        filtered = pf.filter(candidates, [phrase])
        assert "doc_001" in filtered
        assert "doc_003" in filtered

    def test_filter_removes_non_matching_docs(self, pf, qp):
        phrase = qp.parse('"search engine"').phrases[0]
        candidates = {"doc_001", "doc_002", "doc_003", "doc_004"}
        filtered = pf.filter(candidates, [phrase])
        assert "doc_002" not in filtered
        assert "doc_004" not in filtered

    def test_filter_no_phrases_returns_candidates_unchanged(self, pf):
        candidates = {"doc_001", "doc_002"}
        assert pf.filter(candidates, []) == candidates

    def test_filter_empty_candidates_returns_empty(self, pf, qp):
        phrase = qp.parse('"search engine"').phrases[0]
        assert pf.filter(set(), [phrase]) == set()

    def test_multiple_phrases_all_must_match(self, pf, qp):
        # Both "search engine" AND "inverted index" must appear — no single doc has both
        phrases = [
            qp.parse('"search engine"').phrases[0],
            qp.parse('"inverted index"').phrases[0],
        ]
        candidates = {"doc_001", "doc_002", "doc_003", "doc_004"}
        filtered = pf.filter(candidates, phrases)
        assert len(filtered) == 0

    def test_filter_result_is_subset_of_candidates(self, pf, qp):
        phrase = qp.parse('"search engine"').phrases[0]
        candidates = {"doc_001", "doc_002", "doc_003"}
        filtered = pf.filter(candidates, [phrase])
        assert filtered.issubset(candidates)


# ---------------------------------------------------------------------------
# find_phrase_positions()
# ---------------------------------------------------------------------------

class TestFindPhrasePositions:
    def test_returns_start_position(self, pf, qp):
        phrase = qp.parse('"search engine"').phrases[0]
        positions = pf.find_phrase_positions("doc_001", phrase)
        assert isinstance(positions, list)
        assert len(positions) >= 1

    def test_position_is_integer(self, pf, qp):
        phrase = qp.parse('"search engine"').phrases[0]
        positions = pf.find_phrase_positions("doc_001", phrase)
        assert all(isinstance(p, int) for p in positions)

    def test_no_positions_when_phrase_absent(self, pf, qp):
        phrase = qp.parse('"search engine"').phrases[0]
        positions = pf.find_phrase_positions("doc_002", phrase)
        assert positions == []

    def test_empty_phrase_returns_empty(self, pf):
        assert pf.find_phrase_positions("doc_001", []) == []

    def test_multiple_occurrences_all_returned(self, pf):
        # Build a corpus where a phrase appears twice
        idx2 = InvertedIndex()
        idx2.build([{
            "id": "doc_x",
            "title": "repeat",
            "body": "search engine works well and search engine scales too",
            "category": "cs", "date": "2024-01-01",
        }])
        pf2 = PhraseFilter(idx2)
        positions = pf2.find_phrase_positions("doc_x", ["search", "engin"])
        assert len(positions) == 2


# ---------------------------------------------------------------------------
# Integration: full pipeline with phrase filtering
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    def test_phrase_query_end_to_end(self, pf, qp, ret):
        pq = qp.parse('"search engine"')
        assert pq.is_phrase

        # Retrieve candidates based on terms
        candidates = ret.retrieve(pq)
        # Filter to phrase matches
        matched = pf.filter(candidates, pq.phrases)

        assert isinstance(matched, set)
        # All matched docs should actually contain the phrase
        for doc_id in matched:
            assert pf.matches(doc_id, pq.phrases[0]) is True

    def test_phrase_plus_term_query(self, pf, qp, ret):
        # '"search engine" ranking' — phrase + extra term
        pq = qp.parse('"search engine" ranking')
        candidates = ret.retrieve(pq)
        matched = pf.filter(candidates, pq.phrases)
        # doc_003 has "search engine" and "ranking" → should survive
        assert "doc_003" in matched

    def test_full_dataset_phrase_search(self):
        dataset = Path(__file__).parent.parent.parent / \
            "data" / "sample_documents.json"
        if not dataset.exists():
            pytest.skip("Dataset not present")

        with open(dataset) as f:
            docs = json.load(f)

        idx = InvertedIndex()
        idx.build(docs)
        pf = PhraseFilter(idx)
        qp = QueryParser()
        ret = Retriever(idx)

        pq = qp.parse('"inverted index"')
        assert pq.phrases == [["invert", "index"]]

        # For phrase-only queries pq.terms is empty — gather candidates by
        # intersecting the posting lists of all phrase tokens directly.
        phrase = pq.phrases[0]
        candidates = ret.retrieve_for_term(phrase[0])
        for term in phrase[1:]:
            candidates &= ret.retrieve_for_term(term)

        matched = pf.filter(candidates, pq.phrases)

        # The "Inverted Index" document must be in the results
        assert len(matched) > 0
        meta_titles = [idx.get_metadata(d)["title"] for d in matched]
        assert any("inverted" in t.lower() or "index" in t.lower()
                   for t in meta_titles)
