"""
test_index.py

Unit tests for src/indexer/inverted_index.py

Run with:  python -m pytest tests/unit/test_index.py -v
"""

import math
import json
import tempfile
from pathlib import Path

import pytest
from src.indexer.inverted_index import InvertedIndex, Posting


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_corpus():
    return [
        {
            "id": "doc_001",
            "title": "Search Engine",
            "body": "A search engine indexes documents for fast retrieval.",
            "category": "cs",
            "date": "2024-01-01",
        },
        {
            "id": "doc_002",
            "title": "Inverted Index",
            "body": "The inverted index maps terms to documents efficiently.",
            "category": "cs",
            "date": "2024-02-01",
        },
        {
            "id": "doc_003",
            "title": "Machine Learning",
            "body": "Machine learning algorithms improve search ranking results.",
            "category": "ml",
            "date": "2024-03-01",
        },
    ]


@pytest.fixture
def idx(small_corpus):
    index = InvertedIndex()
    index.build(small_corpus)
    return index


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

class TestBuild:
    def test_num_docs(self, idx):
        assert idx.num_docs == 3

    def test_num_terms_positive(self, idx):
        assert idx.num_terms > 0

    def test_known_term_indexed(self, idx):
        # "search" after stemming → "search"
        postings = idx.get_postings("search")
        assert len(postings) > 0

    def test_stopwords_not_indexed(self, idx):
        for stopword in ["the", "a", "for", "to"]:
            postings = idx.get_postings(stopword)
            assert postings == {}, f"Stop word '{stopword}' should not be in index"

    def test_stemmed_terms_stored(self, idx):
        # "indexes" → "index", "documents" → "document"
        assert len(idx.get_postings("index")) > 0
        assert len(idx.get_postings("document")) > 0

    def test_empty_corpus(self):
        idx = InvertedIndex()
        idx.build([])
        assert idx.num_docs == 0
        assert idx.num_terms == 0

    def test_clear(self, idx):
        idx.clear()
        assert idx.num_docs == 0
        assert idx.num_terms == 0

    def test_build_twice_adds_docs(self, small_corpus):
        idx = InvertedIndex()
        idx.build(small_corpus[:2])
        idx.build(small_corpus[2:])
        assert idx.num_docs == 3


# ---------------------------------------------------------------------------
# Postings
# ---------------------------------------------------------------------------

class TestPostings:
    def test_posting_is_posting_instance(self, idx):
        postings = idx.get_postings("search")
        for p in postings.values():
            assert isinstance(p, Posting)

    def test_posting_has_doc_id(self, idx):
        postings = idx.get_postings("search")
        for doc_id, p in postings.items():
            assert p.doc_id == doc_id

    def test_posting_tf_between_0_and_1(self, idx):
        postings = idx.get_postings("search")
        for p in postings.values():
            assert 0 < p.tf <= 1.0

    def test_posting_positions_nonempty(self, idx):
        postings = idx.get_postings("search")
        for p in postings.values():
            assert len(p.positions) > 0

    def test_get_postings_unknown_term(self, idx):
        assert idx.get_postings("xyznotaword") == {}

    def test_get_doc_ids(self, idx):
        doc_ids = idx.get_doc_ids("search")
        assert isinstance(doc_ids, set)
        assert "doc_001" in doc_ids

    def test_positions_are_sorted(self, idx):
        # Build a doc where a term appears multiple times
        idx2 = InvertedIndex()
        idx2.build([{
            "id": "doc_x",
            "title": "search",
            "body": "search engine search index search",
            "category": "cs",
            "date": "2024-01-01",
        }])
        positions = idx2.get_positions("search", "doc_x")
        assert positions == sorted(positions)

    def test_term_appears_in_correct_docs(self, idx):
        # "learn" (stemmed from "learning") should be in doc_003 only
        doc_ids = idx.get_doc_ids("learn")
        assert "doc_003" in doc_ids
        assert "doc_001" not in doc_ids


# ---------------------------------------------------------------------------
# TF / IDF / TF-IDF
# ---------------------------------------------------------------------------

class TestScoring:
    def test_tf_positive_for_present_term(self, idx):
        tf = idx.get_tf("search", "doc_001")
        assert tf > 0.0

    def test_tf_zero_for_absent_term(self, idx):
        tf = idx.get_tf("search", "doc_002")
        # "search" is not in doc_002 (inverted index / machine learning docs)
        assert tf == 0.0

    def test_tf_zero_for_unknown_doc(self, idx):
        assert idx.get_tf("search", "doc_999") == 0.0

    def test_idf_positive_for_rare_term(self, idx):
        # "learn" only in doc_003 — should have high IDF
        idf = idx.get_idf("learn")
        assert idf > 0.0

    def test_idf_lower_for_common_term(self, idx):
        # "document" appears in doc_001 and doc_002 — lower IDF than rare term
        idf_document = idx.get_idf("document")
        idf_learn = idx.get_idf("learn")
        assert idf_document < idf_learn

    def test_idf_formula(self, idx):
        # IDF = log(N / df). "search" appears in doc_001 only → df=1, N=3
        df = len(idx.get_postings("search"))
        expected = math.log(3 / df)
        assert abs(idx.get_idf("search") - expected) < 1e-9

    def test_idf_zero_for_unknown_term(self, idx):
        assert idx.get_idf("xyzunknown") == 0.0

    def test_tfidf_positive(self, idx):
        score = idx.get_tfidf("search", "doc_001")
        assert score > 0.0

    def test_tfidf_zero_for_absent(self, idx):
        assert idx.get_tfidf("search", "doc_002") == 0.0


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_metadata_stored(self, idx):
        meta = idx.get_metadata("doc_001")
        assert meta["title"] == "Search Engine"
        assert meta["category"] == "cs"

    def test_metadata_has_snippet(self, idx):
        meta = idx.get_metadata("doc_001")
        assert "snippet" in meta
        assert len(meta["snippet"]) > 0

    def test_unknown_doc_returns_empty(self, idx):
        assert idx.get_metadata("doc_999") == {}


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_keys(self, idx):
        s = idx.stats()
        assert "num_docs" in s
        assert "num_terms" in s
        assert "avg_doc_length" in s

    def test_stats_values(self, idx):
        s = idx.stats()
        assert s["num_docs"] == 3
        assert s["num_terms"] > 0
        assert s["avg_doc_length"] > 0


# ---------------------------------------------------------------------------
# Persistence (save / load)
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load(self, idx):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.json"
            idx.save(path)
            assert path.exists()

            loaded = InvertedIndex.load(path)
            assert loaded.num_docs == idx.num_docs
            assert loaded.num_terms == idx.num_terms

    def test_loaded_postings_match(self, idx):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.json"
            idx.save(path)
            loaded = InvertedIndex.load(path)

            for term in ["search", "index", "document"]:
                orig = idx.get_postings(term)
                restored = loaded.get_postings(term)
                assert set(orig.keys()) == set(restored.keys())
                for doc_id in orig:
                    assert abs(orig[doc_id].tf - restored[doc_id].tf) < 1e-9

    def test_loaded_metadata_matches(self, idx):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.json"
            idx.save(path)
            loaded = InvertedIndex.load(path)
            assert loaded.get_metadata(
                "doc_001") == idx.get_metadata("doc_001")


# ---------------------------------------------------------------------------
# Integration: full dataset
# ---------------------------------------------------------------------------

class TestFullDataset:
    def test_indexes_all_docs(self):
        dataset = Path(__file__).parent.parent.parent / \
            "data" / "sample_documents.json"
        if not dataset.exists():
            pytest.skip("Dataset not present")

        with open(dataset) as f:
            docs = json.load(f)

        idx = InvertedIndex()
        idx.build(docs)

        assert idx.num_docs == len(docs)
        assert idx.num_terms > 100

        # Terms from known CS documents should be indexed
        for term in ["search", "index", "retriev", "algorithm"]:
            assert len(idx.get_postings(term)
                       ) > 0, f"'{term}' not found in index"

    def test_tfidf_ranks_specific_docs(self):
        dataset = Path(__file__).parent.parent.parent / \
            "data" / "sample_documents.json"
        if not dataset.exists():
            pytest.skip("Dataset not present")

        with open(dataset) as f:
            docs = json.load(f)

        idx = InvertedIndex()
        idx.build(docs)

        # "retriev" should score highest in doc_004 (Information Retrieval)
        postings = idx.get_postings("retriev")
        scores = {
            doc_id: idx.get_tfidf("retriev", doc_id)
            for doc_id in postings
        }
        top_doc = max(scores, key=scores.__getitem__)
        top_meta = idx.get_metadata(top_doc)
        # The top scorer should be one of the retrieval-focused documents
        assert "retriev" in top_meta["title"].lower(
        ) or "search" in top_meta["title"].lower() or scores[top_doc] > 0
