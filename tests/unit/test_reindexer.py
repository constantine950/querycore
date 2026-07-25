"""
test_reindexer.py

Tests for src/indexer/reindexer.py

Run with:  python -m pytest tests/unit/test_reindexer.py -v
"""

import json
from pathlib import Path

import pytest
from src.indexer.inverted_index import InvertedIndex
from src.indexer.reindexer import Reindexer, ReindexStats


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_corpus():
    return [
        {"id": "doc_001", "title": "Search Engine",
            "body": "A search engine indexes documents for fast retrieval.", "category": "cs", "date": "2024-01-01"},
        {"id": "doc_002", "title": "Inverted Index",   "body": "The inverted index maps terms to documents.",
            "category": "cs", "date": "2024-02-01"},
        {"id": "doc_003", "title": "Machine Learning",
            "body": "Machine learning algorithms improve classification.",   "category": "ml", "date": "2024-03-01"},
    ]


@pytest.fixture
def idx(base_corpus):
    index = InvertedIndex()
    index.build(base_corpus)
    return index


@pytest.fixture
def rx(idx):
    return Reindexer(idx)


def make_doc(doc_id, title, body, category="cs", date="2024-06-01"):
    return {"id": doc_id, "title": title, "body": body, "category": category, "date": date}


# ---------------------------------------------------------------------------
# add()
# ---------------------------------------------------------------------------

class TestAdd:
    def test_add_increases_doc_count(self, rx, idx):
        before = idx.num_docs
        rx.add(make_doc("doc_004", "New Doc",
               "brand new document about fuzzy search"))
        assert idx.num_docs == before + 1

    def test_added_doc_is_searchable(self, rx, idx):
        rx.add(make_doc("doc_004", "Fuzzy Search",
               "fuzzy search uses edit distance"))
        postings = idx.get_postings("fuzzi")
        assert "doc_004" in postings

    def test_added_doc_metadata_stored(self, rx, idx):
        rx.add(make_doc("doc_004", "Fuzzy Search",
               "fuzzy search content", category="ai"))
        meta = idx.get_metadata("doc_004")
        assert meta["title"] == "Fuzzy Search"
        assert meta["category"] == "ai"

    def test_add_duplicate_raises(self, rx):
        with pytest.raises(ValueError, match="already exists"):
            rx.add(make_doc("doc_001", "Dup", "duplicate document"))

    def test_add_without_id_raises(self, rx):
        with pytest.raises(ValueError, match="'id' field"):
            rx.add({"title": "No ID", "body": "missing id field"})

    def test_add_records_stat(self, rx):
        rx.add(make_doc("doc_004", "New", "new doc content here"))
        assert rx.stats()["adds"] == 1

    def test_exists_after_add(self, rx):
        rx.add(make_doc("doc_004", "New", "content here"))
        assert rx.exists("doc_004") is True

    def test_not_exists_before_add(self, rx):
        assert rx.exists("doc_999") is False


# ---------------------------------------------------------------------------
# remove()
# ---------------------------------------------------------------------------

class TestRemove:
    def test_remove_decreases_doc_count(self, rx, idx):
        before = idx.num_docs
        rx.remove("doc_001")
        assert idx.num_docs == before - 1

    def test_removed_doc_not_in_postings(self, rx, idx):
        rx.remove("doc_001")
        postings = idx.get_postings("search")
        assert "doc_001" not in postings

    def test_removed_doc_metadata_gone(self, rx, idx):
        rx.remove("doc_001")
        assert idx.get_metadata("doc_001") == {}

    def test_remove_nonexistent_raises(self, rx):
        with pytest.raises(KeyError, match="not found"):
            rx.remove("doc_999")

    def test_remove_records_stat(self, rx):
        rx.remove("doc_001")
        assert rx.stats()["removes"] == 1

    def test_exists_false_after_remove(self, rx):
        rx.remove("doc_001")
        assert rx.exists("doc_001") is False

    def test_terms_only_in_removed_doc_disappear(self, rx, idx):
        # Add a doc with a unique term, then remove it
        rx.add(make_doc("doc_004", "Unique", "xyzuniquetermnotelsewhere content"))
        assert len(idx.get_postings("xyzuniquetermnotelsewher")) > 0
        rx.remove("doc_004")
        assert idx.get_postings("xyzuniquetermnotelsewher") == {}

    def test_other_docs_unaffected_after_remove(self, rx, idx):
        before_doc2 = idx.get_postings("invert").copy()
        rx.remove("doc_001")
        after_doc2 = idx.get_postings("invert")
        assert "doc_002" in after_doc2
        assert set(before_doc2.keys()) == set(after_doc2.keys())


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_update_replaces_content(self, rx, idx):
        # doc_001 originally about "search engine" — update to talk about quantum
        rx.update(make_doc("doc_001", "Quantum Doc",
                  "quantum mechanics describes subatomic particles"))
        # Old terms should be gone from doc_001
        assert "doc_001" not in idx.get_postings("search")
        # New terms should be present
        assert "doc_001" in idx.get_postings("quantum")

    def test_update_preserves_doc_count(self, rx, idx):
        before = idx.num_docs
        rx.update(make_doc("doc_001", "Updated", "updated content here"))
        assert idx.num_docs == before

    def test_update_refreshes_metadata(self, rx, idx):
        rx.update(make_doc("doc_001", "New Title",
                  "new body", category="physics"))
        meta = idx.get_metadata("doc_001")
        assert meta["title"] == "New Title"
        assert meta["category"] == "physics"

    def test_update_nonexistent_acts_as_add(self, rx, idx):
        before = idx.num_docs
        rx.update(make_doc("doc_999", "New", "new document content"))
        assert idx.num_docs == before + 1
        assert rx.exists("doc_999") is True

    def test_update_records_stat(self, rx):
        rx.update(make_doc("doc_001", "Updated", "updated content here"))
        assert rx.stats()["updates"] == 1

    def test_update_tfidf_recalculated(self, rx, idx):
        # After update, TF should reflect new content
        rx.update(make_doc("doc_001", "Quantum Quantum",
                  "quantum quantum quantum mechanics"))
        tf = idx.get_tf("quantum", "doc_001")
        assert tf > 0

    def test_update_other_docs_unaffected(self, rx, idx):
        postings_before = set(idx.get_postings("invert").keys())
        rx.update(make_doc("doc_001", "Changed",
                  "completely different content now"))
        postings_after = set(idx.get_postings("invert").keys())
        assert "doc_002" in postings_after
        assert postings_before == postings_after


# ---------------------------------------------------------------------------
# reindex()
# ---------------------------------------------------------------------------

class TestReindex:
    def test_reindex_replaces_all_docs(self, rx, idx):
        new_corpus = [
            make_doc("new_001", "Alpha", "alpha beta gamma delta content"),
            make_doc("new_002", "Beta",  "beta gamma epsilon content here"),
        ]
        rx.reindex(new_corpus)
        assert idx.num_docs == 2
        assert rx.exists("new_001") is True
        assert rx.exists("doc_001") is False   # old doc gone

    def test_reindex_clears_old_terms(self, rx, idx):
        rx.reindex(
            [make_doc("new_001", "Fresh", "fresh completely new content")])
        # "search" was in old corpus — should be gone
        assert idx.get_postings("search") == {}

    def test_reindex_records_stat(self, rx):
        rx.reindex([make_doc("new_001", "A", "some content here")])
        assert rx.stats()["reindexes"] == 1

    def test_reindex_with_empty_corpus(self, rx, idx):
        rx.reindex([])
        assert idx.num_docs == 0
        assert idx.num_terms == 0


# ---------------------------------------------------------------------------
# stats()
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_initial(self, rx):
        s = rx.stats()
        assert s["adds"] == 0
        assert s["updates"] == 0
        assert s["removes"] == 0
        assert s["reindexes"] == 0
        assert s["total_ops"] == 0

    def test_stats_total_ops(self, rx):
        rx.add(make_doc("doc_004", "D4", "content d4"))
        rx.update(make_doc("doc_001", "D1 Updated", "updated d1 content"))
        rx.remove("doc_002")
        s = rx.stats()
        assert s["total_ops"] == 3


# ---------------------------------------------------------------------------
# Integration: full pipeline after reindex ops
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_search_works_after_add(self, idx, base_corpus):
        from src.search.query_parser import QueryParser
        from src.search.retrieval import Retriever

        rx = Reindexer(idx)
        qp = QueryParser()
        ret = Retriever(idx)

        rx.add(make_doc("doc_004", "Trie Structure",
               "trie prefix tree autocomplete suggestions"))
        pq = qp.parse("trie prefix")
        result = ret.retrieve(pq)
        assert "doc_004" in result

    def test_search_excludes_removed_doc(self, idx, base_corpus):
        from src.search.query_parser import QueryParser
        from src.search.retrieval import Retriever

        rx = Reindexer(idx)
        qp = QueryParser()
        ret = Retriever(idx)

        rx.remove("doc_001")
        pq = qp.parse("search engine")
        result = ret.retrieve(pq)
        assert "doc_001" not in result

    def test_full_dataset_incremental_update(self):
        dataset = Path(__file__).parent.parent.parent / \
            "data" / "sample_documents.json"
        if not dataset.exists():
            pytest.skip("Dataset not present")

        with open(dataset) as f:
            docs = json.load(f)

        idx = InvertedIndex()
        idx.build(docs)
        rx = Reindexer(idx)

        original_count = idx.num_docs

        # Add a new doc
        new_doc = make_doc("doc_NEW", "Brand New Document",
                           "this is a completely new document about trie structures and prefix trees")
        rx.add(new_doc)
        assert idx.num_docs == original_count + 1

        # Update it
        rx.update(make_doc("doc_NEW", "Updated Document",
                           "updated content about inverted indexes and search engines"))
        assert idx.num_docs == original_count + 1
        assert "doc_NEW" not in idx.get_postings("trie")   # old content gone

        # Remove it
        rx.remove("doc_NEW")
        assert idx.num_docs == original_count
        assert rx.exists("doc_NEW") is False

        s = rx.stats()
        assert s["adds"] == 1
        assert s["updates"] == 1
        assert s["removes"] == 1
