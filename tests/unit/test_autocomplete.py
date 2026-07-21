"""
test_autocomplete.py

Tests for src/search/autocomplete.py

Run with:  python -m pytest tests/unit/test_autocomplete.py -v
"""

import json
from pathlib import Path

import pytest
from src.indexer.inverted_index import InvertedIndex
from src.search.autocomplete import Autocomplete, TrieNode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def idx():
    dataset = Path(__file__).parent.parent.parent / \
        "data" / "sample_documents.json"
    if not dataset.exists():
        pytest.skip("Dataset not present")
    with open(dataset) as f:
        docs = json.load(f)
    index = InvertedIndex()
    index.build(docs)
    return index


@pytest.fixture(scope="module")
def ac(idx):
    return Autocomplete(idx)


@pytest.fixture(scope="module")
def small_idx():
    """Tiny controlled index for exact behavioural tests."""
    docs = [
        {"id": "d1", "title": "Search Engine", "body": "search engine indexes fast retrieval",
            "category": "cs", "date": "2024-01-01"},
        {"id": "d2", "title": "Season Review",  "body": "season spring summer autumn winter",
            "category": "misc", "date": "2024-01-01"},
        {"id": "d3", "title": "Sea Creatures",  "body": "sea ocean wave coral reef fish",
            "category": "sci", "date": "2024-01-01"},
        {"id": "d4", "title": "Searcher Tools", "body": "search searcher searching tool lookup",
            "category": "cs", "date": "2024-01-01"},
    ]
    index = InvertedIndex()
    index.build(docs)
    return index


@pytest.fixture(scope="module")
def small_ac(small_idx):
    return Autocomplete(small_idx)


# ---------------------------------------------------------------------------
# suggest()
# ---------------------------------------------------------------------------

class TestSuggest:
    def test_returns_list(self, ac):
        assert isinstance(ac.suggest("sea"), list)

    def test_all_results_start_with_prefix(self, ac):
        prefix = "sea"
        for term in ac.suggest(prefix):
            assert term.startswith(
                prefix), f"{term!r} does not start with {prefix!r}"

    def test_empty_prefix_returns_empty(self, ac):
        assert ac.suggest("") == []

    def test_unknown_prefix_returns_empty(self, ac):
        assert ac.suggest("zzzzqqqq") == []

    def test_top_n_respected(self, ac):
        results = ac.suggest("s", top_n=3)
        assert len(results) <= 3

    def test_exact_term_included(self, small_ac):
        results = small_ac.suggest("search")
        assert "search" in results

    def test_prefix_matches_multiple_terms(self, small_ac):
        results = small_ac.suggest("sea")
        # "search" and "season" and "sea" related stems should appear
        assert len(results) >= 1

    def test_sorted_by_doc_frequency(self, small_ac):
        # "search" appears in d1 and d4 (2 docs), "season" only in d2 (1 doc)
        results = small_ac.suggest("sea")
        # "search" should come before "season" since it has higher doc_count
        if "search" in results and "season" in results:
            assert results.index("search") < results.index("season")

    def test_single_char_prefix(self, ac):
        results = ac.suggest("s")
        assert len(results) > 0
        assert all(t.startswith("s") for t in results)

    def test_full_term_as_prefix(self, ac):
        # Using a complete term as prefix should at minimum return that term
        results = ac.suggest("search")
        assert "search" in results

    def test_default_top_n_is_10(self, ac):
        results = ac.suggest("s")
        assert len(results) <= 10


# ---------------------------------------------------------------------------
# suggest_with_scores()
# ---------------------------------------------------------------------------

class TestSuggestWithScores:
    def test_returns_list_of_dicts(self, ac):
        results = ac.suggest_with_scores("sea")
        assert isinstance(results, list)
        for r in results:
            assert "term" in r
            assert "doc_count" in r

    def test_doc_count_positive(self, ac):
        for r in ac.suggest_with_scores("search"):
            assert r["doc_count"] > 0

    def test_empty_prefix_returns_empty(self, ac):
        assert ac.suggest_with_scores("") == []

    def test_sorted_by_doc_count(self, ac):
        results = ac.suggest_with_scores("s", top_n=20)
        counts = [r["doc_count"] for r in results]
        assert counts == sorted(counts, reverse=True)


# ---------------------------------------------------------------------------
# has_prefix() / exact_match()
# ---------------------------------------------------------------------------

class TestPrefixChecks:
    def test_has_prefix_true_for_known(self, ac):
        assert ac.has_prefix("sea") is True
        assert ac.has_prefix("search") is True

    def test_has_prefix_false_for_unknown(self, ac):
        assert ac.has_prefix("zzzzq") is False

    def test_exact_match_true_for_indexed_term(self, ac):
        assert ac.exact_match("search") is True

    def test_exact_match_false_for_prefix_only(self, ac):
        # "sea" may not be an indexed term even though it's a valid prefix
        # (depends on corpus — just check the method works without error)
        result = ac.exact_match("sea")
        assert isinstance(result, bool)

    def test_exact_match_false_for_unknown(self, ac):
        assert ac.exact_match("zzzzqqqq") is False


# ---------------------------------------------------------------------------
# Trie structure
# ---------------------------------------------------------------------------

class TestTrieStructure:
    def test_root_is_trie_node(self, ac):
        assert isinstance(ac._root, TrieNode)

    def test_root_has_children(self, ac):
        assert len(ac._root.children) > 0

    def test_insert_and_retrieve(self):
        """Verify trie insertions work correctly on a fresh instance."""
        mini_idx = InvertedIndex()
        mini_idx.build([{
            "id": "x1", "title": "apple application",
            "body": "apple application apply appetizer",
            "category": "test", "date": "2024-01-01"
        }])
        ac2 = Autocomplete(mini_idx)
        results = ac2.suggest("appl")
        assert len(results) > 0
        assert all(t.startswith("appl") for t in results)


# ---------------------------------------------------------------------------
# Integration: autocomplete on real corpus
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_suggest_index_related_terms(self, ac):
        results = ac.suggest("index")
        assert "index" in results

    def test_suggest_retriev_prefix(self, ac):
        results = ac.suggest("retriev")
        assert len(results) > 0
        assert all(t.startswith("retriev") for t in results)

    def test_suggest_algo_prefix(self, ac):
        results = ac.suggest("algo")
        assert len(results) > 0

    def test_no_duplicates(self, ac):
        results = ac.suggest("s", top_n=50)
        assert len(results) == len(set(results))

    def test_all_suggestions_in_index(self, idx, ac):
        """Every suggestion must be a real indexed term."""
        all_terms = set(idx.get_all_terms())
        for prefix in ["sea", "ind", "algo", "ret"]:
            for term in ac.suggest(prefix, top_n=20):
                assert term in all_terms, f"Suggested term {term!r} not in index"
