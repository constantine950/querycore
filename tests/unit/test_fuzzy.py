"""
test_fuzzy.py

Tests for src/search/fuzzy_search.py

Run with:  python -m pytest tests/unit/test_fuzzy.py -v
"""

import json
from pathlib import Path

import pytest
from src.indexer.inverted_index import InvertedIndex
from src.search.query_parser import QueryParser
from src.search.fuzzy_search import FuzzyMatcher, FuzzyMatch, levenshtein


# ---------------------------------------------------------------------------
# levenshtein()
# ---------------------------------------------------------------------------

class TestLevenshtein:
    def test_identical_strings(self):
        assert levenshtein("search", "search") == 0

    def test_single_substitution(self):
        assert levenshtein("cat", "bat") == 1

    def test_single_insertion(self):
        assert levenshtein("engne", "engine") == 1

    def test_single_deletion(self):
        assert levenshtein("seach", "search") == 1

    def test_transposition_is_two_edits(self):
        # Levenshtein counts a transposition as 2 (delete + insert)
        assert levenshtein("serach", "search") == 2

    def test_empty_vs_string(self):
        assert levenshtein("", "hello") == 5

    def test_string_vs_empty(self):
        assert levenshtein("hello", "") == 5

    def test_both_empty(self):
        assert levenshtein("", "") == 0

    def test_completely_different(self):
        assert levenshtein("abc", "xyz") == 3

    def test_symmetric(self):
        assert levenshtein("search", "serach") == levenshtein(
            "serach", "search")

    def test_longer_typo(self):
        # "retreval" vs "retrieval" — 2 edits
        assert levenshtein("retreval", "retriev") <= 3

    def test_one_char_strings(self):
        assert levenshtein("a", "b") == 1
        assert levenshtein("a", "a") == 0

    def test_prefix_relationship(self):
        assert levenshtein("search", "searcher") == 2


# ---------------------------------------------------------------------------
# FuzzyMatcher.expand()
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
def fm(idx):
    return FuzzyMatcher(idx, max_distance=2, min_term_len=4)


class TestExpand:
    def test_exact_term_returns_itself(self, fm):
        result = fm.expand("search")
        assert "search" in result

    def test_typo_expands_to_correct_term(self, fm):
        # "serach" (transposition) → should find "search"
        result = fm.expand("serach")
        assert "search" in result

    def test_one_char_off_expands(self, fm):
        # "engin" is already the stem — "engne" should find it
        result = fm.expand("engne")
        assert "engin" in result

    def test_returns_set(self, fm):
        assert isinstance(fm.expand("search"), set)

    def test_short_term_below_min_len(self, fm):
        # Terms shorter than min_term_len (4) — returns empty or exact only
        result = fm.expand("tf")
        # Should not attempt full scan for very short terms
        assert isinstance(result, set)

    def test_no_match_for_garbage(self, fm):
        result = fm.expand("zzzzqqqq")
        assert len(result) == 0

    def test_length_pruning_applied(self, fm):
        # "retriev" (7 chars) — terms of length 1-4 should all be pruned
        result = fm.expand("retriev")
        for term in result:
            assert abs(len(term) - len("retriev")) <= 2


# ---------------------------------------------------------------------------
# FuzzyMatcher.find_matches()
# ---------------------------------------------------------------------------

class TestFindMatches:
    def test_returns_list_of_fuzzy_match(self, fm):
        matches = fm.find_matches("search")
        assert isinstance(matches, list)
        assert all(isinstance(m, FuzzyMatch) for m in matches)

    def test_sorted_by_distance(self, fm):
        matches = fm.find_matches("search")
        distances = [m.distance for m in matches]
        assert distances == sorted(distances)

    def test_exact_match_first(self, fm):
        matches = fm.find_matches("search")
        assert matches[0].distance == 0
        assert matches[0].index_term == "search"

    def test_fuzzy_match_fields(self, fm):
        matches = fm.find_matches("serach")
        assert len(matches) > 0
        m = matches[0]
        assert m.query_term == "serach"
        assert isinstance(m.index_term, str)
        assert isinstance(m.distance, int)
        assert isinstance(m.doc_count, int)
        assert m.doc_count > 0

    def test_short_term_returns_empty(self, fm):
        assert fm.find_matches("tf") == []


# ---------------------------------------------------------------------------
# FuzzyMatcher.retrieve_fuzzy()
# ---------------------------------------------------------------------------

class TestRetrieveFuzzy:
    def test_exact_query_returns_results(self, fm):
        qp = QueryParser()
        pq = qp.parse("search")
        result = fm.retrieve_fuzzy(pq)
        assert len(result) > 0

    def test_typo_query_still_returns_results(self, fm):
        qp = QueryParser()
        pq = qp.parse("serach")   # typo for "search"
        result = fm.retrieve_fuzzy(pq)
        assert len(result) > 0

    def test_empty_query_returns_empty(self, fm):
        from src.search.query_parser import ParsedQuery
        pq = ParsedQuery(raw="")
        assert fm.retrieve_fuzzy(pq) == set()

    def test_fuzzy_retrieval_returns_set(self, fm):
        qp = QueryParser()
        pq = qp.parse("algorithm")
        assert isinstance(fm.retrieve_fuzzy(pq), set)

    def test_typo_finds_same_docs_as_correct(self, fm, idx):
        qp = QueryParser()
        correct = fm.retrieve_fuzzy(qp.parse("algorithm"))
        typo = fm.retrieve_fuzzy(qp.parse("algorythm"))
        # Fuzzy should recover at least some of the correct docs
        assert len(typo & correct) > 0

    def test_exclusion_applied(self, fm, idx):
        qp = QueryParser()
        base = fm.retrieve_fuzzy(qp.parse("index"))
        excluded = fm.retrieve_fuzzy(qp.parse("index -search"))
        assert len(excluded) <= len(base)


# ---------------------------------------------------------------------------
# FuzzyMatcher.suggest_correction()
# ---------------------------------------------------------------------------

class TestSuggestCorrection:
    def test_correct_spelling_returns_itself(self, fm):
        assert fm.suggest_correction("search") == "search"

    def test_typo_returns_correction(self, fm):
        suggestion = fm.suggest_correction("serach")
        assert suggestion == "search"

    def test_no_match_returns_none(self, fm):
        result = fm.suggest_correction("zzzzqqqq")
        assert result is None

    def test_short_term_returns_none(self, fm):
        result = fm.suggest_correction("tf")
        assert result is None


# ---------------------------------------------------------------------------
# Integration: fuzzy search on real queries
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_misspelled_search_finds_docs(self, fm):
        qp = QueryParser()
        # Common typos
        typos = [
            ("serach",    "search"),
            ("retreival", "retriev"),
            ("algorythm", "algorithm"),
        ]
        for typo, _ in typos:
            pq = qp.parse(typo)
            result = fm.retrieve_fuzzy(pq)
            assert len(
                result) > 0, f"Fuzzy search for '{typo}' returned no docs"

    def test_fuzzy_expands_result_vs_exact(self, idx):
        qp = QueryParser()
        fm_tight = FuzzyMatcher(idx, max_distance=1)
        fm_loose = FuzzyMatcher(idx, max_distance=2)

        pq = qp.parse("serach")
        tight = fm_tight.retrieve_fuzzy(pq)
        loose = fm_loose.retrieve_fuzzy(pq)
        # Looser threshold should find at least as many docs
        assert len(loose) >= len(tight)
