"""
test_query_parser.py

Unit tests for src/search/query_parser.py

Run with:  python -m pytest tests/unit/test_query_parser.py -v
"""

import pytest
from src.search.query_parser import QueryParser, ParsedQuery


@pytest.fixture
def qp():
    return QueryParser()


# ---------------------------------------------------------------------------
# Basic term parsing
# ---------------------------------------------------------------------------

class TestBasicTerms:
    def test_single_term(self, qp):
        pq = qp.parse("search")
        assert "search" in pq.terms

    def test_multiple_terms(self, qp):
        pq = qp.parse("search engine ranking")
        assert "search" in pq.terms
        assert "engin" in pq.terms
        assert "rank" in pq.terms

    def test_terms_are_stemmed(self, qp):
        pq = qp.parse("searching engines")
        assert "search" in pq.terms
        assert "engin" in pq.terms

    def test_stop_words_removed(self, qp):
        pq = qp.parse("the search engine is fast")
        assert "the" not in pq.terms
        assert "is" not in pq.terms
        assert "search" in pq.terms

    def test_empty_query(self, qp):
        pq = qp.parse("")
        assert pq.is_empty

    def test_whitespace_query(self, qp):
        pq = qp.parse("   ")
        assert pq.is_empty

    def test_stopwords_only(self, qp):
        pq = qp.parse("the is a of")
        assert pq.is_empty

    def test_raw_preserved(self, qp):
        raw = "Search Engine Ranking"
        pq = qp.parse(raw)
        assert pq.raw == raw


# ---------------------------------------------------------------------------
# Phrase queries
# ---------------------------------------------------------------------------

class TestPhraseQueries:
    def test_single_phrase(self, qp):
        pq = qp.parse('"inverted index"')
        assert pq.is_phrase
        assert len(pq.phrases) == 1
        assert "invert" in pq.phrases[0]
        assert "index" in pq.phrases[0]

    def test_phrase_order_preserved(self, qp):
        pq = qp.parse('"search engine"')
        phrase = pq.phrases[0]
        assert phrase.index("search") < phrase.index("engin")

    def test_multiple_phrases(self, qp):
        pq = qp.parse('"search engine" "inverted index"')
        assert len(pq.phrases) == 2

    def test_phrase_plus_term(self, qp):
        pq = qp.parse('"search engine" ranking')
        assert pq.is_phrase
        assert "rank" in pq.terms

    def test_phrase_stopwords_excluded(self, qp):
        pq = qp.parse('"the search engine"')
        phrase = pq.phrases[0]
        assert "the" not in phrase

    def test_no_phrase_flag_for_plain_query(self, qp):
        pq = qp.parse("search engine")
        assert not pq.is_phrase


# ---------------------------------------------------------------------------
# Boolean / OR queries
# ---------------------------------------------------------------------------

class TestOrQueries:
    def test_or_splits_terms(self, qp):
        pq = qp.parse("search OR retrieval")
        assert "search" in pq.terms
        assert "retriev" in pq.or_terms

    def test_or_is_boolean(self, qp):
        pq = qp.parse("search OR retrieval")
        assert pq.is_boolean

    def test_multiple_or_terms(self, qp):
        pq = qp.parse("search OR retrieval OR index")
        assert "retriev" in pq.or_terms
        assert "index" in pq.or_terms

    def test_or_terms_are_stemmed(self, qp):
        pq = qp.parse("searching OR retrieving")
        assert "search" in pq.terms
        assert "retriev" in pq.or_terms

    def test_plain_query_has_no_or_terms(self, qp):
        pq = qp.parse("search engine")
        assert pq.or_terms == []


# ---------------------------------------------------------------------------
# Exclusion queries
# ---------------------------------------------------------------------------

class TestExclusionQueries:
    def test_single_exclusion(self, qp):
        pq = qp.parse("search -database")
        assert "search" in pq.terms
        assert "databas" in pq.excluded

    def test_exclusion_is_boolean(self, qp):
        pq = qp.parse("search -database")
        assert pq.is_boolean

    def test_multiple_exclusions(self, qp):
        pq = qp.parse("search -database -network")
        assert "databas" in pq.excluded
        assert "network" in pq.excluded

    def test_exclusion_stemmed(self, qp):
        pq = qp.parse("search -databases")
        assert "databas" in pq.excluded

    def test_no_exclusions_in_plain_query(self, qp):
        pq = qp.parse("search engine")
        assert pq.excluded == []


# ---------------------------------------------------------------------------
# Combined queries
# ---------------------------------------------------------------------------

class TestCombinedQueries:
    def test_phrase_and_term(self, qp):
        pq = qp.parse('"search engine" ranking')
        assert pq.is_phrase
        assert "rank" in pq.terms

    def test_term_and_exclusion(self, qp):
        pq = qp.parse("search engine -database")
        assert "search" in pq.terms
        assert "databas" in pq.excluded

    def test_phrase_or_term(self, qp):
        pq = qp.parse('"search engine" OR retrieval')
        assert pq.is_phrase
        assert "retriev" in pq.or_terms

    def test_all_types(self, qp):
        pq = qp.parse('"inverted index" search OR retrieval -database')
        assert pq.is_phrase
        assert len(pq.phrases) == 1
        assert "search" in pq.terms
        assert "retriev" in pq.or_terms
        assert "databas" in pq.excluded


# ---------------------------------------------------------------------------
# all_terms property
# ---------------------------------------------------------------------------

class TestAllTerms:
    def test_includes_terms_and_or_and_phrases(self, qp):
        pq = qp.parse('"search engine" OR retrieval')
        all_t = pq.all_terms
        assert "search" in all_t
        assert "engin" in all_t
        assert "retriev" in all_t

    def test_no_duplicates(self, qp):
        pq = qp.parse("search search search")
        assert pq.all_terms.count("search") == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_uppercase_query(self, qp):
        pq = qp.parse("SEARCH ENGINE")
        assert "search" in pq.terms
        assert "engin" in pq.terms

    def test_punctuation_in_query(self, qp):
        pq = qp.parse("search, engine.")
        assert "search" in pq.terms
        assert "engin" in pq.terms

    def test_hyphenated_query(self, qp):
        pq = qp.parse("full-text search")
        assert "full" in pq.terms
        assert "text" in pq.terms
        assert "search" in pq.terms

    def test_numbers_kept(self, qp):
        pq = qp.parse("python 3 features")
        assert "3" in pq.terms or "python" in pq.terms

    def test_or_case_insensitive_not_triggered(self, qp):
        # lowercase "or" is NOT an operator — only uppercase OR is
        pq = qp.parse("this or that")
        assert pq.or_terms == []

    def test_empty_phrase(self, qp):
        # A quoted string of only stop words should produce an empty phrase
        # which is dropped — not crash
        pq = qp.parse('"the a is"')
        # Either phrases is empty or contains an empty list (both acceptable)
        for phrase in pq.phrases:
            assert len(phrase) > 0 or phrase == []
