"""
test_tokenizer.py

Unit tests for src/indexer/tokenizer.py

Run with:  python -m pytest tests/unit/test_tokenizer.py -v
"""

import pytest
from src.indexer.tokenizer import Tokenizer, TokenizerConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def t():
    return Tokenizer()


# ---------------------------------------------------------------------------
# normalize()
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_lowercases(self, t):
        assert t.normalize("HELLO") == "hello"

    def test_strips_accents(self, t):
        assert t.normalize("café") == "cafe"
        assert t.normalize("naïve") == "naive"
        assert t.normalize("résumé") == "resume"

    def test_nfkd_ligatures(self, t):
        # NFKD decomposes some ligatures
        assert t.normalize("ﬁle") == "file"

    def test_empty_string(self, t):
        assert t.normalize("") == ""

    def test_already_lowercase(self, t):
        assert t.normalize("hello world") == "hello world"


# ---------------------------------------------------------------------------
# tokenize()
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_basic_sentence(self, t):
        assert t.tokenize("Search Engines are Fast!") == [
            "search", "engines", "are", "fast"]

    def test_punctuation_split(self, t):
        assert t.tokenize("hello, world.") == ["hello", "world"]

    def test_hyphenated_words(self, t):
        assert t.tokenize("state-of-the-art") == ["state", "of", "the", "art"]

    def test_keeps_numbers_by_default(self, t):
        assert "2024" in t.tokenize("Version 2024 released")

    def test_empty_string(self, t):
        assert t.tokenize("") == []

    def test_whitespace_only(self, t):
        assert t.tokenize("   ") == []

    def test_unicode_input(self, t):
        tokens = t.tokenize("Héllo Wörld")
        assert tokens == ["hello", "world"]

    def test_multiple_spaces(self, t):
        assert t.tokenize("hello   world") == ["hello", "world"]

    def test_newlines_and_tabs(self, t):
        assert t.tokenize("hello\nworld\ttab") == ["hello", "world", "tab"]

    def test_numbers_disabled(self):
        cfg = TokenizerConfig(keep_numbers=False)
        t2 = Tokenizer(cfg)
        tokens = t2.tokenize("version 42 released")
        assert "42" not in tokens
        assert "version" in tokens

    def test_min_token_length(self):
        cfg = TokenizerConfig(min_token_length=3)
        t2 = Tokenizer(cfg)
        tokens = t2.tokenize("a is an example")
        assert "a" not in tokens
        assert "is" not in tokens
        assert "an" not in tokens
        assert "example" in tokens

    def test_apostrophe_split(self, t):
        # "don't" → ["don", "t"]
        tokens = t.tokenize("don't")
        assert "don" in tokens

    def test_slash_split(self, t):
        tokens = t.tokenize("and/or")
        assert "and" in tokens
        assert "or" in tokens


# ---------------------------------------------------------------------------
# tokenize_document()
# ---------------------------------------------------------------------------

class TestTokenizeDocument:
    def test_combines_title_and_body(self, t):
        doc = {"title": "Search Engine", "body": "Fast retrieval system."}
        tokens = t.tokenize_document(doc)
        assert "search" in tokens
        assert "engine" in tokens
        assert "fast" in tokens
        assert "retrieval" in tokens

    def test_missing_title(self, t):
        doc = {"body": "Only body here."}
        tokens = t.tokenize_document(doc)
        assert "body" in tokens

    def test_missing_body(self, t):
        doc = {"title": "Only Title"}
        tokens = t.tokenize_document(doc)
        assert "only" in tokens
        assert "title" in tokens

    def test_empty_doc(self, t):
        assert t.tokenize_document({}) == []


# ---------------------------------------------------------------------------
# tokenize_query()
# ---------------------------------------------------------------------------

class TestTokenizeQuery:
    def test_basic_query(self, t):
        assert t.tokenize_query("search engine") == ["search", "engine"]

    def test_query_with_punctuation(self, t):
        assert t.tokenize_query(
            "what is TF-IDF?") == ["what", "is", "tf", "idf"]

    def test_empty_query(self, t):
        assert t.tokenize_query("") == []


# ---------------------------------------------------------------------------
# Integration: tokenize real dataset documents
# ---------------------------------------------------------------------------

class TestIntegrationWithDataset:
    def test_tokenizes_many_docs(self, t):
        import json
        from pathlib import Path

        dataset = Path(__file__).parent.parent.parent / \
            "data" / "sample_documents.json"
        if not dataset.exists():
            pytest.skip("Dataset not present")

        with open(dataset) as f:
            docs = json.load(f)

        for doc in docs:
            tokens = t.tokenize_document(doc)
            assert isinstance(tokens, list)
            assert len(tokens) > 0
            assert all(isinstance(tok, str) for tok in tokens)
            assert all(tok == tok.lower() for tok in tokens)
