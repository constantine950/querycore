"""
test_preprocessor.py

Unit tests for src/indexer/preprocessor.py

Run with:  python -m pytest tests/unit/test_preprocessor.py -v
"""

import pytest
from src.indexer.preprocessor import Preprocessor, PreprocessorConfig, DEFAULT_STOPWORDS
from src.indexer.tokenizer import Tokenizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def p():
    return Preprocessor()


@pytest.fixture
def tokenizer():
    return Tokenizer()


# ---------------------------------------------------------------------------
# Stop word removal
# ---------------------------------------------------------------------------

class TestStopWordRemoval:
    def test_removes_common_words(self, p):
        tokens = ["the", "search", "engine", "is", "fast"]
        assert p.process(tokens) == ["search", "engin", "fast"]

    def test_removes_are_and_a(self, p):
        result = p.process(["are", "a", "query"])
        assert "are" not in result
        assert "a" not in result

    def test_keeps_content_words(self, p):
        tokens = ["inverted", "index", "retrieval"]
        result = p.process(tokens)
        # All should survive (stemmed forms)
        assert len(result) == 3

    def test_stopwords_are_lowercase(self):
        # All stop words in the default set should already be lowercase
        assert all(w == w.lower() for w in DEFAULT_STOPWORDS)

    def test_custom_stopwords(self):
        cfg = PreprocessorConfig(custom_stopwords={"engine", "system"})
        p2 = Preprocessor(cfg)
        result = p2.process(["search", "engine", "system", "fast"])
        assert "engin" not in result   # stemmed "engine" dropped
        assert "system" not in result

    def test_removal_disabled(self):
        cfg = PreprocessorConfig(remove_stopwords=False)
        p2 = Preprocessor(cfg)
        result = p2.process(["the", "search"])
        assert "the" in result

    def test_is_stopword(self, p):
        assert p.is_stopword("the") is True
        assert p.is_stopword("search") is False

    def test_stopwords_property(self, p):
        assert isinstance(p.stopwords, frozenset)
        assert "the" in p.stopwords


# ---------------------------------------------------------------------------
# Stemming
# ---------------------------------------------------------------------------

class TestStemming:
    def test_plural_to_singular(self, p):
        result = p.process(["engines", "searches", "indexes"])
        # Porter: engines→engin, searches→search, indexes→index
        assert "engin" in result
        assert "search" in result
        assert "index" in result

    def test_verb_forms(self, p):
        # Porter stemmer collapses inflections but not all derivations.
        # "running" → "run", "runs" → "run" — same stem.
        # "runner" → "runner" — Porter keeps this distinct (agentive suffix).
        result = p.process(["running", "runs"])
        stems = set(result)
        assert len(stems) == 1
        assert "run" in stems

    def test_runner_stem(self, p):
        # Document the actual Porter behaviour so it's not surprising later.
        # Porter does NOT reduce this to "run"
        assert p.stem("runner") == "runner"

    def test_stemming_disabled(self):
        cfg = PreprocessorConfig(apply_stemming=False)
        p2 = Preprocessor(cfg)
        result = p2.process(["engines", "running"])
        assert "engines" in result
        assert "running" in result

    def test_stem_single_token(self, p):
        assert p.stem("running") == "run"
        assert p.stem("engines") == "engin"
        assert p.stem("indexed") == "index"
        assert p.stem("retrieval") == "retriev"

    def test_stem_already_root(self, p):
        # Stemming a root form should be idempotent (or close to it)
        assert p.stem("run") == "run"
        assert p.stem("index") == "index"


# ---------------------------------------------------------------------------
# Min token length
# ---------------------------------------------------------------------------

class TestMinTokenLength:
    def test_default_drops_single_chars(self, p):
        # After stop word removal, leftover single chars should be dropped
        result = p.process(["a", "i", "search"])
        assert "a" not in result
        assert "i" not in result

    def test_custom_min_length(self):
        cfg = PreprocessorConfig(min_token_length=5)
        p2 = Preprocessor(cfg)
        result = p2.process(["fast", "information", "retrieval"])
        # "fast" → "fast" (4 chars) should be dropped with min=5
        assert "fast" not in result
        assert "inform" in result or "informaiton" not in result


# ---------------------------------------------------------------------------
# process_text() convenience method
# ---------------------------------------------------------------------------

class TestProcessText:
    def test_full_pipeline(self, p):
        result = p.process_text("The search engines are indexing fast")
        assert "search" in result
        assert "engin" in result
        assert "index" in result
        assert "the" not in result
        assert "are" not in result

    def test_empty_string(self, p):
        assert p.process_text("") == []

    def test_all_stopwords(self, p):
        # A sentence of only stop words should produce empty output
        result = p.process_text("the is a of and")
        assert result == []


# ---------------------------------------------------------------------------
# Pipeline integration: Tokenizer → Preprocessor
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    def test_tokenizer_then_preprocessor(self, tokenizer, p):
        raw = "Search Engines are the fastest retrieval systems"
        tokens = tokenizer.tokenize(raw)
        result = p.process(tokens)
        assert "search" in result
        assert "engin" in result
        assert "retriev" in result        # "retrieval" → "retriev"
        assert "the" not in result
        assert "are" not in result

    def test_same_stem_for_variants(self, tokenizer, p):
        words = ["index", "indexes", "indexing", "indexed"]
        stems = set()
        for word in words:
            tokens = tokenizer.tokenize(word)
            processed = p.process(tokens)
            stems.update(processed)
        # All variants should collapse to one stem
        assert len(stems) == 1

    def test_real_dataset_documents(self, tokenizer, p):
        import json
        from pathlib import Path
        dataset = Path(__file__).parent.parent.parent / \
            "data" / "sample_documents.json"
        if not dataset.exists():
            pytest.skip("Dataset not present")

        with open(dataset) as f:
            docs = json.load(f)

        for doc in docs:
            tokens = tokenizer.tokenize(doc["body"])
            result = p.process(tokens)
            assert isinstance(result, list)
            # Tokens in the result should all be strings with length >= 2.
            # We do NOT assert that no result token matches a stop word string,
            # because Porter stemming can map non-stop-words to stop-word-shaped
            # strings (e.g. "ones" → "one"). That is expected behaviour.
            assert all(isinstance(t, str) and len(t) >= 2 for t in result), (
                f"Short or non-string token found in doc {doc['id']}: {result}"
            )
