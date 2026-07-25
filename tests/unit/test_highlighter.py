"""
test_highlighter.py

Tests for src/search/highlighter.py

Run with:  python -m pytest tests/unit/test_highlighter.py -v
"""

import json
from pathlib import Path

import pytest
from src.indexer.preprocessor import Preprocessor
from src.search.highlighter import Highlighter, HighlightResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def h():
    return Highlighter()


@pytest.fixture(scope="module")
def body():
    return (
        "A search engine is a software system designed to carry out web searches. "
        "It searches the World Wide Web for information specified in a query. "
        "Search results are presented as a list of results on a search engine results page. "
        "Modern search engines use sophisticated algorithms to rank results by relevance."
    )


# ---------------------------------------------------------------------------
# HighlightResult
# ---------------------------------------------------------------------------

class TestHighlightResult:
    def test_fields_exist(self):
        hr = HighlightResult(snippet="hello", raw="hello",
                             positions=[], match_count=0)
        assert hr.snippet == "hello"
        assert hr.raw == "hello"
        assert hr.positions == []
        assert hr.match_count == 0


# ---------------------------------------------------------------------------
# highlight() — basic cases
# ---------------------------------------------------------------------------

class TestHighlight:
    def test_returns_highlight_result(self, h, body):
        result = h.highlight(body, ["search"])
        assert isinstance(result, HighlightResult)

    def test_mark_tags_present(self, h, body):
        result = h.highlight(body, ["search"])
        assert "<mark>" in result.snippet
        assert "</mark>" in result.snippet

    def test_matched_word_wrapped(self, h):
        result = h.highlight("The search engine is fast.", ["search"])
        assert "<mark>search</mark>" in result.snippet

    def test_case_insensitive_match(self, h):
        result = h.highlight("The Search Engine is fast.", ["search"])
        assert "<mark>Search</mark>" in result.snippet

    def test_multiple_terms_highlighted(self, h):
        result = h.highlight("A search engine indexes documents.", [
                             "search", "engin"])
        assert "<mark>search</mark>" in result.snippet
        assert "<mark>engine</mark>" in result.snippet

    def test_stemmed_term_matches_surface_form(self, h):
        # "engin" stem matches "engine", "engines", "engineering"
        result = h.highlight("Search engines index documents.", ["engin"])
        assert "<mark>engines</mark>" in result.snippet

    def test_match_count_correct(self, h):
        result = h.highlight("Search engine search index.", ["search"])
        assert result.match_count == 2

    def test_raw_has_no_tags(self, h, body):
        result = h.highlight(body, ["search"])
        assert "<mark>" not in result.raw
        assert "</mark>" not in result.raw

    def test_empty_text_returns_empty(self, h):
        result = h.highlight("", ["search"])
        assert result.snippet == ""
        assert result.match_count == 0

    def test_empty_terms_returns_snippet(self, h):
        result = h.highlight("Some text here.", [])
        assert result.snippet == "Some text here."
        assert result.match_count == 0

    def test_no_match_returns_snippet_without_tags(self, h):
        result = h.highlight("Quantum mechanics is fascinating.", ["search"])
        assert "<mark>" not in result.snippet

    def test_positions_are_sorted(self, h, body):
        result = h.highlight(body, ["search", "engin"])
        positions = result.positions
        assert positions == sorted(positions)

    def test_positions_non_overlapping(self, h, body):
        result = h.highlight(body, ["search", "engin"])
        for i in range(len(result.positions) - 1):
            assert result.positions[i][1] <= result.positions[i + 1][0]

    def test_custom_tags(self):
        h2 = Highlighter(open_tag="<b>", close_tag="</b>")
        result = h2.highlight("search engine", ["search"])
        assert "<b>search</b>" in result.snippet


# ---------------------------------------------------------------------------
# Snippet extraction (window)
# ---------------------------------------------------------------------------

class TestSnippetExtraction:
    def test_short_text_returned_whole(self, h):
        text = "Short text with search term."
        result = h.highlight(text, ["search"], window=300)
        assert result.raw == text

    def test_long_text_truncated_to_window(self, h, body):
        result = h.highlight(body, ["search"], window=100)
        # allow small snap overshoot
        assert len(result.raw.replace("…", "")) <= 110

    def test_window_anchored_near_match(self, h):
        # Place the match far into a long document
        prefix = "x " * 200                              # 400 chars of noise
        target = "The search engine indexes documents."
        text = prefix + target
        result = h.highlight(text, ["search"], window=100)
        # The match word should appear in the snippet
        assert "search" in result.raw or "search" in result.snippet

    def test_ellipsis_added_when_truncated(self, h, body):
        result = h.highlight(body, ["search"], window=80)
        # If text was truncated, ellipsis should be present
        if len(body) > 80:
            assert "…" in result.snippet or len(result.raw) < len(body)

    def test_no_ellipsis_for_full_text(self, h):
        text = "Search engine retrieval."
        result = h.highlight(text, ["search"], window=300)
        assert "…" not in result.snippet


# ---------------------------------------------------------------------------
# Position accuracy
# ---------------------------------------------------------------------------

class TestPositionAccuracy:
    def test_position_spans_correct_word(self, h):
        text = "The search engine is fast."
        result = h.highlight(text, ["search"])
        for start, end in result.positions:
            word = text[start:end]
            assert word.lower() in ("search", "searches", "searching", "searched")

    def test_position_start_end_valid(self, h):
        text = "A search engine."
        result = h.highlight(text, ["search"])
        for start, end in result.positions:
            assert 0 <= start < end <= len(text)


# ---------------------------------------------------------------------------
# highlight_result()
# ---------------------------------------------------------------------------

class TestHighlightResult2:
    def test_updates_snippet_on_result(self, h):
        from src.search.ranking import SearchResult
        r = SearchResult(
            doc_id="doc_001", score=1.0, title="Search Engine",
            snippet="Original snippet.", category="cs", date="2024-01-01",
        )
        h.highlight_result(r, "A search engine indexes documents.", ["search"])
        assert "<mark>" in r.snippet

    def test_returns_same_result_object(self, h):
        from src.search.ranking import SearchResult
        r = SearchResult(
            doc_id="doc_001", score=1.0, title="T",
            snippet="s", category="cs", date="2024-01-01",
        )
        returned = h.highlight_result(r, "search engine", ["search"])
        assert returned is r


# ---------------------------------------------------------------------------
# Integration: real dataset
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_highlight_real_docs(self):
        dataset = Path(__file__).parent.parent.parent / \
            "data" / "sample_documents.json"
        if not dataset.exists():
            pytest.skip("Dataset not present")

        with open(dataset) as f:
            docs = json.load(f)

        from src.indexer.inverted_index import InvertedIndex
        from src.search.query_parser import QueryParser
        from src.search.retrieval import Retriever
        from src.search.ranking import Ranker

        idx = InvertedIndex()
        idx.build(docs)
        qp = QueryParser()
        ret = Retriever(idx)
        rnk = Ranker(idx)
        h = Highlighter()

        pq = qp.parse("information retrieval")
        results = rnk.rank(pq, ret.retrieve(pq))
        doc_map = {d["id"]: d["body"] for d in docs}

        for r in results[:3]:
            body = doc_map.get(r.doc_id, "")
            hr = h.highlight(body, pq.all_terms, window=250)
            assert isinstance(hr, HighlightResult)
            # At least one mark tag in documents that scored > 0
            if hr.match_count > 0:
                assert "<mark>" in hr.snippet

    def test_phrase_terms_highlighted(self):
        h = Highlighter()
        text = "The inverted index maps terms to documents efficiently."
        # Phrase ["invert", "index"] — both stems should highlight
        result = h.highlight(text, ["invert", "index"])
        assert result.match_count >= 2
        assert "<mark>inverted</mark>" in result.snippet
        assert "<mark>index</mark>" in result.snippet
