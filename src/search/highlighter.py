"""
highlighter.py

Extracts a relevant snippet from a document body and wraps matching
query terms in highlight markers.

Two responsibilities
--------------------
1. Snippet extraction
   Rather than always showing the first 200 characters, find the passage
   in the document body that contains the most query terms and return
   that window. This gives users context around why a document matched.

2. Term highlighting
   Wrap each occurrence of a query term (or its unstemmed variants) in
   a configurable marker. Default: <mark>term</mark> for HTML rendering.
   The UI (Day 22) can style <mark> tags however it likes.

Design decisions
----------------
- Highlighting works on the *original* (unstemmed) text because users
  see the raw document, not the stemmed index tokens. We reverse-map
  stemmed query terms back to surface forms by checking which words in
  the snippet reduce to a query stem.
- Snippet window is configurable (default 300 chars). For phrase queries
  the window is anchored on the first phrase occurrence.
- Case-insensitive matching: "Search" and "search" both highlight.
- Overlapping matches are handled correctly (longest match wins).

Usage
-----
    from src.search.highlighter import Highlighter

    h = Highlighter(preprocessor)
    result = h.highlight(
        text   = "A search engine indexes documents for fast retrieval.",
        terms  = ["search", "engin"],   # stemmed query terms
        window = 300,
    )
    result.snippet   # "A <mark>search</mark> <mark>engine</mark> indexes..."
    result.positions # [(2, 8), (9, 15)]   char spans of matches
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.indexer.preprocessor import Preprocessor
from src.indexer.tokenizer import Tokenizer


# ---------------------------------------------------------------------------
# HighlightResult
# ---------------------------------------------------------------------------

@dataclass
class HighlightResult:
    """
    Output of the Highlighter.

    Attributes:
        snippet   : Extracted passage with highlight markers inserted.
        raw       : Same passage without any highlight markers.
        positions : List of (start, end) char spans that were highlighted,
                    relative to the start of `raw`.
        match_count: Number of query term matches found.
    """
    snippet:     str
    raw:         str
    positions:   list[tuple[int, int]] = field(default_factory=list)
    match_count: int = 0


# ---------------------------------------------------------------------------
# Highlighter
# ---------------------------------------------------------------------------

class Highlighter:
    """
    Finds relevant snippets and highlights query term matches.
    """

    def __init__(
        self,
        preprocessor: Preprocessor | None = None,
        open_tag:  str = "<mark>",
        close_tag: str = "</mark>",
    ) -> None:
        self._preprocessor = preprocessor or Preprocessor()
        self._tokenizer = Tokenizer()
        self._open = open_tag
        self._close = close_tag

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def highlight(
        self,
        text:   str,
        terms:  list[str],       # stemmed query terms
        window: int = 300,
    ) -> HighlightResult:
        """
        Extract a relevant snippet from text and highlight query terms.

        Args:
            text   : Full document body (original, unstemmed).
            terms  : Stemmed query terms from ParsedQuery.all_terms.
            window : Maximum character length of the extracted snippet.

        Returns:
            HighlightResult with snippet, raw, positions, match_count.
        """
        if not text or not terms:
            raw = text[:window] if text else ""
            return HighlightResult(snippet=raw, raw=raw)

        # Find the best window in the text
        raw_snippet = self._extract_window(text, terms, window)

        # Find highlight positions within that window
        positions = self._find_positions(raw_snippet, terms)

        # Build highlighted snippet by inserting tags (back-to-front to
        # preserve positions)
        highlighted = self._insert_tags(raw_snippet, positions)

        return HighlightResult(
            snippet=highlighted,
            raw=raw_snippet,
            positions=positions,
            match_count=len(positions),
        )

    def highlight_result(self, result, full_body: str, terms: list[str], window: int = 300):
        """
        Convenience: highlight a SearchResult in place.
        Replaces result.snippet with the highlighted version.

        Args:
            result    : SearchResult object (mutated in place).
            full_body : Full document body text.
            terms     : Stemmed query terms.
            window    : Snippet window size.

        Returns:
            The same SearchResult with snippet updated.
        """
        hr = self.highlight(full_body, terms, window)
        result.snippet = hr.snippet
        return result

    # ------------------------------------------------------------------
    # Snippet extraction
    # ------------------------------------------------------------------

    def _extract_window(self, text: str, terms: list[str], window: int) -> str:
        """
        Find the passage of `window` characters that contains the most
        query term matches, and return it.
        """
        if len(text) <= window:
            return text

        match_offsets = self._find_match_offsets(text, terms)

        if not match_offsets:
            return self._snap_to_word(text, 0, window)

        best_start = 0
        best_count = 0

        for start_offset in match_offsets:
            end_offset = start_offset + window
            count = sum(1 for o in match_offsets if start_offset <=
                        o <= end_offset)
            if count > best_count:
                best_count = count
                best_start = start_offset

        start = max(0, best_start - 40)
        return self._snap_to_word(text, start, window)

    def _find_match_offsets(self, text: str, terms: list[str]) -> list[int]:
        """Return character offsets of words whose stem matches a query term."""
        offsets = []
        for match in re.finditer(r'\b\w+\b', text):
            word = match.group()
            stem = self._preprocessor.stem(word.lower())
            if stem in terms:
                offsets.append(match.start())
        return offsets

    def _snap_to_word(self, text: str, start: int, window: int) -> str:
        """Extract text[start:start+window] snapped to word boundaries."""
        end = min(start + window, len(text))

        if start > 0:
            space = text.find(' ', start)
            if space != -1 and space < start + 20:
                start = space + 1

        if end < len(text):
            space = text.rfind(' ', start, end)
            if space != -1:
                end = space

        snippet = text[start:end].strip()
        if start > 0:
            snippet = "…" + snippet
        if end < len(text):
            snippet = snippet + "…"
        return snippet

    # ------------------------------------------------------------------
    # Position finding
    # ------------------------------------------------------------------

    def _find_positions(self, text: str, terms: list[str]) -> list[tuple[int, int]]:
        """Find all (start, end) char spans where a word stem matches a query term."""
        positions: list[tuple[int, int]] = []
        seen_starts: set[int] = set()

        for match in re.finditer(r'\b\w+\b', text):
            word = match.group()
            stem = self._preprocessor.stem(word.lower())
            if stem in terms and match.start() not in seen_starts:
                positions.append((match.start(), match.end()))
                seen_starts.add(match.start())

        return sorted(positions)

    # ------------------------------------------------------------------
    # Tag insertion
    # ------------------------------------------------------------------

    def _insert_tags(self, text: str, positions: list[tuple[int, int]]) -> str:
        """Insert open/close tags around each position span, back-to-front."""
        result = text
        for start, end in reversed(positions):
            result = (
                result[:start]
                + self._open
                + result[start:end]
                + self._close
                + result[end:]
            )
        return result
