"""
query_parser.py

Parses a raw user query string into a structured ParsedQuery object that
the retrieval layer (Day 9) can act on without re-parsing.

Query types supported
---------------------
1. Standard (default)
   Input : "search engine ranking"
   Result: three term tokens, AND semantics (docs must contain all terms)

2. Phrase query  — terms wrapped in double quotes
   Input : '"inverted index" ranking'
   Result: one phrase ["inverted", "index"] + one term "ranking"
   Phrase matching enforces positional adjacency (enforced in Day 12).

3. Boolean operators  — OR between terms
   Input : "search OR retrieval"
   Result: terms with OR semantics (docs must contain at least one term)

4. Exclusion  — minus prefix
   Input : "search -engine"
   Result: required term "search", excluded term "engine"

All terms are passed through the same Tokenizer → Preprocessor pipeline
used at index time, so stems match what was stored. Phrases are tokenized
but NOT stemmed individually — each phrase is stored as a list of stemmed
tokens so the phrase matcher can check positional adjacency.

Usage
-----
    from src.search.query_parser import QueryParser

    qp = QueryParser()
    pq = qp.parse("search engine OR retrieval -database")
    pq.terms        # ["search", "engin"]        (stemmed, AND semantics)
    pq.or_terms     # ["retriev"]                (stemmed, OR semantics)
    pq.excluded     # ["databas"]                (stemmed, must NOT appear)
    pq.phrases      # []                         (phrase token lists)
    pq.raw          # "search engine OR retrieval -database"
    pq.is_empty     # False
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.indexer.tokenizer import Tokenizer
from src.indexer.preprocessor import Preprocessor


# ---------------------------------------------------------------------------
# ParsedQuery
# ---------------------------------------------------------------------------

@dataclass
class ParsedQuery:
    """
    Structured representation of a user query after parsing and stemming.

    Attributes:
        raw         : The original unmodified query string.
        terms       : Stemmed tokens with AND semantics (all must match).
        or_terms    : Stemmed tokens with OR semantics (any must match).
        excluded    : Stemmed tokens that must NOT appear in results.
        phrases     : List of phrases; each phrase is a list of stemmed tokens
                      that must appear consecutively in a document.
        is_phrase   : True if the query contained at least one phrase.
        is_boolean  : True if the query used OR or exclusion operators.
    """
    raw: str = ""
    terms: list[str] = field(default_factory=list)
    or_terms: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    phrases: list[list[str]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.terms or self.or_terms or self.phrases)

    @property
    def is_phrase(self) -> bool:
        return len(self.phrases) > 0

    @property
    def is_boolean(self) -> bool:
        return len(self.or_terms) > 0 or len(self.excluded) > 0

    @property
    def all_terms(self) -> list[str]:
        """All unique stemmed terms across AND, OR, and phrase slots."""
        seen = set()
        result = []
        for t in self.terms + self.or_terms + [t for p in self.phrases for t in p]:
            if t not in seen:
                seen.add(t)
                result.append(t)
        return result


# ---------------------------------------------------------------------------
# QueryParser
# ---------------------------------------------------------------------------

class QueryParser:
    """
    Parses raw query strings into ParsedQuery objects.

    The parser is intentionally simple and explicit. It handles the four
    patterns listed above using regex extraction rather than a grammar,
    which keeps it readable and debuggable without sacrificing correctness
    for the query types QueryCore supports.
    """

    # Matches "double quoted phrases"
    _PHRASE_RE = re.compile(r'"([^"]+)"')
    # Matches -excluded tokens (must be preceded by whitespace or start of string)
    _EXCLUDE_RE = re.compile(r'(?:^|\s)-(\S+)')
    # Matches OR keyword (case-insensitive, surrounded by whitespace)
    _OR_RE = re.compile(r'\bOR\b')

    def __init__(self) -> None:
        self._tokenizer = Tokenizer()
        self._preprocessor = Preprocessor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, raw: str) -> ParsedQuery:
        """
        Parse a raw query string into a ParsedQuery.

        Args:
            raw: The user's query string, e.g. '"search engine" OR retrieval -database'

        Returns:
            ParsedQuery with terms, or_terms, excluded, and phrases populated.
        """
        if not raw or not raw.strip():
            return ParsedQuery(raw=raw)

        pq = ParsedQuery(raw=raw)
        working = raw

        # 1. Extract phrases first (before tokenizing ruins the quote structure)
        phrases, working = self._extract_phrases(working)
        pq.phrases = phrases

        # 2. Extract excluded terms (-word)
        excluded, working = self._extract_excluded(working)
        pq.excluded = excluded

        # 3. Detect OR operator — split remaining text on OR
        or_terms, and_terms, working = self._extract_or_terms(working)
        pq.or_terms = or_terms
        pq.terms = and_terms

        return pq

    # ------------------------------------------------------------------
    # Private extraction helpers
    # ------------------------------------------------------------------

    def _stem_tokens(self, tokens: list[str]) -> list[str]:
        """Apply preprocessor stemming to a list of already-lowercased tokens."""
        return self._preprocessor.process(tokens)

    def _tokenize_and_stem(self, text: str) -> list[str]:
        """Tokenize raw text and run through the full preprocessing pipeline."""
        tokens = self._tokenizer.tokenize(text)
        return self._preprocessor.process(tokens)

    def _extract_phrases(self, text: str) -> tuple[list[list[str]], str]:
        """
        Find all "quoted phrases", tokenize + stem each one, remove from text.

        Returns (list_of_stemmed_phrase_lists, text_with_phrases_removed).
        """
        phrases: list[list[str]] = []
        matches = list(self._PHRASE_RE.finditer(text))

        # reverse so slicing doesn't shift offsets
        for match in reversed(matches):
            phrase_text = match.group(1)
            tokens = self._tokenizer.tokenize(phrase_text)
            stemmed = [self._preprocessor.stem(t) for t in tokens
                       if not self._preprocessor.is_stopword(t) and len(t) >= 2]
            if stemmed:
                phrases.append(stemmed)
            text = text[:match.start()] + " " + text[match.end():]

        phrases.reverse()   # restore original order after reversed() processing
        return phrases, text

    def _extract_excluded(self, text: str) -> tuple[list[str], str]:
        """
        Find all -excluded tokens, stem them, remove from text.

        Returns (list_of_stemmed_excluded_terms, text_with_exclusions_removed).
        """
        excluded: list[str] = []
        matches = list(self._EXCLUDE_RE.finditer(text))

        for match in reversed(matches):
            raw_term = match.group(1)
            stemmed = self._tokenize_and_stem(raw_term)
            excluded.extend(stemmed)
            # Remove the matched exclusion from text (keep surrounding whitespace clean)
            text = text[:match.start()] + " " + text[match.end():]

        excluded.reverse()
        return excluded, text

    def _extract_or_terms(self, text: str) -> tuple[list[str], list[str], str]:
        """
        Split text on OR keyword. Terms to the right of OR get OR semantics,
        terms to the left get AND semantics.

        Handles multiple OR clauses: "a OR b OR c" → all three get OR semantics.

        Returns (or_terms, and_terms, remaining_text).
        """
        parts = self._OR_RE.split(text)

        if len(parts) == 1:
            # No OR — all terms are AND terms
            and_terms = self._tokenize_and_stem(text)
            return [], and_terms, text

        # First part is AND, remaining parts are OR
        and_terms = self._tokenize_and_stem(parts[0])
        or_terms: list[str] = []
        for part in parts[1:]:
            or_terms.extend(self._tokenize_and_stem(part))

        return or_terms, and_terms, ""
