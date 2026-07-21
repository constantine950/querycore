"""
phrase_match.py

Filters a candidate set down to only documents where a quoted phrase
appears as a consecutive sequence of tokens.

How it works
------------
At index time (Day 6), every Posting stores a `positions` list — the
token positions where that term appears in the document. For example:

    doc: "search engine indexes documents for fast retrieval"
          pos 0     1       2       3         4    5     6

    Posting("search", doc_001, positions=[0])
    Posting("engine", doc_001, positions=[1])

A phrase ["search", "engin"] matches doc_001 because there exists a
starting position p such that:
    - "search" appears at position p     (p=0 ✓)
    - "engin"  appears at position p+1   (p+1=1 ✓)

For a phrase of length k, all k terms must appear at consecutive
positions p, p+1, p+2, ..., p+k-1 for the same p.

Algorithm
---------
1. For each candidate doc_id:
   a. Get the position list for the first phrase term in that doc.
   b. For each candidate start position p in that list:
      - Check that term[1] has position p+1, term[2] has p+2, etc.
      - If all check out → phrase matches this doc.
2. A document must satisfy ALL phrases in the query (AND semantics).

Complexity: O(candidates × phrase_len × avg_positions_per_term)
This is fast in practice because candidates are already a small set.

Usage
-----
    from src.search.phrase_match import PhraseFilter

    pf = PhraseFilter(index)
    matched = pf.filter(candidates, pq.phrases)
    # → subset of candidates where all phrases match positionally
"""

from __future__ import annotations

from src.indexer.inverted_index import InvertedIndex


class PhraseFilter:
    """
    Filters a candidate doc_id set to those where all query phrases
    appear as consecutive token sequences.
    """

    def __init__(self, index: InvertedIndex) -> None:
        self._index = index

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(self, candidates: set[str], phrases: list[list[str]]) -> set[str]:
        """
        Return the subset of candidates that contain all phrases.

        Args:
            candidates : Doc IDs from Retriever.retrieve().
            phrases    : List of stemmed token lists from ParsedQuery.phrases.
                         Each inner list is one phrase; all must match (AND).

        Returns:
            Subset of candidates satisfying all phrase constraints.
            If phrases is empty, returns candidates unchanged.
        """
        if not phrases:
            return candidates

        result = set(candidates)
        for phrase in phrases:
            if not phrase:
                continue
            result = {
                doc_id for doc_id in result
                if self._phrase_matches(doc_id, phrase)
            }
            if not result:
                break   # short-circuit: no docs can satisfy remaining phrases

        return result

    def matches(self, doc_id: str, phrase: list[str]) -> bool:
        """
        Check whether a single phrase appears consecutively in a document.

        Args:
            doc_id : Document to check.
            phrase : Ordered list of stemmed tokens.

        Returns:
            True if the phrase appears consecutively anywhere in the doc.
        """
        return self._phrase_matches(doc_id, phrase)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _phrase_matches(self, doc_id: str, phrase: list[str]) -> bool:
        """
        Core positional adjacency check.

        Strategy: anchor on the first term's positions, then verify that
        each subsequent term appears at the expected offset.
        """
        if not phrase:
            return False

        # Get start positions from the first term
        start_positions = self._index.get_positions(phrase[0], doc_id)
        if not start_positions:
            return False

        # Single-term "phrase" — just check it exists (already confirmed above)
        if len(phrase) == 1:
            return True

        # Pre-fetch position sets for all remaining terms (O(1) lookup per check)
        subsequent: list[set[int]] = []
        for term in phrase[1:]:
            positions = self._index.get_positions(term, doc_id)
            if not positions:
                return False   # term not in doc at all → phrase impossible
            subsequent.append(set(positions))

        # Check each candidate starting position
        for start in start_positions:
            if all(
                (start + offset + 1) in subsequent[offset]
                for offset in range(len(subsequent))
            ):
                return True

        return False

    def find_phrase_positions(self, doc_id: str, phrase: list[str]) -> list[int]:
        """
        Return all starting positions where the phrase occurs in the document.
        Used by the highlighter (Day 18) to mark phrase matches.

        Args:
            doc_id : Document to search.
            phrase : Ordered list of stemmed tokens.

        Returns:
            List of starting positions (may be empty).
        """
        if not phrase:
            return []

        start_positions = self._index.get_positions(phrase[0], doc_id)
        if not start_positions:
            return []

        if len(phrase) == 1:
            return list(start_positions)

        subsequent: list[set[int]] = []
        for term in phrase[1:]:
            positions = self._index.get_positions(term, doc_id)
            if not positions:
                return []
            subsequent.append(set(positions))

        matched_starts = []
        for start in start_positions:
            if all(
                (start + offset + 1) in subsequent[offset]
                for offset in range(len(subsequent))
            ):
                matched_starts.append(start)

        return matched_starts
