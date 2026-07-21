"""
fuzzy_search.py

Typo-tolerant search using Levenshtein (edit) distance.

When a query term matches no documents exactly, the fuzzy layer finds
index terms within a configurable edit distance and expands the query
to include those near-matches.

What is edit distance?
----------------------
The Levenshtein distance between two strings is the minimum number of
single-character edits (insertions, deletions, substitutions) needed
to transform one string into the other.

    "serach"  → "search"   distance = 1  (swap a↔r)
    "engne"   → "engine"   distance = 1  (insert i)
    "retreval"→ "retrieval" distance = 2

Algorithm
---------
The classic dynamic-programming approach builds an (m+1) × (n+1) matrix
where cell [i][j] holds the edit distance between s1[:i] and s2[:j].

Time:  O(m × n) per comparison
Space: O(min(m, n)) — only two rows needed at a time

Practical thresholds used in search engines:
    distance 1 — catches most typos (off-by-one-key, transpositions)
    distance 2 — catches worse typos, but increases false positives
    distance 3+ — rarely useful; too many false matches

QueryCore default: max_distance=2, applied only to terms ≥ 4 chars
(short terms have too many near-neighbours to be useful).

Usage
-----
    from src.search.fuzzy_search import FuzzyMatcher

    fm = FuzzyMatcher(index)
    expansions = fm.expand("serach")
    # → {"search"}  (stemmed terms from the index within edit distance)

    candidates = fm.retrieve_fuzzy(pq)
    # → set of doc_ids from exact + fuzzy matches combined
"""

from __future__ import annotations

from dataclasses import dataclass

from src.indexer.inverted_index import InvertedIndex
from src.search.query_parser import ParsedQuery


# ---------------------------------------------------------------------------
# Edit distance
# ---------------------------------------------------------------------------

def levenshtein(s1: str, s2: str) -> int:
    """
    Compute the Levenshtein edit distance between two strings.

    Uses a space-optimised two-row DP approach: O(min(m,n)) space.

    Args:
        s1, s2: Strings to compare. Case-sensitive — normalise before calling.

    Returns:
        Integer edit distance ≥ 0.
    """
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    # Ensure s1 is the shorter string (saves memory)
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    m, n = len(s1), len(s2)
    prev = list(range(m + 1))   # prev[j] = dist(s1[:j], s2[:0..i-1])
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            if s2[i - 1] == s1[j - 1]:
                curr[j] = prev[j - 1]          # characters match — no edit
            else:
                curr[j] = 1 + min(
                    prev[j],        # deletion
                    curr[j - 1],    # insertion
                    prev[j - 1],    # substitution
                )
        prev, curr = curr, prev

    return prev[m]


# ---------------------------------------------------------------------------
# FuzzyMatch result
# ---------------------------------------------------------------------------

@dataclass
class FuzzyMatch:
    """A single fuzzy match result."""
    query_term:  str    # the original (possibly misspelled) query token
    index_term:  str    # the matching term found in the index
    distance:    int    # edit distance between the two
    doc_count:   int    # number of documents the index_term appears in


# ---------------------------------------------------------------------------
# FuzzyMatcher
# ---------------------------------------------------------------------------

class FuzzyMatcher:
    """
    Finds index terms within edit distance of a query term and expands
    the candidate document set to include fuzzy matches.
    """

    def __init__(
        self,
        index:        InvertedIndex,
        max_distance: int = 2,
        min_term_len: int = 4,
    ) -> None:
        self._index = index
        self._max_distance = max_distance
        self._min_term_len = min_term_len

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expand(self, query_term: str) -> set[str]:
        """
        Find all index terms within max_distance of query_term.

        Args:
            query_term: A single stemmed, lowercase token.

        Returns:
            Set of matching index terms (may include query_term itself
            if it's in the index).
        """
        if len(query_term) < self._min_term_len:
            return {query_term} if query_term in set(self._index.get_all_terms()) else set()

        matches: set[str] = set()
        for index_term in self._index.get_all_terms():
            # Early length pruning: if lengths differ by more than max_distance,
            # edit distance must exceed max_distance — skip without computing.
            if abs(len(index_term) - len(query_term)) > self._max_distance:
                continue
            dist = levenshtein(query_term, index_term)
            if dist <= self._max_distance:
                matches.add(index_term)
        return matches

    def find_matches(self, query_term: str) -> list[FuzzyMatch]:
        """
        Return detailed FuzzyMatch objects for all near-matches of a term,
        sorted by edit distance then by document count descending.

        Useful for autocorrect suggestions and debugging.

        Args:
            query_term: A single stemmed, lowercase token.

        Returns:
            List of FuzzyMatch sorted by (distance, -doc_count).
        """
        if len(query_term) < self._min_term_len:
            return []

        results: list[FuzzyMatch] = []
        for index_term in self._index.get_all_terms():
            if abs(len(index_term) - len(query_term)) > self._max_distance:
                continue
            dist = levenshtein(query_term, index_term)
            if dist <= self._max_distance:
                results.append(FuzzyMatch(
                    query_term=query_term,
                    index_term=index_term,
                    distance=dist,
                    doc_count=len(self._index.get_postings(index_term)),
                ))

        results.sort(key=lambda m: (m.distance, -m.doc_count))
        return results

    def retrieve_fuzzy(self, pq: ParsedQuery) -> set[str]:
        """
        Retrieve candidate doc_ids for a ParsedQuery using fuzzy expansion.

        For each AND term and OR term in the query, expands to all near-matches
        in the index and unions their posting lists. This is used as a fallback
        when exact retrieval returns no results, or as an augmentation layer.

        Args:
            pq: ParsedQuery from QueryParser.parse().

        Returns:
            Set of doc_ids that match any fuzzy expansion of any query term.
        """
        if pq.is_empty:
            return set()

        candidates: set[str] = set()
        all_terms = pq.terms + pq.or_terms

        for term in all_terms:
            for expanded_term in self.expand(term):
                candidates |= self._index.get_doc_ids(expanded_term)

        # Apply exclusions
        for term in pq.excluded:
            for expanded_term in self.expand(term):
                candidates -= self._index.get_doc_ids(expanded_term)

        return candidates

    def suggest_correction(self, query_term: str) -> str | None:
        """
        Return the most likely correct spelling for a misspelled term.
        Picks the index term with the lowest edit distance; breaks ties
        by choosing the term that appears in the most documents.

        Args:
            query_term: A (possibly misspelled) stemmed token.

        Returns:
            The best correction, or None if no close match found.
        """
        matches = self.find_matches(query_term)
        # Prefer exact match if present
        for m in matches:
            if m.distance == 0:
                return m.index_term
        return matches[0].index_term if matches else None
