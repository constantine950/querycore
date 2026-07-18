"""
retrieval.py

Stage 1 of query execution: given a ParsedQuery, find all candidate
document IDs that satisfy the query's boolean constraints.

This module does NOT rank results — ranking is Day 10 (TF-IDF scoring).
The job here is purely set operations on posting lists:

    AND terms  → intersection  (doc must contain ALL required terms)
    OR  terms  → union         (doc must contain AT LEAST ONE or-term)
    excluded   → difference    (doc must contain NONE of the excluded terms)
    phrases    → subset        (candidate set filtered further in Day 12)

Pipeline position:
    ParsedQuery → Retrieval → {candidate doc_ids} → Ranker → [ranked results]

Design
------
- Works entirely on sets of doc_ids — no score computation here.
- AND semantics are strict by default: if a query has three terms and a
  doc only contains two, it is excluded from candidates. This is the correct
  behaviour for precision; Day 13 (fuzzy) relaxes this for typo tolerance.
- OR terms are unioned into the candidate set AFTER the AND intersection,
  so "search OR retrieval" returns docs that contain "search" AND/OR docs
  that contain "retrieval".
- Excluded terms are subtracted last.
- An empty ParsedQuery returns an empty set (not all docs).

Usage
-----
    from src.indexer.inverted_index import InvertedIndex
    from src.search.query_parser import QueryParser
    from src.search.retrieval import Retriever

    idx = InvertedIndex()
    idx.build(docs)

    qp  = QueryParser()
    ret = Retriever(idx)

    pq  = qp.parse("search engine -database")
    candidates = ret.retrieve(pq)
    # → {"doc_001", "doc_004", ...}
"""

from __future__ import annotations

from src.indexer.inverted_index import InvertedIndex
from src.search.query_parser import ParsedQuery


class Retriever:
    """
    Fetches candidate document IDs from the inverted index for a ParsedQuery.

    Does not score or rank — returns a plain set of doc_ids that pass all
    boolean constraints and are ready to be handed to the Ranker.
    """

    def __init__(self, index: InvertedIndex) -> None:
        self._index = index

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, pq: ParsedQuery) -> set[str]:
        """
        Return the set of doc_ids that satisfy the ParsedQuery constraints.

        Args:
            pq: A ParsedQuery from QueryParser.parse().

        Returns:
            Set of document IDs. Empty set if nothing matches or query is empty.
        """
        if pq.is_empty:
            return set()

        candidates = self._resolve_and_terms(pq.terms)
        candidates |= self._resolve_or_terms(pq.or_terms)
        candidates -= self._resolve_excluded(pq.excluded)

        return candidates

    def retrieve_for_term(self, term: str) -> set[str]:
        """
        Return all doc_ids containing a single stemmed term.
        Convenience method used by the fuzzy search layer (Day 13).

        Args:
            term: A stemmed, lowercase token.

        Returns:
            Set of doc_ids containing the term.
        """
        return self._index.get_doc_ids(term)

    def count(self, pq: ParsedQuery) -> int:
        """Return the number of candidate documents for a query."""
        return len(self.retrieve(pq))

    # ------------------------------------------------------------------
    # Private set-operation helpers
    # ------------------------------------------------------------------

    def _resolve_and_terms(self, terms: list[str]) -> set[str]:
        """
        Intersect posting lists for all AND terms.

        Strategy: start with the smallest posting list (cheapest intersection)
        and progressively intersect. If any term has zero postings the result
        is immediately empty.

        Returns empty set if terms list is empty.
        """
        if not terms:
            return set()

        # Sort by posting list size ascending — smallest first for fast pruning
        sorted_terms = sorted(
            terms,
            key=lambda t: len(self._index.get_postings(t))
        )

        result = self._index.get_doc_ids(sorted_terms[0])
        if not result:
            return set()

        for term in sorted_terms[1:]:
            result &= self._index.get_doc_ids(term)
            if not result:
                break   # short-circuit: intersection already empty

        return result

    def _resolve_or_terms(self, or_terms: list[str]) -> set[str]:
        """
        Union posting lists for all OR terms.

        Returns empty set if or_terms list is empty.
        """
        result: set[str] = set()
        for term in or_terms:
            result |= self._index.get_doc_ids(term)
        return result

    def _resolve_excluded(self, excluded: list[str]) -> set[str]:
        """
        Union posting lists for all excluded terms — these doc_ids will be
        subtracted from the candidate set.

        Returns empty set if excluded list is empty.
        """
        result: set[str] = set()
        for term in excluded:
            result |= self._index.get_doc_ids(term)
        return result

    # ------------------------------------------------------------------
    # Diagnostic helpers (useful for debugging / Day 25 perf tests)
    # ------------------------------------------------------------------

    def explain(self, pq: ParsedQuery) -> dict:
        """
        Return a breakdown of how the candidate set was constructed.
        Useful for debugging and understanding retrieval behaviour.

        Returns:
            Dict with keys: and_candidates, or_candidates, excluded_docs,
            final_candidates, counts.
        """
        and_cands = self._resolve_and_terms(pq.terms)
        or_cands = self._resolve_or_terms(pq.or_terms)
        excl_docs = self._resolve_excluded(pq.excluded)
        final = (and_cands | or_cands) - excl_docs

        return {
            "query":            pq.raw,
            "and_terms":        pq.terms,
            "or_terms":         pq.or_terms,
            "excluded_terms":   pq.excluded,
            "and_candidates":   sorted(and_cands),
            "or_candidates":    sorted(or_cands),
            "excluded_docs":    sorted(excl_docs),
            "final_candidates": sorted(final),
            "counts": {
                "and":      len(and_cands),
                "or":       len(or_cands),
                "excluded": len(excl_docs),
                "final":    len(final),
            },
        }
