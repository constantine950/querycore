from __future__ import annotations

from src.indexer.inverted_index import InvertedIndex
from src.search.query_parser import ParsedQuery


class Retriever:
    def __init__(self, index: InvertedIndex) -> None:
        self._index = index

    def retrieve(self, pq: ParsedQuery) -> set[str]:
        if pq.is_empty:
            return set()

        candidates = self._resolve_and_terms(pq.terms)
        candidates |= self._resolve_or_terms(pq.or_terms)
        candidates -= self._resolve_excluded(pq.excluded)

        return candidates

    def retrieve_with_phrases(self, pq: ParsedQuery) -> set[str]:
        if pq.is_empty:
            return set()

        candidates = self._resolve_and_terms(pq.terms)
        candidates |= self._resolve_or_terms(pq.or_terms)

        # Phrase-only query: seed candidates from phrase token intersection
        if pq.is_phrase and not pq.terms and not pq.or_terms:
            for phrase in pq.phrases:
                if not phrase:
                    continue
                phrase_cands = self._index.get_doc_ids(phrase[0])
                for term in phrase[1:]:
                    phrase_cands &= self._index.get_doc_ids(term)
                candidates |= phrase_cands

        candidates -= self._resolve_excluded(pq.excluded)
        return candidates

    def retrieve_for_term(self, term: str) -> set[str]:
        return self._index.get_doc_ids(term)

    def count(self, pq: ParsedQuery) -> int:
        return len(self.retrieve(pq))

    def _resolve_and_terms(self, terms: list[str]) -> set[str]:
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
        result: set[str] = set()
        for term in or_terms:
            result |= self._index.get_doc_ids(term)
        return result

    def _resolve_excluded(self, excluded: list[str]) -> set[str]:
        result: set[str] = set()
        for term in excluded:
            result |= self._index.get_doc_ids(term)
        return result

    def explain(self, pq: ParsedQuery) -> dict:
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
