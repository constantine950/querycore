from __future__ import annotations

from dataclasses import dataclass

from src.indexer.inverted_index import InvertedIndex
from src.search.query_parser import ParsedQuery


def levenshtein(s1: str, s2: str) -> int:
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    if len(s1) > len(s2):
        s1, s2 = s2, s1

    m, n = len(s1), len(s2)
    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            if s2[i - 1] == s1[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev

    return prev[m]


@dataclass
class FuzzyMatch:
    query_term:  str
    index_term:  str
    distance:    int
    doc_count:   int


class FuzzyMatcher:

    def __init__(
        self,
        index:        InvertedIndex,
        max_distance: int = 2,
        min_term_len: int = 4,
    ) -> None:
        self._index = index
        self._max_distance = max_distance
        self._min_term_len = min_term_len

    def expand(self, query_term: str) -> set[str]:
        if len(query_term) < self._min_term_len:
            return {query_term} if query_term in set(self._index.get_all_terms()) else set()

        matches: set[str] = set()
        for index_term in self._index.get_all_terms():
            if abs(len(index_term) - len(query_term)) > self._max_distance:
                continue
            if levenshtein(query_term, index_term) <= self._max_distance:
                matches.add(index_term)
        return matches

    def find_matches(self, query_term: str) -> list[FuzzyMatch]:
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
        if pq.is_empty:
            return set()

        candidates: set[str] = set()
        for term in pq.terms + pq.or_terms:
            for expanded_term in self.expand(term):
                candidates |= self._index.get_doc_ids(expanded_term)

        for term in pq.excluded:
            for expanded_term in self.expand(term):
                candidates -= self._index.get_doc_ids(expanded_term)

        return candidates

    def suggest_correction(self, query_term: str) -> str | None:
        matches = self.find_matches(query_term)
        for m in matches:
            if m.distance == 0:
                return m.index_term
        return matches[0].index_term if matches else None
