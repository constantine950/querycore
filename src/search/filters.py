"""
filters.py

Post-retrieval filtering: narrows a candidate set or result list by
document metadata before (or after) ranking.

Two application points
----------------------
1. Pre-ranking  — filter the candidate set (set of doc_ids) so the Ranker
   only scores documents that pass all filters. Cheaper: avoids scoring
   docs that will be thrown away.

2. Post-ranking — filter a list of SearchResult objects. Useful when the
   API caller wants to apply filters without re-running retrieval.

Both modes are supported.

Filter types
------------
- CategoryFilter   : exact match on doc category (e.g. "computer_science")
- DateRangeFilter  : documents whose date falls within [start, end]
- WordCountFilter  : documents whose word_count is within [min, max]
- CompositeFilter  : AND-chains multiple filters together

Usage
-----
    from src.search.filters import FilterSet

    fs = FilterSet(index)
    fs.add_category("computer_science")
    fs.add_date_range("2024-01-01", "2024-06-30")

    # Pre-rank: filter candidate doc_ids
    narrowed = fs.apply_to_candidates(candidates)

    # Post-rank: filter SearchResult list
    narrowed = fs.apply_to_results(results)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.search.ranking import SearchResult

from src.indexer.inverted_index import InvertedIndex


# ---------------------------------------------------------------------------
# Individual filter dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CategoryFilter:
    """Match documents whose category is in the allowed set."""
    categories: set[str]   # e.g. {"computer_science", "science"}

    def matches_meta(self, meta: dict) -> bool:
        return meta.get("category", "") in self.categories


@dataclass
class DateRangeFilter:
    """Match documents whose date falls within [start, end] inclusive."""
    start: date | None = None   # None means no lower bound
    end:   date | None = None   # None means no upper bound

    def matches_meta(self, meta: dict) -> bool:
        raw = meta.get("date", "")
        if not raw:
            return self.start is None and self.end is None
        try:
            doc_date = date.fromisoformat(raw)
        except ValueError:
            return False
        if self.start and doc_date < self.start:
            return False
        if self.end and doc_date > self.end:
            return False
        return True


@dataclass
class WordCountFilter:
    """Match documents whose word count falls within [min_words, max_words]."""
    min_words: int | None = None
    max_words: int | None = None

    def matches_meta(self, meta: dict) -> bool:
        # word_count may not be in snippet metadata — fall back to snippet length
        raw = meta.get("word_count", None)
        if raw is None:
            # Estimate from snippet if word_count not stored
            snippet = meta.get("snippet", "")
            raw = len(snippet.split())
        wc = int(raw)
        if self.min_words is not None and wc < self.min_words:
            return False
        if self.max_words is not None and wc > self.max_words:
            return False
        return True


# ---------------------------------------------------------------------------
# FilterSet — the public interface
# ---------------------------------------------------------------------------

class FilterSet:
    """
    Composes multiple filters and applies them to candidate sets or results.

    Filters are AND-chained: a document must pass ALL filters to be included.
    """

    def __init__(self, index: InvertedIndex) -> None:
        self._index = index
        self._filters: list = []    # list of filter objects with matches_meta()

    # ------------------------------------------------------------------
    # Builder methods (chainable)
    # ------------------------------------------------------------------

    def add_category(self, *categories: str) -> "FilterSet":
        """
        Add a category filter. Documents must belong to one of the given
        categories to pass.

        Args:
            categories: One or more category strings.
        """
        self._filters.append(CategoryFilter(categories=set(categories)))
        return self

    def add_date_range(
        self,
        start: str | date | None = None,
        end:   str | date | None = None,
    ) -> "FilterSet":
        """
        Add a date range filter.

        Args:
            start: ISO date string or date object (inclusive lower bound).
            end:   ISO date string or date object (inclusive upper bound).
        """
        start_date = date.fromisoformat(
            start) if isinstance(start, str) else start
        end_date = date.fromisoformat(end) if isinstance(end, str) else end
        self._filters.append(DateRangeFilter(start=start_date, end=end_date))
        return self

    def add_word_count(
        self,
        min_words: int | None = None,
        max_words: int | None = None,
    ) -> "FilterSet":
        """
        Add a word count range filter.

        Args:
            min_words: Minimum word count (inclusive).
            max_words: Maximum word count (inclusive).
        """
        self._filters.append(WordCountFilter(
            min_words=min_words, max_words=max_words))
        return self

    def clear(self) -> "FilterSet":
        """Remove all filters."""
        self._filters.clear()
        return self

    @property
    def active(self) -> bool:
        """True if at least one filter is configured."""
        return len(self._filters) > 0

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def apply_to_candidates(self, candidates: set[str]) -> set[str]:
        """
        Filter a set of doc_ids using stored metadata from the index.
        Use this before ranking to avoid scoring filtered-out documents.

        Args:
            candidates: Set of doc_ids from Retriever.retrieve().

        Returns:
            Subset of candidates that pass all filters.
        """
        if not self._filters:
            return candidates

        return {
            doc_id for doc_id in candidates
            if self._passes_all(self._index.get_metadata(doc_id))
        }

    def apply_to_results(self, results: list) -> list:
        """
        Filter a list of SearchResult objects.
        Use this after ranking when you want to preserve rank order within
        the filtered set.

        Args:
            results: List of SearchResult from Ranker.rank().

        Returns:
            Filtered list in the same order.
        """
        if not self._filters:
            return results

        return [
            r for r in results
            if self._passes_all_from_result(r)
        ]

    def matches(self, doc_id: str) -> bool:
        """Check whether a single document passes all filters."""
        meta = self._index.get_metadata(doc_id)
        return self._passes_all(meta)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _passes_all(self, meta: dict) -> bool:
        return all(f.matches_meta(meta) for f in self._filters)

    def _passes_all_from_result(self, result) -> bool:
        """Build a minimal meta dict from a SearchResult for filter checking."""
        meta = {
            "category":   result.category,
            "date":       result.date,
            "snippet":    result.snippet,
        }
        return self._passes_all(meta)
