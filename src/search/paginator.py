"""
paginator.py

Slices a sorted list of SearchResult objects into pages and provides
metadata about the result set for the API and UI layers.

Day 10 (Ranker) returns results already sorted by score descending.
This module handles everything after that:

    - Secondary sort options (by date, by title) on top of score
    - Page slicing (page 1 = results 1-10, page 2 = 11-20, etc.)
    - Pagination metadata (total results, total pages, has_next, has_prev)
    - Result window for "showing X–Y of Z results"

Design
------
Pages are 1-indexed (page=1 is the first page) to match user-facing
conventions. Page size is configurable; default is 10.

The paginator is stateless — it takes a full result list and a page number
and returns a Page object. It does not store any state between calls.

Usage
-----
    from src.search.paginator import Paginator, SortBy

    paginator = Paginator(page_size=10)
    page = paginator.paginate(results, page=1, sort_by=SortBy.SCORE)

    page.results        # list of SearchResult for this page
    page.total          # total number of results across all pages
    page.total_pages    # total number of pages
    page.page           # current page number
    page.has_next       # True if there is a next page
    page.has_prev       # True if there is a previous page
    page.start          # 1-indexed position of first result on this page
    page.end            # 1-indexed position of last result on this page
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.search.ranking import SearchResult


# ---------------------------------------------------------------------------
# SortBy
# ---------------------------------------------------------------------------

class SortBy(str, Enum):
    """Available sort orders for search results."""
    SCORE = "score"     # default: TF-IDF score descending
    DATE = "date"      # most recent first
    TITLE = "title"     # alphabetical ascending


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

@dataclass
class Page:
    """
    A single page of search results with pagination metadata.

    Attributes:
        results     : The SearchResult objects on this page.
        page        : Current page number (1-indexed).
        page_size   : Maximum results per page.
        total       : Total number of results across all pages.
        total_pages : Total number of pages.
        has_next    : Whether a next page exists.
        has_prev    : Whether a previous page exists.
        start       : 1-indexed position of the first result on this page.
        end         : 1-indexed position of the last result on this page.
        sort_by     : Sort order applied to this page.
    """
    results:     list  # list[SearchResult]
    page:        int
    page_size:   int
    total:       int
    total_pages: int
    has_next:    bool
    has_prev:    bool
    start:       int
    end:         int
    sort_by:     SortBy

    def to_dict(self) -> dict:
        return {
            "page":        self.page,
            "page_size":   self.page_size,
            "total":       self.total,
            "total_pages": self.total_pages,
            "has_next":    self.has_next,
            "has_prev":    self.has_prev,
            "start":       self.start,
            "end":         self.end,
            "sort_by":     self.sort_by.value,
            "results":     [r.to_dict() for r in self.results],
        }


# ---------------------------------------------------------------------------
# Paginator
# ---------------------------------------------------------------------------

class Paginator:
    """
    Sorts and paginates a list of SearchResult objects.

    Stateless — all inputs come in, a Page comes out.
    """

    DEFAULT_PAGE_SIZE = 10
    MAX_PAGE_SIZE = 100

    def __init__(self, page_size: int = DEFAULT_PAGE_SIZE) -> None:
        self.page_size = max(1, min(page_size, self.MAX_PAGE_SIZE))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def paginate(
        self,
        results:  list,          # list[SearchResult]
        page:     int = 1,
        sort_by:  SortBy = SortBy.SCORE,
    ) -> Page:
        """
        Sort and slice a result list into a single page.

        Args:
            results : Full list of SearchResult objects (from Ranker.rank()).
            page    : 1-indexed page number to return. Clamped to valid range.
            sort_by : Sort order. SCORE is already applied by the Ranker;
                      DATE and TITLE re-sort the list.

        Returns:
            Page object with this page's results and pagination metadata.
        """
        if not results:
            return self._empty_page(page, sort_by)

        sorted_results = self._sort(results, sort_by)

        total = len(sorted_results)
        total_pages = math.ceil(total / self.page_size)
        page = max(1, min(page, total_pages))   # clamp to valid range

        start_idx = (page - 1) * self.page_size
        end_idx = start_idx + self.page_size
        page_results = sorted_results[start_idx:end_idx]

        return Page(
            results=page_results,
            page=page,
            page_size=self.page_size,
            total=total,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
            start=start_idx + 1,
            end=start_idx + len(page_results),
            sort_by=sort_by,
        )

    def get_page_range(self, total: int) -> int:
        """Return the total number of pages for a given result count."""
        return math.ceil(total / self.page_size) if total > 0 else 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sort(self, results: list, sort_by: SortBy) -> list:
        """Return a new sorted list; does not mutate the input."""
        if sort_by == SortBy.SCORE:
            return sorted(results, key=lambda r: r.score, reverse=True)
        elif sort_by == SortBy.DATE:
            return sorted(results, key=lambda r: r.date or "", reverse=True)
        elif sort_by == SortBy.TITLE:
            return sorted(results, key=lambda r: r.title.lower())
        return results

    def _empty_page(self, page: int, sort_by: SortBy) -> Page:
        return Page(
            results=[],
            page=page,
            page_size=self.page_size,
            total=0,
            total_pages=0,
            has_next=False,
            has_prev=False,
            start=0,
            end=0,
            sort_by=sort_by,
        )
