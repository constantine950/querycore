"""
tracker.py

Records search events and exposes aggregated analytics:
    - query frequency (top searches)
    - latency distribution per query
    - zero-result queries (queries that returned nothing)
    - search volume over time (by day)

Storage
-------
In-memory by default (resets on restart). A JSON file path can be passed
to persist events across restarts — the tracker loads existing events on
init and appends new ones on each log() call.

This is intentionally simple: for a portfolio project, a flat list of
SearchEvent records is sufficient. A production system would use a time-
series database or a log aggregation pipeline.

Usage
-----
    from src.analytics.tracker import SearchTracker

    tracker = SearchTracker()
    tracker.log(query="search engine", result_count=5, latency_ms=12.4)

    tracker.top_queries(n=10)
    tracker.zero_result_queries()
    tracker.latency_stats("search engine")
    tracker.volume_by_day()
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, date, UTC
from pathlib import Path
from statistics import mean, median, stdev
from typing import Iterator


# ---------------------------------------------------------------------------
# SearchEvent
# ---------------------------------------------------------------------------

@dataclass
class SearchEvent:
    """A single recorded search interaction."""
    query:        str
    result_count: int
    latency_ms:   float
    timestamp:    str = field(
        default_factory=lambda: datetime.now(UTC).isoformat())
    # active filters at query time
    filters:      dict = field(default_factory=dict)

    @property
    def date(self) -> str:
        """Return the YYYY-MM-DD date portion of the timestamp."""
        return self.timestamp[:10]

    @property
    def is_zero_result(self) -> bool:
        return self.result_count == 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SearchEvent":
        return cls(**d)


# ---------------------------------------------------------------------------
# SearchTracker
# ---------------------------------------------------------------------------

class SearchTracker:
    """
    Logs search events and computes aggregated analytics.

    Thread safety: not thread-safe. For a concurrent API server, wrap
    log() calls with a lock or use an async queue (Day 20 consideration).
    """

    def __init__(self, persist_path: str | Path | None = None) -> None:
        self._events: list[SearchEvent] = []
        self._path = Path(persist_path) if persist_path else None
        if self._path and self._path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(
        self,
        query:        str,
        result_count: int,
        latency_ms:   float,
        filters:      dict | None = None,
    ) -> SearchEvent:
        """
        Record a search event.

        Args:
            query:        Raw query string as entered by the user.
            result_count: Number of results returned.
            latency_ms:   End-to-end query latency in milliseconds.
            filters:      Dict of active filters (optional).

        Returns:
            The SearchEvent that was logged.
        """
        event = SearchEvent(
            query=query.strip().lower(),
            result_count=result_count,
            latency_ms=round(latency_ms, 3),
            filters=filters or {},
        )
        self._events.append(event)
        if self._path:
            self._append(event)
        return event

    # ------------------------------------------------------------------
    # Query analytics
    # ------------------------------------------------------------------

    def top_queries(self, n: int = 10) -> list[dict]:
        """
        Return the n most frequently searched queries.

        Returns:
            List of dicts: [{"query": str, "count": int, "avg_results": float}]
            sorted by count descending.
        """
        counts: Counter[str] = Counter(e.query for e in self._events)
        result_totals: dict[str, int] = defaultdict(int)
        for e in self._events:
            result_totals[e.query] += e.result_count

        return [
            {
                "query":       query,
                "count":       count,
                "avg_results": round(result_totals[query] / count, 1),
            }
            for query, count in counts.most_common(n)
        ]

    def zero_result_queries(self, n: int | None = None) -> list[dict]:
        """
        Return queries that returned zero results, with their frequency.

        Args:
            n: Maximum number to return (None = all).

        Returns:
            List of dicts: [{"query": str, "count": int}] sorted by count desc.
        """
        zero_events = [e for e in self._events if e.is_zero_result]
        counts: Counter[str] = Counter(e.query for e in zero_events)
        results = [
            {"query": query, "count": count}
            for query, count in counts.most_common(n)
        ]
        return results

    def latency_stats(self, query: str | None = None) -> dict:
        """
        Return latency statistics (mean, median, p95, p99, stdev).

        Args:
            query: If provided, stats for that specific query only.
                   If None, stats across all queries.

        Returns:
            Dict with keys: count, mean_ms, median_ms, p95_ms, p99_ms,
            min_ms, max_ms, stdev_ms.
        """
        events = (
            [e for e in self._events if e.query == query.strip().lower()]
            if query else self._events
        )
        latencies = [e.latency_ms for e in events]

        if not latencies:
            return {"count": 0}

        sorted_l = sorted(latencies)
        n = len(sorted_l)

        return {
            "count":     n,
            "mean_ms":   round(mean(latencies), 3),
            "median_ms": round(median(latencies), 3),
            "p95_ms":    round(sorted_l[int(n * 0.95)], 3),
            "p99_ms":    round(sorted_l[min(int(n * 0.99), n - 1)], 3),
            "min_ms":    round(sorted_l[0], 3),
            "max_ms":    round(sorted_l[-1], 3),
            "stdev_ms":  round(stdev(latencies), 3) if n > 1 else 0.0,
        }

    def volume_by_day(self) -> list[dict]:
        """
        Return search volume grouped by calendar day.

        Returns:
            List of dicts: [{"date": "YYYY-MM-DD", "count": int}]
            sorted by date ascending.
        """
        counts: Counter[str] = Counter(e.date for e in self._events)
        return [
            {"date": d, "count": c}
            for d, c in sorted(counts.items())
        ]

    def summary(self) -> dict:
        """
        High-level summary of all tracked searches.

        Returns:
            Dict with total_searches, unique_queries, zero_result_rate,
            avg_results, avg_latency_ms.
        """
        if not self._events:
            return {
                "total_searches":   0,
                "unique_queries":   0,
                "zero_result_rate": 0.0,
                "avg_results":      0.0,
                "avg_latency_ms":   0.0,
            }

        zero = sum(1 for e in self._events if e.is_zero_result)
        return {
            "total_searches":   len(self._events),
            "unique_queries":   len({e.query for e in self._events}),
            "zero_result_rate": round(zero / len(self._events), 4),
            "avg_results":      round(mean(e.result_count for e in self._events), 1),
            "avg_latency_ms":   round(mean(e.latency_ms for e in self._events), 3),
        }

    # ------------------------------------------------------------------
    # Event access
    # ------------------------------------------------------------------

    def all_events(self) -> list[SearchEvent]:
        """Return all logged events (read-only copy)."""
        return list(self._events)

    def recent(self, n: int = 20) -> list[SearchEvent]:
        """Return the n most recent events."""
        return self._events[-n:]

    def clear(self) -> None:
        """Remove all in-memory events (does not delete persisted file)."""
        self._events.clear()

    def __len__(self) -> int:
        return len(self._events)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load events from the JSON file."""
        try:
            data = json.loads(self._path.read_text())
            self._events = [SearchEvent.from_dict(d) for d in data]
        except (json.JSONDecodeError, KeyError):
            self._events = []

    def _append(self, event: SearchEvent) -> None:
        """Append a single event to the JSON file (rewrite full file)."""
        data = [e.to_dict() for e in self._events]
        self._path.write_text(json.dumps(data, indent=2))

    def save(self, path: str | Path | None = None) -> None:
        """Save all events to a JSON file."""
        target = Path(path) if path else self._path
        if not target:
            raise ValueError("No path specified for save()")
        data = [e.to_dict() for e in self._events]
        target.write_text(json.dumps(data, indent=2))

    @classmethod
    def load_from(cls, path: str | Path) -> "SearchTracker":
        """Load a tracker from an existing JSON file."""
        return cls(persist_path=path)
