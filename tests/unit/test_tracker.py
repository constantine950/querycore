"""
test_tracker.py

Tests for src/analytics/tracker.py

Run with:  python -m pytest tests/unit/test_tracker.py -v
"""

import json
import tempfile
from pathlib import Path

import pytest
from src.analytics.tracker import SearchTracker, SearchEvent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tracker():
    return SearchTracker()


@pytest.fixture
def populated(tracker):
    """Tracker with a mix of events for analytics testing."""
    events = [
        ("search engine",        5,  12.1),
        ("search engine",        5,  11.8),
        ("search engine",        5,  13.0),
        ("inverted index",       3,   9.5),
        ("inverted index",       3,   9.2),
        ("machine learning",     7,  15.3),
        ("quantum mechanics",    2,  10.0),
        ("xyznotaword",          0,   8.0),   # zero-result
        ("zzz garbage query",    0,   7.5),   # zero-result
        ("xyznotaword",          0,   8.2),   # zero-result again
    ]
    for query, count, latency in events:
        tracker.log(query=query, result_count=count, latency_ms=latency)
    return tracker


# ---------------------------------------------------------------------------
# SearchEvent
# ---------------------------------------------------------------------------

class TestSearchEvent:
    def test_fields_stored(self):
        e = SearchEvent(query="search", result_count=5, latency_ms=12.3)
        assert e.query == "search"
        assert e.result_count == 5
        assert e.latency_ms == 12.3

    def test_timestamp_auto_set(self):
        e = SearchEvent(query="search", result_count=5, latency_ms=12.0)
        assert len(e.timestamp) > 0
        assert "T" in e.timestamp   # ISO format

    def test_date_property(self):
        e = SearchEvent(query="q", result_count=0, latency_ms=1.0,
                        timestamp="2024-06-15T10:30:00")
        assert e.date == "2024-06-15"

    def test_is_zero_result_true(self):
        e = SearchEvent(query="q", result_count=0, latency_ms=1.0)
        assert e.is_zero_result is True

    def test_is_zero_result_false(self):
        e = SearchEvent(query="q", result_count=3, latency_ms=1.0)
        assert e.is_zero_result is False

    def test_to_dict_roundtrip(self):
        e = SearchEvent(query="search", result_count=5, latency_ms=12.3)
        d = e.to_dict()
        restored = SearchEvent.from_dict(d)
        assert restored.query == e.query
        assert restored.result_count == e.result_count
        assert restored.latency_ms == e.latency_ms


# ---------------------------------------------------------------------------
# log()
# ---------------------------------------------------------------------------

class TestLog:
    def test_log_adds_event(self, tracker):
        tracker.log("search", 5, 10.0)
        assert len(tracker) == 1

    def test_log_normalises_query(self, tracker):
        tracker.log("  Search Engine  ", 5, 10.0)
        events = tracker.all_events()
        assert events[0].query == "search engine"

    def test_log_returns_event(self, tracker):
        event = tracker.log("search", 5, 10.0)
        assert isinstance(event, SearchEvent)

    def test_log_stores_filters(self, tracker):
        event = tracker.log("search", 5, 10.0, filters={"category": "science"})
        assert event.filters == {"category": "science"}

    def test_multiple_logs(self, tracker):
        for i in range(5):
            tracker.log(f"query {i}", i, float(i))
        assert len(tracker) == 5


# ---------------------------------------------------------------------------
# top_queries()
# ---------------------------------------------------------------------------

class TestTopQueries:
    def test_returns_list(self, populated):
        result = populated.top_queries()
        assert isinstance(result, list)

    def test_sorted_by_count(self, populated):
        result = populated.top_queries()
        counts = [r["count"] for r in result]
        assert counts == sorted(counts, reverse=True)

    def test_top_query_is_search_engine(self, populated):
        result = populated.top_queries(n=1)
        assert result[0]["query"] == "search engine"
        assert result[0]["count"] == 3

    def test_has_required_keys(self, populated):
        for r in populated.top_queries():
            assert "query" in r
            assert "count" in r
            assert "avg_results" in r

    def test_n_respected(self, populated):
        result = populated.top_queries(n=2)
        assert len(result) <= 2

    def test_empty_tracker(self, tracker):
        assert tracker.top_queries() == []


# ---------------------------------------------------------------------------
# zero_result_queries()
# ---------------------------------------------------------------------------

class TestZeroResultQueries:
    def test_returns_zero_result_queries(self, populated):
        zeros = populated.zero_result_queries()
        queries = [z["query"] for z in zeros]
        assert "xyznotaword" in queries
        assert "zzz garbage query" in queries

    def test_excludes_successful_queries(self, populated):
        zeros = populated.zero_result_queries()
        queries = [z["query"] for z in zeros]
        assert "search engine" not in queries

    def test_sorted_by_count(self, populated):
        zeros = populated.zero_result_queries()
        counts = [z["count"] for z in zeros]
        assert counts == sorted(counts, reverse=True)

    def test_repeated_zero_result_counted(self, populated):
        zeros = {z["query"]: z["count"]
                 for z in populated.zero_result_queries()}
        assert zeros["xyznotaword"] == 2

    def test_empty_tracker_returns_empty(self, tracker):
        assert tracker.zero_result_queries() == []


# ---------------------------------------------------------------------------
# latency_stats()
# ---------------------------------------------------------------------------

class TestLatencyStats:
    def test_global_stats_keys(self, populated):
        stats = populated.latency_stats()
        for key in ("count", "mean_ms", "median_ms", "p95_ms", "p99_ms",
                    "min_ms", "max_ms", "stdev_ms"):
            assert key in stats

    def test_count_matches_events(self, populated):
        stats = populated.latency_stats()
        assert stats["count"] == len(populated)

    def test_per_query_stats(self, populated):
        stats = populated.latency_stats("search engine")
        assert stats["count"] == 3
        assert 11.0 < stats["mean_ms"] < 14.0

    def test_unknown_query_returns_zero_count(self, populated):
        stats = populated.latency_stats("this query was never searched")
        assert stats["count"] == 0

    def test_min_lte_mean_lte_max(self, populated):
        stats = populated.latency_stats()
        assert stats["min_ms"] <= stats["mean_ms"] <= stats["max_ms"]

    def test_empty_tracker_returns_zero_count(self, tracker):
        assert tracker.latency_stats()["count"] == 0


# ---------------------------------------------------------------------------
# volume_by_day()
# ---------------------------------------------------------------------------

class TestVolumeByDay:
    def test_returns_list(self, populated):
        vol = populated.volume_by_day()
        assert isinstance(vol, list)

    def test_sorted_by_date(self, populated):
        vol = populated.volume_by_day()
        dates = [v["date"] for v in vol]
        assert dates == sorted(dates)

    def test_total_count_matches_events(self, populated):
        vol = populated.volume_by_day()
        total = sum(v["count"] for v in vol)
        assert total == len(populated)

    def test_has_required_keys(self, populated):
        for v in populated.volume_by_day():
            assert "date" in v
            assert "count" in v


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_keys(self, populated):
        s = populated.summary()
        for key in ("total_searches", "unique_queries", "zero_result_rate",
                    "avg_results", "avg_latency_ms"):
            assert key in s

    def test_total_correct(self, populated):
        assert populated.summary()["total_searches"] == len(populated)

    def test_zero_result_rate(self, populated):
        s = populated.summary()
        # 3 out of 10 events had zero results
        assert abs(s["zero_result_rate"] - 0.3) < 0.01

    def test_empty_tracker(self, tracker):
        s = tracker.summary()
        assert s["total_searches"] == 0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load(self, populated):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.json"
            populated.save(path)
            assert path.exists()

            loaded = SearchTracker.load_from(path)
            assert len(loaded) == len(populated)

    def test_loaded_events_match(self, populated):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.json"
            populated.save(path)

            loaded = SearchTracker.load_from(path)
            orig_queries = {e.query for e in populated.all_events()}
            loaded_queries = {e.query for e in loaded.all_events()}
            assert orig_queries == loaded_queries

    def test_persist_on_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "live.json"
            t = SearchTracker(persist_path=path)
            t.log("search", 5, 10.0)
            t.log("index", 3, 8.0)

            # Load fresh from disk — should have both events
            t2 = SearchTracker.load_from(path)
            assert len(t2) == 2

    def test_clear_removes_memory_only(self, populated):
        count_before = len(populated)
        populated.clear()
        assert len(populated) == 0
        # Re-populate for other tests (fixture is function-scoped so OK)
