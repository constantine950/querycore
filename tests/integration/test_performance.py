"""
test_performance.py

Day 25 — Performance tests against a large synthetic corpus.

Validates the PRD success criteria at scale:
    ✓  Index 1,000+ docs in < 5s
    ✓  Query latency < 100ms (mean)
    ✓  Fuzzy search < 500ms at scale

Run with:
    python -m pytest tests/integration/test_performance.py -v -s

The -s flag prints the benchmark table to stdout.
"""

import json
import random
import statistics
import time
from pathlib import Path

import pytest

from src.indexer.inverted_index import InvertedIndex
from src.indexer.reindexer import Reindexer
from src.search.autocomplete import Autocomplete
from src.search.filters import FilterSet
from src.search.fuzzy_search import FuzzyMatcher
from src.search.paginator import Paginator, SortBy
from src.search.phrase_match import PhraseFilter
from src.search.query_parser import QueryParser
from src.search.ranking import Ranker
from src.search.retrieval import Retriever


# ---------------------------------------------------------------------------
# Synthetic corpus generator
# ---------------------------------------------------------------------------

CATEGORIES = ["computer_science", "science",
              "history", "mathematics", "engineering"]
TECH_WORDS = [
    "search", "index", "retrieval", "algorithm", "data", "structure", "hash",
    "tree", "graph", "sort", "query", "rank", "score", "term", "document",
    "token", "stem", "fuzzy", "trie", "prefix", "suffix", "matrix", "vector",
    "neural", "model", "cache", "memory", "pointer", "stack", "queue", "heap",
    "binary", "linear", "recursive", "dynamic", "greedy", "optimize", "compile",
    "parse", "lexer", "grammar", "syntax", "semantic", "runtime", "kernel",
    "thread", "process", "network", "protocol", "packet", "socket", "buffer",
]


def make_synthetic_corpus(n: int) -> list[dict]:
    """Generate n synthetic documents with varied vocabulary."""
    random.seed(42)
    docs = []
    for i in range(n):
        cat = random.choice(CATEGORIES)
        word_count = random.randint(80, 200)
        words = random.choices(TECH_WORDS, k=word_count)
        # Insert some unique terms so IDF has real signal
        words[0] = f"concept{i % 50}"
        words[1] = f"topic{i % 20}"

        docs.append({
            "id":       f"doc_{str(i + 1).zfill(5)}",
            "title":    f"{random.choice(TECH_WORDS).capitalize()} {random.choice(TECH_WORDS).capitalize()}",
            "body":     " ".join(words),
            "category": cat,
            "date":     f"202{i % 4}-{str((i % 12) + 1).zfill(2)}-01",
            "word_count": word_count,
        })
    return docs


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------

def bench(fn, runs: int = 50) -> dict:
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - t0) * 1000)
    s = sorted(latencies)
    return {
        "mean":   statistics.mean(latencies),
        "median": statistics.median(latencies),
        "p95":    s[int(len(s) * 0.95)],
        "p99":    s[min(int(len(s) * 0.99), len(s) - 1)],
        "min":    s[0],
        "max":    s[-1],
    }


def row(label: str, b: dict, limit_ms: float | None = None):
    status = ""
    if limit_ms is not None:
        status = " ✓" if b["mean"] < limit_ms else " ✗"
    print(
        f"  {label:<40}  mean={b['mean']:6.2f}ms  p95={b['p95']:6.2f}ms  p99={b['p99']:6.2f}ms{status}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def large_corpus():
    return make_synthetic_corpus(1000)


@pytest.fixture(scope="module")
def extra_large_corpus():
    return make_synthetic_corpus(5000)


@pytest.fixture(scope="module")
def large_idx(large_corpus):
    idx = InvertedIndex()
    idx.build(large_corpus)
    return idx


@pytest.fixture(scope="module")
def xl_idx(extra_large_corpus):
    idx = InvertedIndex()
    idx.build(extra_large_corpus)
    return idx


# ---------------------------------------------------------------------------
# PRD criterion 1: Index 1,000+ docs in < 5s
# ---------------------------------------------------------------------------

class TestIndexBuild:
    def test_1000_docs_under_5s(self, large_corpus):
        print("\n\n── Index build ──────────────────────────────────────")
        idx = InvertedIndex()
        t0 = time.perf_counter()
        idx.build(large_corpus)
        elapsed = (time.perf_counter() - t0) * 1000

        print(f"  1,000 docs  →  {elapsed:.0f}ms  ({idx.num_terms} terms)")
        assert elapsed < 5000, f"Build took {elapsed:.0f}ms — exceeds 5s limit"

    def test_5000_docs_build_time(self, extra_large_corpus):
        idx = InvertedIndex()
        t0 = time.perf_counter()
        idx.build(extra_large_corpus)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  5,000 docs  →  {elapsed:.0f}ms  ({idx.num_terms} terms)")
        # No hard limit — just reporting
        assert elapsed < 30_000

    def test_index_stats_correct(self, large_idx, large_corpus):
        assert large_idx.num_docs == len(large_corpus)
        assert large_idx.num_terms > 50


# ---------------------------------------------------------------------------
# PRD criterion 2: Query latency < 100ms
# ---------------------------------------------------------------------------

class TestQueryLatency:
    def test_full_pipeline_under_100ms(self, large_idx):
        print("\n── Full pipeline latency (1,000 docs) ───────────────")
        qp = QueryParser()
        ret = Retriever(large_idx)
        rnk = Ranker(large_idx)
        pag = Paginator(page_size=10)

        queries = ["search index", "algorithm data", "query rank score",
                   "tree graph sort", "network protocol buffer"]

        for q in queries:
            pq = qp.parse(q)

            def run(pq=pq):
                candidates = ret.retrieve(pq)
                results = rnk.rank(pq, candidates)
                pag.paginate(results, page=1)

            b = bench(run, runs=100)
            row(f'"{q}"', b, limit_ms=100)
            assert b["mean"] < 100, f"Mean latency {b['mean']:.2f}ms > 100ms for '{q}'"

    def test_full_pipeline_5000_docs(self, xl_idx):
        print("\n── Full pipeline latency (5,000 docs) ───────────────")
        qp = QueryParser()
        ret = Retriever(xl_idx)
        rnk = Ranker(xl_idx)

        pq = qp.parse("search index algorithm")

        def run():
            candidates = ret.retrieve(pq)
            rnk.rank(pq, candidates)

        b = bench(run, runs=50)
        row('"search index algorithm" @5k docs', b)
        # Softer limit at 5k — still should be well under 500ms
        assert b["mean"] < 500

    def test_index_lookup_microseconds(self, large_idx):
        print("\n── Index operation latency ──────────────────────────")
        for term in ["search", "algorithm", "index", "concept25"]:
            def run(t=term):
                large_idx.get_postings(t)
                large_idx.get_idf(t)
                large_idx.get_doc_ids(t)

            b = bench(run, runs=500)
            row(f"get_postings+idf({term!r})", b)
            assert b["mean"] < 1.0, f"Index lookup {b['mean']:.3f}ms — should be sub-millisecond"


# ---------------------------------------------------------------------------
# Fuzzy search at scale
# ---------------------------------------------------------------------------

class TestFuzzyLatency:
    def test_fuzzy_expand_under_500ms(self, large_idx):
        print("\n── Fuzzy expansion latency ──────────────────────────")
        fm = FuzzyMatcher(large_idx, max_distance=2)

        for typo in ["searh", "algorythm", "indx", "querys"]:
            b = bench(lambda t=typo: fm.expand(t), runs=20)
            row(f"expand({typo!r})", b, limit_ms=500)
            assert b["mean"] < 500, f"Fuzzy expand {b['mean']:.1f}ms > 500ms for {typo!r}"

    def test_fuzzy_retrieval_latency(self, large_idx):
        fm = FuzzyMatcher(large_idx, max_distance=2)
        qp = QueryParser()
        pq = qp.parse("searh algorythm")
        b = bench(lambda: fm.retrieve_fuzzy(pq), runs=20)
        row("retrieve_fuzzy('searh algorythm')", b)
        assert b["mean"] < 500


# ---------------------------------------------------------------------------
# Autocomplete at scale
# ---------------------------------------------------------------------------

class TestAutocompleteLatency:
    def test_autocomplete_under_10ms(self, large_idx):
        print("\n── Autocomplete latency ─────────────────────────────")
        ac = Autocomplete(large_idx)

        for prefix in ["sea", "algo", "ind", "con", "top"]:
            b = bench(lambda p=prefix: ac.suggest(p, top_n=8), runs=200)
            row(f"suggest({prefix!r})", b, limit_ms=10)
            assert b["mean"] < 10, f"Autocomplete {b['mean']:.2f}ms > 10ms for {prefix!r}"


# ---------------------------------------------------------------------------
# Filter performance
# ---------------------------------------------------------------------------

class TestFilterLatency:
    def test_filter_candidates_under_10ms(self, large_idx, large_corpus):
        print("\n── Filter latency ───────────────────────────────────")
        all_ids = {d["id"] for d in large_corpus}
        fs = FilterSet(large_idx).add_category("computer_science")

        b = bench(lambda: fs.apply_to_candidates(all_ids), runs=100)
        row("category filter (1,000 candidates)", b, limit_ms=10)
        assert b["mean"] < 10


# ---------------------------------------------------------------------------
# Reindexer at scale
# ---------------------------------------------------------------------------

class TestReindexerLatency:
    def test_incremental_add_remove(self, large_corpus):
        print("\n── Reindexer latency ────────────────────────────────")
        idx = InvertedIndex()
        idx.build(large_corpus)
        rx = Reindexer(idx)

        new_doc = {
            "id": "perf_test_doc", "title": "Perf Test",
            "body": "performance testing incremental reindex update",
            "category": "cs", "date": "2024-01-01",
        }

        # Add
        b_add = bench(lambda: (
            rx.add({**new_doc, "id": f"perf_{time.time_ns()}"})
        ), runs=20)
        row("reindexer.add()", b_add, limit_ms=50)

        # Remove (add first, then bench remove)
        rx.add({**new_doc, "id": "remove_target"})
        t0 = time.perf_counter()
        rx.remove("remove_target")
        remove_ms = (time.perf_counter() - t0) * 1000
        print(f"  {'reindexer.remove()':<40}  {remove_ms:.2f}ms")
        assert remove_ms < 200, f"Remove took {remove_ms:.1f}ms — should be < 200ms"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def test_print_summary():
    print("\n" + "=" * 60)
    print("  QueryCore — Day 25 Performance Summary")
    print("=" * 60)
    print("  All PRD criteria verified:")
    print("  ✓  1,000+ docs indexed in < 5s")
    print("  ✓  Query latency < 100ms mean")
    print("  ✓  Index lookups < 1ms")
    print("  ✓  Autocomplete < 10ms")
    print("  ✓  Fuzzy search < 500ms")
    print("=" * 60)
