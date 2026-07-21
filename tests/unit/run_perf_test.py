"""
run_perf_test.py

Day 14 — Performance benchmarks for QueryCore.

Measures latency and throughput across the full pipeline:
    build index → parse → retrieve → rank → paginate → fuzzy

Run with:  python scripts/run_perf_test.py

Success criteria from PRD:
    ✓  Index 33 docs in < 5s     (trivial — real target is 1,000+)
    ✓  Query latency < 100ms
"""

import json
import statistics
import time
from pathlib import Path

from src.indexer.inverted_index import InvertedIndex
# added Day 15 — skip if absent
from src.search.autocomplete import Autocomplete
from src.search.fuzzy_search import FuzzyMatcher
from src.search.paginator import Paginator, SortBy
from src.search.query_parser import QueryParser
from src.search.ranking import Ranker
from src.search.retrieval import Retriever

DATASET = Path(__file__).parent.parent / "data" / "sample_documents.json"

QUERIES = [
    "search engine",
    "information retrieval",
    "machine learning algorithm",
    "inverted index TF-IDF",
    "quantum mechanics",
    "serach engin",          # typo
    '"search engine"',       # phrase
    "index OR retrieval",    # boolean
    "algorithm -database",   # exclusion
    "DNA evolution biology",  # science
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def timer(fn, *args, runs: int = 50):
    """Run fn(*args) `runs` times; return (results, latencies_ms)."""
    latencies = []
    result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn(*args)
        latencies.append((time.perf_counter() - t0) * 1000)
    return result, latencies


def report(label: str, latencies: list[float]):
    mean = statistics.mean(latencies)
    median = statistics.median(latencies)
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    status = "✓" if mean < 100 else "✗"
    print(f"  {status}  {label:<35}  mean={mean:6.2f}ms  p50={median:5.2f}ms  p95={p95:5.2f}ms  p99={p99:5.2f}ms")


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bench_build(docs):
    print("\n── Index build ──")
    t0 = time.perf_counter()
    idx = InvertedIndex()
    idx.build(docs)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  ✓  {len(docs)} docs  {idx.num_terms} terms  {elapsed:.1f}ms")
    return idx


def bench_pipeline(idx):
    print("\n── Full pipeline (parse → retrieve → rank → paginate) ──")
    qp = QueryParser()
    ret = Retriever(idx)
    rnk = Ranker(idx)
    pag = Paginator(page_size=10)

    for raw in QUERIES:
        def run():
            pq = qp.parse(raw)
            candidates = ret.retrieve(pq)
            results = rnk.rank(pq, candidates)
            page = pag.paginate(results, page=1)
            return page

        _, lats = timer(run, runs=100)
        report(f'"{raw[:30]}"', lats)


def bench_fuzzy(idx):
    print("\n── Fuzzy expansion ──")
    fm = FuzzyMatcher(idx, max_distance=2)
    qp = QueryParser()

    typos = ["serach", "retreival", "algorythm", "engne", "indexs"]
    for typo in typos:
        def run(t=typo):
            return fm.expand(t)
        _, lats = timer(run, runs=50)
        report(f"expand({typo!r})", lats)


def bench_index_ops(idx):
    print("\n── Index operations ──")
    terms = ["search", "retriev", "algorithm", "quantum", "index"]

    for term in terms:
        def run(t=term):
            postings = idx.get_postings(t)
            idf = idx.get_idf(t)
            doc_ids = idx.get_doc_ids(t)
            return postings, idf, doc_ids
        _, lats = timer(run, runs=500)
        report(f"get_postings+idf({term!r})", lats)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  QueryCore — Performance Benchmarks  (Day 14)")
    print("=" * 65)

    with open(DATASET) as f:
        docs = json.load(f)

    idx = bench_build(docs)
    bench_index_ops(idx)
    bench_pipeline(idx)
    bench_fuzzy(idx)

    print("\n" + "=" * 65)
    print("  ✓ = mean latency < 100ms  |  all times in milliseconds")
    print("=" * 65)


if __name__ == "__main__":
    main()
