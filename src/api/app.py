from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.analytics.tracker import SearchTracker
from src.indexer.inverted_index import InvertedIndex
from src.indexer.reindexer import Reindexer
from src.search.autocomplete import Autocomplete
from src.search.filters import FilterSet
from src.search.fuzzy_search import FuzzyMatcher
from src.search.highlighter import Highlighter
from src.search.paginator import Paginator, SortBy
from src.search.phrase_match import PhraseFilter
from src.search.query_parser import QueryParser
from src.search.ranking import Ranker
from src.search.retrieval import Retriever

DATASET_PATH = Path(__file__).parent.parent.parent / \
    "data" / "sample_documents.json"

# Module-level singletons — built once on import
_index = InvertedIndex()
_docs: list[dict] = []


def _load_and_build():
    global _docs
    if DATASET_PATH.exists():
        with open(DATASET_PATH) as f:
            _docs = json.load(f)
    _index.build(_docs)


_load_and_build()

_doc_bodies: dict[str, str] = {d["id"]: d.get("body", "") for d in _docs}

_qp = QueryParser()
_retriever = Retriever(_index)
_ranker = Ranker(_index)
_paginator = Paginator(page_size=10)
_phrase_f = PhraseFilter(_index)
_fuzzy = FuzzyMatcher(_index, max_distance=2)
_autocomplete = Autocomplete(_index)
_highlighter = Highlighter()
_tracker = SearchTracker()
_reindexer = Reindexer(_index)


class DocumentIn(BaseModel):
    id:       str
    title:    str
    body:     str
    category: str = ""
    date:     str = ""
    url:      str = ""


class SearchResponse(BaseModel):
    query:       str
    page:        int
    page_size:   int
    total:       int
    total_pages: int
    has_next:    bool
    has_prev:    bool
    start:       int
    end:         int
    sort_by:     str
    latency_ms:  float
    results:     list[dict]


class AutocompleteResponse(BaseModel):
    prefix:      str
    suggestions: list[str]


class IndexResponse(BaseModel):
    doc_id:  str
    action:  str
    success: bool


class HealthResponse(BaseModel):
    status:    str
    num_docs:  int
    num_terms: int


app = FastAPI(
    title="QueryCore API",
    description="Text indexing and retrieval engine with TF-IDF ranking.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return HealthResponse(
        status="ok",
        num_docs=_index.num_docs,
        num_terms=_index.num_terms,
    )


@app.get("/search", response_model=SearchResponse, tags=["Search"])
def search(
    q:         str = Query(..., min_length=1, description="Search query"),
    page:      int = Query(
        1,   ge=1,         description="Page number (1-indexed)"),
    sort:      str = Query(
        "score",            description="Sort order: score | date | title"),
    category:  str | None = Query(
        None,               description="Filter by category"),
    date_from: str | None = Query(
        None,               description="Filter from date (YYYY-MM-DD)"),
    date_to:   str | None = Query(
        None,               description="Filter to date (YYYY-MM-DD)"),
    fuzzy:     bool = Query(
        False,              description="Enable fuzzy (typo-tolerant) search"),
    highlight: bool = Query(
        True,               description="Highlight matching terms in snippets"),
):
    t0 = time.perf_counter()

    # Parse sort order
    try:
        sort_by = SortBy(sort)
    except ValueError:
        sort_by = SortBy.SCORE

    # Parse query
    pq = _qp.parse(q)
    if pq.is_empty:
        raise HTTPException(
            status_code=400, detail="Query produced no searchable terms.")

    # Build filter set
    fs = FilterSet(_index)
    active_filters: dict = {}
    if category:
        fs.add_category(category)
        active_filters["category"] = category
    if date_from or date_to:
        fs.add_date_range(date_from, date_to)
        active_filters["date_from"] = date_from
        active_filters["date_to"] = date_to

    # Retrieve candidates
    if fuzzy:
        candidates = _fuzzy.retrieve_fuzzy(pq)
    else:
        candidates = _retriever.retrieve_with_phrases(pq)

    # Phrase filtering
    if pq.is_phrase:
        candidates = _phrase_f.filter(candidates, pq.phrases)

    # Apply filters
    if fs.active:
        candidates = fs.apply_to_candidates(candidates)

    # Rank
    results = _ranker.rank(pq, candidates)

    # Highlight snippets
    if highlight:
        for r in results:
            body = _doc_bodies.get(r.doc_id, r.snippet)
            hr = _highlighter.highlight(body, pq.all_terms, window=250)
            r.snippet = hr.snippet

    # Paginate
    page_obj = _paginator.paginate(results, page=page, sort_by=sort_by)

    latency_ms = (time.perf_counter() - t0) * 1000

    # Log to analytics
    _tracker.log(
        query=q,
        result_count=page_obj.total,
        latency_ms=latency_ms,
        filters=active_filters,
    )

    return SearchResponse(
        query=q,
        page=page_obj.page,
        page_size=page_obj.page_size,
        total=page_obj.total,
        total_pages=page_obj.total_pages,
        has_next=page_obj.has_next,
        has_prev=page_obj.has_prev,
        start=page_obj.start,
        end=page_obj.end,
        sort_by=page_obj.sort_by.value,
        latency_ms=round(latency_ms, 3),
        results=[r.to_dict() for r in page_obj.results],
    )


@app.get("/autocomplete", response_model=AutocompleteResponse, tags=["Search"])
def autocomplete(
    q: str = Query(..., min_length=1, description="Prefix to complete"),
    n: int = Query(8,  ge=1, le=20,  description="Max suggestions"),
):
    suggestions = _autocomplete.suggest(q.lower().strip(), top_n=n)
    return AutocompleteResponse(prefix=q, suggestions=suggestions)


@app.post("/index", response_model=IndexResponse, tags=["Index"])
def add_document(doc: DocumentIn):
    d = doc.model_dump()
    try:
        _reindexer.add(d)
        _doc_bodies[d["id"]] = d.get("body", "")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return IndexResponse(doc_id=doc.id, action="add", success=True)


@app.put("/index/{doc_id}", response_model=IndexResponse, tags=["Index"])
def update_document(doc_id: str, doc: DocumentIn):
    if doc.id != doc_id:
        raise HTTPException(
            status_code=400, detail="doc_id in path must match body id.")
    d = doc.model_dump()
    _reindexer.update(d)
    _doc_bodies[doc_id] = d.get("body", "")
    return IndexResponse(doc_id=doc_id, action="update", success=True)


@app.delete("/index/{doc_id}", response_model=IndexResponse, tags=["Index"])
def delete_document(doc_id: str):
    try:
        _reindexer.remove(doc_id)
        _doc_bodies.pop(doc_id, None)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return IndexResponse(doc_id=doc_id, action="delete", success=True)


@app.get("/analytics/summary", tags=["Analytics"])
def analytics_summary():
    return _tracker.summary()


@app.get("/analytics/top", tags=["Analytics"])
def analytics_top(n: int = Query(10, ge=1, le=50, description="Number of top queries")):
    return {"top_queries": _tracker.top_queries(n=n)}


@app.get("/analytics/zero", tags=["Analytics"])
def analytics_zero(n: int = Query(10, ge=1, le=50)):
    return {"zero_result_queries": _tracker.zero_result_queries(n=n)}


@app.get("/analytics/latency", tags=["Analytics"])
def analytics_latency(q: str | None = Query(None, description="Filter to a specific query")):
    return _tracker.latency_stats(query=q)


@app.get("/analytics/volume", tags=["Analytics"])
def analytics_volume():
    return {"volume": _tracker.volume_by_day()}
