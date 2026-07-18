"""
ranking.py

Stage 2 of query execution: takes a set of candidate doc_ids from the
Retriever and scores each one against the query using TF-IDF, returning
a sorted list of SearchResult objects.

Scoring model
-------------
For a multi-term query, the score for a document is the sum of TF-IDF
scores across all query terms:

    score(doc, query) = Σ  TF(t, doc) × IDF(t)
                       t ∈ query_terms

where:
    TF(t, doc)  = count(t in doc) / total_tokens(doc)   [stored in Posting]
    IDF(t)      = log(N / df(t))                         [computed from index]

For OR queries, all or_terms are scored alongside regular terms — a doc
that matches both AND and OR terms will score higher than one that only
matches OR terms.

Title boost
-----------
Terms that appear in the document title are given a configurable score
multiplier (default 1.5×). Title matches are more likely to be the
primary topic of the document, so they deserve extra weight.

Pipeline position:
    ParsedQuery + {candidate doc_ids} → Ranker → [SearchResult, ...]

Usage
-----
    from src.search.ranking import Ranker, SearchResult

    ranker  = Ranker(index)
    results = ranker.rank(pq, candidates)
    # → [SearchResult(doc_id, score, metadata), ...]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from src.indexer.inverted_index import InvertedIndex
from src.search.query_parser import ParsedQuery


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """
    A single ranked result returned by the Ranker.

    Attributes:
        doc_id   : Document identifier.
        score    : TF-IDF relevance score (higher = more relevant).
        title    : Document title (from stored metadata).
        snippet  : Short body excerpt for display (first 200 chars).
        category : Document category (for filtering on Day 16).
        date     : Document date string.
        url      : Source URL if available.
    """
    doc_id:   str
    score:    float
    title:    str = ""
    snippet:  str = ""
    category: str = ""
    date:     str = ""
    url:      str = ""

    def to_dict(self) -> dict:
        return {
            "doc_id":   self.doc_id,
            "score":    round(self.score, 6),
            "title":    self.title,
            "snippet":  self.snippet,
            "category": self.category,
            "date":     self.date,
            "url":      self.url,
        }


# ---------------------------------------------------------------------------
# RankerConfig
# ---------------------------------------------------------------------------

@dataclass
class RankerConfig:
    """Tunable parameters for the Ranker."""
    title_boost: float = 1.5    # multiplier for terms found in the title
    min_score:   float = 0.0    # results below this threshold are dropped


# ---------------------------------------------------------------------------
# Ranker
# ---------------------------------------------------------------------------

class Ranker:
    """
    Scores and sorts candidate documents using TF-IDF.

    Accepts the output of Retriever.retrieve() (a set of doc_ids) and the
    ParsedQuery, and returns a sorted list of SearchResult objects.
    """

    def __init__(self, index: InvertedIndex, config: RankerConfig | None = None) -> None:
        self._index = index
        self._config = config or RankerConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(self, pq: ParsedQuery, candidates: set[str]) -> list[SearchResult]:
        """
        Score and sort candidate documents by TF-IDF relevance.

        Args:
            pq:         ParsedQuery containing the terms to score against.
            candidates: Set of doc_ids from Retriever.retrieve().

        Returns:
            List of SearchResult sorted by score descending.
            Empty list if candidates is empty or query is empty.
        """
        if not candidates or pq.is_empty:
            return []

        # All terms that contribute to scoring: AND terms + OR terms + phrase tokens
        scoring_terms = list({
            t for t in pq.terms + pq.or_terms + [t for p in pq.phrases for t in p]
        })

        results: list[SearchResult] = []

        for doc_id in candidates:
            score = self._score(doc_id, scoring_terms)
            if score <= self._config.min_score:
                continue

            meta = self._index.get_metadata(doc_id)
            results.append(SearchResult(
                doc_id=doc_id,
                score=score,
                title=meta.get("title", ""),
                snippet=meta.get("snippet", ""),
                category=meta.get("category", ""),
                date=meta.get("date", ""),
                url=meta.get("url", ""),
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def score_document(self, doc_id: str, terms: list[str]) -> float:
        """
        Compute the TF-IDF score for a single document against a list of terms.
        Exposed for testing and for the explain() helper.

        Args:
            doc_id: Document identifier.
            terms:  List of stemmed query terms.

        Returns:
            Summed TF-IDF score.
        """
        return self._score(doc_id, terms)

    def explain(self, doc_id: str, pq: ParsedQuery) -> dict:
        """
        Return a per-term score breakdown for a document.
        Useful for debugging why a document scored where it did.
        """
        scoring_terms = list({
            t for t in pq.terms + pq.or_terms + [t for p in pq.phrases for t in p]
        })
        meta = self._index.get_metadata(doc_id)
        title_tokens = set(meta.get("title", "").lower().split())

        breakdown = {}
        total = 0.0
        for term in scoring_terms:
            tf = self._index.get_tf(term, doc_id)
            idf = self._index.get_idf(term)
            base = tf * idf
            boosted = self._apply_title_boost(base, term, title_tokens)
            breakdown[term] = {
                "tf": round(tf, 6),
                "idf": round(idf, 6),
                "base_score": round(base, 6),
                "boosted_score": round(boosted, 6),
                "title_hit": term in title_tokens,
            }
            total += boosted

        return {
            "doc_id":    doc_id,
            "title":     meta.get("title", ""),
            "total":     round(total, 6),
            "breakdown": breakdown,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _score(self, doc_id: str, terms: list[str]) -> float:
        """Sum TF-IDF scores across all query terms for one document."""
        meta = self._index.get_metadata(doc_id)
        title_tokens = set(meta.get("title", "").lower().split())

        total = 0.0
        for term in terms:
            tf = self._index.get_tf(term, doc_id)
            idf = self._index.get_idf(term)
            base = tf * idf
            total += self._apply_title_boost(base, term, title_tokens)

        return total

    def _apply_title_boost(self, base_score: float, term: str, title_tokens: set[str]) -> float:
        """Multiply score by title_boost if the term appears in the document title."""
        if term in title_tokens:
            return base_score * self._config.title_boost
        return base_score
