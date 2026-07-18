"""
inverted_index.py

The core data structure of QueryCore. Builds and stores a mapping from
stemmed tokens to the documents that contain them, with enough metadata
to support TF-IDF ranking, phrase matching, and fuzzy search.

Structure
---------
The index is a two-level dict:

    {
        "search": {
            "doc_001": Posting(doc_id="doc_001", tf=0.04, positions=[1, 9]),
            "doc_003": Posting(doc_id="doc_003", tf=0.12, positions=[0, 3, 5]),
        },
        "engin": {
            "doc_001": Posting(doc_id="doc_001", tf=0.02, positions=[2]),
        },
        ...
    }

Each Posting stores:
    - doc_id    : str   document identifier
    - tf        : float normalized term frequency (count / total_tokens)
    - positions : list  token positions where the term appears (for phrase search)

The index also stores per-document metadata needed for TF-IDF:
    - doc_lengths  : {doc_id: total_token_count}
    - doc_metadata : {doc_id: {title, category, date, ...}}

Pipeline position:
    Tokenizer → Preprocessor → [clean tokens + positions] → InvertedIndex

Usage:
    from src.indexer.inverted_index import InvertedIndex

    idx = InvertedIndex()
    idx.build(docs)          # docs from load_dataset.load_documents()

    postings = idx.get_postings("search")
    # → {"doc_001": Posting(...), "doc_003": Posting(...)}

    doc_ids = idx.get_doc_ids("engin")
    # → {"doc_001", "doc_003"}
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from src.indexer.tokenizer import Tokenizer
from src.indexer.preprocessor import Preprocessor


# ---------------------------------------------------------------------------
# Posting
# ---------------------------------------------------------------------------

@dataclass
class Posting:
    """
    One entry in a posting list — the occurrence of a term in one document.

    Attributes:
        doc_id    : Unique document identifier.
        tf        : Normalized term frequency = count(term, doc) / total_tokens(doc).
                    Stored as a float so TF-IDF scoring doesn't need to re-derive it.
        positions : Sorted list of token positions where the term appears.
                    Position 0 = first token in the combined title+body stream.
                    Required for phrase matching (Day 12).
    """
    doc_id: str
    tf: float
    positions: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# InvertedIndex
# ---------------------------------------------------------------------------

class InvertedIndex:
    """
    Builds and queries the inverted index over a document corpus.

    Responsibilities:
      - Tokenize and preprocess each document
      - Record term→doc mappings with TF and positions
      - Track corpus-level stats (N, df) needed for IDF at query time
      - Expose a clean read API for the retrieval layer
    """

    def __init__(self) -> None:
        # term → {doc_id → Posting}
        self._index: dict[str, dict[str, Posting]] = defaultdict(dict)

        # doc_id → total token count (denominator for TF)
        self._doc_lengths: dict[str, int] = {}

        # doc_id → original metadata (title, category, date, body snippet)
        self._doc_metadata: dict[str, dict] = {}

        # Number of documents indexed
        self._num_docs: int = 0

        self._tokenizer = Tokenizer()
        self._preprocessor = Preprocessor()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, docs: list[dict]) -> None:
        """
        Index a list of document dicts.

        Each doc must have: 'id', 'title', 'body'.
        Optional fields stored in metadata: 'category', 'date', 'url'.

        Args:
            docs: List of document dicts, e.g. from load_dataset.load_documents().

        Side effects:
            Populates self._index, self._doc_lengths, self._doc_metadata.
            Calling build() again on a populated index ADDS to it (use clear() first
            to reindex from scratch).
        """
        for doc in docs:
            self._index_document(doc)
        self._num_docs = len(self._doc_metadata)

    def _index_document(self, doc: dict) -> None:
        doc_id = doc["id"]

        # Store metadata for result rendering
        self._doc_metadata[doc_id] = {
            "title":    doc.get("title", ""),
            "category": doc.get("category", ""),
            "date":     doc.get("date", ""),
            "url":      doc.get("url", ""),
            # Store a short snippet for result display (first 200 chars of body)
            "snippet":  doc.get("body", "")[:200],
        }

        # Tokenize title + body together
        raw_tokens = self._tokenizer.tokenize_document(doc)

        # Build positional map: position → stemmed token
        # We need positions BEFORE stop word removal so phrase positions are
        # consistent with what the user would expect.
        # Strategy: stem each raw token but keep its original position index.
        # Stop words are still removed — their positions are skipped.
        # positional[i] = stemmed token at position i (or "" if stop word)
        positional: list[str] = []
        clean_tokens: list[str] = []

        for raw_token in raw_tokens:
            if self._preprocessor.is_stopword(raw_token):
                # placeholder — position exists but no index entry
                positional.append("")
            else:
                stemmed = self._preprocessor.stem(raw_token)
                if len(stemmed) >= 2:
                    positional.append(stemmed)
                    clean_tokens.append(stemmed)
                else:
                    positional.append("")

        total_tokens = len(clean_tokens)
        if total_tokens == 0:
            return

        self._doc_lengths[doc_id] = total_tokens

        # Count term frequencies and collect positions
        term_counts: dict[str, int] = defaultdict(int)
        term_positions: dict[str, list[int]] = defaultdict(list)

        for pos, term in enumerate(positional):
            if term:  # skip placeholders
                term_counts[term] += 1
                term_positions[term].append(pos)

        # Write postings
        for term, count in term_counts.items():
            tf = count / total_tokens
            self._index[term][doc_id] = Posting(
                doc_id=doc_id,
                tf=tf,
                positions=term_positions[term],
            )

        self._num_docs = len(self._doc_metadata)

    def clear(self) -> None:
        """Reset the index to empty."""
        self._index.clear()
        self._doc_lengths.clear()
        self._doc_metadata.clear()
        self._num_docs = 0

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_postings(self, term: str) -> dict[str, Posting]:
        """
        Return all postings for a stemmed term.

        Args:
            term: A stemmed, lowercase token (same form used at index time).

        Returns:
            Dict mapping doc_id → Posting. Empty dict if term not in index.
        """
        return self._index.get(term, {})

    def get_doc_ids(self, term: str) -> set[str]:
        """Return the set of doc IDs that contain the given term."""
        return set(self._index.get(term, {}).keys())

    def get_tf(self, term: str, doc_id: str) -> float:
        """Return TF(term, doc), or 0.0 if not present."""
        postings = self._index.get(term, {})
        posting = postings.get(doc_id)
        return posting.tf if posting else 0.0

    def get_idf(self, term: str) -> float:
        """
        Compute IDF for a term using the standard log formula:
            IDF(t) = log(N / df(t))

        where N = total documents, df(t) = documents containing the term.
        Returns 0.0 if the term is not in the index.
        """
        df = len(self._index.get(term, {}))
        if df == 0 or self._num_docs == 0:
            return 0.0
        return math.log(self._num_docs / df)

    def get_tfidf(self, term: str, doc_id: str) -> float:
        """
        Compute TF-IDF(term, doc) = TF(term, doc) × IDF(term).
        Returns 0.0 if either is missing.
        """
        return self.get_tf(term, doc_id) * self.get_idf(term)

    def get_positions(self, term: str, doc_id: str) -> list[int]:
        """Return token positions for a term in a specific document."""
        postings = self._index.get(term, {})
        posting = postings.get(doc_id)
        return posting.positions if posting else []

    def get_metadata(self, doc_id: str) -> dict:
        """Return stored metadata for a document."""
        return self._doc_metadata.get(doc_id, {})

    def get_all_terms(self) -> Iterator[str]:
        """Iterate over every term in the index."""
        return iter(self._index.keys())

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def num_docs(self) -> int:
        """Total number of indexed documents."""
        return self._num_docs

    @property
    def num_terms(self) -> int:
        """Total number of unique terms in the index."""
        return len(self._index)

    def stats(self) -> dict:
        """Return a summary dict of index statistics."""
        return {
            "num_docs":  self._num_docs,
            "num_terms": self.num_terms,
            "avg_doc_length": (
                sum(self._doc_lengths.values()) / self._num_docs
                if self._num_docs else 0
            ),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Serialise the index to a JSON file.
        Useful for saving a built index so it doesn't need rebuilding on startup.
        """
        path = Path(path)
        data = {
            "num_docs": self._num_docs,
            "doc_lengths": self._doc_lengths,
            "doc_metadata": self._doc_metadata,
            "index": {
                term: {
                    doc_id: {
                        "tf": p.tf,
                        "positions": p.positions,
                    }
                    for doc_id, p in postings.items()
                }
                for term, postings in self._index.items()
            },
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "InvertedIndex":
        """Load a previously saved index from a JSON file."""
        path = Path(path)
        data = json.loads(path.read_text())

        idx = cls()
        idx._num_docs = data["num_docs"]
        idx._doc_lengths = data["doc_lengths"]
        idx._doc_metadata = data["doc_metadata"]

        for term, postings in data["index"].items():
            idx._index[term] = {
                doc_id: Posting(
                    doc_id=doc_id,
                    tf=p["tf"],
                    positions=p["positions"],
                )
                for doc_id, p in postings.items()
            }
        return idx
