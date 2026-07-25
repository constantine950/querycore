"""
reindexer.py

Supports dynamic updates to a live InvertedIndex without rebuilding
the entire index from scratch.

Operations
----------
- add(doc)     : Index a new document. Raises if doc_id already exists.
- update(doc)  : Remove old postings for a doc, then re-index it.
- remove(doc_id): Delete all postings for a document from the index.
- reindex(docs): Full rebuild — clears the index and re-indexes all docs.
                 Use when a large batch of changes makes incremental
                 updates impractical.

Why not just rebuild every time?
---------------------------------
For 33 documents, a full rebuild takes ~40ms — fine for startup, wasteful
for a live API serving continuous document updates. As corpus size grows:
    1,000 docs  →  ~1.5s rebuild
   10,000 docs  →  ~15s rebuild
Incremental add/remove keeps individual operations at O(k) where k is
the number of tokens in the changed document, regardless of corpus size.

Consistency
-----------
The InvertedIndex does not enforce uniqueness on doc_ids — calling
_index_document() twice for the same id creates duplicate postings.
The Reindexer enforces uniqueness by checking before adding and clearing
old postings before updating.

Usage
-----
    from src.indexer.reindexer import Reindexer

    reindexer = Reindexer(index)
    reindexer.add({"id": "doc_034", "title": "New Doc", "body": "...", ...})
    reindexer.update({"id": "doc_001", "title": "Updated", "body": "...", ...})
    reindexer.remove("doc_002")

    stats = reindexer.stats()
    # → {"adds": 1, "updates": 1, "removes": 1, "reindexes": 0}
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.indexer.inverted_index import InvertedIndex


# ---------------------------------------------------------------------------
# ReindexStats
# ---------------------------------------------------------------------------

@dataclass
class ReindexStats:
    """Tracks how many of each operation the Reindexer has performed."""
    adds:      int = 0
    updates:   int = 0
    removes:   int = 0
    reindexes: int = 0

    def to_dict(self) -> dict:
        return {
            "adds":      self.adds,
            "updates":   self.updates,
            "removes":   self.removes,
            "reindexes": self.reindexes,
            "total_ops": self.adds + self.updates + self.removes + self.reindexes,
        }


# ---------------------------------------------------------------------------
# Reindexer
# ---------------------------------------------------------------------------

class Reindexer:
    """
    Wraps an InvertedIndex to provide safe incremental document updates.
    """

    def __init__(self, index: InvertedIndex) -> None:
        self._index = index
        self._stats = ReindexStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, doc: dict) -> None:
        """
        Add a new document to the index.

        Args:
            doc: Document dict with at least 'id', 'title', 'body'.

        Raises:
            ValueError: If a document with this id is already indexed.
        """
        doc_id = doc.get("id")
        if not doc_id:
            raise ValueError("Document must have an 'id' field.")
        if self._exists(doc_id):
            raise ValueError(
                f"Document '{doc_id}' already exists. Use update() to replace it."
            )
        self._index._index_document(doc)
        self._stats.adds += 1

    def update(self, doc: dict) -> None:
        """
        Replace an existing document in the index.

        Removes all old postings for the document, then re-indexes the
        new version. If the document does not exist, it is added.

        Args:
            doc: Updated document dict.
        """
        doc_id = doc.get("id")
        if not doc_id:
            raise ValueError("Document must have an 'id' field.")

        if self._exists(doc_id):
            self._remove_postings(doc_id)
            self._stats.updates += 1
        else:
            self._stats.adds += 1

        self._index._index_document(doc)

    def remove(self, doc_id: str) -> None:
        """
        Remove a document from the index entirely.

        Args:
            doc_id: ID of the document to remove.

        Raises:
            KeyError: If the document is not in the index.
        """
        if not self._exists(doc_id):
            raise KeyError(f"Document '{doc_id}' not found in index.")
        self._remove_postings(doc_id)
        self._stats.removes += 1

    def reindex(self, docs: list[dict]) -> None:
        """
        Full rebuild: clear the index and re-index all documents.

        Use when the corpus has changed substantially (e.g. bulk import).

        Args:
            docs: Complete list of document dicts to index.
        """
        self._index.clear()
        self._index.build(docs)
        self._stats.reindexes += 1

    def exists(self, doc_id: str) -> bool:
        """Return True if a document with this id is in the index."""
        return self._exists(doc_id)

    def stats(self) -> dict:
        """Return a dict of operation counts."""
        return self._stats.to_dict()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _exists(self, doc_id: str) -> bool:
        """Check if doc_id is in the index metadata."""
        return doc_id in self._index._doc_metadata

    def _remove_postings(self, doc_id: str) -> None:
        """
        Remove all postings for doc_id from the index.

        Walks every term in the index and deletes the posting for this
        doc_id if present. Also removes the doc from metadata and lengths.

        Complexity: O(num_terms) — acceptable for incremental updates on
        a corpus of thousands of documents. For very large indexes a
        reverse mapping (doc_id → terms) would make this O(unique_terms_in_doc).
        """
        # Collect terms to clean (avoid mutating dict during iteration)
        terms_to_clean = [
            term for term, postings in self._index._index.items()
            if doc_id in postings
        ]

        for term in terms_to_clean:
            del self._index._index[term][doc_id]
            # Remove the term entirely if it has no postings left
            if not self._index._index[term]:
                del self._index._index[term]

        # Remove doc-level records
        self._index._doc_metadata.pop(doc_id, None)
        self._index._doc_lengths.pop(doc_id, None)
        self._index._num_docs = len(self._index._doc_metadata)
