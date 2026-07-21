"""
autocomplete.py

Prefix-based query suggestion using a trie (prefix tree) built from
all terms in the inverted index.

What is a trie?
---------------
A trie is a tree where each edge represents a character and each path
from the root to a node spells out a prefix. Every term in the index
is inserted as a path from the root; leaf nodes (or marked nodes) are
complete terms.

Structure for ["search", "searcher", "season"]:

    root
     └─ s
        └─ e
           ├─ a
           │  ├─ r
           │  │  ├─ c
           │  │  │  └─ h  [TERM] doc_count=5
           │  │  │     └─ e
           │  │  │        └─ r  [TERM] doc_count=2
           │  └─ s
           │     └─ o
           │        └─ n  [TERM] doc_count=1

Lookup: traverse the trie following the prefix characters, then collect
all [TERM] nodes in the subtree — that's the suggestion list.

Time complexity:
    insert : O(k)  where k = term length
    lookup : O(k + m)  where m = number of terms with that prefix

Usage
-----
    from src.search.autocomplete import Autocomplete

    ac = Autocomplete(index)
    suggestions = ac.suggest("sea", top_n=5)
    # → ["search", "season", ...]   (sorted by doc frequency)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from src.indexer.inverted_index import InvertedIndex


# ---------------------------------------------------------------------------
# TrieNode
# ---------------------------------------------------------------------------

@dataclass
class TrieNode:
    """A single node in the prefix trie."""
    children:  dict[str, "TrieNode"] = field(default_factory=dict)
    is_term:   bool = False    # True if this node marks a complete indexed term
    # the full term string (only set when is_term=True)
    term:      str = ""
    # number of docs containing this term (for ranking)
    doc_count: int = 0


# ---------------------------------------------------------------------------
# Autocomplete
# ---------------------------------------------------------------------------

class Autocomplete:
    """
    Trie-based autocomplete engine built from an InvertedIndex.

    Suggests completions for a prefix, ranked by document frequency
    (terms that appear in more documents surface first — they are more
    likely to be useful suggestions).
    """

    def __init__(self, index: InvertedIndex) -> None:
        self._root = TrieNode()
        self._built = False
        self._build(index)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self, index: InvertedIndex) -> None:
        """Insert every term from the index into the trie."""
        for term in index.get_all_terms():
            doc_count = len(index.get_postings(term))
            self._insert(term, doc_count)
        self._built = True

    def _insert(self, term: str, doc_count: int) -> None:
        """Insert a single term into the trie."""
        node = self._root
        for char in term:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_term = True
        node.term = term
        node.doc_count = doc_count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def suggest(self, prefix: str, top_n: int = 10) -> list[str]:
        """
        Return up to top_n term suggestions for a given prefix.

        Suggestions are sorted by document frequency descending — more
        common terms appear first since they're more likely what the
        user wants.

        Args:
            prefix : Lowercase prefix string to complete.
            top_n  : Maximum number of suggestions to return.

        Returns:
            List of stemmed index terms that start with prefix.
            Empty list if prefix matches no terms.

        Example:
            >>> ac.suggest("sea")
            ["search", "season"]
        """
        if not prefix:
            return []

        prefix = prefix.lower().strip()
        node = self._find_prefix_node(prefix)
        if node is None:
            return []

        # Collect all terms in the subtree rooted at this node
        matches: list[tuple[int, str]] = []
        self._collect(node, matches)

        # Sort by doc_count descending, then alphabetically for ties
        matches.sort(key=lambda x: (-x[0], x[1]))
        return [term for _, term in matches[:top_n]]

    def suggest_with_scores(self, prefix: str, top_n: int = 10) -> list[dict]:
        """
        Like suggest(), but returns dicts with term and doc_count.
        Useful for the API layer to expose suggestion metadata.
        """
        if not prefix:
            return []

        prefix = prefix.lower().strip()
        node = self._find_prefix_node(prefix)
        if node is None:
            return []

        matches: list[tuple[int, str]] = []
        self._collect(node, matches)
        matches.sort(key=lambda x: (-x[0], x[1]))

        return [
            {"term": term, "doc_count": count}
            for count, term in matches[:top_n]
        ]

    def has_prefix(self, prefix: str) -> bool:
        """Return True if any indexed term starts with this prefix."""
        return self._find_prefix_node(prefix.lower()) is not None

    def exact_match(self, term: str) -> bool:
        """Return True if term is an exact indexed term (not just a prefix)."""
        node = self._find_prefix_node(term.lower())
        return node is not None and node.is_term

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_prefix_node(self, prefix: str) -> TrieNode | None:
        """
        Traverse the trie following prefix characters.
        Returns the node at the end of the prefix, or None if prefix
        is not in the trie.
        """
        node = self._root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def _collect(self, node: TrieNode, results: list[tuple[int, str]]) -> None:
        """
        DFS from node, collecting all (doc_count, term) pairs where is_term=True.
        Mutates results in place.
        """
        if node.is_term:
            results.append((node.doc_count, node.term))
        for child in node.children.values():
            self._collect(child, results)
