"""
tokenizer.py

Splits raw text into a list of lowercase tokens.
This is the first stage of the indexing pipeline:

  Raw Text → Tokenizer → [tokens] → Preprocessor → Inverted Index

Design decisions:
  - Regex-based splitting on non-alphanumeric characters
  - Unicode NFKD normalization to handle accented chars and ligatures
  - Lowercasing applied here (not in preprocessor) so everything downstream
    works with a uniform casing assumption
  - Hyphenated words are split: "state-of-the-art" → ["state", "of", "the", "art"]
  - Numbers are kept as tokens (useful for date/version queries)
  - Minimum token length of 1 enforced; empty strings filtered out

Usage:
    from src.indexer.tokenizer import Tokenizer

    t = Tokenizer()
    tokens = t.tokenize("Search Engines are Fast!")
    # → ["search", "engines", "are", "fast"]
"""

import re
import unicodedata
from dataclasses import dataclass, field


@dataclass
class TokenizerConfig:
    """Configuration for the Tokenizer."""
    min_token_length: int = 1          # tokens shorter than this are dropped
    keep_numbers: bool = True          # whether to keep numeric tokens
    # regex pattern to split on (post-lowercase)
    split_pattern: str = r"[^a-z0-9]"


class Tokenizer:
    """
    Converts raw text into a flat list of lowercase string tokens.

    This class is intentionally simple and stateless — it has no knowledge
    of stop words or stemming. Those concerns belong to the Preprocessor.
    """

    def __init__(self, config: TokenizerConfig | None = None):
        self.config = config or TokenizerConfig()
        self._pattern = re.compile(self.config.split_pattern)

    def normalize(self, text: str) -> str:
        """
        Unicode-normalize text and lowercase it.

        NFKD decomposition converts characters like 'é' → 'e' + combining accent,
        then we strip the non-ASCII combining marks, leaving just 'e'.
        """
        # Decompose unicode characters
        normalized = unicodedata.normalize("NFKD", text)
        # Drop combining characters (accents, diacritics)
        ascii_only = "".join(
            c for c in normalized
            if not unicodedata.combining(c)
        )
        return ascii_only.lower()

    def split(self, text: str) -> list[str]:
        """
        Split normalized text into raw tokens using the split pattern.
        Filters out empty strings and tokens below min_token_length.
        """
        raw_tokens = self._pattern.split(text)
        tokens = [
            t for t in raw_tokens
            if len(t) >= self.config.min_token_length
        ]
        if not self.config.keep_numbers:
            tokens = [t for t in tokens if not t.isdigit()]
        return tokens

    def tokenize(self, text: str) -> list[str]:
        """
        Full pipeline: normalize → split → return token list.

        Args:
            text: Raw input string (any encoding, any case).

        Returns:
            List of lowercase string tokens.

        Example:
            >>> Tokenizer().tokenize("Search Engines are Fast!")
            ['search', 'engines', 'are', 'fast']
        """
        if not text or not text.strip():
            return []
        normalized = self.normalize(text)
        return self.split(normalized)

    def tokenize_document(self, doc: dict) -> list[str]:
        """
        Tokenize both the title and body of a document dict.
        Title tokens are included so title matches are possible.

        Args:
            doc: dict with at least 'title' and 'body' keys.

        Returns:
            Combined list of tokens from title + body.
        """
        title_tokens = self.tokenize(doc.get("title", ""))
        body_tokens = self.tokenize(doc.get("body", ""))
        return title_tokens + body_tokens

    def tokenize_query(self, query: str) -> list[str]:
        """
        Tokenize a user query string.
        Identical to tokenize() but semantically distinct — query tokenization
        may diverge from doc tokenization in future (e.g. query expansion).

        Args:
            query: Raw user query string.

        Returns:
            List of lowercase query tokens.
        """
        return self.tokenize(query)
