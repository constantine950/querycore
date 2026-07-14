"""
preprocessor.py

Stage 2 of the indexing pipeline. Takes a list of tokens from the Tokenizer
and returns a cleaned list ready to be inserted into the inverted index.

Pipeline position:
    Raw Text → Tokenizer → [tokens] → Preprocessor → [clean tokens] → Inverted Index

Two operations are applied in order:

    1. Stop word removal
       Common words that carry no discriminating signal ("the", "is", "a", "of")
       are filtered out. This reduces index size and improves TF-IDF quality
       because stop words would otherwise dominate term frequency counts.

    2. Stemming (Porter algorithm)
       Words are reduced to their root form so that "running", "runs", and "ran"
       all map to "run" in the index. This improves recall — a query for "search"
       will match documents containing "searches" or "searching".

       Trade-off: stemming can hurt precision. "universe" and "university" both
       stem to "univers". We accept this trade-off for a portfolio search engine.

Usage:
    from src.indexer.tokenizer import Tokenizer
    from src.indexer.preprocessor import Preprocessor

    tokens = Tokenizer().tokenize("The search engines are searching fast")
    clean  = Preprocessor().process(tokens)
    # → ["search", "engin", "search", "fast"]
    #   ("the", "are" removed; "engines"→"engin", "searching"→"search")
"""

from dataclasses import dataclass, field
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords as nltk_stopwords


# ---------------------------------------------------------------------------
# Default English stop words (NLTK list + common extras)
# ---------------------------------------------------------------------------

def _build_default_stopwords() -> frozenset[str]:
    """
    Return the default stop word set: NLTK's English list plus a few
    domain-specific additions that carry no search signal.
    """
    base = set(nltk_stopwords.words("english"))
    extras = {
        "also", "use", "used", "using", "one", "two", "three",
        "many", "much", "well", "even", "first", "second",
        "new", "known", "called", "often", "may", "can",
        "however", "therefore", "thus", "since", "while",
    }
    return frozenset(base | extras)


DEFAULT_STOPWORDS: frozenset[str] = _build_default_stopwords()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PreprocessorConfig:
    """Configuration for the Preprocessor."""
    remove_stopwords: bool = True
    apply_stemming: bool = True
    custom_stopwords: set[str] = field(default_factory=set)
    # Tokens shorter than this after stemming are dropped
    min_token_length: int = 2


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class Preprocessor:
    """
    Filters stop words and applies Porter stemming to a token list.

    Designed to be composed with Tokenizer, not subclassed:

        tokens = Tokenizer().tokenize(text)
        clean  = Preprocessor().process(tokens)
    """

    def __init__(self, config: PreprocessorConfig | None = None):
        self.config = config or PreprocessorConfig()
        self._stemmer = PorterStemmer()
        self._stopwords = DEFAULT_STOPWORDS | frozenset(
            self.config.custom_stopwords)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, tokens: list[str]) -> list[str]:
        """
        Apply the full preprocessing pipeline to a token list.

        Args:
            tokens: List of lowercase strings from Tokenizer.

        Returns:
            Filtered and stemmed token list.

        Example:
            >>> Preprocessor().process(["the", "search", "engines", "are", "fast"])
            ['search', 'engin', 'fast']
        """
        result = tokens

        if self.config.remove_stopwords:
            result = self._remove_stopwords(result)

        if self.config.apply_stemming:
            result = self._stem(result)

        # Drop tokens that became too short after stemming
        result = [t for t in result if len(t) >= self.config.min_token_length]

        return result

    def process_text(self, text: str) -> list[str]:
        """
        Convenience: tokenize + preprocess in one call.
        Imports Tokenizer inline to avoid circular dependency.

        Args:
            text: Raw input string.

        Returns:
            Fully preprocessed token list.
        """
        from src.indexer.tokenizer import Tokenizer
        tokens = Tokenizer().tokenize(text)
        return self.process(tokens)

    def stem(self, token: str) -> str:
        """
        Stem a single token. Exposed so the query parser can stem
        query terms with the same algorithm used at index time.

        Args:
            token: A single lowercase string.

        Returns:
            Stemmed form of the token.
        """
        return self._stemmer.stem(token)

    def is_stopword(self, token: str) -> bool:
        """Return True if the token is in the stop word set."""
        return token in self._stopwords

    @property
    def stopwords(self) -> frozenset[str]:
        """Read-only access to the active stop word set."""
        return self._stopwords

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _remove_stopwords(self, tokens: list[str]) -> list[str]:
        return [t for t in tokens if t not in self._stopwords]

    def _stem(self, tokens: list[str]) -> list[str]:
        return [self._stemmer.stem(t) for t in tokens]
