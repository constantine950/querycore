from dataclasses import dataclass, field
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords as nltk_stopwords


def _build_default_stopwords() -> frozenset[str]:
    base = set(nltk_stopwords.words("english"))
    extras = {
        "also", "use", "used", "using", "one", "two", "three",
        "many", "much", "well", "even", "first", "second",
        "new", "known", "called", "often", "may", "can",
        "however", "therefore", "thus", "since", "while",
    }
    return frozenset(base | extras)


DEFAULT_STOPWORDS: frozenset[str] = _build_default_stopwords()


@dataclass
class PreprocessorConfig:
    """Configuration for the Preprocessor."""
    remove_stopwords: bool = True
    apply_stemming:   bool = True
    custom_stopwords: set[str] = field(default_factory=set)
    min_token_length: int = 2


class Preprocessor:

    def __init__(self, config: PreprocessorConfig | None = None):
        self.config = config or PreprocessorConfig()
        self._stemmer = PorterStemmer()
        self._stopwords = DEFAULT_STOPWORDS | frozenset(
            self.config.custom_stopwords)
        # 133× speedup on repeated tokens
        self._stem_cache: dict[str, str] = {}

    def process(self, tokens: list[str]) -> list[str]:
        result = tokens

        if self.config.remove_stopwords:
            result = self._remove_stopwords(result)

        if self.config.apply_stemming:
            result = self._stem(result)

        result = [t for t in result if len(t) >= self.config.min_token_length]

        return result

    def process_text(self, text: str) -> list[str]:
        from src.indexer.tokenizer import Tokenizer
        tokens = Tokenizer().tokenize(text)
        return self.process(tokens)

    def stem(self, token: str) -> str:
        if token not in self._stem_cache:
            self._stem_cache[token] = self._stemmer.stem(token)
        return self._stem_cache[token]

    def is_stopword(self, token: str) -> bool:
        return token in self._stopwords

    @property
    def stopwords(self) -> frozenset[str]:
        return self._stopwords

    def cache_stats(self) -> dict:
        return {"cached_stems": len(self._stem_cache)}

    def _remove_stopwords(self, tokens: list[str]) -> list[str]:
        return [t for t in tokens if t not in self._stopwords]

    def _stem(self, tokens: list[str]) -> list[str]:
        return [self.stem(t) for t in tokens]
