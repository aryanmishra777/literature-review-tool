from retrieval.acm import ACMScraper
from retrieval.base import BaseTranslator
from retrieval.crossref import CrossrefTranslator


_REGISTRY: dict[str, type[BaseTranslator]] = {
    "crossref": CrossrefTranslator,
    "acm": ACMScraper,
}

DEFAULT_SOURCE = "crossref"


def get_translator(source: str = DEFAULT_SOURCE) -> BaseTranslator:
    cls = _REGISTRY.get(source)
    if cls is None:
        raise ValueError(f"Unknown source '{source}'. Available: {sorted(_REGISTRY)}")
    return cls()
