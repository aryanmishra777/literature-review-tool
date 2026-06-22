"""Build the string we actually send to Crossref's relevance search.

The LLM gives us a refined query (a short canonical phrase) plus a keyword list. We
prefer the *refined phrase* over a bag of keywords because Crossref ranks across all
terms, so connective context matters: "natural language *to* SQL" retrieves far more
precisely than the three bare words. We only strip the most generic container words.
"""
import re

# Generic words that describe a *document* rather than a *topic*. They add no retrieval
# signal and only dilute relevance, so we drop them from the search string.
_NOISE_WORDS = {
    "universities", "university", "books", "book", "papers", "paper",
    "research", "study", "studies", "survey", "literature", "review",
    "introduction", "overview", "tutorial",
}


def build_search_query(refined_query: str, keywords: list[str]) -> str:
    """Return the ``query.bibliographic`` string for Crossref.

    Resolution order, falling back when a step yields nothing usable:
      1. the refined phrase with only noise words removed (the common, best case);
      2. the technical keywords joined together;
      3. the raw refined phrase, untouched.
    """
    words = [
        w for w in re.split(r"\s+", refined_query)
        if w and w.lower() not in _NOISE_WORDS
    ]
    if words:
        return " ".join(words)

    technical = [k for k in keywords if k.lower() not in _NOISE_WORDS]
    if technical:
        return " ".join(technical)

    return refined_query
