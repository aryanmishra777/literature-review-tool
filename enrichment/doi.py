"""The one DOI normalizer used as the universal join + cache key.

Different APIs return DOIs in different shapes (`https://doi.org/10.x`, `doi:10.x`,
mixed case). Normalizing to a bare, lowercased DOI means a record fetched from Crossref
and the same work returned by OpenAlex collapse to the exact same key.
"""

# Prefixes a DOI may be wrapped in, longest-first so we strip the right one.
_PREFIXES = ("https://doi.org/", "http://doi.org/", "doi:")


def norm_doi(doi: str | None) -> str | None:
    """Return a bare lowercased DOI (e.g. ``10.1109/icpc.2021.00053``), or ``None``."""
    if not doi:
        return None
    d = doi.strip().lower()
    for prefix in _PREFIXES:
        if d.startswith(prefix):
            d = d[len(prefix):]
            break
    return d or None
