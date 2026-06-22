"""On-disk DOI cache: ``{normalized_doi: {abstract, tldr, oa_url, cited_by_count, ...}}``.

The cache does two jobs:

  * *Positive* caching — store fields we fetched (abstract, oa_url, ...) so a re-run
    fills them instantly with no network call.
  * *Negative* caching — store boolean markers (``openalex_checked`` etc.) recording
    "we already asked this source about this DOI and it had nothing". Without these,
    every re-run would re-query the same DOIs that we know return nothing.

The file lives next to the package and is gitignored.
"""
import json
import sys
from pathlib import Path

from models import CSLRecord
from enrichment.doi import norm_doi

# Stored alongside the package root (…/enrichment/ -> project root).
CACHE_PATH = Path(__file__).resolve().parent.parent / "enrichment_cache.json"


class Cache:
    """A thin JSON-backed dict. Load once, consult before any network call, flush once."""

    def __init__(self, path: Path = CACHE_PATH):
        self._path = path
        self._data: dict[str, dict] = {}
        self._dirty = False
        try:
            if path.exists():
                self._data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            # A corrupt cache is not worth crashing over — start empty and rewrite it.
            self._data = {}

    def get(self, doi: str) -> dict:
        return self._data.get(doi, {})

    def update(self, doi: str, fields: dict) -> None:
        """Merge ``fields`` into the DOI's entry, but only fill blanks (never overwrite)."""
        if not fields:
            return
        entry = self._data.setdefault(doi, {})
        for key, value in fields.items():
            if value is not None and entry.get(key) is None:
                entry[key] = value
                self._dirty = True

    def flush(self) -> None:
        """Write back to disk only if something actually changed."""
        if not self._dirty:
            return
        try:
            self._path.write_text(json.dumps(self._data), encoding="utf-8")
            self._dirty = False
        except Exception as exc:
            print(f"[enrich] cache write failed: {exc}", file=sys.stderr)


def apply_cached(records: list[CSLRecord], cache: Cache) -> None:
    """Fill record fields from the cache before any network call is made."""
    for rec in records:
        doi = norm_doi(rec.DOI)
        if not doi:
            continue
        cached = cache.get(doi)
        if not cached:
            continue
        if rec.abstract is None and cached.get("abstract"):
            rec.abstract = cached["abstract"]
            rec.metadata_missingness.abstract_missing = False
        if rec.oa_url is None and cached.get("oa_url"):
            rec.oa_url = cached["oa_url"]
            rec.metadata_missingness.oa_missing = False
        if rec.tldr is None and cached.get("tldr"):
            rec.tldr = cached["tldr"]
            rec.metadata_missingness.tldr_missing = False
        if rec.cited_by_count is None and cached.get("cited_by_count") is not None:
            rec.cited_by_count = cached["cited_by_count"]
