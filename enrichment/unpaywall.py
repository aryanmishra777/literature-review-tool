"""Unpaywall enricher — open-access links, as a fallback when OpenAlex found none.

Unpaywall has no batch endpoint, so we look up one DOI per request and parallelize with
a small thread pool. Each worker gets its OWN ``requests.Session`` because a session's
connection pool is not guaranteed thread-safe.
"""
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from models import CSLRecord
from enrichment.cache import Cache
from enrichment.doi import norm_doi
from enrichment.http import TIMEOUT, new_session

_URL = "https://api.unpaywall.org/v2/"
_WORKERS = 6  # modest fan-out — we only hit Unpaywall for the OA gaps OpenAlex left.


def _lookup_one(doi: str, email: str) -> str | None:
    """Fetch the best open-access URL for a single DOI, or ``None``.

    Creates a session per call so concurrent workers never share one (see module note).
    """
    try:
        resp = new_session().get(f"{_URL}{doi}", params={"email": email}, timeout=TIMEOUT)
        if resp.status_code != 200:
            return None
        loc = resp.json().get("best_oa_location") or {}
        return loc.get("url_for_pdf") or loc.get("url")
    except (requests.RequestException, ValueError):
        return None


def enrich_oa(records: list[CSLRecord], cache: Cache, email: str) -> int:
    """Fill ``oa_url`` for records OpenAlex left without one. Returns # links filled."""
    targets: dict[str, CSLRecord] = {}
    for rec in records:
        doi = norm_doi(rec.DOI)
        if doi and rec.oa_url is None and not cache.get(doi).get("unpaywall_checked"):
            targets[doi] = rec
    if not targets:
        return 0

    filled = 0
    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        futures = {pool.submit(_lookup_one, doi, email): doi for doi in targets}
        for fut in as_completed(futures):
            doi = futures[fut]
            try:
                url = fut.result()
            except Exception as exc:  # a worker should never take the whole run down.
                print(f"[enrich] Unpaywall worker error for {doi}: {exc}", file=sys.stderr)
                url = None
            cache.update(doi, {"unpaywall_checked": True})
            if url:
                cache.update(doi, {"oa_url": url})
                rec = targets[doi]
                rec.oa_url = url
                rec.metadata_missingness.oa_missing = False
                filled += 1
    return filled
