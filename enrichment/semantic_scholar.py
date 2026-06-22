"""Semantic Scholar enricher — one-line TLDR summaries (+ abstract fallback).

Semantic Scholar's batch endpoint takes up to 500 DOIs in a single POST, so the whole
top-K slice fits in one call. TLDRs are a "nice to have", so any failure here is silently
non-fatal. The endpoint rate-limits aggressively, hence the 429-aware retry loop.
"""
import sys
import time

import requests

from models import CSLRecord
from enrichment.cache import Cache
from enrichment.doi import norm_doi
from enrichment.http import TIMEOUT, new_session

_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
_MAX_IDS = 500
_MAX_ATTEMPTS = 3


def _fetch_batch(session: requests.Session, dois: list[str]) -> dict[str, dict]:
    """Return ``{norm_doi: {tldr, abstract}}`` for up to ``_MAX_IDS`` DOIs."""
    ids = [f"DOI:{d}" for d in dois[:_MAX_IDS]]
    out: dict[str, dict] = {}
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            resp = session.post(
                _URL,
                params={"fields": "externalIds,tldr,abstract"},
                json={"ids": ids},
                timeout=TIMEOUT,
            )
            if resp.status_code == 429:  # rate-limited — wait a bit and retry.
                time.sleep(2.0 * attempt)
                continue
            if resp.status_code != 200:
                print(f"[enrich] Semantic Scholar HTTP {resp.status_code} — skipping TLDRs", file=sys.stderr)
                return out
            for paper in resp.json() or []:
                if not paper:
                    continue  # S2 returns null for DOIs it doesn't know.
                doi = norm_doi((paper.get("externalIds") or {}).get("DOI"))
                if not doi:
                    continue
                out[doi] = {
                    "tldr": (paper.get("tldr") or {}).get("text"),
                    "abstract": paper.get("abstract"),
                }
            return out
        except (requests.RequestException, ValueError) as exc:
            print(f"[enrich] Semantic Scholar request error (attempt {attempt}/{_MAX_ATTEMPTS}): {exc}", file=sys.stderr)
            time.sleep(1.5 * attempt)
    return out


def enrich_tldr(records: list[CSLRecord], cache: Cache) -> int:
    """Fill TLDR summaries (and abstract as a fallback) for the given records."""
    by_doi: dict[str, list[CSLRecord]] = {}
    for rec in records:
        doi = norm_doi(rec.DOI)
        if doi and rec.tldr is None and not cache.get(doi).get("s2_checked"):
            by_doi.setdefault(doi, []).append(rec)
    if not by_doi:
        return 0

    results = _fetch_batch(new_session(), list(by_doi))
    for doi in by_doi:
        cache.update(doi, {"s2_checked": True})  # negative cache for every queried DOI.

    filled = 0
    for doi, fields in results.items():
        cache.update(doi, fields)
        for rec in by_doi.get(doi, []):
            if rec.tldr is None and fields.get("tldr"):
                rec.tldr = fields["tldr"]
                rec.metadata_missingness.tldr_missing = False
                filled += 1
            if rec.abstract is None and fields.get("abstract"):
                rec.abstract = fields["abstract"]
                rec.metadata_missingness.abstract_missing = False
    return filled
