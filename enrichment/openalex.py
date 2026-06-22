"""OpenAlex enricher — abstracts (+ OA url + citation count), batched by DOI.

OpenAlex lets us OR many DOIs into one ``filter`` query, so we fetch up to 50 records
per request. Abstracts come back as an *inverted index* (word -> positions) rather than
plain text, so we reconstruct the prose ourselves.
"""
import sys

import requests

from models import CSLRecord
from enrichment.cache import Cache
from enrichment.doi import norm_doi
from enrichment.http import TIMEOUT, new_session

_URL = "https://api.openalex.org/works"
_BATCH = 50  # max DOIs OR-ed into a single filter / page of results.


def _rebuild_abstract(inv: dict | None) -> str | None:
    """Reconstruct plain text from OpenAlex's ``abstract_inverted_index``.

    The index maps each word to the list of positions it occupies; we flatten that back
    into ``(position, word)`` pairs, sort by position, and join.
    """
    if not inv:
        return None
    positions: list[tuple[int, str]] = []
    for word, idxs in inv.items():
        for i in idxs:
            positions.append((i, word))
    if not positions:
        return None
    positions.sort()
    return " ".join(word for _, word in positions) or None


def _fetch_batch(session: requests.Session, dois: list[str], mailto: str) -> dict[str, dict]:
    """Return ``{norm_doi: {abstract, oa_url, cited_by_count}}`` for one batch of DOIs."""
    params = {
        "filter": "doi:" + "|".join(f"https://doi.org/{d}" for d in dois),
        "per-page": _BATCH,
        "select": "doi,abstract_inverted_index,open_access,cited_by_count",
        "mailto": mailto,
    }
    out: dict[str, dict] = {}
    try:
        resp = session.get(_URL, params=params, timeout=TIMEOUT)
        if resp.status_code != 200:
            print(f"[enrich] OpenAlex HTTP {resp.status_code} for a batch — skipping", file=sys.stderr)
            return out
        for work in resp.json().get("results", []):
            doi = norm_doi(work.get("doi"))
            if not doi:
                continue
            out[doi] = {
                "abstract": _rebuild_abstract(work.get("abstract_inverted_index")),
                "oa_url": (work.get("open_access") or {}).get("oa_url"),
                "cited_by_count": work.get("cited_by_count"),
            }
    except (requests.RequestException, ValueError) as exc:
        # ValueError covers a malformed-JSON 200 from resp.json().
        print(f"[enrich] OpenAlex request error — skipping batch: {exc}", file=sys.stderr)
    return out


def _apply(rec: CSLRecord, fields: dict) -> bool:
    """Fill a record's blanks from a result. Returns True iff an abstract was filled."""
    filled_abstract = False
    if rec.abstract is None and fields.get("abstract"):
        rec.abstract = fields["abstract"]
        rec.metadata_missingness.abstract_missing = False
        filled_abstract = True
    if rec.oa_url is None and fields.get("oa_url"):
        rec.oa_url = fields["oa_url"]
        rec.metadata_missingness.oa_missing = False
    if rec.cited_by_count is None and fields.get("cited_by_count") is not None:
        rec.cited_by_count = fields["cited_by_count"]
    return filled_abstract


def enrich_abstracts(records: list[CSLRecord], cache: Cache, mailto: str) -> int:
    """Fill missing abstracts (+ oa_url, citations) from OpenAlex. Returns # abstracts filled.

    DOIs already looked up in a prior run are skipped via the ``openalex_checked``
    negative-cache marker, so repeat runs make no redundant calls for records OpenAlex
    can't help with.
    """
    by_doi: dict[str, list[CSLRecord]] = {}
    for rec in records:
        doi = norm_doi(rec.DOI)
        if not doi or (rec.abstract is not None and rec.oa_url is not None):
            continue
        if cache.get(doi).get("openalex_checked"):
            continue
        by_doi.setdefault(doi, []).append(rec)
    if not by_doi:
        return 0

    session = new_session()
    dois = list(by_doi)
    filled = 0
    for start in range(0, len(dois), _BATCH):
        chunk = dois[start:start + _BATCH]
        results = _fetch_batch(session, chunk, mailto)
        # Mark every queried DOI as checked, even ones with no result (negative cache).
        for doi in chunk:
            cache.update(doi, {"openalex_checked": True})
        for doi, fields in results.items():
            cache.update(doi, fields)
            for rec in by_doi.get(doi, []):
                filled += int(_apply(rec, fields))
    return filled
