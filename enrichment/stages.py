"""The two public enrichment stages the pipeline calls.

``enrich_pre_rank`` runs over ALL candidates before ranking, so that abstracts are
present when the ranker computes its abstract-similarity term. ``enrich_tldr`` runs
after ranking over just the top-K slice that synthesis will actually read.

Both create their own cache, wrap each source in a try/except, flush once, and return
the (mutated) records. Neither ever raises.
"""
import sys

from config import CONTACT_EMAIL
from models import CSLRecord
from enrichment.cache import Cache, apply_cached
from enrichment.openalex import enrich_abstracts
from enrichment.unpaywall import enrich_oa
from enrichment.semantic_scholar import enrich_tldr as _enrich_tldr_s2


def enrich_pre_rank(records: list[CSLRecord]) -> list[CSLRecord]:
    """Fill abstracts / OA links / citations for every record, before ranking.

    Order of resolution: cache → OpenAlex (batched) → Unpaywall (OA fallback).
    """
    if not records:
        return records
    cache = Cache()
    apply_cached(records, cache)

    try:
        n_abstracts = enrich_abstracts(records, cache, CONTACT_EMAIL)
    except Exception as exc:
        print(f"[enrich] OpenAlex stage failed: {exc}", file=sys.stderr)
        n_abstracts = 0
    try:
        n_oa = enrich_oa(records, cache, CONTACT_EMAIL)
    except Exception as exc:
        print(f"[enrich] Unpaywall stage failed: {exc}", file=sys.stderr)
        n_oa = 0

    cache.flush()
    have_abstracts = sum(1 for r in records if r.abstract)
    print(
        f"[enrich] abstracts: {have_abstracts}/{len(records)} present "
        f"(+{n_abstracts} via OpenAlex); OA links +{n_oa} via Unpaywall fallback",
        file=sys.stderr,
    )
    return records


def enrich_tldr(records: list[CSLRecord]) -> list[CSLRecord]:
    """Fill TLDR summaries for a (typically top-K) record slice."""
    if not records:
        return records
    cache = Cache()
    apply_cached(records, cache)
    try:
        n = _enrich_tldr_s2(records, cache)
    except Exception as exc:
        print(f"[enrich] Semantic Scholar stage failed: {exc}", file=sys.stderr)
        n = 0
    cache.flush()
    print(f"[enrich] TLDRs: +{n} via Semantic Scholar (top {len(records)})", file=sys.stderr)
    return records
