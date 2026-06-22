"""DOI-keyed metadata enrichment.

Crossref gives a clean bibliographic skeleton but is thin on abstracts. We fill the
gaps by joining other free scholarly APIs on the *normalized DOI*:

  * OpenAlex        → abstract (rebuilt from an inverted index), OA url, citations  [batched]
  * Unpaywall       → open-access url, as a fallback when OpenAlex had none         [parallel]
  * Semantic Scholar→ a one-line TLDR summary for the top-K papers                  [batched]

Three rules hold everywhere in this package:

  1. Only ever fill a MISSING field — never overwrite what a source already provided.
  2. Every enricher degrades gracefully: on any failure it logs and returns the
     records untouched, so enrichment can never crash the pipeline.
  3. An on-disk DOI cache (with negative-result markers) short-circuits repeat calls.

Outside callers use just the two stage entry points re-exported here.
"""
from enrichment.stages import enrich_pre_rank, enrich_tldr

__all__ = ["enrich_pre_rank", "enrich_tldr"]
