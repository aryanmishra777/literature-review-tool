"""Crossref retrieval source.

Crossref is the canonical DOI metadata registry: publishers (ACM included) deposit
their bibliographic metadata here at publication time, so it is fresh, complete for
the core fields, and — unlike scraping a publisher site — has no anti-bot wall.

The package is split into three small, single-purpose modules:

  * ``client``     — the HTTP layer (session, retries, the works endpoint).
  * ``parse``      — turning a raw Crossref JSON item into a clean ``CSLRecord``.
  * ``translator`` — the cursor-paging loop and the public ``CrossrefTranslator``.

Only ``CrossrefTranslator`` is meant to be imported from outside.
"""
from retrieval.crossref.translator import CrossrefTranslator

__all__ = ["CrossrefTranslator"]
