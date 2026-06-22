"""Stage 2b — normalize raw translator output to a strict CSL-JSON contract.

Translators (Crossref, ACM) each have their own quirks. This stage gives the rest
of the pipeline one predictable shape to work with: whitespace collapsed, authors
trimmed, and every field either populated or an explicit ``None``.

Guiding principle: *consistency over completeness*. A single malformed record must
never abort the run, so each record is rebuilt inside a try/except and simply
skipped if it cannot be validated.
"""
import re

from models import Author, CSLRecord


def _norm_title(title: str) -> str:
    """Collapse runs of whitespace (newlines, tabs, double spaces) into one space."""
    return re.sub(r"\s+", " ", title).strip()


def _norm_authors(authors: list[Author]) -> list[Author]:
    """Strip stray whitespace from each author's name parts."""
    return [Author(given=a.given.strip(), family=a.family.strip()) for a in authors]


def process_records(records: list[CSLRecord]) -> list[CSLRecord]:
    """Return a cleaned copy of ``records``, dropping any that fail validation.

    Every field the translators populate is carried through. In particular the
    enrichment-related fields (``container_title``, ``cited_by_count``, ``tldr``,
    ``oa_url``) MUST be preserved here: Crossref already fills ``container_title``
    and ``cited_by_count`` at retrieval time, and silently dropping them would make
    the venue and citation count vanish from the synthesis payload and JSON output.
    """
    out: list[CSLRecord] = []
    for rec in records:
        try:
            out.append(CSLRecord(
                id=rec.id,
                type=rec.type,
                title=_norm_title(rec.title),
                author=_norm_authors(rec.author),
                abstract=rec.abstract.strip() if rec.abstract else None,
                issued=rec.issued,
                DOI=rec.DOI,
                URL=rec.URL,
                # Carried through so enrichment / output don't lose them (see docstring).
                container_title=rec.container_title,
                cited_by_count=rec.cited_by_count,
                tldr=rec.tldr,
                oa_url=rec.oa_url,
                source=rec.source,
                metadata_missingness=rec.metadata_missingness,
            ))
        except Exception:
            # A record that won't validate is dropped, never fatal to the batch.
            continue
    return out
