"""Assemble the final result dict (the shape used by both ``--json`` and the saved file)."""
from models import CSLRecord, RankedRecord, StructuredQuery


def _authors(rec: CSLRecord) -> list[str]:
    """Flatten structured authors to display strings ("Given Family")."""
    return [f"{a.given} {a.family}".strip() for a in rec.author]


def build_result(
    structured: StructuredQuery,
    records: list[CSLRecord],
    ranked: list[RankedRecord],
    review: str,
    top_k: int,
) -> dict:
    """Combine query understanding, the top-K ranked papers, and the review into one dict.

    Only the top-K papers are serialized — that's the slice the user asked to act on.
    ``records_retrieved`` still reports the full retrieved count for context.
    """
    return {
        "structured_query": structured.model_dump(),
        "records_retrieved": len(records),
        "ranked": [
            {
                "title": r.record.title,
                "authors": _authors(r.record),
                "year": r.record.issued,
                "venue": r.record.container_title,
                "DOI": r.record.DOI,
                "URL": r.record.URL,
                "oa_url": r.record.oa_url,
                "tldr": r.record.tldr,
                "cited_by_count": r.record.cited_by_count,
                "score": round(r.final_score, 4),
                "title_score": round(r.title_score, 4),
                "abstract_score": round(r.abstract_score, 4),
                "semantic_score": round(r.semantic_score, 4),
                "abstract_missing": r.record.metadata_missingness.abstract_missing,
                "tier": r.tier,
            }
            for r in ranked[:top_k]
        ],
        "review": review,
    }
