from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


class QueryInterpretation(BaseModel):
    """One distinct research sense a query could carry, used to disambiguate.

    Populated only when the query is genuinely ambiguous (the model sees ≥2 senses).
    Each interpretation carries its own search terms so picking one fully redirects retrieval.
    """
    label: str                             # human-readable sense, shown in the picker
    refined_query: str
    focus_query: Optional[str] = None
    keywords: list[str] = Field(default_factory=list)


class StructuredQuery(BaseModel):
    refined_query: str                     # broad canonical topic — drives the recall pass
    keywords: list[str]
    focus_query: Optional[str] = None      # the user's specific angle — drives the precision pass
    intent: Optional[Literal["survey", "implementation", "theory"]] = None
    # Distinct senses the query could mean; empty when unambiguous. The first entry mirrors
    # the top-level refined_query/focus_query (the model's primary guess).
    interpretations: list[QueryInterpretation] = Field(default_factory=list)


class Author(BaseModel):
    given: str = ""
    family: str = ""


class MetadataMissingness(BaseModel):
    abstract_missing: bool = False
    tldr_missing: bool = True
    oa_missing: bool = True


class CSLRecord(BaseModel):
    id: str
    type: str = "article-journal"
    title: str
    author: list[Author] = Field(default_factory=list)
    abstract: Optional[str] = None
    issued: Optional[str] = None
    DOI: Optional[str] = None
    URL: Optional[str] = None
    container_title: Optional[str] = None
    cited_by_count: Optional[int] = None
    tldr: Optional[str] = None
    oa_url: Optional[str] = None
    source: str = "acm"
    metadata_missingness: MetadataMissingness = Field(default_factory=MetadataMissingness)


class RankedRecord(BaseModel):
    record: CSLRecord
    title_score: float = 0.0
    abstract_score: float = 0.0
    semantic_score: float = 0.0     # embedding cosine; 0.0 when semantic ranking is off
    final_score: float = 0.0
    # LLM relevance tier ("high" / "moderate" / "tangential"); None until the tiering pass
    # runs. Purely additive — it labels and re-orders, never removes a paper.
    tier: Optional[str] = None
