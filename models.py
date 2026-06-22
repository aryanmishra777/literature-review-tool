from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


class StructuredQuery(BaseModel):
    refined_query: str
    keywords: list[str]
    intent: Optional[Literal["survey", "implementation", "theory"]] = None


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
    final_score: float = 0.0
