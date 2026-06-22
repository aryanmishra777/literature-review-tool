"""Turn a raw Crossref JSON ``item`` into a clean ``CSLRecord`` (or drop it).

Crossref's search returns more than just papers — supplementary figures, datasets,
peer-review reports, whole-journal container records, and conference front-matter all
share keywords with real articles and would otherwise pollute the results. The filters
in this module keep only authored, article-like records.
"""
import re

from bs4 import BeautifulSoup

from models import Author, CSLRecord, MetadataMissingness

# Crossref `type` values are already CSL-JSON types, so we pass them straight through
# (e.g. "journal-article", "proceedings-article"). This default only kicks in for the
# rare record with no type at all.
_DEFAULT_TYPE = "article-journal"

# Container / non-article record types that share search keywords but aren't papers.
_EXCLUDED_TYPES = {
    "component",        # supplementary figures, tables, datasets attached to a paper
    "dataset",
    "peer-review",
    "grant",
    "standard",
    "standard-series",
    "journal",          # whole-journal container records
    "journal-issue",
    "journal-volume",
    "book-series",
    "book-set",
    "proceedings",      # the whole proceedings, not an article within it
    "proceedings-series",
    "report-series",
    "other",
}

# Conference front-matter (covers, copyright pages, tables of contents) is often
# registered as a `proceedings-article` with no authors and a container-style title.
# We only drop a title matching this when it ALSO has zero authors, so a real authored
# paper can never be filtered out by accident.
_FRONTMATTER_RE = re.compile(
    r"""
      ^\s*proceedings\b
    | \[?\s*cover\s*art\s*\]?
    | -\s*cover\b
    | \bfront\s*matter\b
    | -\s*copyright\b | ^\s*copyright\b
    | \btable\s+of\s+contents\b
    | \bauthor\s+index\b
    | \bprogram\s+committee\b
    | \btitle\s+page\b
    | \b\d+(?:st|nd|rd|th)\s+(?:ieee\s+)?international\s+
        (?:conference|workshop|symposium)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _first(seq) -> str | None:
    """Crossref wraps single values (title, container-title) in a list — unwrap it."""
    return seq[0] if isinstance(seq, list) and seq else None


def _strip_jats(abstract: str | None) -> str | None:
    """Crossref abstracts arrive as JATS XML; reduce them to plain text."""
    if not abstract:
        return None
    text = BeautifulSoup(abstract, "lxml").get_text(" ", strip=True)
    # The markup frequently embeds the literal word "Abstract" as a heading.
    if text.lower().startswith("abstract "):
        text = text[len("abstract "):].strip()
    return text or None


def _parse_authors(raw: list | None) -> list[Author]:
    authors: list[Author] = []
    for a in raw or []:
        given = (a.get("given") or "").strip()
        family = (a.get("family") or "").strip()
        if not given and not family:
            # Organisations / consortia carry a single "name" instead of given/family.
            family = (a.get("name") or "").strip()
        if given or family:
            authors.append(Author(given=given, family=family))
    return authors


def _parse_year(issued: dict | None) -> str | None:
    """Extract the publication year as a string, or ``None`` if unknown.

    Crossref represents an unknown date as ``{"date-parts": [[None]]}`` (a list whose
    only element is ``None``). Calling ``str()`` on that ``None`` would yield the
    literal string ``"None"`` — which is *truthy*, so it would defeat every
    ``year or "n.d."`` fallback downstream and render as "None" in the output. Guard
    against it explicitly.
    """
    try:
        year = issued["date-parts"][0][0]
    except (TypeError, KeyError, IndexError):
        return None
    return str(year) if year is not None else None


def _is_frontmatter(title: str, n_authors: int) -> bool:
    return n_authors == 0 and bool(_FRONTMATTER_RE.search(title))


def parse_item(item: dict) -> CSLRecord | None:
    """Map one Crossref item to a ``CSLRecord``, or ``None`` if it should be dropped."""
    doi = (item.get("DOI") or "").strip().lower()
    title = _first(item.get("title"))
    if not doi or not title:
        return None
    if (item.get("type") or "").lower() in _EXCLUDED_TYPES:
        return None

    authors = _parse_authors(item.get("author"))
    if _is_frontmatter(title, len(authors)):
        return None

    abstract = _strip_jats(item.get("abstract"))
    cites = item.get("is-referenced-by-count")

    return CSLRecord(
        id=doi,
        type=item.get("type") or _DEFAULT_TYPE,
        title=title,
        author=authors,
        abstract=abstract,
        issued=_parse_year(item.get("issued")),
        DOI=doi,
        URL=item.get("URL") or f"https://doi.org/{doi}",
        container_title=_first(item.get("container-title")),
        cited_by_count=cites if isinstance(cites, int) else None,
        source="crossref",
        metadata_missingness=MetadataMissingness(abstract_missing=abstract is None),
    )
