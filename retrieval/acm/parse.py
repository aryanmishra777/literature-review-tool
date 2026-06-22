"""Extract ``CSLRecord``s from an ACM search-results HTML page.

ACM's markup varies across result layouts, so most helpers try several CSS selectors
and fall back gracefully. Container records (whole proceedings/journals/books) are
skipped so only individual articles survive.
"""
import hashlib
import re

from bs4 import BeautifulSoup, Tag

from models import Author, CSLRecord, MetadataMissingness
from retrieval.acm.config import PAGE_SIZE, RESULTS_SELECTOR

# Result "types" that are collections rather than papers — skip them.
_SKIP_TYPES = {"proceedings", "book", "journal", "magazine", "newsletter", "report"}
_YEAR_SELECTORS = (".bookPubDate", "span.bookPubDate", ".issue-item__detail", ".issue-item__pubdate")
_PROCEEDINGS_TITLE_RE = re.compile(r"^\s*\w.*'?\d{2}:.*proceeding|^proceedings\s+of\b", re.IGNORECASE)


def _parse_total(soup: BeautifulSoup) -> int:
    """Read the reported total hit count, defaulting to one page if not found."""
    for sel in (".hitsLength", ".result__count", "[class*='hitsLength']", "[class*='result-count']"):
        el = soup.select_one(sel)
        if el and (m := re.search(r"[\d,]+", el.get_text())):
            return int(m.group().replace(",", ""))
    if m := re.search(r"([\d,]+)\s*[Rr]esult", soup.get_text()):
        return int(m.group(1).replace(",", ""))
    return PAGE_SIZE


def _extract_doi(href: str) -> str:
    m = re.search(r"10\.\d{4,}/\S+", href)
    return m.group().rstrip("/") if m else ""


def _parse_name(raw: str) -> Author:
    """Split "Given Family" on the last space; treat single tokens as a family name."""
    parts = raw.strip().rsplit(" ", 1)
    return Author(given=parts[0], family=parts[1]) if len(parts) == 2 else Author(given="", family=raw.strip())


def _is_collection(item: Tag) -> bool:
    """True for whole-proceedings / book / journal entries we want to drop."""
    el = item.select_one(".issue-heading")
    heading = el.get_text(strip=True).lower() if el else ""
    if heading in _SKIP_TYPES or "proceeding" in heading:
        return True
    title_el = item.select_one("h3.issue-item__title a, .issue-item__title a")
    title = title_el.get_text(strip=True) if title_el else ""
    return bool(_PROCEEDINGS_TITLE_RE.search(title))


def _extract_abstract(item: Tag) -> str | None:
    container = (
        item.select_one("div.item__abstract")
        or item.select_one(".abstractSection")
        or item.select_one("[class*='abstract']")
    )
    if not container:
        return None
    raw = (container.select_one("p") or container).get_text(strip=True)
    return raw or None


def _extract_year(item: Tag) -> str | None:
    for sel in _YEAR_SELECTORS:
        el = item.select_one(sel)
        if el and (m := re.search(r"\b(19|20)\d{2}\b", el.get_text())):
            return m.group()
    return None


def _extract_url(href: str, doi: str) -> str | None:
    if href.startswith("/"):
        return f"https://dl.acm.org{href}"
    if href:
        return href
    return f"https://doi.org/{doi}" if doi else None


def _parse_item(item: Tag) -> CSLRecord | None:
    """Turn one result ``<li>`` into a ``CSLRecord``, or ``None`` to skip it."""
    if _is_collection(item):
        return None
    title_el = (
        item.select_one("h3.issue-item__title a")
        or item.select_one(".issue-item__title a")
        or item.select_one(".hlFld-Title a")
    )
    if not title_el or not title_el.get_text(strip=True):
        return None

    title = title_el.get_text(strip=True)
    href = title_el.get("href", "")
    doi = _extract_doi(href)
    authors = [
        _parse_name(a.get("title") or a.get_text(strip=True))
        for a in item.select(".hlFld-ContribAuthor a")
        if a.get("title") or a.get_text(strip=True)
    ]
    abstract = _extract_abstract(item)
    # No DOI? Fall back to a stable hash of the title so dedupe still works.
    uid = doi or hashlib.md5(title.encode()).hexdigest()[:12]

    return CSLRecord(
        id=uid,
        type="article-journal",
        title=title,
        author=authors,
        abstract=abstract,
        issued=_extract_year(item),
        DOI=doi or None,
        URL=_extract_url(href, doi),
        source="acm",
        metadata_missingness=MetadataMissingness(abstract_missing=(abstract is None)),
    )


def parse_html(html: str) -> tuple[list[CSLRecord], int]:
    """Parse a results page into ``(records, reported_total)``."""
    soup = BeautifulSoup(html, "lxml")
    total = _parse_total(soup)
    items = soup.select(RESULTS_SELECTOR)
    return [r for item in items if (r := _parse_item(item)) is not None], total
