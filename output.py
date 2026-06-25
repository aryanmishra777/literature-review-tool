"""Render a result dict to a Markdown file.

The layout is: a metadata header (query, keywords, intent, model), a ranked papers table
with links + an open-access column, and finally the generated review. URL shortening is
delegated to :mod:`urls`.
"""
import re
import sys
from datetime import date
from pathlib import Path

from tiering import TIER_LABELS, TIER_SEQUENCE
from urls import shorten_urls


def _slugify(text: str, max_len: int = 60) -> str:
    """Turn a query into a safe, short filename stem."""
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s_]+", "-", slug).strip("-")
    return slug[:max_len]


def _esc(text: str) -> str:
    """Escape pipe characters so they don't break Markdown table cells."""
    return text.replace("|", "\\|")


def _paper_row(index: int, paper: dict, paper_url: str, doi_url: str, oa_url: str) -> str:
    """Format a single ranked paper as one Markdown table row."""
    title = _esc(paper["title"])
    title_cell = f"[{title}]({paper_url})" if paper_url else title

    authors = paper["authors"]
    author_str = _esc(", ".join(authors[:2])) + (" et al." if len(authors) > 2 else "")

    year = paper["year"] or "n.d."
    score = f"{paper['score']:.4f}"
    abstract_cell = "—" if paper["abstract_missing"] else "✓"

    doi = paper.get("DOI") or ""
    doi_cell = f"[{doi}]({doi_url})" if doi and doi_url else (doi or "—")
    oa_cell = f"[PDF]({oa_url})" if oa_url else "—"

    return f"| {index} | {title_cell} | {author_str} | {year} | {score} | {abstract_cell} | {doi_cell} | {oa_cell} |"


def _tier_section(papers: list[dict], paper_urls: list[str]) -> list[str]:
    """Group the ranked papers under relevance-tier headings (additive — keeps every paper).

    The ranked table above stays in score order; this is a second view that buckets the same
    papers into highly / moderately / tangentially relevant so the structure of the output is
    visible at a glance. Each entry keeps its rank number to tie back to the table. Returns an
    empty list (renders nothing) when the papers were not tiered.
    """
    if not any(p.get("tier") for p in papers):
        return []

    # Bucket by tier, preserving rank order (papers is already score-sorted). Carry the 1-based
    # rank and the shortened title link so entries point back to the table row.
    buckets: dict[str, list[str]] = {t: [] for t in TIER_SEQUENCE}
    for rank, (paper, p_url) in enumerate(zip(papers, paper_urls), 1):
        tier = paper.get("tier")
        if tier not in buckets:
            continue
        title = _esc(paper["title"])
        title_cell = f"[{title}]({p_url})" if p_url else title
        authors = paper["authors"]
        author_str = _esc(", ".join(authors[:2])) + (" et al." if len(authors) > 2 else "")
        year = paper["year"] or "n.d."
        meta = f" — {author_str} ({year})" if author_str else f" ({year})"
        buckets[tier].append(f"- **#{rank}** {title_cell}{meta}")

    lines = ["", "---", "", "## Relevance Tiers", "",
             "_Every ranked paper above, grouped by how relevant the model judged it to the "
             "research goal. Nothing is removed — this only labels the list; the review is "
             "built from the highly-relevant papers first._", ""]
    for tier in TIER_SEQUENCE:
        rows = buckets[tier]
        lines.append(f"### {TIER_LABELS[tier]} ({len(rows)})")
        lines += rows or ["- _none_"]
        lines.append("")
    return lines


def build_markdown(query, result, model, short_paper_urls, short_doi_urls, short_oa_urls) -> str:
    sq = result["structured_query"]
    papers = result["ranked"]
    review = result.get("review", "")

    lines: list[str] = [
        f"# Literature Review: {query}",
        "",
        f"> **Date:** {date.today().isoformat()} &nbsp;|&nbsp; "
        f"**Source:** Crossref (+ OpenAlex / Unpaywall / Semantic Scholar enrichment) "
        f"&nbsp;|&nbsp; **Model:** {model}  ",
        f"> **Refined query:** {sq['refined_query']}  ",
        f"> **Keywords:** {', '.join(sq['keywords']) or '—'}  ",
        f"> **Intent:** {sq['intent'] or 'not classified'}",
        "",
        "---",
        "",
        f"## Papers ({result['records_retrieved']} retrieved, top {len(papers)} ranked)",
        "",
        "| # | Title | Authors | Year | Score | Abstract | DOI | OA |",
        "|--:|-------|---------|-----:|------:|:--------:|-----|:--:|",
    ]
    for i, (paper, p_url, d_url, o_url) in enumerate(
        zip(papers, short_paper_urls, short_doi_urls, short_oa_urls), 1
    ):
        lines.append(_paper_row(i, paper, p_url, d_url, o_url))

    lines += _tier_section(papers, short_paper_urls)

    if review:
        lines += ["", "---", "", "## Literature Review", "", review]
    lines.append("")
    return "\n".join(lines)


def save_markdown(query: str, result: dict, model: str, out_dir: Path) -> Path:
    """Shorten all the links, render the Markdown, and write ``<slug>.md``."""
    papers = result["ranked"]
    n = len(papers)

    paper_urls = [p.get("URL") or "" for p in papers]
    doi_urls = [f"https://doi.org/{p['DOI']}" if p.get("DOI") else "" for p in papers]
    oa_urls = [p.get("oa_url") or "" for p in papers]

    print("[out] Shortening URLs...", file=sys.stderr, flush=True)
    # Shorten all three columns in one parallel pass, then split back out by section.
    short = shorten_urls(paper_urls + doi_urls + oa_urls)
    md = build_markdown(query, result, model, short[:n], short[n:2 * n], short[2 * n:])

    out_path = out_dir / f"{_slugify(query)}.md"
    out_path.write_text(md, encoding="utf-8")
    return out_path
