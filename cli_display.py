"""Terminal rendering + prompts for the interactive CLI flow."""
from cli_args import TOP_K_DEFAULT


def print_ranked(ranked: list, cap: int | None = None) -> None:
    """Print the ranked papers as a numbered list with scores, authors, year, and DOI."""
    papers = ranked[:cap] if cap else ranked
    print("=" * 72)
    print(f"RANKED PAPERS  ({len(papers)} shown, score = relevance × citation impact)")
    print("=" * 72)
    for i, r in enumerate(papers, 1):
        rec = r.record
        authors = [f"{a.given} {a.family}".strip() for a in rec.author]
        author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
        year = rec.issued or "n.d."
        flag = "  [no abstract]" if rec.metadata_missingness.abstract_missing else ""
        print(f"\n{i:2}. [{r.final_score:.4f}] {rec.title}{flag}")
        print(f"    {author_str} ({year})")
        if rec.DOI:
            print(f"    https://doi.org/{rec.DOI}")


def ask_top_k(n_total: int, default: int = TOP_K_DEFAULT) -> int:
    """Prompt for how many papers to synthesize, clamped to ``1..n_total``.

    A blank line, EOF (piped/no TTY), or non-numeric input all fall back to the default.
    """
    if n_total == 0:
        return 0
    cap = min(default, n_total)
    try:
        raw = input(f"\nHow many papers to include in the synthesis? (1–{n_total}) [{cap}]: ").strip()
        if not raw:
            return cap
        return max(1, min(int(raw), n_total))
    except (ValueError, EOFError):
        return cap
