"""Terminal rendering + prompts for the interactive CLI flow."""
from cli_args import TOP_K_DEFAULT
from models import QueryInterpretation, StructuredQuery


def choose_interpretation(structured: StructuredQuery) -> QueryInterpretation | None:
    """Ask the user which sense of an ambiguous query to search.

    Returns the chosen interpretation, or ``None`` to accept the model's primary guess
    (the caller then falls back to it). A blank line, EOF (piped/no TTY), or out-of-range
    input all select the first/primary sense. Only called when ≥2 interpretations exist.
    """
    interps = structured.interpretations
    print("\nThis query is ambiguous — it could mean different things to a paper search:")
    for i, it in enumerate(interps, 1):
        marker = "  (best guess)" if i == 1 else ""
        print(f"  {i}. {it.label}{marker}")
        print(f'       → searches: "{it.refined_query}"')
    try:
        raw = input(f"\nWhich sense did you mean? (1–{len(interps)}) [1]: ").strip()
    except EOFError:
        return None
    if not raw:
        return None
    try:
        idx = int(raw)
    except ValueError:
        return None
    if 1 <= idx <= len(interps):
        return interps[idx - 1]
    return None


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
