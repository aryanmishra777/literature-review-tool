"""Terminal rendering + prompts for the interactive CLI flow."""
import re

from cli_args import TOP_K_DEFAULT
from models import QueryInterpretation, StructuredQuery


def choose_interpretations(structured: StructuredQuery) -> list[QueryInterpretation]:
    """Ask the user which sense(s) of an ambiguous query to search.

    Returns the chosen interpretations — one *or several*, since related senses can be
    combined (their literatures are unioned and the rest become contrastive negatives).
    Returns ``[]`` to accept the model's primary guess: a blank line or EOF (piped/no TTY)
    does that. Unparseable or out-of-range input re-prompts rather than silently overriding
    the user. Only called when ≥2 interpretations exist.
    """
    interps = structured.interpretations
    print("\nThis query is ambiguous — it could mean different things to a paper search:")
    for i, it in enumerate(interps, 1):
        marker = "  (best guess)" if i == 1 else ""
        print(f"  {i}. {it.label}{marker}")
        print(f'       → searches: "{it.refined_query}"')
    prompt = f"\nWhich sense(s) did you mean? (1–{len(interps)}, comma-separated for several) [1]: "
    while True:
        try:
            raw = input(prompt).strip()
        except EOFError:
            return []
        if not raw:
            return []
        parts = [p for p in re.split(r"[,\s]+", raw) if p]
        if not all(p.isdigit() for p in parts):
            print("  Please enter number(s) like 1 or 1,2.")
            continue
        idxs = [int(p) for p in parts]
        if not all(1 <= i <= len(interps) for i in idxs):
            print(f"  Pick number(s) between 1 and {len(interps)}.")
            continue
        # De-dupe, preserve the order the user typed.
        seen: set[int] = set()
        return [interps[i - 1] for i in idxs if not (i in seen or seen.add(i))]


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
