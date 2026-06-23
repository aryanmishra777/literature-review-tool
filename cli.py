#!/usr/bin/env python3
"""Command-line entry point.

Two modes share one pipeline:
  * ``--json``      — fully non-interactive; prints a JSON result to stdout.
  * default         — interactive; shows the ranked list, asks how many papers to
                      synthesize, prints the review, and saves a Markdown file.
"""
import json
import sys
from pathlib import Path

from cli_args import TOP_K_DEFAULT, build_parser
from cli_display import ask_top_k, print_ranked
from config import OLLAMA_API_KEY
from output import save_markdown
from pipeline import LiteratureReviewPipeline


def _normalize_limit(raw: int) -> int | None:
    """0 (or negative) means "no limit" — normalize to None for uniform paging."""
    return raw if (raw and raw > 0) else None


def _run_json(pipeline: LiteratureReviewPipeline, args, limit: int | None) -> None:
    """Non-interactive path: run everything and print the JSON result."""
    try:
        result = pipeline.run(
            query=args.query,
            limit=limit,
            top_k=TOP_K_DEFAULT,
            skip_synthesis=args.no_synthesis,
            workers=args.workers,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(result, indent=2))


def _run_interactive(pipeline: LiteratureReviewPipeline, args, limit: int | None) -> None:
    """Interactive path: retrieve, show, prompt, synthesize, save."""
    try:
        structured, records, ranked = pipeline.run_retrieval(
            query=args.query, limit=limit, workers=args.workers
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    sq = structured.model_dump()
    print(f"\nRefined query : {sq['refined_query']}")
    print(f"Keywords      : {', '.join(sq['keywords']) or '—'}")
    print(f"Intent        : {sq['intent'] or 'not classified'}")
    print(f"Retrieved     : {len(records)} papers\n")
    print_ranked(ranked, cap=len(ranked))

    # Decide how many papers to synthesize (skipped entirely with --no-synthesis).
    if args.no_synthesis:
        top_k, review = TOP_K_DEFAULT, ""
    else:
        top_k = ask_top_k(n_total=len(ranked), default=TOP_K_DEFAULT)
        review = pipeline.synthesize(
            args.query, ranked, top_k=top_k, intent=sq.get("intent") or ""
        ) if ranked else ""

    if review:
        print("\n" + "=" * 72)
        print("LITERATURE REVIEW")
        print("=" * 72)
        print(review)

    result = pipeline.build_result(structured, records, ranked, review, top_k)
    if not args.no_save and ranked:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_markdown(args.query, result, args.model, out_dir)
        print(f"\nSaved → {out_path}", file=sys.stderr)


def main() -> None:
    args = build_parser().parse_args()
    limit = _normalize_limit(args.limit)

    if not OLLAMA_API_KEY:
        print(
            "Warning: OLLAMA_API_KEY not found in .env — "
            "falling back to local Ollama at http://localhost:11434",
            file=sys.stderr,
        )

    pipeline = LiteratureReviewPipeline(
        model=args.model,
        ollama_host=args.ollama_host,
        source=args.source,
        enrich=not args.no_enrich,
        semantic=not args.no_semantic,
    )

    if args.output_json:
        _run_json(pipeline, args, limit)
    else:
        _run_interactive(pipeline, args, limit)


if __name__ == "__main__":
    main()
