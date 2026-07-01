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
from cli_display import ask_top_k, choose_interpretations, print_ranked
from config import OLLAMA_HOST, provider_defaults
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
            min_relevance=args.min_relevance,
            contrast=args.contrast,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(result, indent=2))


def _run_interactive(pipeline: LiteratureReviewPipeline, args, limit: int | None) -> None:
    """Interactive path: retrieve, show, prompt, synthesize, save."""
    # Offer the disambiguation picker only when prompting is enabled AND there's a human
    # at a TTY; --no-interactive or a piped/redirected stdin auto-picks the primary sense.
    can_prompt = args.interactive and sys.stdin.isatty()
    disambiguate = choose_interpretations if can_prompt else None
    try:
        structured, records, ranked = pipeline.run_retrieval(
            query=args.query, limit=limit, workers=args.workers,
            disambiguate=disambiguate, min_relevance=args.min_relevance, contrast=args.contrast,
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

    # Decide how many papers to act on. --no-synthesis skips the review but NOT tiering —
    # tiering is a default stage (like query understanding, it uses the LLM regardless).
    if args.no_synthesis:
        top_k = TOP_K_DEFAULT
    else:
        top_k = (
            ask_top_k(n_total=len(ranked), default=TOP_K_DEFAULT)
            if can_prompt
            else min(TOP_K_DEFAULT, len(ranked))
        )
    # Tier the slice the result will expose (labels the table/grouped section and lets
    # synthesis draw from the highly-relevant papers first). No-op under --no-tier.
    if ranked:
        pipeline.tier_papers(args.query, ranked[:top_k], intent=sq.get("intent") or "")
    review = ""
    if not args.no_synthesis and ranked:
        review = pipeline.synthesize(
            args.query, ranked, top_k=top_k, intent=sq.get("intent") or ""
        )

    if review:
        print("\n" + "=" * 72)
        print("LITERATURE REVIEW")
        print("=" * 72)
        print(review)

    result = pipeline.build_result(structured, records, ranked, review, top_k)
    if not args.no_save and ranked:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_markdown(args.query, result, args.model, out_dir,
                                 show_metadata=args.show_metadata)
        print(f"\nSaved → {out_path}", file=sys.stderr)


def main() -> None:
    args = build_parser().parse_args()
    limit = _normalize_limit(args.limit)

    # Resolve provider settings: host/key/default-model come from the chosen provider,
    # with --model and --ollama-host as optional overrides.
    host, api_key, default_model = provider_defaults(args.provider)
    if args.ollama_host and args.provider == "ollama":
        host = args.ollama_host
    args.model = args.model or default_model

    if args.provider == "ollama" and not api_key:
        print(
            "Warning: OLLAMA_API_KEY not found in .env — "
            f"falling back to local Ollama at {OLLAMA_HOST}",
            file=sys.stderr,
        )
    if args.provider == "groq" and not api_key:
        print("Error: --provider groq requires GROQ_API_KEY in .env", file=sys.stderr)
        sys.exit(1)

    pipeline = LiteratureReviewPipeline(
        model=args.model,
        host=host,
        api_key=api_key,
        source=args.source,
        enrich=not args.no_enrich,
        semantic=not args.no_semantic,
        tier=not args.no_tier,
        provider=args.provider,
    )

    if args.output_json:
        _run_json(pipeline, args, limit)
    else:
        _run_interactive(pipeline, args, limit)


if __name__ == "__main__":
    main()
