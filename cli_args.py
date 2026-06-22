"""Argument parser definition for the CLI, kept separate so ``cli.py`` stays focused."""
import argparse

from config import DEFAULT_MODEL, OLLAMA_HOST

TOP_K_DEFAULT = 10


def build_parser() -> argparse.ArgumentParser:
    """Construct the full argument parser. See ``--help`` for the user-facing docs."""
    parser = argparse.ArgumentParser(
        description="Generate a literature review from scholarly metadata using an Ollama model."
    )
    parser.add_argument("query", help="Research query in natural language")
    parser.add_argument(
        "--source",
        default="crossref",
        choices=["crossref", "acm"],
        help="Paper source to search (default: crossref; acm is a best-effort scraper fallback)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model for query understanding and synthesis (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--ollama-host",
        default=OLLAMA_HOST,
        help=f"Ollama server URL (default: {OLLAMA_HOST})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        metavar="N",
        help="Maximum papers to retrieve (default: 50; use 0 for no limit). Crossref "
             "returns by relevance, so the first ~50 carry most of the signal.",
    )
    parser.add_argument(
        "--no-synthesis",
        action="store_true",
        help="Skip LLM synthesis and only show the ranked paper list",
    )
    parser.add_argument(
        "--no-enrich",
        action="store_true",
        help="Skip metadata enrichment (abstracts/OA/TLDR); faster, offline-friendly",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write the output Markdown file to disk",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Directory to save the output Markdown file (default: current directory)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent page fetches during retrieval; only used by the acm source (default: 4)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Emit full structured JSON to stdout; non-interactive, uses the default top-k",
    )
    return parser
