"""Argument parser definition for the CLI, kept separate so ``cli.py`` stays focused."""
import argparse

from config import DEFAULT_MODEL, GROQ_DEFAULT_MODEL, OLLAMA_HOST

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
        "--provider",
        default="ollama",
        choices=["ollama", "groq"],
        help="LLM provider for query understanding and synthesis (default: ollama). "
             "groq is cloud-only and needs GROQ_API_KEY in .env.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model for query understanding and synthesis. Defaults per provider "
             f"(ollama: {DEFAULT_MODEL}; groq: {GROQ_DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--ollama-host",
        default=None,
        help=f"Override the Ollama server URL (default: {OLLAMA_HOST}). Ignored for "
             "--provider groq (Groq uses its fixed cloud endpoint).",
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
        help="Skip the LLM review and only produce the ranked paper list (papers are still "
             "tiered unless --no-tier; add both for a fully synthesis-free, label-free run)",
    )
    parser.add_argument(
        "--no-enrich",
        action="store_true",
        help="Skip metadata enrichment (abstracts/OA/TLDR); faster, offline-friendly",
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Skip SBERT embedding ranking and use lexical scoring only "
             "(deterministic, no model load)",
    )
    parser.add_argument(
        "--no-tier",
        action="store_true",
        help="Skip the LLM relevance-tiering pass. By default each ranked paper is labelled "
             "highly / moderately / tangentially relevant (a grouped section is added to the "
             "output and synthesis draws from the highly-relevant papers first). Tiering runs "
             "by default even with --no-synthesis. Nothing is ever removed — this only adds "
             "labels.",
    )
    parser.add_argument(
        "--min-relevance",
        type=float,
        default=0.25,
        metavar="X",
        help="Soft relevance floor: drop candidates scoring below X (0–1) before the "
             "citation boost; trims off-topic tail noise. Default 0.25; use 0 to disable, "
             "raise toward 0.30–0.35 for a tighter, higher-precision list.",
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=0.4,
        metavar="X",
        help="Contrastive down-weighting: when you disambiguate, subtract X × similarity to "
             "the rejected sense(s) so papers in the wrong sense sink. Only active when a "
             "query was ambiguous and a sense was chosen. Default 0.4; 0 disables, raise "
             "toward 0.5 for a stricter cut (at some cost to borderline-relevant recall).",
    )
    parser.add_argument(
        "--interactive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prompt during the run: pick a sense for ambiguous queries and choose how "
             "many papers to synthesize. On by default; --no-interactive accepts defaults "
             "without prompting (auto-pick the primary sense, default top-k). Always off "
             "under --json, and prompts are skipped automatically when stdin isn't a TTY.",
    )
    parser.add_argument(
        "--show-metadata",
        action="store_true",
        help="Add a 'Query Understanding' block to the saved Markdown (before the papers): the "
             "refined/focus queries, keywords, intent, every sense the model considered (marking "
             "the one(s) searched), and the exact search strings issued. Transparency aid; does "
             "not affect ranking. The same fields are always present in --json output.",
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
