"""Stage orchestration: query → retrieve → enrich → rank → synthesize.

The pipeline exposes its stages individually (``run_retrieval`` / ``synthesize``) so the
CLI can run retrieval, show the ranked list, ask the user how many papers to synthesize,
and only then call the model. ``run`` is a convenience wrapper for non-interactive use.
"""
import sys

from config import DEFAULT_MODEL, OLLAMA_API_KEY, OLLAMA_HOST
from enrichment import enrich_pre_rank, enrich_tldr
from models import CSLRecord, RankedRecord, StructuredQuery
from processing import process_records
from query_understanding import QueryUnderstanding
from ranking import rank
from result import build_result
from retrieval import get_translator
from search_query import build_search_query
from semantic import semantic_scores
from synthesis import Synthesizer, select_for_synthesis


def _log(msg: str) -> None:
    """All progress chatter goes to stderr so stdout stays clean for ``--json``."""
    print(msg, file=sys.stderr, flush=True)


def _merge_by_doi(*passes: list[CSLRecord]) -> list[CSLRecord]:
    """Union several retrieval passes into one candidate pool, deduped by DOI.

    A faceted query is searched twice — a broad topic pass (recall) and a focused pass
    (precision) — and the same work can surface in both. We keep the first occurrence of
    each DOI (metadata is identical across passes, both being Crossref). Order is
    irrelevant downstream because ranking re-sorts the whole pool anyway.
    """
    seen: set[str] = set()
    merged: list[CSLRecord] = []
    for records in passes:
        for rec in records:
            key = rec.DOI or rec.id
            if key in seen:
                continue
            seen.add(key)
            merged.append(rec)
    return merged


class LiteratureReviewPipeline:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        ollama_host: str = OLLAMA_HOST,
        api_key: str = OLLAMA_API_KEY,
        source: str = "crossref",
        enrich: bool = True,
        semantic: bool = True,
    ):
        self._qu = QueryUnderstanding(model=model, host=ollama_host, api_key=api_key)
        self._translator = get_translator(source)
        self._synthesizer = Synthesizer(model=model, host=ollama_host, api_key=api_key)
        self._source = source
        self._enrich = enrich
        self._semantic = semantic
        self.model = model

    def run_retrieval(
        self, query: str, limit: int | None = None, workers: int = 2
    ) -> tuple[StructuredQuery, list[CSLRecord], list[RankedRecord]]:
        """Stages 1–3: understand → retrieve → (enrich) → rank. No LLM synthesis."""
        _log("[1/4] Understanding query...")
        structured = self._qu.transform(query)
        _log(f'      refined  : "{structured.refined_query}"')
        _log(f"      keywords : {structured.keywords}")
        _log(f"      intent   : {structured.intent}")

        search = build_search_query(structured.refined_query, structured.keywords)
        shown_limit = limit if limit is not None else "unlimited"
        _log(f"[2/4] Retrieving from {self._source} (limit={shown_limit}, workers={workers})...")
        _log(f'      search   : "{search}"  (broad / recall)')
        raw_records = self._translator.search(search, limit=limit, workers=workers)

        # Faceted queries get a second, narrower pass so the user's specific angle
        # actually reaches Crossref instead of being flattened to the bare topic.
        if structured.focus_query:
            focus_search = build_search_query(structured.focus_query, structured.keywords)
            _log(f'      focus    : "{focus_search}"  (focused / precision)')
            focus_records = self._translator.search(focus_search, limit=limit, workers=workers)
            raw_records = _merge_by_doi(focus_records, raw_records)
        _log(f"      fetched {len(raw_records)} unique records")

        _log("[2b]  Processing (schema enforcement)...")
        records = process_records(raw_records)
        _log(f"      kept {len(records)} valid records")

        if self._enrich:
            _log("[2c]  Enriching metadata (abstracts, OA links, citations)...")
            records = enrich_pre_rank(records)

        sem = None
        if self._semantic:
            _log("[3a]  Scoring semantic similarity (SBERT embeddings)...")
            sem = semantic_scores(query or structured.refined_query, records)

        _log("[3/4] Ranking by relevance (semantic + lexical, citation-aware)...")
        ranked = rank(records, structured.refined_query, structured.keywords,
                      original_query=query, semantic_scores=sem)
        # The two passes can union to more than the caller asked for; keep the best `limit`
        # after ranking so a richer, more on-intent pool still honours --limit.
        if limit is not None:
            ranked = ranked[:limit]
        return structured, records, ranked

    def synthesize(
        self, query: str, ranked: list[RankedRecord], top_k: int = 10, intent: str = ""
    ) -> str:
        """Stage 4: synthesize a review over a bounded slice of the ranked papers.

        ``select_for_synthesis`` caps both the paper count and the prompt size, so the
        request can never exceed the model's context window regardless of ``top_k``.
        """
        selected = select_for_synthesis(ranked, top_k)
        k = len(selected)
        requested = min(top_k, len(ranked))
        if k < requested:
            _log(f"      note: capped synthesis to {k} papers (requested {requested}) "
                 f"to fit the model context window")
        if self._enrich and k:
            _log(f"[3b]  Enriching top {k} with TLDR summaries...")
            enrich_tldr([r.record for r in selected])
        _log(f"[4/4] Synthesizing review (top {k})...")
        return self._synthesizer.synthesize(query, selected, intent=intent)

    def build_result(self, structured, records, ranked, review, top_k) -> dict:
        """Thin pass-through to :func:`result.build_result` (kept for call-site stability)."""
        return build_result(structured, records, ranked, review, top_k)

    def run(
        self,
        query: str,
        limit: int | None = None,
        top_k: int = 10,
        skip_synthesis: bool = False,
        workers: int = 2,
    ) -> dict:
        """Non-interactive convenience wrapper: retrieve, synthesize, and build the result."""
        structured, records, ranked = self.run_retrieval(query, limit, workers=workers)
        review = ""
        if not skip_synthesis and ranked:
            review = self.synthesize(query, ranked, top_k=top_k, intent=structured.intent or "")
        return build_result(structured, records, ranked, review, top_k)
