"""Stage orchestration: query → retrieve → enrich → rank → synthesize.

The pipeline exposes its stages individually (``run_retrieval`` / ``synthesize``) so the
CLI can run retrieval, show the ranked list, ask the user how many papers to synthesize,
and only then call the model. ``run`` is a convenience wrapper for non-interactive use.
"""
import sys
from typing import Callable, Optional

from config import DEFAULT_MODEL, OLLAMA_API_KEY, OLLAMA_HOST
from enrichment import enrich_pre_rank, enrich_tldr
from models import CSLRecord, QueryInterpretation, RankedRecord, StructuredQuery

# A disambiguation hook: given an ambiguous StructuredQuery, return the interpretation the
# user picked, or None to accept the model's primary guess. The CLI supplies one in
# interactive runs; non-interactive callers pass nothing (auto-pick the first sense).
DisambiguateFn = Callable[[StructuredQuery], Optional[QueryInterpretation]]
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
        host: str = OLLAMA_HOST,
        api_key: str = OLLAMA_API_KEY,
        source: str = "crossref",
        enrich: bool = True,
        semantic: bool = True,
        provider: str = "ollama",
    ):
        self._qu = QueryUnderstanding(model=model, host=host, api_key=api_key, provider=provider)
        self._translator = get_translator(source)
        self._synthesizer = Synthesizer(model=model, host=host, api_key=api_key, provider=provider)
        self._source = source
        self._enrich = enrich
        self._semantic = semantic
        self.model = model

    @staticmethod
    def _sense_text(interp) -> str:
        """Flatten an interpretation into embedding text: label + phrase + its own keywords."""
        parts = [interp.label, interp.refined_query]
        if interp.keywords:
            parts.append(", ".join(interp.keywords))
        return ". ".join(p for p in parts if p)

    def _resolve_interpretation(
        self, structured: StructuredQuery, disambiguate: DisambiguateFn | None
    ) -> tuple[StructuredQuery, str | None, list[str]]:
        """Pick a sense when the query is ambiguous, then redirect retrieval to it.

        Returns the (possibly redirected) query, a *positive hint* naming the chosen sense
        (appended to the embedding text so the pick reaches the dominant semantic ranker),
        and the *rejected* senses as text (negative anchors the ranker subtracts, so papers
        hugging a wrong sense are pushed down). Hint is ``None`` / negatives empty when there
        was nothing to disambiguate.

        With <2 interpretations there is nothing to resolve. Otherwise ask ``disambiguate``
        (interactive runs); if it declines or there is no hook, fall back to the model's
        primary guess (first interpretation) and log which sense was chosen plus the
        alternatives, so an auto-pick is never silent.
        """
        interps = structured.interpretations
        if len(interps) < 2:
            return structured, None, []

        chosen = disambiguate(structured) if disambiguate else None
        if chosen is None:
            chosen = interps[0]
            _log('      note: query is ambiguous — auto-picked one sense (run interactively to choose):')
            _log(f'            → {chosen.label}')
            for other in interps[1:]:
                _log(f'              other sense: {other.label}')

        redirected = structured.model_copy(update={
            "refined_query": chosen.refined_query,
            "focus_query": chosen.focus_query,
            "keywords": chosen.keywords or structured.keywords,
        })
        hint = self._sense_text(chosen)
        negatives = [self._sense_text(it) for it in interps if it is not chosen]
        return redirected, hint, negatives

    def run_retrieval(
        self,
        query: str,
        limit: int | None = None,
        workers: int = 2,
        disambiguate: DisambiguateFn | None = None,
        min_relevance: float = 0.0,
        contrast: float = 0.0,
    ) -> tuple[StructuredQuery, list[CSLRecord], list[RankedRecord]]:
        """Stages 1–3: understand → retrieve → (enrich) → rank. No LLM synthesis."""
        _log("[1/4] Understanding query...")
        structured = self._qu.transform(query)
        structured, sem_hint, sem_negatives = self._resolve_interpretation(structured, disambiguate)
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
            # Embed the raw query (keeps its rich intent anchors); when the user disambiguated,
            # append the chosen sense so it reaches the dominant 0.8 semantic signal, not just
            # the search/lexical layers. Unambiguous queries embed the raw query unchanged.
            semantic_query = query or structured.refined_query
            negatives = sem_negatives if contrast > 0 else []
            if sem_hint:
                semantic_query = f"{semantic_query}. {sem_hint}"
                _log(f'      semantic anchor: "{sem_hint}"')
            if negatives:
                _log(f"      contrastive down-weighting vs {len(negatives)} rejected sense(s) "
                     f"(penalty {contrast})")
            _log("[3a]  Scoring semantic similarity (SBERT embeddings)...")
            sem = semantic_scores(semantic_query, records,
                                  negative_texts=negatives, penalty=contrast)

        _log("[3/4] Ranking by relevance (semantic + lexical, citation-aware)...")
        ranked = rank(records, structured.refined_query, structured.keywords,
                      original_query=query, semantic_scores=sem, min_relevance=min_relevance)
        if min_relevance > 0:
            _log(f"      floor: kept {len(ranked)} of {len(records)} above relevance {min_relevance}")
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
        disambiguate: DisambiguateFn | None = None,
        min_relevance: float = 0.0,
        contrast: float = 0.0,
    ) -> dict:
        """Non-interactive convenience wrapper: retrieve, synthesize, and build the result."""
        structured, records, ranked = self.run_retrieval(
            query, limit, workers=workers, disambiguate=disambiguate,
            min_relevance=min_relevance, contrast=contrast,
        )
        review = ""
        if not skip_synthesis and ranked:
            review = self.synthesize(query, ranked, top_k=top_k, intent=structured.intent or "")
        return build_result(structured, records, ranked, review, top_k)
