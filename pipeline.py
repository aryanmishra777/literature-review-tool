"""Stage orchestration: query → retrieve → enrich → rank → synthesize.

The pipeline exposes its stages individually (``run_retrieval`` / ``synthesize``) so the
CLI can run retrieval, show the ranked list, ask the user how many papers to synthesize,
and only then call the model. ``run`` is a convenience wrapper for non-interactive use.
"""
import sys
from typing import Callable

from config import DEFAULT_MODEL, OLLAMA_API_KEY, OLLAMA_HOST
from enrichment import enrich_pre_rank, enrich_tldr
from models import CSLRecord, QueryInterpretation, RankedRecord, StructuredQuery

# A disambiguation hook: given an ambiguous StructuredQuery, return the interpretation(s) the
# user picked (one or several — senses can be combined), or an empty list to accept the
# model's primary guess. The CLI supplies one in interactive runs; non-interactive callers
# pass nothing (auto-pick the first sense).
DisambiguateFn = Callable[[StructuredQuery], list[QueryInterpretation]]
from processing import process_records
from query_understanding import QueryUnderstanding
from ranking import rank
from result import build_result
from retrieval import get_translator
from search_query import build_search_query
from semantic import semantic_scores
from synthesis import Synthesizer, select_for_synthesis
from tiering import TIER_SEQUENCE, Tierer, prioritize_by_tier


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
        tier: bool = True,
        provider: str = "ollama",
    ):
        self._qu = QueryUnderstanding(model=model, host=host, api_key=api_key, provider=provider)
        self._translator = get_translator(source)
        self._synthesizer = Synthesizer(model=model, host=host, api_key=api_key, provider=provider)
        self._tierer = Tierer(model=model, host=host, api_key=api_key, provider=provider)
        self._source = source
        self._enrich = enrich
        self._semantic = semantic
        self._tier = tier
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
    ) -> tuple[StructuredQuery, list[tuple[str, str | None, list[str]]], str | None, list[str]]:
        """Pick one or more senses when the query is ambiguous, then steer retrieval to them.

        Returns four things:
          • the (possibly redirected) query for display/lexical scoring,
          • the **retrieval specs** — one ``(refined_query, focus_query, keywords)`` per chosen
            sense, so each sense gets its own two-pass search (merged by DOI downstream),
          • a *positive hint* (the chosen sense(s) flattened to text, appended to the embedding
            so the pick reaches the dominant semantic ranker), or ``None``,
          • the *rejected* senses as text (negative anchors the ranker subtracts).

        With <2 interpretations there is nothing to resolve — one spec, no hint, no negatives.
        Otherwise ask ``disambiguate``: the user may pick **several** senses (their literatures
        are unioned; the rest become negatives). If it declines or there is no hook, fall back
        to the model's primary guess and log it, so an auto-pick is never silent.
        """
        interps = structured.interpretations
        if len(interps) < 2:
            spec = (structured.refined_query, structured.focus_query, structured.keywords)
            return structured, [spec], None, []

        chosen = (disambiguate(structured) if disambiguate else None) or []
        if not chosen:
            chosen = [interps[0]]
            _log('      note: query is ambiguous — auto-picked one sense (run interactively to choose):')
            _log(f'            → {chosen[0].label}')
            for other in interps[1:]:
                _log(f'              other sense: {other.label}')
        elif len(chosen) > 1:
            _log(f'      note: combining {len(chosen)} senses:')
            for c in chosen:
                _log(f'            + {c.label}')

        # Union the chosen senses' keywords (order-preserving) for the lexical/display fields.
        kw_seen: set[str] = set()
        kw_union: list[str] = []
        for c in chosen:
            for k in c.keywords or []:
                if k.lower() not in kw_seen:
                    kw_seen.add(k.lower())
                    kw_union.append(k)
        kw_union = kw_union or structured.keywords

        redirected = structured.model_copy(update={
            "refined_query": chosen[0].refined_query,
            "focus_query": chosen[0].focus_query,
            "keywords": kw_union,
        })
        specs = [(c.refined_query, c.focus_query, c.keywords or kw_union) for c in chosen]
        hint = ". ".join(self._sense_text(c) for c in chosen)
        negatives = [self._sense_text(it) for it in interps
                     if all(it is not c for c in chosen)]
        return redirected, specs, hint, negatives

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
        structured, specs, sem_hint, sem_negatives = self._resolve_interpretation(
            structured, disambiguate)
        _log(f'      refined  : "{structured.refined_query}"')
        _log(f"      keywords : {structured.keywords}")
        _log(f"      intent   : {structured.intent}")

        shown_limit = limit if limit is not None else "unlimited"
        _log(f"[2/4] Retrieving from {self._source} (limit={shown_limit}, workers={workers})...")
        # One two-pass search (broad recall + focused precision) per chosen sense; the broad
        # pass keeps the user's specific angle from being flattened, and multiple senses union
        # their literatures. All passes merge by DOI — order is irrelevant (ranking re-sorts).
        passes: list[list[CSLRecord]] = []
        for refined, focus, kws in specs:
            search = build_search_query(refined, kws)
            _log(f'      search   : "{search}"  (broad / recall)')
            passes.append(self._translator.search(search, limit=limit, workers=workers))
            if focus:
                focus_search = build_search_query(focus, kws)
                _log(f'      focus    : "{focus_search}"  (focused / precision)')
                passes.append(self._translator.search(focus_search, limit=limit, workers=workers))
        raw_records = _merge_by_doi(*passes)
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

    def tier_papers(
        self, query: str, ranked: list[RankedRecord], intent: str = ""
    ) -> list[RankedRecord]:
        """Stage 3c: label each ranked paper high / moderate / tangential (LLM).

        Additive only — sets ``.tier`` in place and returns the same list, untouched in
        order. Skipped (a no-op) when tiering is disabled or there are no papers.
        """
        if not self._tier or not ranked:
            return ranked
        _log(f"[3c]  Tiering {len(ranked)} papers by relevance (LLM)...")
        self._tierer.assign(query or "", ranked, intent=intent)
        counts = {t: sum(1 for r in ranked if r.tier == t) for t in TIER_SEQUENCE}
        _log(f"      tiers: {counts['high']} highly / {counts['moderate']} moderately / "
             f"{counts['tangential']} tangentially relevant")
        return ranked

    def synthesize(
        self, query: str, ranked: list[RankedRecord], top_k: int = 10, intent: str = ""
    ) -> str:
        """Stage 4: synthesize a review over a bounded slice of the ranked papers.

        When papers have been tiered, the budget is filled from the highly-relevant ones
        first (stable within a tier) — but only by re-ordering *within* the ``top_k`` papers
        the user chose to act on, never by pulling in papers outside that slice (so the
        review can't cite a paper that isn't in the shown list). ``select_for_synthesis``
        caps both the paper count and the prompt size, so the request can never exceed the
        model's context window regardless of ``top_k``.
        """
        pool = prioritize_by_tier(ranked[:top_k])
        selected = select_for_synthesis(pool, top_k)
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
        # Tiering is a default stage in its own right (independent of synthesis): label the
        # slice the result will expose — the same papers the table and grouped section show.
        if ranked:
            self.tier_papers(query, ranked[:top_k], intent=structured.intent or "")
        review = ""
        if not skip_synthesis and ranked:
            review = self.synthesize(query, ranked, top_k=top_k, intent=structured.intent or "")
        return build_result(structured, records, ranked, review, top_k)
