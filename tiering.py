"""LLM relevance triage: label each ranked paper high / moderate / tangential.

This is a purely *additive* layer over ranking — it never removes a paper. The cosine
score measures textual *similarity* to the query; this pass asks the model to judge actual
*relevance to the research goal*, which catches papers that share vocabulary but miss the
point (e.g. a 1980s neuroscience paper surfacing under an LLM-hallucination query). The
tiers drive two things downstream: a grouped "Relevance Tiers" section in the output, and
the order synthesis fills its budget in (high-tier papers first).

Recall-safe by design: when the model is unsure between two tiers it is told to pick the
HIGHER one. A paper the model fails to classify altogether falls to the weak "tangential"
tier — it shouldn't inherit a strong label it was never given — but nothing is ever dropped,
so even tangential papers stay in the list (just last in the synthesis fill order).
"""
import json
import re
import sys

import ollama

from config import DEFAULT_MODEL, OLLAMA_API_KEY, OLLAMA_HOST
from llm import ChatError, make_client
from models import RankedRecord

TIER_HIGH = "high"
TIER_MODERATE = "moderate"
TIER_TANGENTIAL = "tangential"
TIER_IRRELEVANT = "irrelevant"

# Priority order for filling the synthesis budget (lower = synthesized first). Untiered
# records (tiering disabled or a failed batch) sort with "moderate" so the order is stable.
_TIER_ORDER = {TIER_HIGH: 0, TIER_MODERATE: 1, TIER_TANGENTIAL: 2, TIER_IRRELEVANT: 3}
_VALID_TIERS = set(_TIER_ORDER)

TIER_LABELS = {
    TIER_HIGH: "Highly relevant",
    TIER_MODERATE: "Moderately relevant",
    TIER_TANGENTIAL: "Tangentially relevant",
    TIER_IRRELEVANT: "Irrelevant",
}
# Stable display/iteration order for the grouped section.
TIER_SEQUENCE = (TIER_HIGH, TIER_MODERATE, TIER_TANGENTIAL, TIER_IRRELEVANT)


def prioritize_by_tier(ranked: list[RankedRecord]) -> list[RankedRecord]:
    """Return a new list ordered high → moderate → tangential, stable within each tier.

    Used to fill the synthesis budget from the most relevant papers first while preserving
    the original (score) order inside a tier. Untiered records keep their order (treated as
    "moderate"), so this is a safe no-op when tiering didn't run.
    """
    return sorted(ranked, key=lambda r: _TIER_ORDER.get(r.tier, 1))


def _snippet(rec, cap: int = 320) -> str:
    """Title-supporting context for the classifier: abstract, else TLDR, trimmed."""
    text = rec.abstract or rec.tldr or ""
    text = " ".join(text.split())
    return text[:cap] + ("…" if len(text) > cap else "")


_SYSTEM = """\
You are a relevance triage assistant for a literature-review tool. You are given a research
query (and its intent) plus a list of candidate papers, each with a title and a short
abstract snippet. Classify EACH paper's relevance to the research GOAL behind the query into
exactly one of three tiers:

  • "high"        — directly on the query's specific topic/goal; a core paper the researcher
                    would read and cite for this review.
  • "moderate"    — genuinely related and useful for context: an adjacent sub-topic, a
                    component or method of the topic, or the topic applied in a neighbouring
                    area. Worth including, but not central.
  • "tangential"  — shares vocabulary or the broad domain but is not really about this goal:
                    background, a different sense of an ambiguous term, or peripheral.
  • "irrelevant"  — not about this topic at all: a different field or subject that surfaced
                    only through incidental keyword overlap or retrieval noise.

Judge relevance to the GOAL, not mere keyword overlap — a paper can repeat the query's words
and still be tangential or irrelevant, and a paper can be highly relevant without sharing
them. Honour the intent: e.g. for an "implementation" query, purely theoretical or linguistic
papers are usually moderate or tangential, not high. Reserve "irrelevant" for genuine
off-topic noise — papers from the wrong domain — not for on-topic-but-peripheral work.

This only LABELS papers; nothing is removed from the results. So when you are genuinely
unsure between two tiers, choose the HIGHER one.

Return ONLY valid JSON — a list of objects
{"i": <index>, "tier": "high|moderate|tangential|irrelevant"}, one per input paper, using the
exact index given. No prose, no markdown fences.\
"""

_USER_TEMPLATE = """\
Research query: {query}
Intent: {intent}

Papers (classify every one):
{papers_json}
"""


class Tierer:
    """Assigns a relevance tier to each ranked paper via one or more batched LLM calls."""

    _TEMPERATURE = 0.1     # classification is convergent — keep it low for consistency
    _BATCH = 50            # cap items per call so the JSON stays small and well-aligned

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = OLLAMA_HOST,
        api_key: str = OLLAMA_API_KEY,
        provider: str = "ollama",
    ):
        self._client = make_client(provider, host, api_key)
        self._model = model

    def assign(self, query: str, ranked: list[RankedRecord], intent: str = "") -> None:
        """Set ``.tier`` on every record in ``ranked`` (mutates in place).

        Papers are classified in batches; a batch that fails to parse leaves its records at
        the "moderate" default rather than aborting the run (pipeline never crashes).
        """
        for start in range(0, len(ranked), self._BATCH):
            self._assign_batch(query, ranked[start:start + self._BATCH], intent)
        # Belt-and-braces: anything the model skipped gets the WEAK tier. If a paper wasn't
        # worth classifying it shouldn't inherit a strong label — and "tangential" keeps it
        # out of the high-priority synthesis fill while still leaving it visible in the list.
        for r in ranked:
            if r.tier not in _VALID_TIERS:
                r.tier = TIER_TANGENTIAL

    def _assign_batch(self, query: str, batch: list[RankedRecord], intent: str) -> None:
        items = [
            {"i": i, "title": r.record.title, "abstract": _snippet(r.record)}
            for i, r in enumerate(batch)
        ]
        papers_json = json.dumps(items, ensure_ascii=False, indent=1)
        try:
            response = self._client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": _USER_TEMPLATE.format(
                        query=query, intent=intent or "unspecified", papers_json=papers_json,
                    )},
                ],
                options={"temperature": self._TEMPERATURE},
            )
        except (ollama.ResponseError, ChatError) as exc:
            print(f"[tiering] model error: {exc}", file=sys.stderr)
            return
        for idx, tier in self._parse(response.message.content).items():
            if 0 <= idx < len(batch):
                batch[idx].tier = tier

    @staticmethod
    def _parse(raw: str) -> dict[int, str]:
        """Pull {index: tier} from the model reply; ignore anything malformed."""
        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {}
        if isinstance(data, dict):
            # Tolerate a wrapper like {"papers": [...]} or {"results": [...]}.
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break
        if not isinstance(data, list):
            return {}
        out: dict[int, str] = {}
        for obj in data:
            if not isinstance(obj, dict):
                continue
            try:
                idx = int(obj["i"])
            except (KeyError, TypeError, ValueError):
                continue
            tier = str(obj.get("tier", "")).strip().lower()
            if tier in _VALID_TIERS:
                out[idx] = tier
        return out
