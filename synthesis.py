import json
import sys

import ollama

from config import OLLAMA_API_KEY, OLLAMA_HOST, DEFAULT_MODEL
from llm import ChatError, make_client
from models import RankedRecord


_SYSTEM = """\
You are a research assistant writing a literature review.
You receive a set of ranked academic papers (JSON) and a research query.
Synthesize a clear, structured literature review based ONLY on the provided papers.
Do not invent papers, authors, findings, or claims not present in the input.
Cite papers inline as [Author et al., Year] or [Author, Year]. When a paper's year
is "n.d." (no date), cite it exactly as [Author, n.d.] — never write the word "None".\
"""

_USER_TEMPLATE = """\
Research Query: {query}
{intent_line}\
Top-{k} Ranked Papers:
{papers_json}

Write a concise literature review (3–5 paragraphs) covering key themes, methods, and findings \
relevant to the query. Group related work thematically, not chronologically.\
"""

# ── Synthesis input bounds ─────────────────────────────────────────────────── #
# A literature review summarises a focused set of papers in a few paragraphs;
# feeding hundreds is both unsynthesizable and blows past the model context
# window. We cap the count and trim each paper to the fields synthesis needs,
# then enforce a hard token budget as a final safety net.
_MAX_SYNTH_PAPERS = 50
_ABSTRACT_CHAR_CAP = 1200          # truncate long abstracts in the payload
_CHARS_PER_TOKEN = 4               # rough estimate for budgeting
_PAYLOAD_TOKEN_BUDGET = 200_000    # leave headroom under a 262k context window


def _paper_payload(rec) -> dict:
    """Compact, synthesis-relevant view of a record (drops ids/flags/urls)."""
    abstract = rec.abstract
    if abstract and len(abstract) > _ABSTRACT_CHAR_CAP:
        abstract = abstract[:_ABSTRACT_CHAR_CAP].rstrip() + "…"
    return {
        "title": rec.title,
        "authors": [f"{a.given} {a.family}".strip() for a in rec.author] or ["Anon."],
        "year": rec.issued or "n.d.",
        "venue": rec.container_title,
        "abstract": abstract,
        "tldr": rec.tldr,
        "DOI": rec.DOI,
    }


def select_for_synthesis(ranked: list[RankedRecord], top_k: int) -> list[RankedRecord]:
    """Pick the ranked slice that will actually be synthesized.

    Applies, in order: the user's top_k, an absolute paper cap, and a token
    budget so the assembled prompt can never exceed the model context window.
    """
    candidates = ranked[: min(top_k, _MAX_SYNTH_PAPERS)]
    selected: list[RankedRecord] = []
    chars = 0
    for r in candidates:
        size = len(json.dumps(_paper_payload(r.record)))
        if selected and (chars + size) / _CHARS_PER_TOKEN > _PAYLOAD_TOKEN_BUDGET:
            break
        selected.append(r)
        chars += size
    return selected


class Synthesizer:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = OLLAMA_HOST,
        api_key: str = OLLAMA_API_KEY,
        provider: str = "ollama",
    ):
        self._client = make_client(provider, host, api_key)
        self._model = model

    def synthesize(self, query: str, papers: list[RankedRecord], intent: str = "") -> str:
        """Generate a review over a pre-selected, budget-bounded set of papers."""
        if not papers:
            return ""
        papers_json = json.dumps([_paper_payload(r.record) for r in papers], indent=2)
        intent_line = f"Research Intent: {intent}\n" if intent else ""
        try:
            response = self._client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": _USER_TEMPLATE.format(
                        query=query,
                        intent_line=intent_line,
                        k=len(papers),
                        papers_json=papers_json,
                    )},
                ],
                options={"temperature": 0.3},
            )
        except (ollama.ResponseError, ChatError) as exc:
            print(f"[synthesis] model error: {exc}", file=sys.stderr)
            return ""
        return response.message.content.strip()
