import json
import re
from config import OLLAMA_API_KEY, OLLAMA_HOST, DEFAULT_MODEL
from llm import make_client
from models import StructuredQuery

# Enumerating the possible *senses* of a query is a divergent/brainstorming task: greedy
# (temp 0) decoding commits to the single most-salient framing and under-lists alternatives,
# so a missed sense never becomes a contrastive negative. A modest temperature gives the
# enumeration room to surface more senses. Kept low (not high) to protect the convergent
# fields in the same call — refined_query/keywords feed a search engine and want precision.
_TEMPERATURE = 0.3


_SYSTEM = """\
You are a query analysis assistant for an academic literature search tool.
Researchers often phrase questions in everyday, conversational language, but
scholarly papers use specific field terminology. Your job is to translate the
user's question into the terminology the academic literature actually uses, so a
bibliographic search engine can match the right papers.

Map the user's lay phrasing to the established academic term for the SAME concept.
Choose the term that most faithfully names what the user is actually asking about;
do NOT substitute a different but adjacent field just because it is well-known.

You produce TWO search phrases at different breadths, because retrieval runs both:
  • a BROAD topic phrase for recall (the bare canonical term), and
  • a FOCUSED phrase for precision that keeps the SPECIFIC angle the user asked about.
When the question targets one aspect of a topic (how to assess it, one method within it,
one sub-population), the focused phrase must preserve that aspect, not collapse to the
bare topic.

Stay faithful to the user's intent. Do NOT drift to unrelated concepts or invent
topics the user did not ask about. If the topic has no standard academic term,
keep the user's clearest phrasing.
Return ONLY valid JSON — no explanation, no markdown fences.\
"""

_USER_TEMPLATE = """\
Analyze this research query and return a JSON object with exactly these fields:
- "refined_query": a concise phrase (about 2-6 words) NAMING THE TOPIC in the
  field's standard terminology. Name the subject itself; do NOT add methodological
  framing words such as "metrics", "evaluation", "framework", "approach", "analysis",
  or "methods" unless the user explicitly asked for them. This string is sent
  directly to the search engine, so prefer canonical terms over the user's literal words.
- "focus_query": a more specific phrase capturing the user's PARTICULAR angle on the
  topic. Combine the canonical term with the specific aspect they asked about, and HERE
  you DO keep the qualifier words refined_query drops. Set this to null
  when the query is a plain topic with no narrowing aspect (then refined_query alone is used).
- "keywords": list of the canonical technical/field terms for the topic (the
  established academic terms — not just words copied verbatim from the query).
  Related sub-topics are fine here; keep them out of refined_query.
- "intent": one of "survey", "implementation", "theory", or null (null if ambiguous).
- "interpretations": a list of DISTINCT research senses the query could mean — senses that
  would retrieve DIFFERENT BODIES OF LITERATURE from a bibliographic search. Populate this
  ONLY when the query is genuinely ambiguous; return an EMPTY list [] when it has a single
  clear reading. Do NOT pad with near-synonyms or sub-aspects of one topic — those are the
  same sense, not separate interpretations. But DO list EVERY plausible distinct sense
  (commonly two to four), not just the two most obvious — completeness matters, because the
  senses you DON'T pick are used to steer the search AWAY from their literature.
  For a computing/technical term, work through whether each of these distinct senses applies,
  and list every one that genuinely does:
    • COMPUTATIONAL / formal — the thing as a mathematical or computer-science object;
    • LEARNING / cognitive — a learner's comprehension of that object;
    • SOFTWARE-ENGINEERING — practitioners working with it (program comprehension, debugging);
    • SOCIOTECHNICAL / critical — the thing as a system embedded in society (algorithmic
      management, governance, bias, fairness, media, policy, the arts).
  Even when the user clearly wants one of these, list the OTHERS that plausibly collide in
  search results as separate interpretations, so retrieval can be pushed away from them.
  Each entry is an object with:
    - "label": a short description of that sense and the literature it targets.
    - "refined_query": the canonical search phrase for THAT sense. It MUST be discriminating —
      a phrase that retrieves that sense and NOT the others. Center it on the ACTUAL SUBJECT
      the user is asking about (e.g. "students' comprehension of algorithms"); do NOT use a
      vague locative qualifier like "algorithmic understanding in education", which a search
      engine reads as "algorithmic systems used IN education" and pulls the wrong literature.
      Do NOT reuse the bare ambiguous term for any entry, and keep the entries clearly distinct.
    - "focus_query": the focused phrase for that sense, or null.
    - "keywords": the field terms SPECIFIC to that sense, distinct from the other entries'.
  The FIRST entry must be your single best guess and MUST match the top-level
  refined_query / focus_query / keywords above (so the top-level fields are themselves
  discriminating, never the bare ambiguous term).

Query: {query}
"""


class QueryUnderstanding:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = OLLAMA_HOST,
        api_key: str = OLLAMA_API_KEY,
        provider: str = "ollama",
    ):
        self._client = make_client(provider, host, api_key)
        self._model = model

    def transform(self, query: str) -> StructuredQuery:
        response = self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _USER_TEMPLATE.format(query=query)},
            ],
            options={"temperature": _TEMPERATURE},
        )
        raw = response.message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
        data = json.loads(raw)
        return StructuredQuery(**data)
