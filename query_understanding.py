import json
import re
import ollama
from config import OLLAMA_API_KEY, OLLAMA_HOST, DEFAULT_MODEL
from models import StructuredQuery


_SYSTEM = """\
You are a query analysis assistant for an academic literature search tool.
Researchers often phrase questions in everyday, conversational language, but
scholarly papers use specific field terminology. Your job is to translate the
user's question into the terminology the academic literature actually uses, so a
bibliographic search engine can match the right papers.

Map lay phrasing to the established academic term for the SAME concept, e.g.:
  "making a computer understand what I type"      -> "natural language understanding"
  "how well someone grasps an algorithm"          -> "program comprehension"
  "teaching kids to code"                          -> "computing education / CS education"
  "spotting fake reviews online"                   -> "fake review detection / opinion spam"

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
  (e.g. "how well someone grasps an algorithm" -> "program comprehension", NOT
  "metrics for program comprehension".)
- "keywords": list of the canonical technical/field terms for the topic (the
  established academic terms — not just words copied verbatim from the query).
  Related sub-topics are fine here; keep them out of refined_query.
- "intent": one of "survey", "implementation", "theory", or null (null if ambiguous).

Query: {query}
"""


def _make_client(host: str, api_key: str) -> ollama.Client:
    kwargs: dict = {"host": host}
    if api_key:
        kwargs["headers"] = {"Authorization": f"Bearer {api_key}"}
    return ollama.Client(**kwargs)


class QueryUnderstanding:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = OLLAMA_HOST,
        api_key: str = OLLAMA_API_KEY,
    ):
        self._client = _make_client(host, api_key)
        self._model = model

    def transform(self, query: str) -> StructuredQuery:
        response = self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _USER_TEMPLATE.format(query=query)},
            ],
            options={"temperature": 0.0},
        )
        raw = response.message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
        data = json.loads(raw)
        return StructuredQuery(**data)
