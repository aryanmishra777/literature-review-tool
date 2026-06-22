import math
import re
from collections import Counter
from models import CSLRecord, RankedRecord


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _cosine(q: Counter, d: Counter) -> float:
    if not q or not d:
        return 0.0
    dot = sum(q[t] * d[t] for t in q)
    norm_q = math.sqrt(sum(v * v for v in q.values()))
    norm_d = math.sqrt(sum(v * v for v in d.values()))
    if norm_q == 0 or norm_d == 0:
        return 0.0
    return dot / (norm_q * norm_d)


def rank(
    records: list[CSLRecord],
    refined_query: str,
    keywords: list[str],
    original_query: str = "",
) -> list[RankedRecord]:
    """Late-fusion cosine ranking: 0.8 * title_score + 0.2 * abstract_score.

    Query vector = refined query tokens + extracted keywords + original query tokens.
    The original query restores intent words ("taught", "universities") that are
    stripped from the bibliographic search string but still matter for ranking.
    Abstract score is only included when the abstract is actually present.
    """
    query_tokens = _tokenize(refined_query) + [kw.lower() for kw in keywords]
    if original_query:
        query_tokens += _tokenize(original_query)
    q_tf = Counter(query_tokens)

    ranked: list[RankedRecord] = []
    for rec in records:
        title_score = _cosine(q_tf, Counter(_tokenize(rec.title)))

        if rec.abstract and not rec.metadata_missingness.abstract_missing:
            abstract_score = _cosine(q_tf, Counter(_tokenize(rec.abstract)))
            final_score = 0.8 * title_score + 0.2 * abstract_score
        else:
            abstract_score = 0.0
            final_score = title_score

        ranked.append(RankedRecord(
            record=rec,
            title_score=title_score,
            abstract_score=abstract_score,
            final_score=final_score,
        ))

    ranked.sort(key=lambda r: r.final_score, reverse=True)
    return ranked
