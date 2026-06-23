import math
import re
from collections import Counter
from models import CSLRecord, RankedRecord


# How strongly citation impact may amplify a paper's relevance. The boost is
# *multiplicative* — final = relevance * (1 + β·impact) — so citations only ever
# reorder papers that already match the query; a highly-cited but off-topic paper
# still scores ~0 because its relevance gates it. β=1.0 lets the most-cited paper in
# the pool roughly double its score over an equally-relevant but uncited peer.
_IMPACT_WEIGHT = 1.0


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


def _impact(citations: int | None, max_log: float) -> float:
    """Citation count → [0, 1] impact, log-compressed and normalised to the pool.

    Citations are heavily skewed (a few seminal papers, a long tail on 0–5), so we
    take log1p before normalising — otherwise one 2000-cite paper would flatten the
    whole field to ~0. The pool's most-cited paper defines the top of the scale.
    Returns 0 when the candidate pool carries no citation data at all.
    """
    if max_log <= 0:
        return 0.0
    return math.log1p(citations or 0) / max_log


def rank(
    records: list[CSLRecord],
    refined_query: str,
    keywords: list[str],
    original_query: str = "",
) -> list[RankedRecord]:
    """Rank papers by lexical relevance, then amplify by citation impact.

    relevance   = 0.8 * title_cosine + 0.2 * abstract_cosine  (title-only if no abstract)
    final_score = relevance * (1 + β * impact)

    Query vector = refined query tokens + extracted keywords + original query tokens.
    The original query restores intent words ("taught", "universities") that are
    stripped from the bibliographic search string but still matter for ranking.

    The impact term breaks the ties that pure title cosine leaves all over a pool of
    near-identically-titled papers, pulling the field's seminal work up and burying
    uncited minor notes — without ever surfacing a famous paper that doesn't match.
    """
    query_tokens = _tokenize(refined_query) + [kw.lower() for kw in keywords]
    if original_query:
        query_tokens += _tokenize(original_query)
    q_tf = Counter(query_tokens)

    # Pool-relative citation scale: the most-cited candidate sets impact = 1.0.
    max_log = max((math.log1p(r.cited_by_count or 0) for r in records), default=0.0)

    ranked: list[RankedRecord] = []
    for rec in records:
        title_score = _cosine(q_tf, Counter(_tokenize(rec.title)))

        if rec.abstract and not rec.metadata_missingness.abstract_missing:
            abstract_score = _cosine(q_tf, Counter(_tokenize(rec.abstract)))
            relevance = 0.8 * title_score + 0.2 * abstract_score
        else:
            abstract_score = 0.0
            relevance = title_score

        final_score = relevance * (1 + _IMPACT_WEIGHT * _impact(rec.cited_by_count, max_log))

        ranked.append(RankedRecord(
            record=rec,
            title_score=title_score,
            abstract_score=abstract_score,
            final_score=final_score,
        ))

    ranked.sort(key=lambda r: r.final_score, reverse=True)
    return ranked
