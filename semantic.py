"""Semantic ranking signal: dense sentence embeddings via Sentence-BERT (SBERT).

SBERT runs fully in-process — ``model.encode()`` turns titles/abstracts into dense
vectors whose cosine similarity captures *meaning*. That is what separates "program
comprehension" (software) from "reading comprehension program" (education): token
overlap rates them alike, dense vectors place them far apart. The lexical ranker
(``ranking.py``) can't do this on its own; this module is the upgrade.

Everything degrades gracefully. If ``sentence-transformers`` isn't installed or the
model can't load/encode, every entry point returns ``None`` and the caller falls back
to lexical ranking. Document vectors are cached on disk by DOI (keyed per model), so
re-runs and overlapping queries skip re-encoding; the query vector is always fresh.

Acceleration: if a PyTorch build exposing an Intel XPU (Arc iGPU) or CUDA device is
present, encoding uses it automatically; otherwise CPU. The batch is small and cached,
so CPU is usually plenty.
"""
import json
import sys
from pathlib import Path

MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, fast, strong on short-text semantic search
_CACHE_PATH = Path(__file__).resolve().parent / "embedding_cache.json"
_ABSTRACT_CHARS = 1000            # cap abstract length fed to the encoder, for speed

_model = None  # lazily loaded SentenceTransformer; set to False once known-unavailable


def _pick_device() -> str:
    """Prefer an available accelerator (Intel XPU / CUDA), else CPU. Never raises."""
    try:
        import torch
        if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
            return "xpu"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _get_model():
    """Load the SBERT model once. Returns the model, or None if SBERT is unavailable."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            device = _pick_device()
            _model = SentenceTransformer(MODEL_NAME, device=device)
            print(f"[semantic] SBERT {MODEL_NAME} on {device}", file=sys.stderr)
        except Exception as exc:
            print(f"[semantic] embeddings unavailable ({str(exc)[:90]}); using lexical "
                  f"ranking. `pip install sentence-transformers` to enable.", file=sys.stderr)
            _model = False
    return _model or None


def _encode(texts: list[str]) -> list[list[float]] | None:
    """Encode texts to unit-norm vectors. None if SBERT/encoding is unavailable."""
    model = _get_model()
    if model is None:
        return None
    if not texts:
        return []
    try:
        return model.encode(texts, normalize_embeddings=True, batch_size=64).tolist()
    except Exception as exc:
        print(f"[semantic] encode failed ({str(exc)[:90]}); using lexical ranking",
              file=sys.stderr)
        return None


def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        try:
            return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    try:
        _CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    except OSError:
        pass


def _doc_text(rec) -> str:
    """The text we embed for a paper: title, plus abstract when we actually have one."""
    text = rec.title or ""
    if rec.abstract and not rec.metadata_missingness.abstract_missing:
        text = f"{text}. {rec.abstract[:_ABSTRACT_CHARS]}"
    return text


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def semantic_scores(query_text: str, records: list) -> dict[str, float] | None:
    """Map each ``record.id`` to cosine(query, title+abstract), or None if unavailable.

    Cached doc vectors are reused; only cache misses are encoded. Any encode failure
    returns None so the caller cleanly falls back to deterministic lexical ranking.
    """
    cache = _load_cache()
    key = lambda rec: f"{MODEL_NAME}:{rec.DOI or rec.id}"

    misses = [r for r in records if key(r) not in cache]
    if misses:
        vectors = _encode([_doc_text(r) for r in misses])
        if vectors is None:
            return None
        for rec, vec in zip(misses, vectors):
            cache[key(rec)] = vec
        _save_cache(cache)

    query_vec = _encode([query_text])
    if not query_vec:
        return None

    qv = query_vec[0]
    return {r.id: _dot(qv, cache[key(r)]) for r in records if key(r) in cache}
