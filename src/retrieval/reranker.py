from functools import lru_cache

from sentence_transformers import CrossEncoder


@lru_cache(maxsize=1)
def _get_model() -> CrossEncoder:
    """Lazy-load the cross-encoder — downloaded once, cached in memory."""
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Re-rank candidates using a cross-encoder. Truncates code to 512 chars."""
    if not candidates:
        return []
    model = _get_model()
    pairs = [(query, c.get("code", "")[:512]) for c in candidates]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [item for _, item in ranked[:top_k]]