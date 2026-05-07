from src.retrieval.sparse import BM25Retriever
from src.indexing.embedder import embed_query
from src.indexing.pinecone_store import query_index

# Singleton — loaded once at app startup, then `.build()` or `.load()` called
bm25 = BM25Retriever()


def reciprocal_rank_fusion(
    dense: list[dict],
    sparse: list[dict],
    k: int = 60,
) -> list[dict]:
    """Fuse dense + sparse results using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    all_items: dict[str, dict] = {}

    for rank, item in enumerate(dense):
        key = f"{item['file_path']}::{item['name']}::{item['start_line']}"
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        all_items[key] = item

    for rank, item in enumerate(sparse):
        key = f"{item['file_path']}::{item['name']}::{item['start_line']}"
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        all_items[key] = item

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [all_items[k] for k in sorted_keys]


def hybrid_search(
    query: str,
    top_k: int = 10,
    language_filter: str | None = None,
) -> list[dict]:
    """Dense (Pinecone) + sparse (BM25) search fused via RRF."""
    # Dense via Pinecone
    qvec = embed_query(query)
    pinecone_filter = (
        {"language": {"$eq": language_filter}} if language_filter else None
    )
    dense_results = query_index(qvec, top_k=top_k * 2, filter=pinecone_filter)

    # Sparse via BM25 (falls back gracefully if index not built)
    sparse_results = bm25.search(query, top_k=top_k * 2)

    fused = reciprocal_rank_fusion(dense_results, sparse_results)
    return fused[:top_k]