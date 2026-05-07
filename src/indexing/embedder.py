import voyageai
from tenacity import retry, stop_after_attempt, wait_exponential
from rich.console import Console

from src.config.settings import settings

console = Console()

_client = voyageai.Client(api_key=settings.voyage_api_key)

MAX_TOKENS_PER_CHUNK = 16_000  # voyage-code-2 context window


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _embed_batch_raw(texts: list[str], input_type: str) -> list[list[float]]:
    response = _client.embed(texts, model=settings.voyage_model, input_type=input_type)
    return response.embeddings


def embed_chunks_batched(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Embed documents in batches — use at index time."""
    all_embeddings: list[list[float]] = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        embeddings = _embed_batch_raw(batch, input_type="document")
        all_embeddings.extend(embeddings)
        console.print(
            f"  [dim]Embedded {min(i + batch_size, total)}/{total} chunks[/dim]"
        )
    return all_embeddings


def embed_query(query: str) -> list[float]:
    """Embed a single query — uses input_type='query' for better retrieval.

    This is intentionally different from embed_chunks_batched.
    voyage-code-2 is asymmetric: the document/query distinction matters.
    """
    response = _client.embed(
        [query],
        model=settings.voyage_model,
        input_type="query",
    )
    return response.embeddings[0]