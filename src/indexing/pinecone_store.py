import hashlib

from pinecone import Pinecone, ServerlessSpec
from rich.console import Console

from src.config.settings import settings
from src.ingestion.ast_parser import CodeChunk

console = Console()

VECTOR_DIM = 1536  # voyage-code-2 output dimension

_pc = Pinecone(api_key=settings.pinecone_api_key)


def get_or_create_index():
    names = _pc.list_indexes().names()
    if settings.pinecone_index_name not in names:
        console.print(
            f"[cyan]Creating Pinecone index '{settings.pinecone_index_name}'...[/cyan]"
        )
        _pc.create_index(
            name=settings.pinecone_index_name,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=settings.pinecone_environment,
            ),
        )
        console.print("[green]✓ Index created[/green]")
    return _pc.Index(settings.pinecone_index_name)


def _chunk_to_id(chunk: CodeChunk) -> str:
    """Stable, collision-resistant ID — same chunk always maps to same ID."""
    key = f"{chunk.repo}::{chunk.file_path}::{chunk.name}::{chunk.start_line}"
    return hashlib.md5(key.encode()).hexdigest()


def upsert_chunks(chunks: list[CodeChunk], embeddings: list[list[float]]) -> None:
    index = get_or_create_index()

    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vectors.append(
            {
                "id": _chunk_to_id(chunk),
                "values": embedding,
                "metadata": {
                    "name":       chunk.name,
                    "kind":       chunk.kind,
                    "language":   chunk.language,
                    "file_path":  chunk.file_path,
                    "repo":       chunk.repo,
                    "start_line": chunk.start_line,
                    "end_line":   chunk.end_line,
                    # Pinecone metadata cap ~40KB total per vector — truncate large chunks
                    "code":       chunk.code[:4000],
                    "docstring":  chunk.docstring[:500],
                },
            }
        )

    batch_size = 100  # Pinecone recommended batch size
    total = len(vectors)
    for i in range(0, total, batch_size):
        index.upsert(vectors=vectors[i : i + batch_size])
        console.print(
            f"  [dim]Upserted {min(i + batch_size, total)}/{total} vectors[/dim]"
        )

    console.print(f"[green]✅ Indexed {total} chunks into '{settings.pinecone_index_name}'[/green]")


def query_index(
    embedding: list[float],
    top_k: int = 10,
    filter: dict | None = None,
) -> list[dict]:
    index = get_or_create_index()
    results = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter,  # e.g. {"language": {"$eq": "python"}}
    )
    return [
        {**match.metadata, "score": match.score, "id": match.id}
        for match in results.matches
    ]