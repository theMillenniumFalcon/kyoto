from contextlib import asynccontextmanager

from fastapi import FastAPI
from rich.console import Console

from src.api.routes import query, debug

console = Console()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up the cross-encoder at startup so the first request isn't slow
    console.print("[cyan]Loading cross-encoder re-ranker...[/cyan]")
    from src.retrieval.reranker import _get_model
    _get_model()
    console.print("[green]✓ Re-ranker ready[/green]")
    yield


app = FastAPI(
    title="Kyoto — Codebase RAG",
    description="Ask questions about your codebase. Hybrid retrieval + Claude.",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(query.router, prefix="/api/v1", tags=["Query"])
app.include_router(debug.router, prefix="/api/v1", tags=["Debug"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "0.2.0"}