"""
Celery tasks for async ingestion.

Dispatch from anywhere with:
    from src.worker.tasks import ingest_repo
    result = ingest_repo.delay("https://github.com/org/repo", "my-repo")
"""
import shutil
from src.worker.celery_app import celery_app
from src.ingestion.repo_loader import clone_repo, walk_repo
from src.ingestion.ast_parser import parse_file
from src.ingestion.chunker import chunk_files
from src.indexing.embedder import embed_chunks_batched
from src.indexing.pinecone_store import upsert_chunks


@celery_app.task(bind=True, name="src.worker.tasks.ingest_repo", max_retries=2)
def ingest_repo(self, repo_url: str, repo_name_override: str | None = None) -> dict:
    """
    Full ingestion pipeline as a background task.
    Returns a summary dict stored in the Celery result backend.
    """
    repo_path = None
    try:
        repo_path, repo_name = clone_repo(repo_url)
        if repo_name_override:
            repo_name = repo_name_override

        files = walk_repo(repo_path, repo_name)
        if not files:
            return {"status": "empty", "repo": repo_name, "chunks": 0}

        all_chunks = []
        for f in files:
            all_chunks.extend(parse_file(f))

        all_chunks = chunk_files(all_chunks)
        embeddings = embed_chunks_batched([c.code for c in all_chunks])
        upsert_chunks(all_chunks, embeddings)

        return {
            "status": "ok",
            "repo": repo_name,
            "files": len(files),
            "chunks": len(all_chunks),
        }

    except Exception as exc:
        raise self.retry(exc=exc, countdown=30)

    finally:
        if repo_path:
            shutil.rmtree(repo_path, ignore_errors=True)