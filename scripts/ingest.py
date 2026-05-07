"""
Ingestion CLI — clone a GitHub repo and index it into Pinecone.

Usage:
    uv run python scripts/ingest.py --repo https://github.com/tiangolo/fastapi
    uv run python scripts/ingest.py --repo https://github.com/org/repo --repo-name my-alias
"""
import sys
import argparse
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel

from src.ingestion.repo_loader import clone_repo, walk_repo
from src.ingestion.ast_parser import parse_file
from src.ingestion.chunker import chunk_files
from src.indexing.embedder import embed_chunks_batched
from src.indexing.pinecone_store import upsert_chunks

console = Console()


def ingest(repo_url: str, repo_name_override: str | None = None) -> None:
    console.print(Panel("[bold cyan]🔍 Kyoto — Ingestion Pipeline[/bold cyan]", expand=False))

    # ── Step 1: Clone ────────────────────────────────────────────────────────
    console.rule("[bold]Step 1 / 5  Clone[/bold]")
    repo_path, repo_name = clone_repo(repo_url)
    if repo_name_override:
        repo_name = repo_name_override

    try:
        # ── Step 2: Walk ─────────────────────────────────────────────────────
        console.rule("[bold]Step 2 / 5  Walk[/bold]")
        files = walk_repo(repo_path, repo_name)
        if not files:
            console.print("[red]No supported files found. Exiting.[/red]")
            return

        # ── Step 3: Parse ────────────────────────────────────────────────────
        console.rule("[bold]Step 3 / 5  Parse (AST)[/bold]")
        all_chunks = []
        failed = 0
        for f in files:
            chunks = parse_file(f)
            if chunks:
                all_chunks.extend(chunks)
            else:
                failed += 1

        console.print(
            f"[green]✓ {len(all_chunks)} chunks from {len(files)} files "
            f"[dim]({failed} files skipped)[/dim][/green]"
        )

        if not all_chunks:
            console.print("[red]No chunks produced. Exiting.[/red]")
            return

        # Token-budget split: oversized chunks → overlapping sub-chunks
        all_chunks = chunk_files(all_chunks)
        console.print(f"[green]✓ {len(all_chunks)} chunks after token-budget splitting[/green]")

        # ── Step 4: Embed ────────────────────────────────────────────────────
        console.rule("[bold]Step 4 / 5  Embed (Voyage AI)[/bold]")
        console.print(
            f"[cyan]Embedding {len(all_chunks)} chunks — this may take a few minutes...[/cyan]"
        )
        texts = [c.code for c in all_chunks]
        embeddings = embed_chunks_batched(texts)

        # ── Step 5: Upsert ───────────────────────────────────────────────────
        console.rule("[bold]Step 5 / 5  Upsert (Pinecone)[/bold]")
        upsert_chunks(all_chunks, embeddings)

    finally:
        # Always clean up the temp clone
        shutil.rmtree(repo_path, ignore_errors=True)
        console.print(f"[dim]Cleaned up temp dir {repo_path}[/dim]")

    console.print(Panel("[bold green]✅ Ingestion Complete[/bold green]", expand=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index a GitHub repo into Kyoto/Pinecone")
    parser.add_argument("--repo", required=True, help="GitHub URL to clone and index")
    parser.add_argument(
        "--repo-name",
        default=None,
        help="Override the repo name stored in metadata (default: inferred from URL)",
    )
    args = parser.parse_args()
    ingest(args.repo, args.repo_name)