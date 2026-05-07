"""
Incremental re-indexer — only re-indexes files changed since a given git ref.

Usage:
    uv run python scripts/reindex_changed.py --repo ./path/to/repo --since HEAD~1 --name fastapi
    uv run python scripts/reindex_changed.py --repo ./fastapi --since abc1234 --name fastapi
"""
import subprocess
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

from src.ingestion.ast_parser import parse_file
from src.indexing.embedder import embed_chunks_batched
from src.indexing.pinecone_store import upsert_chunks, get_or_create_index, _chunk_to_id

console = Console()

SUPPORTED = {".py", ".js", ".ts", ".tsx", ".jsx"}

LANG_MAP = {
    ".py":  "python",
    ".js":  "javascript",
    ".jsx": "javascript",
    ".ts":  "typescript",
    ".tsx": "typescript",
}


def get_changed_files(repo_path: str, since: str) -> list[str]:
    """Return relative paths of files changed since `since` ref."""
    result = subprocess.run(
        ["git", "diff", "--name-only", since],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        console.print(f"[red]git diff failed: {result.stderr}[/red]")
        return []
    return [
        f for f in result.stdout.strip().splitlines()
        if Path(f).suffix in SUPPORTED
    ]


def delete_old_chunks(repo_path: str, repo_name: str, rel_paths: list[str]) -> None:
    """
    Delete previously indexed chunks for files that no longer exist
    (renamed/deleted). Chunks for existing files are overwritten by upsert.
    """
    index = get_or_create_index()
    deleted = 0
    for rel_path in rel_paths:
        full = Path(repo_path) / rel_path
        if not full.exists():
            # File was deleted — remove all its vectors by metadata filter
            try:
                index.delete(filter={"file_path": {"$eq": rel_path}, "repo": {"$eq": repo_name}})
                console.print(f"  [dim]Deleted vectors for removed file: {rel_path}[/dim]")
                deleted += 1
            except Exception as e:
                console.print(f"  [yellow]⚠ Could not delete {rel_path}: {e}[/yellow]")
    if deleted:
        console.print(f"[green]✓ Deleted vectors for {deleted} removed files[/green]")


def reindex(repo_path: str, since: str, repo_name: str) -> None:
    console.print(f"[cyan]Checking changed files since {since!r} in {repo_path!r}...[/cyan]")
    changed = get_changed_files(repo_path, since)

    if not changed:
        console.print("[green]No supported files changed — nothing to re-index.[/green]")
        return

    console.print(f"[cyan]{len(changed)} changed file(s):[/cyan]")
    for f in changed:
        console.print(f"  [dim]{f}[/dim]")

    # Delete vectors for any removed files first
    delete_old_chunks(repo_path, repo_name, changed)

    # Parse changed files that still exist
    all_chunks = []
    skipped = 0
    for rel_path in changed:
        full = Path(repo_path) / rel_path
        if not full.exists():
            skipped += 1
            continue

        try:
            content = full.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            console.print(f"  [yellow]⚠ Could not read {rel_path}: {e}[/yellow]")
            continue

        file_info = {
            "path": rel_path,
            "content": content,
            "language": LANG_MAP.get(full.suffix, "unknown"),
            "repo": repo_name,
            "size_lines": len(content.splitlines()),
        }
        chunks = parse_file(file_info)
        if chunks:
            all_chunks.extend(chunks)
            console.print(f"  [dim]Parsed {len(chunks)} chunks from {rel_path}[/dim]")

    if not all_chunks:
        console.print("[yellow]No chunks produced from changed files.[/yellow]")
        return

    console.print(f"\n[cyan]Embedding {len(all_chunks)} chunks...[/cyan]")
    embeddings = embed_chunks_batched([c.code for c in all_chunks])

    console.print("[cyan]Upserting to Pinecone (idempotent via stable IDs)...[/cyan]")
    upsert_chunks(all_chunks, embeddings)

    console.print(
        f"\n[bold green]✅ Re-indexed {len(all_chunks)} chunks from "
        f"{len(changed) - skipped} files ({skipped} deleted/skipped)[/bold green]"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Incrementally re-index changed files")
    parser.add_argument("--repo", required=True, help="Path to local git repo")
    parser.add_argument(
        "--since",
        default="HEAD~1",
        help="Git ref to diff against (default: HEAD~1)",
    )
    parser.add_argument(
        "--name",
        default="myrepo",
        help="Repo name as stored in Pinecone metadata (must match original ingest)",
    )
    args = parser.parse_args()
    reindex(args.repo, args.since, args.name)