import tempfile
from pathlib import Path

from git import Repo
from rich.console import Console

console = Console()

SUPPORTED = {".py", ".js", ".ts", ".jsx", ".tsx"}
SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".venv",
    "dist", "build", ".next", "coverage", ".pytest_cache",
}


def clone_repo(url: str) -> tuple[str, str]:
    """Clone repo to a temp dir. Returns (temp_dir, repo_name)."""
    repo_name = url.rstrip("/").split("/")[-1].replace(".git", "")
    tmp = tempfile.mkdtemp(prefix=f"rag_{repo_name}_")
    console.print(f"[cyan]Cloning {url}...[/cyan]")
    Repo.clone_from(url, tmp, depth=1)  # shallow clone = faster
    console.print(f"[green]✓ Cloned to {tmp}[/green]")
    return tmp, repo_name


def walk_repo(repo_path: str, repo_name: str) -> list[dict]:
    """Walk repo and return list of file dicts."""
    files = []
    root = Path(repo_path)

    for path in root.rglob("*"):
        if any(skip in path.parts for skip in SKIP_DIRS):
            continue
        if path.suffix not in SUPPORTED:
            continue
        if not path.is_file():
            continue

        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if len(content.strip()) < 50:  # skip near-empty files
            continue

        relative = str(path.relative_to(root))
        lang = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
        }.get(path.suffix, "unknown")

        files.append({
            "path": relative,
            "content": content,
            "language": lang,
            "repo": repo_name,
            "size_lines": len(content.splitlines()),
        })

    console.print(f"[green]✓ {len(files)} files found[/green]")
    return files