"""
Phase 1 smoke test — verifies all three API clients initialise correctly.
Run with: uv run python scripts/smoke_test.py
"""

import sys
from rich.console import Console
from rich.panel import Panel

console = Console()


def test_anthropic() -> None:
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=32,
        messages=[{"role": "user", "content": "say hi in exactly two words"}],
    )
    reply = msg.content[0].text.strip()
    console.print(f"[green]✅ Anthropic[/green]  → [dim]{reply}[/dim]")


def test_voyage() -> None:
    import voyageai

    vc = voyageai.Client()
    result = vc.embed(["def hello(): pass"], model="voyage-code-2")
    dim = len(result.embeddings[0])
    assert dim == 1536, f"Expected 1536 dims, got {dim}"
    console.print(f"[green]✅ Voyage AI[/green]  → embedding dim = {dim}")


def test_pinecone() -> None:
    from pinecone import Pinecone

    pc = Pinecone()
    indexes = pc.list_indexes().names()
    console.print(f"[green]✅ Pinecone[/green]   → indexes = {list(indexes)}")


def test_tree_sitter() -> None:
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser

    PY_LANGUAGE = Language(tspython.language())
    parser = Parser(PY_LANGUAGE)
    tree = parser.parse(b"def hello(): pass")
    root = tree.root_node
    assert root.type == "module"
    console.print(f"[green]✅ tree-sitter[/green] → parsed root = {root.type!r}, children = {root.child_count}")


TESTS = [
    ("Anthropic", test_anthropic),
    ("Voyage AI", test_voyage),
    ("Pinecone", test_pinecone),
    ("tree-sitter", test_tree_sitter),
]

if __name__ == "__main__":
    console.print(Panel("[bold]Kyoto — Phase 1 Smoke Test[/bold]", expand=False))
    failed = []

    for name, fn in TESTS:
        try:
            fn()
        except Exception as e:
            console.print(f"[red]❌ {name}[/red] → {e}")
            failed.append(name)

    console.print()
    if failed:
        console.print(f"[bold red]{len(failed)} check(s) failed:[/bold red] {', '.join(failed)}")
        sys.exit(1)
    else:
        console.print("[bold green]All checks passed — Phase 1 complete.[/bold green]")