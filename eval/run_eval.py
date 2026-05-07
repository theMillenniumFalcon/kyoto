"""
RAGAS evaluation for Kyoto retrieval + Q&A quality.

Usage:
    uv run python eval/run_eval.py
    uv run python eval/run_eval.py --queries eval/test_queries.json --output eval/results.json
"""
import json
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from rich.console import Console
from rich.table import Table

from src.config.settings import settings
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import rerank
from src.api.routes.query import SYSTEM_PROMPT

console = Console()

_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)


def run_eval(queries_path: str = "eval/test_queries.json", output_path: str | None = None):
    console.print("[bold cyan]Kyoto — RAGAS Evaluation[/bold cyan]")

    test_cases = json.loads(Path(queries_path).read_text())
    console.print(f"[dim]Loaded {len(test_cases)} test cases from {queries_path}[/dim]")

    rows = []
    for i, tc in enumerate(test_cases):
        q = tc["question"]
        console.print(f"\n[cyan]({i + 1}/{len(test_cases)}) {q}[/cyan]")

        # Retrieve + rerank
        candidates = hybrid_search(q, top_k=10)
        chunks = rerank(q, candidates, top_k=5)
        contexts = [c["code"] for c in chunks]

        if not contexts:
            console.print("  [yellow]⚠ No context retrieved — skipping[/yellow]")
            continue

        # Build context string for Claude
        context_str = "\n\n".join(
            f"# {c['file_path']} (lines {c['start_line']}–{c['end_line']})\n{c['code']}"
            for c in chunks
        )

        # Ask Claude
        response = _client.messages.create(
            model=settings.claude_model,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"## Code Context\n\n{context_str}\n\n## Question\n\n{q}",
                }
            ],
        )
        answer = response.content[0].text
        console.print(f"  [dim]Answer ({len(answer)} chars)[/dim]")

        rows.append(
            {
                "question": q,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": tc["ground_truth"],
            }
        )

    if not rows:
        console.print("[red]No rows to evaluate — is the Pinecone index populated?[/red]")
        return

    console.print(f"\n[cyan]Running RAGAS on {len(rows)} rows...[/cyan]")
    dataset = Dataset.from_list(rows)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    # Pretty-print results
    table = Table(title="RAGAS Scorecard", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Interpretation")

    thresholds = {
        "faithfulness":      (0.85, "% of answer claims grounded in context"),
        "answer_relevancy":  (0.80, "% relevance of answer to the question"),
        "context_precision": (0.75, "% of retrieved context that was useful"),
    }

    scores = result.to_pandas().mean().to_dict()
    for metric, (threshold, description) in thresholds.items():
        score = scores.get(metric, 0.0)
        colour = "green" if score >= threshold else "yellow" if score >= threshold - 0.1 else "red"
        table.add_row(
            metric,
            f"[{colour}]{score:.3f}[/{colour}]",
            f"{description} (target ≥ {threshold})",
        )

    console.print(table)
    console.print(f"\n[dim]Full result object: {result}[/dim]")

    # Optionally persist results
    if output_path:
        output = {
            "scores": scores,
            "num_questions": len(rows),
            "queries_path": queries_path,
        }
        Path(output_path).write_text(json.dumps(output, indent=2))
        console.print(f"[green]✓ Results saved to {output_path}[/green]")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation for Kyoto")
    parser.add_argument(
        "--queries",
        default="eval/test_queries.json",
        help="Path to test_queries.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save JSON results (e.g. eval/results.json)",
    )
    args = parser.parse_args()
    run_eval(args.queries, args.output)