import anthropic
from rich.console import Console

from src.config.settings import settings
from src.debugger.error_parser import parse_traceback
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import rerank

console = Console()

_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

MAX_ITERATIONS = 4

TOOLS = [
    {
        "name": "search_codebase",
        "description": (
            "Search the indexed codebase for code relevant to the error. "
            "Use this to find the source of a failing function, trace a call chain, "
            "or look up how something is implemented."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language search query "
                        "(e.g. 'authenticate user function', 'database connection pool')"
                    ),
                },
                "language": {
                    "type": "string",
                    "enum": ["python", "javascript", "typescript"],
                    "description": "Optionally filter results by language",
                },
            },
            "required": ["query"],
        },
    }
]

SYSTEM_PROMPT = """\
You are an expert debugger. Given a stack trace and access to the codebase, your job is to:

1. Identify the ROOT CAUSE of the error (not just the surface symptom)
2. Explain clearly WHY this error occurs
3. Suggest a specific code fix

You have a tool `search_codebase` to look up any function or module.
Use it proactively — search for functions mentioned in the stack trace, then search for their dependencies.

Format your final answer with:
- **Root Cause:** one-sentence summary
- **Why it happens:** detailed explanation
- **Suggested Fix:** code block showing the corrected code
- **Files affected:** list of files\
"""


def _do_search(query: str, language: str | None = None) -> str:
    """Run hybrid search + rerank and return formatted markdown."""
    console.print(f"  [dim]🔍 search_codebase({query!r})[/dim]")
    candidates = hybrid_search(query, top_k=6, language_filter=language)
    top = rerank(query, candidates, top_k=3)
    if not top:
        return "No relevant code found."

    parts = []
    for c in top:
        parts.append(
            f"### `{c['file_path']}` — `{c['name']}` "
            f"(lines {c['start_line']}–{c['end_line']})\n"
            f"```{c['language']}\n{c['code'][:1500]}\n```"
        )
    return "\n\n".join(parts)


def debug_traceback(traceback: str) -> dict:
    """
    Full agentic debug loop:
      1. Parse the traceback
      2. Pre-fetch context for the top frames
      3. Run Claude tool-use loop (max MAX_ITERATIONS rounds)
      4. Return structured diagnosis
    """
    console.print("[cyan]Parsing traceback...[/cyan]")
    error = parse_traceback(traceback)
    console.print(
        f"[green]✓ Detected:[/green] {error.language} / "
        f"{error.error_type}: {error.message[:80]}"
    )

    # Pre-fetch context: last 3 frames + the error type+message
    search_seeds = [f.function for f in error.frames[-3:]] + [
        f"{error.error_type} {error.message}"
    ]
    console.print("[cyan]Pre-fetching initial context...[/cyan]")
    initial_context_parts = [_do_search(q) for q in search_seeds[:3]]
    initial_context = "\n\n".join(initial_context_parts)

    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                f"## Stack Trace\n```\n{traceback}\n```\n\n"
                f"## Initial Code Context\n{initial_context}\n\n"
                "Please diagnose this error. Use `search_codebase` if you need more context."
            ),
        }
    ]

    console.print("[cyan]Starting Claude agentic loop...[/cyan]")

    for iteration in range(MAX_ITERATIONS):
        console.print(f"  [dim]Iteration {iteration + 1}/{MAX_ITERATIONS}[/dim]")

        response = _client.messages.create(
            model=settings.claude_model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Append assistant turn to history
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            final_text = next(
                (b.text for b in response.content if hasattr(b, "text")), ""
            )
            console.print(f"[green]✓ Diagnosis complete ({iteration + 1} iterations)[/green]")
            return {
                "diagnosis": final_text,
                "error": {
                    "type": error.error_type,
                    "message": error.message,
                    "language": error.language,
                    "frames": [
                        {
                            "file": f.file,
                            "line": f.line,
                            "function": f.function,
                            "snippet": f.snippet,
                        }
                        for f in error.frames
                    ],
                },
                "iterations": iteration + 1,
            }

        # Handle tool_use blocks → feed results back
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                args = block.input
                result = _do_search(args["query"], args.get("language"))
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )

        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        else:
            # No tool calls and stop_reason wasn't end_turn — shouldn't happen, break safely
            break

    return {
        "diagnosis": "Max iterations reached without a conclusive diagnosis.",
        "error": {
            "type": error.error_type,
            "message": error.message,
            "language": error.language,
        },
        "iterations": MAX_ITERATIONS,
    }