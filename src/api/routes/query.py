import anthropic
from fastapi import APIRouter
from pydantic import BaseModel

from src.config.settings import settings
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import rerank

router = APIRouter()
_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

SYSTEM_PROMPT = """\
You are an expert code assistant. You help developers understand codebases.

Rules:
- Answer ONLY using the provided code context
- Always cite the specific file and function name when referencing code
- If the answer is not in the context, say "I couldn't find this in the indexed codebase"
- Be precise and technical — your audience is software engineers
- Format code snippets with markdown code blocks\
"""


class QueryRequest(BaseModel):
    question: str
    language: str | None = None  # optional filter: "python" | "typescript" | "javascript"
    top_k: int = 5


class SourceRef(BaseModel):
    file: str
    function: str
    lines: str
    language: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceRef]


@router.post("/query", response_model=QueryResponse)
async def query_codebase(req: QueryRequest) -> QueryResponse:
    # ── Retrieve ──────────────────────────────────────────────────────────────
    candidates = hybrid_search(
        req.question,
        top_k=req.top_k * 2,
        language_filter=req.language,
    )
    top_chunks = rerank(req.question, candidates, top_k=req.top_k)

    # ── Build context for Claude ──────────────────────────────────────────────
    context_parts = []
    for i, chunk in enumerate(top_chunks):
        context_parts.append(
            f"### [{i + 1}] `{chunk['file_path']}` — "
            f"`{chunk['name']}()` (lines {chunk['start_line']}–{chunk['end_line']})\n"
            f"```{chunk['language']}\n{chunk['code']}\n```"
        )
    context = "\n\n".join(context_parts)

    # ── Ask Claude ────────────────────────────────────────────────────────────
    response = _client.messages.create(
        model=settings.claude_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"## Code Context\n\n{context}\n\n## Question\n\n{req.question}",
            }
        ],
    )

    sources = [
        SourceRef(
            file=c["file_path"],
            function=c["name"],
            lines=f"{c['start_line']}–{c['end_line']}",
            language=c["language"],
        )
        for c in top_chunks
    ]

    return QueryResponse(answer=response.content[0].text, sources=sources)