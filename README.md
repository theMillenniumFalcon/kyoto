# Kyoto

**Enterprise Codebase Q&A + Auto-Debugger**

Ask natural-language questions about any GitHub repository and get cited answers. Drop in a stack trace and get a root-cause analysis with a suggested fix. Powered by Claude (Anthropic), Voyage AI embeddings, and Pinecone vector search.

```
POST /api/v1/query  →  "How does authentication work?"  →  cited answer + source files
POST /api/v1/debug  →  stack trace                      →  root cause + suggested fix
```

---

## Table of contents

- [Features](#features)
- [Architecture](#architecture)
- [How it works](#how-it-works)
  - [Ingestion pipeline](#ingestion-pipeline)
  - [Retrieval pipeline](#retrieval-pipeline)
  - [Auto-debugger](#auto-debugger)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
  - [Local (uv)](#local-uv)
  - [Docker](#docker)
- [Usage](#usage)
  - [Ingest a repository](#ingest-a-repository)
  - [Query the codebase](#query-the-codebase)
  - [Debug a stack trace](#debug-a-stack-trace)
  - [Incremental re-indexing](#incremental-re-indexing)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Project structure](#project-structure)
- [Tech stack](#tech-stack)

---

## Features

| Feature | Details |
|---|---|
| **AST-aware ingestion** | tree-sitter parses Python, JavaScript, and TypeScript at function/class boundaries — not arbitrary line splits |
| **Token-budget chunking** | Overlapping sub-chunks with configurable size (default 512 tokens) ensure large classes never exceed Voyage's context window |
| **Hybrid retrieval** | Dense vector search (Pinecone) + BM25 sparse search fused via Reciprocal Rank Fusion |
| **Cross-encoder re-ranking** | `ms-marco-MiniLM-L-6-v2` re-scores the top candidates before sending to Claude |
| **Cited Q&A** | Every answer references the exact file, function, and line range it was drawn from |
| **Agentic debugger** | Claude uses a `search_codebase` tool in a loop (up to 4 iterations) to trace the root cause of any Python or JS/TS stack trace |
| **Async ingestion** | Celery + Redis task queue — ingest large repos without blocking the API |
| **Incremental re-indexing** | `git diff`-based script re-indexes only changed files; deleted files are purged from Pinecone |
| **RAGAS evaluation** | Automated retrieval quality scoring: faithfulness, answer relevancy, context precision |

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                     Docker stack                     │
│                                                      │
│  ┌───────────────┐  ┌─────────────┐  ┌───────────┐   │
│  │  API  :8000   │  │   Worker    │  │  Redis    │   │
│  │  (FastAPI)    │  │  (Celery)   │  │  :6379    │   │
│  └───────┬───────┘  └──────┬──────┘  └─────┬─────┘   │
└──────────┼────────────────-┼───────────────┼─────────┘
           │                 │               │
           ▼                 ▼               │
     ┌──────────┐      ┌──────────┐          │
     │ Pinecone │      │ Voyage   │◄─────────┘
     │ (vectors)│      │ AI       │  Celery broker
     └──────────┘      └──────────┘
           ▲
           │ context
     ┌──────────┐
     │  Claude  │
     │ (Sonnet) │
     └──────────┘
```

Three containers share a network. The API handles queries synchronously. Heavy ingestion jobs are dispatched to the Celery worker via Redis. Both services share a named Docker volume for the BM25 index pickle so it survives restarts.

---

## How it works

### Ingestion pipeline

```
GitHub URL
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  1. Clone                                            │
│     Shallow clone (depth=1) via GitPython            │
│     → temp directory, cleaned up after indexing      │
└────────────────────────┬─────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│  2. Walk                                             │
│     Recurse all .py / .js / .ts / .jsx / .tsx files  │
│     Skip: node_modules, .git, __pycache__, dist …    │
└────────────────────────┬─────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│  3. AST parse  (tree-sitter)                         │
│     Extract function_definition, class_definition,   │
│     method_definition, interface_declaration etc.    │
│     Each becomes a CodeChunk:                        │
│       name, kind, code, language, file_path,         │
│       start_line, end_line, imports, docstring       │
└────────────────────────┬─────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│  4. Token-budget chunking                            │
│     Count tokens with tiktoken (cl100k_base)         │
│     Chunks > 512 tokens → overlapping windows        │
│     64-token overlap, import header on every window  │
└────────────────────────┬─────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│  5. Embed  (Voyage AI)                               │
│     voyage-code-2, 1536 dimensions                   │
│     input_type="document" at index time              │
│     Batched in groups of 64 with tenacity retry      │
└────────────────────────┬─────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│  6. Upsert  (Pinecone serverless)                    │
│     Stable MD5 ID per chunk → idempotent re-indexing │
│     Code stored in metadata (truncated at 4000 chars)│
│     Batch size 100                                   │
└──────────────────────────────────────────────────────┘
```

**Why AST-aware chunking?** Splitting on arbitrary line counts breaks functions mid-body, discards import context, and mixes unrelated logic. Parsing at declaration boundaries means each vector represents a complete, semantically coherent unit of code.

---

### Retrieval pipeline

Every `/query` request runs three stages before Claude sees any code:

```
User question
      │
      ├──────────────────────────────────┐
      │                                  │
      ▼                                  ▼
┌───────────────┐              ┌───────────────────┐
│ Dense search  │              │  Sparse search    │
│ (Pinecone)    │              │  (BM25 in-memory) │
│               │              │                   │
│ embed_query() │              │  camelCase-aware  │
│ input_type=   │              │  tokenizer splits │
│ "query"       │              │  getUserById →    │
│ top_k × 2     │              │  [get,user,by,id] │
└───────┬───────┘              └────────┬──────────┘
        │                               │
        └───────────────┬───────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │  RRF fusion      │
              │                  │
              │  score(d) =      │
              │  Σ 1/(60 + rank) │
              │  dense + sparse  │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  Cross-encoder   │
              │  re-rank         │
              │                  │
              │  ms-marco-MiniLM │
              │  (query, code)   │
              │  pairs → scores  │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  Claude Sonnet   │
              │                  │
              │  top_k chunks as │
              │  numbered context│
              │  cites file +    │
              │  function name   │
              └──────────────────┘
```

**Why hybrid?** Dense search excels at semantic similarity ("find where errors are caught") but misses exact keyword matches ("find `authenticate_user`"). BM25 handles exact identifiers and camelCase tokens well. RRF fusion rewards documents that rank highly in both lists without requiring score normalisation.

**Why re-rank?** The bi-encoder (Voyage) scores query and document independently — fast but approximate. The cross-encoder sees the query and code together, producing more accurate relevance scores at the cost of speed. Running it only on the top 20 candidates keeps latency manageable.

---

### Auto-debugger

The debugger is a Claude tool-use agent. It receives the stack trace, searches the codebase with a `search_codebase` tool, and iterates until it has enough context to diagnose the root cause.

```
Stack trace input
      │
      ▼
┌─────────────────────────────┐
│  Parse  (error_parser.py)   │
│                             │
│  Detects: Python or JS/TS   │
│  Extracts:                  │
│    error_type, message      │
│    StackFrame list          │
│    (file, line, function)   │
│  Strips env prefixes        │
│  /app/auth.py → auth.py     │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Pre-fetch context          │
│                             │
│  hybrid_search() on:        │
│  • last 3 frame functions   │
│  • error_type + message     │
└──────────────┬──────────────┘
               │
               ▼
┌──────────────────────────────────────────────────┐
│  Claude agentic loop  (max 4 iterations)         │
│                                                  │
│  ┌────────────┐  stop_reason   ┌──────────────┐  │
│  │   Claude   │── end_turn ───►│  Diagnosis   │  │
│  │   Sonnet   │                │  returned    │  │
│  └─────┬──────┘                └──────────────┘  │
│        │ tool_use: search_codebase               │
│        ▼                                         │
│  ┌─────────────────┐                             │
│  │  hybrid_search  │  top_k=6                    │
│  │  + rerank       │  top 3 chunks returned      │
│  └────────┬────────┘                             │
│           │ tool_result (markdown code blocks)   │
│           └──────────────────────► back to Claude│
└──────────────────────────────────────────────────┘
               │
               ▼
  {
    "diagnosis": "**Root Cause:** ...",
    "error": { type, message, language, frames },
    "iterations": 2
  }
```

---

## Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Python | ≥ 3.11 | Runtime |
| [uv](https://docs.astral.sh/uv/) | latest | Package manager |
| Docker + Docker Compose | v2+ | Container stack |
| [Anthropic API key](https://console.anthropic.com) | — | Claude Sonnet |
| [Voyage AI API key](https://dash.voyageai.com) | — | Code embeddings |
| [Pinecone API key](https://app.pinecone.io) | — | Vector store |

---

## Setup

### Local (uv)

```bash
# 1. Clone
git clone https://github.com/your-org/kyoto.git
cd kyoto

# 2. Install dependencies
uv sync
uv sync --extra dev    # pytest, ruff, ragas

# 3. Configure environment
cp .env.example .env
# Fill in ANTHROPIC_API_KEY, VOYAGE_API_KEY, PINECONE_API_KEY

# 4. Smoke test — verifies all three API clients and tree-sitter
uv run python scripts/smoke_test.py
```

Expected output:
```
✅ Anthropic   → hi there
✅ Voyage AI   → embedding dim = 1536
✅ Pinecone    → indexes = []
✅ tree-sitter → parsed root = 'module', children = 1
All checks passed — Phase 1 complete.
```

Redis is required for the Celery worker. Start it locally:
```bash
docker run -d -p 6379:6379 redis:7-alpine
```

### Docker

```bash
# 1. Configure environment
cp .env.example .env

# 2. Build and start (dev — hot reload enabled via docker-compose.override.yml)
docker compose up --build

# 3. Verify
curl http://localhost:8000/health
# → {"status":"ok","version":"0.2.0"}
```

**Production:**
```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

Production differences vs dev: no `--reload`, uvicorn runs `--workers 2`, Celery concurrency raised to 4, Redis port not exposed externally, AOF persistence enabled.

---

## Usage

### Ingest a repository

```bash
uv run python scripts/ingest.py --repo https://github.com/tiangolo/fastapi

# With a custom name stored in Pinecone metadata
uv run python scripts/ingest.py \
  --repo https://github.com/tiangolo/fastapi \
  --repo-name fastapi
```

Or dispatch as an async Celery task:

```python
from src.worker.tasks import ingest_repo
result = ingest_repo.delay("https://github.com/tiangolo/fastapi", "fastapi")
print(result.get())  # {"status":"ok","repo":"fastapi","files":94,"chunks":1203}
```

> **Cost note:** A ~50k-line repo costs roughly $0.10–$0.50 in Voyage API credits and takes 2–5 minutes. Test on a smaller repo first (e.g. `https://github.com/encode/starlette`).

---

### Query the codebase

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does request validation work?"}'
```

With optional language filter and result count:

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Where are database models defined?",
    "language": "python",
    "top_k": 5
  }'
```

Response:

```json
{
  "answer": "Request validation in FastAPI is handled by Pydantic models. When you declare a path operation function with a parameter typed as a Pydantic model...",
  "sources": [
    {
      "file": "fastapi/routing.py",
      "function": "get_request_handler",
      "lines": "163–201",
      "language": "python"
    },
    {
      "file": "fastapi/dependencies/utils.py",
      "function": "request_body_to_args",
      "lines": "341–389",
      "language": "python"
    }
  ]
}
```

---

### Debug a stack trace

```bash
curl -X POST http://localhost:8000/api/v1/debug \
  -H "Content-Type: application/json" \
  -d '{
    "traceback": "Traceback (most recent call last):\n  File \"/app/auth/views.py\", line 42, in login\n    user = authenticate_user(db, form.username, form.password)\n  File \"/app/auth/utils.py\", line 18, in authenticate_user\n    return db.query(User).filter(User.email == username).first()\nAttributeError: \"NoneType\" object has no attribute \"query\""
  }'
```

Response:

```json
{
  "diagnosis": "**Root Cause:** The `db` session is `None` when passed to `authenticate_user`.\n\n**Why it happens:** ...\n\n**Suggested Fix:**\n```python\n...\n```\n\n**Files affected:** auth/views.py, auth/utils.py",
  "error": {
    "type": "AttributeError",
    "message": "\"NoneType\" object has no attribute \"query\"",
    "language": "python",
    "frames": [
      { "file": "auth/views.py", "line": 42, "function": "login", "snippet": "user = authenticate_user(db, ...)" },
      { "file": "auth/utils.py", "line": 18, "function": "authenticate_user", "snippet": "return db.query(User)..." }
    ]
  },
  "iterations": 2
}
```

Both Python (`Traceback (most recent call last)`) and JavaScript/TypeScript (`at functionName (file.js:line:col)`) formats are supported.

---

### Incremental re-indexing

After pulling new commits to a previously-indexed repo:

```bash
# Re-index files changed since the last commit
uv run python scripts/reindex_changed.py \
  --repo ./fastapi \
  --since HEAD~1 \
  --name fastapi

# Re-index since a specific commit hash
uv run python scripts/reindex_changed.py \
  --repo ./fastapi \
  --since abc1234 \
  --name fastapi
```

Deleted files are purged from Pinecone automatically. Upserts are idempotent — re-running is always safe.

---

## Configuration

All settings are in `src/config/settings.py`, loaded from `.env`:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | **required** | Claude API key |
| `CLAUDE_MODEL` | `claude-sonnet-4-20250514` | Model for Q&A and debug |
| `VOYAGE_API_KEY` | **required** | Voyage AI key |
| `VOYAGE_MODEL` | `voyage-code-2` | Embedding model (1536 dims) |
| `PINECONE_API_KEY` | **required** | Pinecone API key |
| `PINECONE_INDEX_NAME` | `kyoto-codebase` | Pinecone index name |
| `PINECONE_ENVIRONMENT` | `us-east-1` | Pinecone serverless region |
| `REDIS_URL` | `redis://localhost:6379/0` | Celery broker + result backend |
| `MAX_CHUNK_TOKENS` | `512` | Max tokens per chunk before splitting |
| `CHUNK_OVERLAP_TOKENS` | `64` | Overlap between split sub-chunks |

> When creating your Pinecone index, set dimensions to **1536** explicitly. The UI default is different.

---

## Evaluation

```bash
# Run RAGAS against the populated index
uv run python eval/run_eval.py

# Persist scores for regression tracking
uv run python eval/run_eval.py --output eval/results.json
```

Sample output:

```
┌──────────────────────────────────────────────────────────────────┐
│                         RAGAS Scorecard                          │
├───────────────────────┬───────┬──────────────────────────────────┤
│ Metric                │ Score │ Interpretation                   │
├───────────────────────┼───────┼──────────────────────────────────┤
│ faithfulness          │ 0.891 │ % of claims grounded in context  │
│ answer_relevancy      │ 0.843 │ % relevance to the question      │
│ context_precision     │ 0.812 │ % of retrieved context useful    │
└───────────────────────┴───────┴──────────────────────────────────┘
```

Targets: faithfulness ≥ 0.85, answer relevancy ≥ 0.80, context precision ≥ 0.75.

Add questions to `eval/test_queries.json`. 20 hand-crafted questions covering the key concepts of your repo produces more actionable signal than a large auto-generated set.

---

## Project structure

```
kyoto/
├── Dockerfile
├── docker-compose.yml            ← full stack: api + worker + redis
├── docker-compose.override.yml   ← dev: hot reload + src volume mounts
├── docker-compose.prod.yml       ← prod: workers=2, AOF persistence
├── pyproject.toml
├── .env.example
│
├── eval/
│   ├── test_queries.json         ← ground-truth Q&A pairs (RAGAS)
│   └── run_eval.py               ← scorecard runner
│
├── scripts/
│   ├── smoke_test.py             ← verify API keys + deps
│   ├── ingest.py                 ← full ingestion CLI
│   └── reindex_changed.py        ← incremental re-index via git diff
│
└── src/
    ├── config/
    │   └── settings.py           ← pydantic-settings, loads .env
    │
    ├── ingestion/
    │   ├── repo_loader.py        ← git clone + file walk
    │   ├── ast_parser.py         ← tree-sitter → CodeChunk
    │   └── chunker.py            ← token-budget split + overlap
    │
    ├── indexing/
    │   ├── embedder.py           ← Voyage AI (doc vs query type)
    │   └── pinecone_store.py     ← upsert, query, stable IDs
    │
    ├── retrieval/
    │   ├── sparse.py             ← BM25 + camelCase tokenizer
    │   ├── hybrid.py             ← RRF fusion of dense + sparse
    │   └── reranker.py           ← cross-encoder, lazy-loaded
    │
    ├── debugger/
    │   ├── error_parser.py       ← Python + JS/TS stack trace parser
    │   └── agent.py              ← Claude tool-use agentic loop
    │
    ├── api/
    │   ├── main.py               ← FastAPI app + lifespan warmup
    │   └── routes/
    │       ├── query.py          ← POST /api/v1/query
    │       └── debug.py          ← POST /api/v1/debug
    │
    └── worker/
        ├── celery_app.py         ← Celery app configuration
        └── tasks.py              ← ingest_repo async task
```

---

## Tech stack

| Layer | Technology |
|---|---|
| LLM | Claude Sonnet (Anthropic) |
| Embeddings | Voyage AI `voyage-code-2` — 1536 dimensions |
| Vector store | Pinecone serverless |
| Sparse search | BM25 (`rank-bm25`) |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Code parsing | tree-sitter (Python, JavaScript, TypeScript) |
| API | FastAPI + uvicorn |
| Task queue | Celery + Redis |
| Package manager | uv |
| Containerisation | Docker + Docker Compose |
| Evaluation | RAGAS |