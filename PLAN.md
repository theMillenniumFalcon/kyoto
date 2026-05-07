# 🔨 Kyoto: Enterprise Codebase Q&A + Auto-Debugger
## Implementation Phases (post `uv init`)

> **Stack:** Claude (Anthropic) · Pinecone · Python + JS/TS parsing  
> **Starting point:** `uv init` already run

---

## Current State of Your Project

```
your-project/
├── .python-version
├── pyproject.toml
├── README.md
└── main.py          ← we'll delete/replace this
```

---

## 📦 Phase 1 — Project Scaffold + Dependencies

**Goal:** Set up the full folder structure, install all dependencies, and verify everything is wired correctly with a smoke test.

**✅ Testable outcome:** Run `uv run python -c "import anthropic, pinecone, tree_sitter; print('all good')"` without errors.

---

### 1.1 Add Dependencies

```toml
# pyproject.toml — add these under [project] dependencies
[project]
name = "codebase-rag"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    # LLM
    "anthropic>=0.25.0",

    # Vector DB
    "pinecone-client>=3.0.0",

    # Embeddings (Voyage is Anthropic's recommended code embedding model)
    "voyageai>=0.2.0",

    # Code parsing
    "tree-sitter>=0.23.0",
    "tree-sitter-python>=0.23.0",
    "tree-sitter-javascript>=0.23.0",
    "tree-sitter-typescript>=0.23.0",

    # Git ingestion
    "gitpython>=3.1.40",
    "PyGithub>=2.1.1",

    # Sparse search
    "rank-bm25>=0.2.2",

    # API server
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.29.0",

    # Async + task queue
    "celery>=5.3.6",
    "redis>=5.0.3",

    # Env + config
    "python-dotenv>=1.0.1",
    "pydantic-settings>=2.2.1",

    # Re-ranking
    "sentence-transformers>=2.7.0",

    # Utilities
    "tenacity>=8.2.3",
    "rich>=13.7.1",
    "tiktoken>=0.6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.1.1",
    "pytest-asyncio>=0.23.6",
    "httpx>=0.27.0",      # for testing FastAPI
    "ruff>=0.4.1",
    "ragas>=0.1.9",       # evaluation (Phase 5)
    "datasets>=2.19.0",
]
```

```bash
uv sync
uv sync --extra dev      # for dev tools
```

---

### 1.2 Create Folder Structure

```bash
mkdir -p src/{ingestion,indexing,retrieval,debugger,api/routes,config}
mkdir -p tests/{unit,integration}
mkdir -p eval scripts
touch src/__init__.py
touch src/{ingestion,indexing,retrieval,debugger,api,config}/__init__.py
touch src/api/routes/__init__.py
touch .env .env.example
```

Final structure:
```
your-project/
├── pyproject.toml
├── .env                        ← secrets (never commit)
├── .env.example                ← template to commit
├── .gitignore
├── src/
│   ├── config/
│   │   └── settings.py         ← all env vars in one place
│   ├── ingestion/
│   │   ├── repo_loader.py      ← clone + walk repos
│   │   ├── ast_parser.py       ← tree-sitter for py/js/ts
│   │   └── chunker.py          ← AST-aware chunking
│   ├── indexing/
│   │   ├── embedder.py         ← Voyage AI embeddings
│   │   └── pinecone_store.py   ← upsert / query Pinecone
│   ├── retrieval/
│   │   ├── dense.py            ← vector search
│   │   ├── sparse.py           ← BM25
│   │   ├── hybrid.py           ← RRF fusion
│   │   └── reranker.py         ← cross-encoder
│   ├── debugger/
│   │   ├── error_parser.py     ← stack trace parsing
│   │   └── agent.py            ← Claude tool-use agent
│   └── api/
│       ├── main.py             ← FastAPI app
│       └── routes/
│           ├── query.py
│           └── debug.py
├── tests/
├── eval/
│   └── test_queries.json
└── scripts/
    └── ingest.py               ← CLI to index a repo
```

---

### 1.3 Settings / Config

```python
# src/config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Anthropic
    anthropic_api_key: str
    claude_model: str = "claude-3-5-sonnet-20241022"

    # Voyage AI (embeddings)
    voyage_api_key: str
    voyage_model: str = "voyage-code-2"   # code-specific, 1536 dims

    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str = "codebase-rag"
    pinecone_environment: str = "us-east-1"  # your Pinecone region

    # Redis (for Celery, Phase 3+)
    redis_url: str = "redis://localhost:6379/0"

    class Config:
        env_file = ".env"

settings = Settings()
```

```bash
# .env.example  (commit this)
ANTHROPIC_API_KEY=sk-ant-...
VOYAGE_API_KEY=pa-...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=codebase-rag
PINECONE_ENVIRONMENT=us-east-1
```

---

### 1.4 Smoke Test

```bash
# scripts/smoke_test.py
import anthropic
from pinecone import Pinecone
import voyageai

# Test Anthropic
client = anthropic.Anthropic()
msg = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=32,
    messages=[{"role": "user", "content": "say hi"}]
)
print("✅ Anthropic:", msg.content[0].text)

# Test Voyage
vc = voyageai.Client()
emb = vc.embed(["def hello(): pass"], model="voyage-code-2")
print(f"✅ Voyage: embedding dim = {len(emb.embeddings[0])}")

# Test Pinecone
pc = Pinecone()
print(f"✅ Pinecone: indexes = {pc.list_indexes().names()}")
```

```bash
uv run python scripts/smoke_test.py
```

### Phase 1 Watch-outs
- `voyage-code-2` produces **1536-dimensional** vectors. Set this when creating your Pinecone index (not 1536 by default on their UI — double check).
- Keep `.env` in `.gitignore` from the very start. Leaked API keys are painful.
- `tree-sitter` grammars changed their API in v0.23 — the imports above use the new style. Don't copy old Stack Overflow snippets.

---

## 📦 Phase 2 — Ingestion Pipeline

**Goal:** Clone a GitHub repo, parse it with AST, chunk at function/class boundaries, embed with Voyage, and store in Pinecone.

**✅ Testable outcome:** Run `uv run python scripts/ingest.py --repo https://github.com/tiangolo/fastapi` and see chunks appearing in your Pinecone index.

---

### 2.1 Repo Loader

```python
# src/ingestion/repo_loader.py
import os
import tempfile
from pathlib import Path
from git import Repo
from rich.console import Console

console = Console()

SUPPORTED = {".py", ".js", ".ts", ".jsx", ".tsx"}
SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".venv",
    "dist", "build", ".next", "coverage", ".pytest_cache"
}

def clone_repo(url: str) -> tuple[str, str]:
    """Clone repo to a temp dir. Returns (temp_dir, repo_name)."""
    repo_name = url.rstrip("/").split("/")[-1].replace(".git", "")
    tmp = tempfile.mkdtemp(prefix=f"rag_{repo_name}_")
    console.print(f"[cyan]Cloning {url}...[/cyan]")
    Repo.clone_from(url, tmp, depth=1)   # shallow clone = faster
    return tmp, repo_name

def walk_repo(repo_path: str, repo_name: str) -> list[dict]:
    """Walk repo and return list of file dicts."""
    files = []
    root = Path(repo_path)

    for path in root.rglob("*"):
        # Skip unwanted directories
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

        if len(content.strip()) < 50:   # skip near-empty files
            continue

        relative = str(path.relative_to(root))
        lang = {
            ".py": "python",
            ".js": "javascript", ".jsx": "javascript",
            ".ts": "typescript", ".tsx": "typescript",
        }.get(path.suffix, "unknown")

        files.append({
            "path": relative,
            "content": content,
            "language": lang,
            "repo": repo_name,
            "size_lines": len(content.splitlines()),
        })
        console.print(f"  [dim]Found: {relative}[/dim]")

    console.print(f"[green]✓ {len(files)} files found[/green]")
    return files
```

---

### 2.2 AST Parser — Python + JS/TS

```python
# src/ingestion/ast_parser.py
from dataclasses import dataclass, field
from tree_sitter import Language, Parser, Node
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript

@dataclass
class CodeChunk:
    name: str               # function/class name
    kind: str               # "function" | "class" | "method" | "module_top"
    code: str               # actual source code
    language: str
    file_path: str
    repo: str
    start_line: int
    end_line: int
    docstring: str = ""
    imports: list[str] = field(default_factory=list)

# Build language objects once at import time
_PARSERS: dict[str, Parser] = {
    "python":     Parser(Language(tspython.language())),
    "javascript": Parser(Language(tsjavascript.language())),
    "typescript": Parser(Language(tstypescript.language_typescript())),
}

# Node types that represent "chunk boundaries" per language
_CHUNK_TYPES = {
    "python":     {"function_definition", "class_definition", "decorated_definition"},
    "javascript": {"function_declaration", "function_expression", "arrow_function",
                   "class_declaration", "method_definition", "export_statement"},
    "typescript": {"function_declaration", "function_expression", "arrow_function",
                   "class_declaration", "method_definition", "export_statement",
                   "interface_declaration", "type_alias_declaration"},
}

def _extract_imports(code: str, language: str) -> list[str]:
    """Pull import lines from the top of the file."""
    imports = []
    for line in code.splitlines()[:30]:
        line = line.strip()
        if language == "python" and (line.startswith("import ") or line.startswith("from ")):
            imports.append(line)
        elif language in ("javascript", "typescript") and line.startswith("import "):
            imports.append(line)
    return imports

def _get_docstring(node: Node, source_bytes: bytes) -> str:
    """Extract docstring from the first child string literal (Python)."""
    for child in node.children:
        if child.type == "block":
            for stmt in child.children:
                if stmt.type == "expression_statement":
                    for inner in stmt.children:
                        if inner.type in ("string", "concatenated_string"):
                            return source_bytes[inner.start_byte:inner.end_byte].decode(errors="ignore")
    return ""

def _traverse(node: Node, source_bytes: bytes, file_info: dict,
               imports: list[str], chunks: list[CodeChunk]):
    lang = file_info["language"]
    chunk_types = _CHUNK_TYPES.get(lang, set())

    if node.type in chunk_types:
        name_node = node.child_by_field_name("name")
        name = source_bytes[name_node.start_byte:name_node.end_byte].decode() if name_node else "anonymous"
        code = source_bytes[node.start_byte:node.end_byte].decode(errors="ignore")
        docstring = _get_docstring(node, source_bytes) if lang == "python" else ""

        kind = "function"
        if "class" in node.type or "interface" in node.type:
            kind = "class"
        elif "method" in node.type:
            kind = "method"

        # Prepend imports for context
        import_header = "\n".join(imports) + "\n\n" if imports else ""

        chunks.append(CodeChunk(
            name=name,
            kind=kind,
            code=import_header + code,
            language=lang,
            file_path=file_info["path"],
            repo=file_info["repo"],
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            docstring=docstring,
            imports=imports,
        ))

    for child in node.children:
        _traverse(child, source_bytes, file_info, imports, chunks)

def parse_file(file_info: dict) -> list[CodeChunk]:
    lang = file_info["language"]
    parser = _PARSERS.get(lang)
    if not parser:
        return []

    source = file_info["content"]
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    imports = _extract_imports(source, lang)
    chunks: list[CodeChunk] = []
    _traverse(tree.root_node, source_bytes, file_info, imports, chunks)

    # If no named chunks found, treat whole file as one chunk (e.g. config files)
    if not chunks and len(source.splitlines()) < 200:
        chunks.append(CodeChunk(
            name=file_info["path"].split("/")[-1],
            kind="module_top",
            code=source,
            language=lang,
            file_path=file_info["path"],
            repo=file_info["repo"],
            start_line=1,
            end_line=len(source.splitlines()),
            imports=imports,
        ))

    return chunks
```

> ⚠️ **Watch out:** For TypeScript, use `language_typescript()` not `language_tsx()` unless you're specifically parsing `.tsx` files. Keep them separate.

---

### 2.3 Embedder (Voyage AI)

```python
# src/indexing/embedder.py
import voyageai
from tenacity import retry, stop_after_attempt, wait_exponential
from src.config.settings import settings

_client = voyageai.Client(api_key=settings.voyage_api_key)

MAX_TOKENS_PER_CHUNK = 16000   # voyage-code-2 context window

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def embed_batch(texts: list[str]) -> list[list[float]]:
    response = _client.embed(
        texts,
        model=settings.voyage_model,
        input_type="document",   # use "query" at query time!
    )
    return response.embeddings

def embed_chunks_batched(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Embed in batches to avoid rate limits."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = embed_batch(batch)
        all_embeddings.extend(embeddings)
        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks")
    return all_embeddings

def embed_query(query: str) -> list[float]:
    """Query embeddings use input_type='query' for better retrieval."""
    response = _client.embed(
        [query],
        model=settings.voyage_model,
        input_type="query",   # ← critical difference from document embedding
    )
    return response.embeddings[0]
```

> ⚠️ **Watch out:** `voyage-code-2` distinguishes between `input_type="document"` (indexing) and `input_type="query"` (querying). Using the wrong one at query time silently degrades retrieval quality.

---

### 2.4 Pinecone Store

```python
# src/indexing/pinecone_store.py
from pinecone import Pinecone, ServerlessSpec
from src.config.settings import settings
from src.ingestion.ast_parser import CodeChunk

VECTOR_DIM = 1536   # voyage-code-2 output dimension

_pc = Pinecone(api_key=settings.pinecone_api_key)

def get_or_create_index():
    names = _pc.list_indexes().names()
    if settings.pinecone_index_name not in names:
        print(f"Creating Pinecone index '{settings.pinecone_index_name}'...")
        _pc.create_index(
            name=settings.pinecone_index_name,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=settings.pinecone_environment),
        )
    return _pc.Index(settings.pinecone_index_name)

def _chunk_to_id(chunk: CodeChunk) -> str:
    """Stable ID — same chunk always gets same ID (for upsert idempotency)."""
    import hashlib
    key = f"{chunk.repo}::{chunk.file_path}::{chunk.name}::{chunk.start_line}"
    return hashlib.md5(key.encode()).hexdigest()

def upsert_chunks(chunks: list[CodeChunk], embeddings: list[list[float]]):
    index = get_or_create_index()

    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vectors.append({
            "id": _chunk_to_id(chunk),
            "values": embedding,
            "metadata": {
                "name":       chunk.name,
                "kind":       chunk.kind,
                "language":   chunk.language,
                "file_path":  chunk.file_path,
                "repo":       chunk.repo,
                "start_line": chunk.start_line,
                "end_line":   chunk.end_line,
                "code":       chunk.code[:4000],   # Pinecone metadata cap: 40KB total
                "docstring":  chunk.docstring[:500],
            }
        })

    # Pinecone recommends batches of 100
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i + batch_size])
        print(f"  Upserted {min(i + batch_size, len(vectors))}/{len(vectors)}")
    print(f"✅ Indexed {len(vectors)} chunks")

def query_index(embedding: list[float], top_k: int = 10,
                filter: dict = None) -> list[dict]:
    index = get_or_create_index()
    results = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter,          # e.g. {"language": {"$eq": "python"}}
    )
    return [
        {**match.metadata, "score": match.score, "id": match.id}
        for match in results.matches
    ]
```

> ⚠️ **Watch out:** Pinecone metadata values per vector are capped at ~40KB total. Storing full code in metadata is fine for small functions but truncate large ones. The embedding itself handles semantic recall — metadata is just for display.

---

### 2.5 Ingestion CLI Script

```python
# scripts/ingest.py
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.repo_loader import clone_repo, walk_repo
from src.ingestion.ast_parser import parse_file
from src.indexing.embedder import embed_chunks_batched
from src.indexing.pinecone_store import upsert_chunks
from rich.console import Console

console = Console()

def ingest(repo_url: str):
    console.rule("[bold cyan]🔍 Ingesting Repository[/bold cyan]")

    # Step 1: Clone
    repo_path, repo_name = clone_repo(repo_url)

    # Step 2: Walk
    files = walk_repo(repo_path, repo_name)

    # Step 3: Parse into chunks
    all_chunks = []
    for f in files:
        chunks = parse_file(f)
        all_chunks.extend(chunks)
    console.print(f"[green]✓ Parsed {len(all_chunks)} chunks from {len(files)} files[/green]")

    # Step 4: Embed
    console.print("[cyan]Embedding chunks (this may take a while)...[/cyan]")
    texts = [c.code for c in all_chunks]
    embeddings = embed_chunks_batched(texts)

    # Step 5: Upsert to Pinecone
    console.print("[cyan]Upserting to Pinecone...[/cyan]")
    upsert_chunks(all_chunks, embeddings)

    console.rule("[bold green]✅ Ingestion Complete[/bold green]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="GitHub URL to ingest")
    args = parser.parse_args()
    ingest(args.repo)
```

```bash
# Run it!
uv run python scripts/ingest.py --repo https://github.com/tiangolo/fastapi
```

### Phase 2 Watch-outs
- Large repos (100k+ lines) can take 5-10 mins and cost ~$0.50–$2 in Voyage API calls. Test on a small repo first.
- Duplicate function names across files are fine — the `_chunk_to_id` hash includes file path and line number, so they won't collide.
- If a file fails to parse (syntax errors are common in real repos), catch the exception and skip — don't crash the whole ingestion.

---

## 📦 Phase 3 — Retrieval Pipeline + Basic Q&A API

**Goal:** Build hybrid retrieval (dense + BM25 + re-rank) and expose it via a FastAPI endpoint that answers questions using Claude.

**✅ Testable outcome:** `curl -X POST http://localhost:8000/query -d '{"question": "how does authentication work?"}'` returns a cited answer.

---

### 3.1 BM25 Sparse Retriever

```python
# src/retrieval/sparse.py
import re
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self):
        self.chunks: list[dict] = []
        self.bm25: BM25Okapi | None = None

    def tokenize(self, text: str) -> list[str]:
        """Tokenize code: split on non-alphanum AND expand camelCase."""
        tokens = re.findall(r'[a-zA-Z0-9_]+', text)
        expanded = []
        for t in tokens:
            # camelCase → ["camel", "case"]
            parts = re.sub(r'([A-Z])', r' \1', t).lower().split()
            expanded.extend(parts)
        return expanded

    def build(self, chunks: list[dict]):
        self.chunks = chunks
        tokenized = [self.tokenize(c.get("code", "")) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        if not self.bm25:
            return []
        tokens = self.tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = scores.argsort()[-top_k:][::-1]
        return [
            {**self.chunks[i], "bm25_score": float(scores[i])}
            for i in top_idx if scores[i] > 0
        ]

    def save(self, path: str = "bm25_index.pkl"):
        with open(path, "wb") as f:
            pickle.dump((self.chunks, self.bm25), f)

    def load(self, path: str = "bm25_index.pkl"):
        if Path(path).exists():
            with open(path, "rb") as f:
                self.chunks, self.bm25 = pickle.load(f)
```

---

### 3.2 Hybrid Retriever with RRF

```python
# src/retrieval/hybrid.py
from src.retrieval.sparse import BM25Retriever
from src.indexing.embedder import embed_query
from src.indexing.pinecone_store import query_index

bm25 = BM25Retriever()   # Load at app startup

def reciprocal_rank_fusion(
    dense: list[dict],
    sparse: list[dict],
    k: int = 60
) -> list[dict]:
    scores: dict[str, float] = {}
    all_items: dict[str, dict] = {}

    for rank, item in enumerate(dense):
        key = f"{item['file_path']}::{item['name']}::{item['start_line']}"
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        all_items[key] = item

    for rank, item in enumerate(sparse):
        key = f"{item['file_path']}::{item['name']}::{item['start_line']}"
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        all_items[key] = item

    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [all_items[k] for k in sorted_keys]

def hybrid_search(
    query: str,
    top_k: int = 10,
    language_filter: str | None = None
) -> list[dict]:
    # Dense search via Pinecone
    qvec = embed_query(query)
    pinecone_filter = {"language": {"$eq": language_filter}} if language_filter else None
    dense_results = query_index(qvec, top_k=top_k * 2, filter=pinecone_filter)

    # Sparse search via BM25
    sparse_results = bm25.search(query, top_k=top_k * 2)

    # Fuse
    fused = reciprocal_rank_fusion(dense_results, sparse_results)
    return fused[:top_k]
```

---

### 3.3 Re-ranker

```python
# src/retrieval/reranker.py
from sentence_transformers import CrossEncoder

_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    if not candidates:
        return []
    pairs = [(query, c.get("code", "")[:512]) for c in candidates]
    scores = _model.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [item for _, item in ranked[:top_k]]
```

---

### 3.4 Claude Q&A

```python
# src/api/routes/query.py
import anthropic
from fastapi import APIRouter
from pydantic import BaseModel
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import rerank
from src.config.settings import settings

router = APIRouter()
client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

class QueryRequest(BaseModel):
    question: str
    language: str | None = None   # optional filter: "python" | "typescript"
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]

SYSTEM_PROMPT = """You are an expert code assistant. You help developers understand codebases.

Rules:
- Answer ONLY using the provided code context
- Always cite the specific file and function name when referencing code
- If the answer is not in the context, say "I couldn't find this in the indexed codebase"
- Be precise and technical — your audience is software engineers
- Format code snippets with markdown code blocks"""

@router.post("/query", response_model=QueryResponse)
async def query_codebase(req: QueryRequest):
    # Retrieve
    candidates = hybrid_search(req.question, top_k=req.top_k * 2, language_filter=req.language)
    top_chunks = rerank(req.question, candidates, top_k=req.top_k)

    # Build context
    context_parts = []
    for i, chunk in enumerate(top_chunks):
        context_parts.append(
            f"### [{i+1}] `{chunk['file_path']}` — `{chunk['name']}()` (lines {chunk['start_line']}–{chunk['end_line']})\n"
            f"```{chunk['language']}\n{chunk['code']}\n```"
        )
    context = "\n\n".join(context_parts)

    # Ask Claude
    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"## Code Context\n\n{context}\n\n## Question\n\n{req.question}"
        }]
    )

    sources = [{
        "file": c["file_path"],
        "function": c["name"],
        "lines": f"{c['start_line']}–{c['end_line']}",
        "language": c["language"],
    } for c in top_chunks]

    return QueryResponse(
        answer=response.content[0].text,
        sources=sources,
    )
```

---

### 3.5 FastAPI App

```python
# src/api/main.py
from fastapi import FastAPI
from src.api.routes import query, debug

app = FastAPI(
    title="Codebase RAG",
    description="Ask questions about your codebase",
    version="0.1.0",
)

app.include_router(query.router, prefix="/api/v1", tags=["Query"])
app.include_router(debug.router, prefix="/api/v1", tags=["Debug"])

@app.get("/health")
def health():
    return {"status": "ok"}
```

```bash
uv run uvicorn src.api.main:app --reload --port 8000
```

```bash
# Test it
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "how does request validation work in FastAPI?"}'
```

### Phase 3 Watch-outs
- The cross-encoder re-ranker loads a ~85MB model on startup. This is fine in dev but consider lazy-loading in production.
- BM25 index is built from chunks loaded in memory. For >500k chunks, this gets heavy — switch to Elasticsearch for sparse search at scale.
- Claude's context window is large but not infinite. If `top_k=5` and each chunk is 200 lines, you may hit limits. Monitor token count with `tiktoken`.

---

## 📦 Phase 4 — Auto-Debugger Agent

**Goal:** Accept a stack trace → parse it → retrieve relevant code → use Claude's tool-use to iteratively diagnose the root cause and suggest a fix.

**✅ Testable outcome:** POST a real Python or JS stack trace → get root cause analysis with the exact function that caused it, plus a suggested fix diff.

---

### 4.1 Stack Trace Parser (Python + JS/TS)

```python
# src/debugger/error_parser.py
import re
from dataclasses import dataclass, field

@dataclass
class StackFrame:
    file: str
    line: int
    function: str
    snippet: str = ""

@dataclass
class ParsedError:
    language: str
    error_type: str
    message: str
    frames: list[StackFrame] = field(default_factory=list)

def parse_python(tb: str) -> ParsedError:
    lines = tb.strip().splitlines()
    error_line = lines[-1]
    parts = error_line.split(":", 1)
    etype = parts[0].strip()
    emsg = parts[1].strip() if len(parts) > 1 else ""

    frames = []
    pattern = re.compile(r'File "(.+?)", line (\d+), in (.+)')
    for i, line in enumerate(lines):
        m = pattern.search(line)
        if m:
            snippet = lines[i + 1].strip() if i + 1 < len(lines) else ""
            frames.append(StackFrame(m.group(1), int(m.group(2)), m.group(3), snippet))

    return ParsedError("python", etype, emsg, frames)

def parse_javascript(tb: str) -> ParsedError:
    lines = tb.strip().splitlines()
    error_line = lines[0]
    parts = error_line.split(":", 1)
    etype = parts[0].strip()
    emsg = parts[1].strip() if len(parts) > 1 else ""

    frames = []
    # Format: "    at functionName (file.js:42:10)"
    pattern = re.compile(r'at (.+?) \((.+?):(\d+):\d+\)')
    for line in lines[1:]:
        m = pattern.search(line)
        if m:
            frames.append(StackFrame(m.group(2), int(m.group(3)), m.group(1)))

    return ParsedError("javascript", etype, emsg, frames)

def parse_traceback(tb: str) -> ParsedError:
    if "Traceback (most recent call last)" in tb:
        return parse_python(tb)
    elif " at " in tb and ("Error:" in tb or "Exception:" in tb):
        return parse_javascript(tb)
    else:
        # Generic fallback
        return ParsedError("unknown", "UnknownError", tb[:200])
```

---

### 4.2 Debug Agent with Claude Tool Use

```python
# src/debugger/agent.py
import json
import anthropic
from src.debugger.error_parser import parse_traceback, ParsedError
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import rerank
from src.config.settings import settings

client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

TOOLS = [
    {
        "name": "search_codebase",
        "description": "Search the codebase for code relevant to the error. Use this to find the source of a failing function, trace a call chain, or look up how something is implemented.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query (e.g. 'authenticate user function', 'database connection pool')"
                },
                "language": {
                    "type": "string",
                    "enum": ["python", "javascript", "typescript"],
                    "description": "Filter by language (optional)"
                }
            },
            "required": ["query"]
        }
    }
]

SYSTEM_PROMPT = """You are an expert debugger. Given a stack trace and access to the codebase, your job is to:

1. Identify the ROOT CAUSE of the error (not just the surface symptom)
2. Explain clearly WHY this error occurs
3. Suggest a specific code fix

You have a tool `search_codebase` to look up any function or module.
Use it proactively — search for functions mentioned in the stack trace, then search for their dependencies.

Format your final answer with:
- **Root Cause:** one-sentence summary
- **Why it happens:** detailed explanation  
- **Suggested Fix:** code block showing the corrected code
- **Files affected:** list of files"""

def _do_search(query: str, language: str | None = None) -> str:
    candidates = hybrid_search(query, top_k=6, language_filter=language)
    top = rerank(query, candidates, top_k=3)
    if not top:
        return "No relevant code found."

    parts = []
    for c in top:
        parts.append(
            f"### `{c['file_path']}` — `{c['name']}` (lines {c['start_line']}–{c['end_line']})\n"
            f"```{c['language']}\n{c['code'][:1500]}\n```"
        )
    return "\n\n".join(parts)

def debug_traceback(traceback: str) -> dict:
    error = parse_traceback(traceback)

    # Initial context: search for each frame function
    initial_queries = [f.function for f in error.frames[-3:]] + [
        f"{error.error_type} {error.message}"
    ]
    initial_context = "\n\n".join([_do_search(q) for q in initial_queries[:3]])

    messages = [{
        "role": "user",
        "content": f"""## Stack Trace
```
{traceback}
```

## Initial Code Context
{initial_context}

Please diagnose this error. Use `search_codebase` if you need more context.
"""
    }]

    # Agentic loop — max 4 iterations
    for iteration in range(4):
        response = client.messages.create(
            model=settings.claude_model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Append Claude's response to history
        messages.append({"role": "assistant", "content": response.content})

        # If Claude is done (no more tool calls)
        if response.stop_reason == "end_turn":
            final_text = next(
                (b.text for b in response.content if hasattr(b, "text")), ""
            )
            return {
                "diagnosis": final_text,
                "error": {
                    "type": error.error_type,
                    "message": error.message,
                    "language": error.language,
                },
                "iterations": iteration + 1,
            }

        # Handle tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                args = block.input
                result = _do_search(args["query"], args.get("language"))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        messages.append({"role": "user", "content": tool_results})

    return {"diagnosis": "Max iterations reached", "error": vars(error), "iterations": 4}
```

---

### 4.3 Debug Route

```python
# src/api/routes/debug.py
from fastapi import APIRouter
from pydantic import BaseModel
from src.debugger.agent import debug_traceback

router = APIRouter()

class DebugRequest(BaseModel):
    traceback: str

@router.post("/debug")
async def debug(req: DebugRequest):
    return debug_traceback(req.traceback)
```

```bash
# Test with a real traceback
curl -X POST http://localhost:8000/api/v1/debug \
  -H "Content-Type: application/json" \
  -d '{
    "traceback": "Traceback (most recent call last):\n  File \"/app/auth/views.py\", line 42, in login\n    user = authenticate_user(db, form.username, form.password)\n  File \"/app/auth/utils.py\", line 18, in authenticate_user\n    return db.query(User).filter(User.email == username).first()\nAttributeError: \"NoneType\" object has no attribute \"query\""
  }'
```

### Phase 4 Watch-outs
- Claude's tool-use agentic loop can call `search_codebase` 10+ times if you don't cap it. `max_iterations=4` is a safe ceiling.
- Stack trace file paths are often absolute (`/app/auth/views.py`). Strip the environment prefix so they match your indexed relative paths.
- For JS/TS, minified stack traces are unreadable — document that source maps need to be resolved first before passing to the debugger.

---

## 📦 Phase 5 — Evaluation + Incremental Indexing

**Goal:** Measure retrieval quality objectively, and handle repo updates without full re-ingestion.

**✅ Testable outcome:** `uv run python eval/run_eval.py` prints a RAGAS scorecard. Pushing a commit to your test repo triggers partial re-indexing via a script.

---

### 5.1 Build Evaluation Dataset

```json
// eval/test_queries.json
[
  {
    "question": "How does FastAPI handle request body validation?",
    "ground_truth": "FastAPI uses Pydantic models for request body validation. When you declare a parameter with a Pydantic model type, FastAPI automatically validates incoming JSON against the schema.",
    "relevant_files": ["fastapi/routing.py", "fastapi/dependencies/utils.py"]
  },
  {
    "question": "Where is the OpenAPI schema generated?",
    "ground_truth": "The OpenAPI schema is generated in fastapi/openapi/utils.py using the get_openapi function which collects route information.",
    "relevant_files": ["fastapi/openapi/utils.py"]
  }
]
```

### 5.2 RAGAS Evaluation

```python
# eval/run_eval.py
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import rerank
from src.api.routes.query import client, SYSTEM_PROMPT, settings

test_cases = json.loads(Path("eval/test_queries.json").read_text())

rows = []
for tc in test_cases:
    q = tc["question"]
    candidates = hybrid_search(q, top_k=10)
    chunks = rerank(q, candidates, top_k=5)
    contexts = [c["code"] for c in chunks]

    context_str = "\n\n".join([
        f"# {c['file_path']}\n{c['code']}" for c in chunks
    ])
    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {q}"}]
    )
    answer = response.content[0].text

    rows.append({
        "question": q,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": tc["ground_truth"],
    })

dataset = Dataset.from_list(rows)
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
print(result)
```

```bash
uv run python eval/run_eval.py
```

### 5.3 Incremental Re-indexing Script

```python
# scripts/reindex_changed.py
"""
Run this after pulling new commits to only re-index changed files.
Usage: uv run python scripts/reindex_changed.py --repo ./path/to/repo --since HEAD~1
"""
import subprocess, sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.ast_parser import parse_file
from src.indexing.embedder import embed_chunks_batched
from src.indexing.pinecone_store import upsert_chunks, get_or_create_index, _chunk_to_id

SUPPORTED = {".py", ".js", ".ts", ".tsx", ".jsx"}

def get_changed_files(repo_path: str, since: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", since],
        cwd=repo_path, capture_output=True, text=True
    )
    return [f for f in result.stdout.strip().splitlines()
            if Path(f).suffix in SUPPORTED]

def reindex(repo_path: str, since: str, repo_name: str):
    changed = get_changed_files(repo_path, since)
    print(f"Changed files: {changed}")

    all_chunks = []
    for rel_path in changed:
        full = Path(repo_path) / rel_path
        if not full.exists():
            continue
        file_info = {
            "path": rel_path,
            "content": full.read_text(errors="ignore"),
            "language": {".py": "python", ".js": "javascript",
                         ".ts": "typescript", ".tsx": "typescript"}.get(full.suffix, "unknown"),
            "repo": repo_name,
        }
        all_chunks.extend(parse_file(file_info))

    if not all_chunks:
        print("No chunks to re-index.")
        return

    embeddings = embed_chunks_batched([c.code for c in all_chunks])
    upsert_chunks(all_chunks, embeddings)   # Pinecone upsert is idempotent via stable IDs
    print(f"✅ Re-indexed {len(all_chunks)} chunks from {len(changed)} files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--since", default="HEAD~1")
    parser.add_argument("--name", default="myrepo")
    args = parser.parse_args()
    reindex(args.repo, args.since, args.name)
```

```bash
uv run python scripts/reindex_changed.py --repo ./fastapi --since HEAD~1 --name fastapi
```

### Phase 5 Watch-outs
- RAGAS scores are relative — a `faithfulness` of 0.85+ is good for code Q&A. Don't chase 1.0.
- Start with 20 hand-crafted eval questions. Quality beats quantity for small evals.
- Pinecone `upsert` is idempotent (same ID = overwrite), so re-indexing a file is safe to run multiple times without duplicates.

---

## 🗺️ Phase Summary

| Phase | Command to test | Key output |
|---|---|---|
| **1 — Scaffold** | `uv run python scripts/smoke_test.py` | All 3 services respond |
| **2 — Ingestion** | `uv run python scripts/ingest.py --repo <url>` | Chunks in Pinecone dashboard |
| **3 — Q&A API** | `curl POST /api/v1/query` | Cited answers from Claude |
| **4 — Debugger** | `curl POST /api/v1/debug` | Root cause + fix suggestion |
| **5 — Eval** | `uv run python eval/run_eval.py` | RAGAS scorecard |

---

## 🔑 Key Commands Reference

```bash
# Install deps
uv sync

# Ingest a repo
uv run python scripts/ingest.py --repo https://github.com/tiangolo/fastapi

# Start API server
uv run uvicorn src.api.main:app --reload --port 8000

# Re-index changed files
uv run python scripts/reindex_changed.py --repo ./local-repo --since HEAD~1

# Run evaluation
uv run python eval/run_eval.py

# Lint
uv run ruff check src/
```