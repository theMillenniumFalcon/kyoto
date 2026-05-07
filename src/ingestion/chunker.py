"""
Token-budget chunker.

Sits between ast_parser.py and the embedder. If a parsed CodeChunk exceeds
MAX_CHUNK_TOKENS it is split into overlapping sub-chunks along line boundaries,
each prefixed with the original import header so every sub-chunk is
self-contained for embedding.

voyage-code-2 context window: 16,000 tokens.
We target MAX_CHUNK_TOKENS=512 per the settings default — this keeps each
vector focused on a tight semantic unit and stays well within Voyage's limit.
"""
import dataclasses

import tiktoken

from src.config.settings import settings
from src.ingestion.ast_parser import CodeChunk

# tiktoken doesn't have a voyage-code-2 tokeniser; cl100k_base (GPT-4)
# is a close-enough approximation for token counting purposes.
_enc = tiktoken.get_encoding("cl100k_base")

MAX_TOKENS = settings.max_chunk_tokens        # default 512
OVERLAP_TOKENS = settings.chunk_overlap_tokens  # default 64


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text, disallowed_special=()))


def _split_lines_by_budget(
    lines: list[str],
    max_tokens: int,
    overlap_tokens: int,
) -> list[list[str]]:
    """
    Greedily pack lines into windows that fit within max_tokens.
    Each window overlaps with the previous by overlap_tokens worth of lines.
    """
    windows: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    overlap_buffer: list[str] = []

    for line in lines:
        line_tokens = _count_tokens(line + "\n")

        # If a single line exceeds budget, hard-split it (rare — very long strings)
        if line_tokens > max_tokens:
            if current:
                windows.append(current)
            windows.append([line])
            current = []
            current_tokens = 0
            overlap_buffer = []
            continue

        if current_tokens + line_tokens > max_tokens:
            windows.append(current)
            # Seed next window with overlap lines from the tail of the current one
            current = list(overlap_buffer)
            current_tokens = sum(_count_tokens(l + "\n") for l in current)
            overlap_buffer = []

        current.append(line)
        current_tokens += line_tokens

        # Maintain a rolling overlap buffer (keep adding until we exceed overlap budget)
        overlap_buffer.append(line)
        while sum(_count_tokens(l + "\n") for l in overlap_buffer) > overlap_tokens:
            overlap_buffer.pop(0)

    if current:
        windows.append(current)

    return windows


def split_chunk(chunk: CodeChunk) -> list[CodeChunk]:
    """
    If `chunk` fits within MAX_TOKENS, return it as-is (list of one).
    Otherwise split it into overlapping sub-chunks and return the list.
    Each sub-chunk carries the original import header for embedding context.
    """
    if _count_tokens(chunk.code) <= MAX_TOKENS:
        return [chunk]

    # Separate the import header (prepended by ast_parser) from the body
    import_header = "\n".join(chunk.imports) + "\n\n" if chunk.imports else ""
    # The body is everything after the header
    body = chunk.code[len(import_header):]
    body_lines = body.splitlines()

    # Budget for the body = max_tokens minus the header cost
    header_tokens = _count_tokens(import_header)
    body_budget = max(MAX_TOKENS - header_tokens, 64)  # at least 64 tokens of body

    windows = _split_lines_by_budget(body_lines, body_budget, OVERLAP_TOKENS)

    sub_chunks: list[CodeChunk] = []
    for i, window_lines in enumerate(windows):
        window_code = import_header + "\n".join(window_lines)
        # Approximate line numbers for the sub-chunk
        lines_before = sum(len(w) for w in windows[:i])
        start = chunk.start_line + lines_before
        end = start + len(window_lines) - 1

        sub_chunks.append(
            dataclasses.replace(
                chunk,
                name=f"{chunk.name}__part{i + 1}",
                code=window_code,
                start_line=start,
                end_line=end,
            )
        )

    return sub_chunks


def chunk_files(chunks: list[CodeChunk]) -> list[CodeChunk]:
    """
    Run all parsed chunks through the token-budget splitter.
    Drop-in between parse_file() and embed_chunks_batched().

    Usage in ingest.py:
        from src.ingestion.chunker import chunk_files
        all_chunks = chunk_files(all_chunks)
    """
    result: list[CodeChunk] = []
    oversized = 0
    for c in chunks:
        split = split_chunk(c)
        if len(split) > 1:
            oversized += 1
        result.extend(split)

    if oversized:
        print(f"  [chunker] Split {oversized} oversized chunks → {len(result)} total")

    return result