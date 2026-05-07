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
        tokens = re.findall(r"[a-zA-Z0-9_]+", text)
        expanded = []
        for t in tokens:
            # camelCase → ["camel", "case"]
            parts = re.sub(r"([A-Z])", r" \1", t).lower().split()
            expanded.extend(parts)
        return expanded

    def build(self, chunks: list[dict]) -> None:
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
            for i in top_idx
            if scores[i] > 0
        ]

    def save(self, path: str = "bm25_index.pkl") -> None:
        with open(path, "wb") as f:
            pickle.dump((self.chunks, self.bm25), f)

    def load(self, path: str = "bm25_index.pkl") -> bool:
        """Load persisted index. Returns True if loaded, False if not found."""
        if Path(path).exists():
            with open(path, "rb") as f:
                self.chunks, self.bm25 = pickle.load(f)
            return True
        return False