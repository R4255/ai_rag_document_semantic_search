from __future__ import annotations

import numpy as np

from .embedders import Embedder
from .types import Chunk, Hit


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class DenseIndex:
    """Exact cosine search over an in-memory matrix. Swap for FAISS/HNSW past ~1M chunks."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.mat = np.zeros((0, embedder.dim), dtype=np.float32)

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vecs = _normalize(self.embedder.embed_documents([c.text for c in chunks]))
        self.mat = np.vstack([self.mat, vecs])
        self.chunks.extend(chunks)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        if not self.chunks:
            return []
        q = _normalize(self.embedder.embed_query(query)[None, :])[0]
        scores = self.mat @ q
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [Hit(self.chunks[i], float(scores[i]), "dense") for i in idx]
