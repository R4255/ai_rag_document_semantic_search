from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

from .chunking import chunk_text
from .fusion import reciprocal_rank_fusion
from .index_bm25 import BM25Index
from .index_dense import DenseIndex
from .rerankers import Reranker
from .types import Hit

MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")


@dataclass
class SearchResult:
    hits: list[Hit]
    timings_ms: dict[str, float]


@contextmanager
def _timed(store: dict[str, float], name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[name] = (time.perf_counter() - t0) * 1000.0


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Index,
        dense: DenseIndex,
        reranker: Reranker | None = None,
        rrf_k: int = 60,
        candidates: int = 50,
        weights: tuple[float, float] = (1.0, 1.0),
        max_words: int = 120,
        overlap: int = 20,
    ):
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.candidates = candidates
        self.weights = list(weights)
        self.max_words = max_words
        self.overlap = overlap

    def add_documents(self, docs: list[dict]) -> int:
        """docs: [{"id": str, "text": str, "meta": {...}?}]. Returns number of chunks indexed."""
        chunks = []
        for d in docs:
            chunks.extend(
                chunk_text(str(d["id"]), d["text"], self.max_words, self.overlap, d.get("meta"))
            )
        self.bm25.add(chunks)
        self.dense.add(chunks)
        return len(chunks)

    def search(self, query: str, k: int = 10, mode: str = "hybrid_rerank") -> SearchResult:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "hybrid_rerank" and self.reranker is None:
            raise ValueError("mode 'hybrid_rerank' requires a reranker")

        t: dict[str, float] = {}
        t0 = time.perf_counter()
        n = max(self.candidates, k)

        if mode == "bm25":
            with _timed(t, "bm25"):
                hits = self.bm25.search(query, k)
        elif mode == "dense":
            with _timed(t, "dense"):
                hits = self.dense.search(query, k)
        else:
            with _timed(t, "bm25"):
                lexical = self.bm25.search(query, n)
            with _timed(t, "dense"):
                semantic = self.dense.search(query, n)
            with _timed(t, "fuse"):
                fused = reciprocal_rank_fusion([lexical, semantic], self.rrf_k, self.weights)
            if mode == "hybrid":
                hits = fused[:k]
            else:
                with _timed(t, "rerank"):
                    hits = self.reranker.rerank(query, fused[:n], k)  # type: ignore[union-attr]

        t["total"] = (time.perf_counter() - t0) * 1000.0
        return SearchResult(hits=hits, timings_ms=t)

    @property
    def available_modes(self) -> tuple[str, ...]:
        return MODES if self.reranker is not None else MODES[:3]
