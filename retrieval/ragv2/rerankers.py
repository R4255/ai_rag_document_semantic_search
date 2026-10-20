from __future__ import annotations

from typing import Protocol

from .index_bm25 import tokenize
from .types import Hit


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]: ...


class CrossEncoderReranker:
    """Real reranker. Needs `pip install sentence-transformers` and a model download."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        scores = self.model.predict([(query, h.chunk.text) for h in hits])
        order = sorted(range(len(hits)), key=lambda i: -float(scores[i]))[:top_k]
        return [Hit(hits[i].chunk, float(scores[i]), "rerank") for i in order]


class LexicalOverlapReranker:
    """Offline stand-in: fraction of query terms present, plus a bigram bonus.

    Exists to exercise the reranker interface in tests without model downloads.
    """

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        q = tokenize(query)
        if not q:
            return hits[:top_k]
        qset = set(q)
        qbi = set(zip(q, q[1:]))
        scored = []
        for h in hits:
            toks = tokenize(h.chunk.text)
            tset = set(toks)
            overlap = len(qset & tset) / len(qset)
            bigram = len(qbi & set(zip(toks, toks[1:]))) / len(qbi) if qbi else 0.0
            scored.append(Hit(h.chunk, overlap + 0.5 * bigram, "rerank"))
        scored.sort(key=lambda h: -h.score)
        return scored[:top_k]
