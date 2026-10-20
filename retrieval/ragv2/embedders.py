from __future__ import annotations

import zlib
from typing import Protocol

import numpy as np

from .index_bm25 import tokenize


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic offline embedder using signed feature hashing of unigrams and bigrams.

    This is lexical, NOT semantic. It exists so tests and demos run with no model
    download. Use SentenceTransformerEmbedder for real quality numbers.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        toks = tokenize(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for f in feats:
            h = zlib.crc32(f.encode())
            sign = 1.0 if (h >> 31) & 1 else -1.0
            vec[h % self.dim] += sign
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._embed(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)


class SentenceTransformerEmbedder:
    """Real embedder. Needs `pip install sentence-transformers` and a model download."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        query_prefix: str = "Represent this sentence for searching relevant passages: ",
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.query_prefix = query_prefix

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True, batch_size=64), dtype=np.float32
        )

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(
            self.model.encode(self.query_prefix + text, normalize_embeddings=True), dtype=np.float32
        )
