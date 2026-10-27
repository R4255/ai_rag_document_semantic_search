"""
Bridges the ragv2 retrieval core (../retrieval) with Postgres + pgvector.

- PgVectorIndex adapts one user's rows in the chunks table to ragv2's dense-index
  interface, so the configured embedding model stays the semantic side of the search.
- build_user_retriever wraps it in a ragv2 HybridRetriever with an in-memory BM25
  index, built from the same rows the first time this process sees the user.
"""
import os

import numpy as np

import db
from providers import EMBED_DIM
from ragv2 import Chunk, Hit, HybridRetriever, MODES
from ragv2.index_bm25 import BM25Index

RETRIEVAL_MODE = os.getenv("RAG_RETRIEVAL_MODE", "hybrid")
RERANKER = os.getenv("RAG_RERANKER", "none")  # none | cross
CANDIDATES = int(os.getenv("RAG_CANDIDATES", "30"))

if RETRIEVAL_MODE not in MODES:
    raise ValueError(f"RAG_RETRIEVAL_MODE must be one of {MODES}")


def _load_reranker():
    if RERANKER == "none":
        return None
    if RERANKER == "cross":
        from ragv2.rerankers import CrossEncoderReranker  # needs sentence-transformers

        return CrossEncoderReranker()
    raise ValueError("RAG_RERANKER must be 'none' or 'cross'")


reranker = _load_reranker()

if RETRIEVAL_MODE == "hybrid_rerank" and reranker is None:
    raise ValueError("RAG_RETRIEVAL_MODE=hybrid_rerank requires RAG_RERANKER=cross")


def _chunk(chunk_id, source: str, page, content: str) -> Chunk:
    cid = str(chunk_id)
    return Chunk(id=cid, doc_id=source, text=content, meta={"source": source, "page": page, "chunk_id": cid})


def _vector(values) -> np.ndarray:
    vec = np.asarray(values, dtype=np.float32)
    if vec.shape[0] != EMBED_DIM:
        raise ValueError(f"embedding has {vec.shape[0]} numbers; the chunks table expects {EMBED_DIM}")
    return vec


class PgVectorIndex:
    """ragv2 dense-index interface over one user's rows in the pgvector chunks table."""

    def __init__(self, embeddings, user_id: str, embed_model: str):
        self.embeddings = embeddings
        self.user_id = user_id
        self.embed_model = embed_model

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = self.embeddings.embed_documents([c.text for c in chunks])
        db.insert_chunks([
            (c.id, c.meta["document_id"], self.user_id, self.embed_model,
             c.meta.get("source"), c.meta.get("page"), c.text, _vector(v))
            for c, v in zip(chunks, vectors)
        ])

    def search(self, query: str, k: int = 10) -> list[Hit]:
        rows = db.search_chunks(self.user_id, self.embed_model, _vector(self.embeddings.embed_query(query)), k)
        # <=> is cosine distance; 1 - distance is cosine similarity, so higher is better
        return [Hit(_chunk(cid, src, page, text), 1.0 - float(dist), "dense") for cid, src, page, text, dist in rows]


def build_user_retriever(embeddings, user_id: str, embed_model: str) -> HybridRetriever:
    """Load this user's chunks once and build BM25 over them; dense search stays in Postgres."""
    bm25 = BM25Index()
    bm25.add([_chunk(*row) for row in db.load_chunks(user_id, embed_model)])
    return HybridRetriever(bm25, PgVectorIndex(embeddings, user_id, embed_model), reranker, candidates=CANDIDATES)


def add_chunks(retriever: HybridRetriever, chunks: list[Chunk]) -> None:
    """Index chunks into both sides (bypasses ragv2's word chunker; LangChain splits upstream)."""
    retriever.dense.add(chunks)
    retriever.bm25.add(chunks)
