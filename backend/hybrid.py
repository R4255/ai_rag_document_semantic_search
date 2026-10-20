"""
Bridges the ragv2 retrieval core (../retrieval) with the LangChain FAISS store.

- FaissDenseIndex adapts a per-user FAISS store to ragv2's dense-index interface,
  so Gemini embeddings stay the semantic side of the search.
- build_user_retriever wraps it in a ragv2 HybridRetriever with an in-memory
  BM25 index, rebuilt from the FAISS docstore whenever a store is loaded.
"""
import os

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from ragv2 import Chunk, Hit, HybridRetriever, MODES
from ragv2.index_bm25 import BM25Index

SYSTEM_SOURCE = "system"

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


class FaissDenseIndex:
    """ragv2 dense-index interface over a LangChain FAISS store."""

    def __init__(self, vs: FAISS):
        self.vs = vs

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        docs = [Document(page_content=c.text, metadata=c.meta) for c in chunks]
        self.vs.add_documents(docs, ids=[c.id for c in chunks])

    def search(self, query: str, k: int = 10) -> list[Hit]:
        # +1 so the placeholder document never costs a real result slot
        results = self.vs.similarity_search_with_score(query, k=k + 1)
        hits = [
            Hit(_to_chunk(doc), -float(dist), "dense")  # L2 distance: lower is closer
            for doc, dist in results
            if doc.metadata.get("source") != SYSTEM_SOURCE
        ]
        return hits[:k]


def _to_chunk(doc: Document) -> Chunk:
    return Chunk(
        id=doc.metadata["chunk_id"],
        doc_id=doc.metadata.get("source", "Unknown"),
        text=doc.page_content,
        meta=doc.metadata,
    )


def build_user_retriever(vs: FAISS) -> HybridRetriever:
    """Wrap a FAISS store and backfill BM25 from the documents it already holds."""
    existing = []
    for doc_id, doc in vs.docstore._dict.items():
        # Stores created before hybrid search have no chunk_id; the docstore key is stable.
        doc.metadata.setdefault("chunk_id", doc_id)
        if doc.metadata.get("source") != SYSTEM_SOURCE:
            existing.append(_to_chunk(doc))

    bm25 = BM25Index()
    bm25.add(existing)
    return HybridRetriever(bm25, FaissDenseIndex(vs), reranker, candidates=CANDIDATES)


def add_chunks(retriever: HybridRetriever, chunks: list[Chunk]) -> None:
    """Index chunks into both sides (bypasses ragv2's word chunker; LangChain splits upstream)."""
    retriever.dense.add(chunks)
    retriever.bm25.add(chunks)
