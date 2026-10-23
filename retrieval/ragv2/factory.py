from __future__ import annotations

from .embedders import HashingEmbedder, SentenceTransformerEmbedder
from .index_bm25 import BM25Index
from .index_dense import DenseIndex
from .rerankers import CrossEncoderReranker, LexicalOverlapReranker
from .retriever import HybridRetriever


def build_retriever(
    embedder: str = "hashing", reranker: str = "lexical", cross_model: str | None = None, **kwargs
) -> HybridRetriever:
    """embedder: hashing | st.  reranker: none | lexical | cross (cross_model picks the HF model)."""
    if embedder == "hashing":
        emb = HashingEmbedder()
    elif embedder == "st":
        emb = SentenceTransformerEmbedder()
    else:
        raise ValueError(f"unknown embedder: {embedder}")

    if reranker == "none":
        rr = None
    elif reranker == "lexical":
        rr = LexicalOverlapReranker()
    elif reranker == "cross":
        rr = CrossEncoderReranker(cross_model) if cross_model else CrossEncoderReranker()
    else:
        raise ValueError(f"unknown reranker: {reranker}")

    return HybridRetriever(BM25Index(), DenseIndex(emb), rr, **kwargs)
