from __future__ import annotations

from .embedders import HashingEmbedder, SentenceTransformerEmbedder
from .index_bm25 import BM25Index
from .index_dense import DenseIndex
from .rerankers import CrossEncoderReranker, LexicalOverlapReranker
from .retriever import HybridRetriever


def build_retriever(embedder: str = "hashing", reranker: str = "lexical", **kwargs) -> HybridRetriever:
    """embedder: hashing | st.  reranker: none | lexical | cross."""
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
        rr = CrossEncoderReranker()
    else:
        raise ValueError(f"unknown reranker: {reranker}")

    return HybridRetriever(BM25Index(), DenseIndex(emb), rr, **kwargs)
