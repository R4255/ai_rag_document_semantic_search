"""
Offline tests for the hybrid (BM25 + FAISS) retrieval path.
Gemini embeddings are swapped for a deterministic fake, so no API key is needed.

    cd backend && pytest tests
"""
import asyncio
import os
import sys

import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import DeterministicFakeEmbedding

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag_service as rag_module  # noqa: E402
from rag_service import RAGService  # noqa: E402

DOCS = {
    "kafka.txt": "Kafka partitions limit consumer parallelism within a consumer group.",
    "redis.txt": "Redis caches hot reads and reduces database round trips. Error code E4021 means eviction.",
}


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_module, "VECTOR_STORE_PATH", str(tmp_path / "faiss_store"))
    svc = RAGService()
    svc.embeddings = DeterministicFakeEmbedding(size=64)
    for name, text in DOCS.items():
        path = tmp_path / name
        path.write_text(text)
        record = asyncio.run(svc.ingest_file(str(path), name, "u1"))
        assert record["status"] == "success"
    return svc


@pytest.mark.parametrize("mode", ["bm25", "hybrid"])
def test_exact_term_is_found(service, mode):
    # Fake embeddings carry no meaning, so this only passes because BM25 is in the mix.
    res = asyncio.run(service.search("E4021", "u1", k=2, mode=mode))
    assert res.hits[0].chunk.meta["source"] == "redis.txt"
    assert "total" in res.timings_ms


def test_placeholder_never_returned(service):
    for mode in ["bm25", "dense", "hybrid"]:
        res = asyncio.run(service.search("user space initialization", "u1", k=10, mode=mode))
        assert all(h.chunk.meta.get("source") != "system" for h in res.hits)
        if mode != "bm25":  # bm25 matches no terms here; dense always returns neighbours
            assert len(res.hits) == 2


def test_bm25_rebuilt_after_restart(service, tmp_path, monkeypatch):
    fresh = RAGService()
    fresh.embeddings = service.embeddings
    res = asyncio.run(fresh.search("consumer parallelism", "u1", k=1, mode="bm25"))
    assert res.hits[0].chunk.meta["source"] == "kafka.txt"
    assert [f["filename"] for f in fresh.get_ingested_files("u1")] == list(DOCS)


def test_legacy_store_without_chunk_ids(tmp_path, monkeypatch):
    """Stores saved before hybrid search have no chunk_id metadata."""
    monkeypatch.setattr(rag_module, "VECTOR_STORE_PATH", str(tmp_path / "faiss_store"))
    emb = DeterministicFakeEmbedding(size=64)
    vs = FAISS.from_texts(
        ["User space initialization.", DOCS["kafka.txt"]],
        emb,
        metadatas=[{"source": "system", "page": 0}, {"source": "kafka.txt", "page": 1}],
    )
    svc = RAGService()
    svc.embeddings = emb
    vs.save_local(svc._get_user_store_path("u1"))
    for mode in ["bm25", "dense", "hybrid"]:
        res = asyncio.run(svc.search("kafka consumer", "u1", k=3, mode=mode))
        assert [h.chunk.meta["source"] for h in res.hits] == ["kafka.txt"]


def test_users_are_isolated(service):
    res = asyncio.run(service.search("kafka", "someone-else", k=5, mode="hybrid"))
    assert res.hits == []
