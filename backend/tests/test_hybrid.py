"""
Tests for the hybrid (BM25 + pgvector) retrieval path against a real Postgres.
Embeddings are a deterministic fake, so no model or API key is needed.

    podman run -d --name nexus_rag_pg -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=nexus_rag \
      -p 5434:5432 docker.io/pgvector/pgvector:pg16
    TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5434/nexus_rag pytest tests

Skipped when TEST_DATABASE_URL is not set.
"""
import asyncio
import os
import sys
import uuid

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEST_DB = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not TEST_DB, reason="TEST_DATABASE_URL not set")

import db  # noqa: E402
from providers import EMBED_DIM, embed_model_name  # noqa: E402
from rag_service import NO_TEXT_ERROR, RAGService  # noqa: E402

DOCS = {
    "kafka.txt": "Kafka partitions limit consumer parallelism within a consumer group.",
    "redis.txt": "Redis caches hot reads and reduces database round trips. Error code E4021 means eviction.",
}


@pytest.fixture(scope="module", autouse=True)
def database():
    db.init_db(TEST_DB)
    yield
    db.close_db()


def _new_service():
    svc = RAGService()
    svc.embeddings = DeterministicFakeEmbedding(size=EMBED_DIM)
    return svc


def _ingest(svc, user, tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return asyncio.run(svc.ingest_file(str(path), name, user))


@pytest.fixture
def user(tmp_path):
    """A fresh user with both DOCS ingested; their rows are removed afterwards."""
    uid = f"test-{uuid.uuid4()}"
    svc = _new_service()
    for name, text in DOCS.items():
        assert _ingest(svc, uid, tmp_path, name, text)["status"] == "success"
    yield uid, svc
    with db._conn() as conn:
        conn.execute("DELETE FROM documents WHERE user_id = %s", (uid,))


@pytest.mark.parametrize("mode", ["bm25", "hybrid"])
def test_exact_term_is_found(user, mode):
    uid, svc = user
    # Fake embeddings carry no meaning, so this only passes because BM25 is in the mix.
    res = asyncio.run(svc.search("E4021", uid, k=2, mode=mode))
    assert res.hits[0].chunk.meta["source"] == "redis.txt"
    assert "total" in res.timings_ms


def test_dense_returns_every_chunk_and_no_placeholder(user):
    uid, svc = user
    res = asyncio.run(svc.search("anything at all", uid, k=10, mode="dense"))
    assert sorted(h.chunk.meta["source"] for h in res.hits) == ["kafka.txt", "redis.txt"]


def test_survives_restart(user):
    uid, _ = user
    fresh = _new_service()  # empty memory, same database
    files = asyncio.run(fresh.get_ingested_files(uid))
    assert [f["filename"] for f in files] == list(DOCS)
    assert all(f["status"] == "success" and f["chunks"] == 1 for f in files)
    res = asyncio.run(fresh.search("consumer parallelism", uid, k=1, mode="bm25"))
    assert res.hits[0].chunk.meta["source"] == "kafka.txt"


def test_same_file_twice_is_skipped(user, tmp_path):
    uid, svc = user
    again = _ingest(svc, uid, tmp_path, "kafka-copy.txt", DOCS["kafka.txt"])
    assert again["status"] == "duplicate"
    assert len(asyncio.run(svc.get_ingested_files(uid))) == 2


def test_file_without_text_is_recorded_as_error(user, tmp_path):
    uid, svc = user
    out = _ingest(svc, uid, tmp_path, "blank.txt", "   \n  ")
    assert out == {"filename": "blank.txt", "status": "error", "error": NO_TEXT_ERROR}
    blank = [f for f in asyncio.run(svc.get_ingested_files(uid)) if f["filename"] == "blank.txt"][0]
    assert (blank["status"], blank["error"], blank["chunks"]) == ("error", NO_TEXT_ERROR, 0)


def test_users_are_isolated(user):
    _, svc = user
    res = asyncio.run(svc.search("kafka", f"someone-else-{uuid.uuid4()}", k=5, mode="hybrid"))
    assert res.hits == []


def test_document_chunks_come_back_in_reading_order(tmp_path):
    uid = f"test-{uuid.uuid4()}"
    svc = _new_service()
    words = lambda tag: " ".join([tag] * 180)  # ~1,000+ characters per paragraph
    text = "\n\n".join([words("alpha"), words("beta"), words("gamma"), words("omega")])
    assert _ingest(svc, uid, tmp_path, "long.txt", text)["status"] == "success"
    chunks = asyncio.run(svc.get_document_chunks(uid, "long.txt"))
    assert len(chunks) >= 4
    assert chunks[0]["text"].startswith("alpha") and chunks[-1]["text"].endswith("omega")
    assert all(c["page"] == 1 for c in chunks)
    assert asyncio.run(svc.get_document_chunks(uid, "missing.txt")) == []
    with db._conn() as conn:
        conn.execute("DELETE FROM documents WHERE user_id = %s", (uid,))


def _doc_id(svc, uid, filename):
    return next(f["id"] for f in asyncio.run(svc.get_ingested_files(uid)) if f["filename"] == filename)


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_deleted_document_is_gone_from_every_retriever(user, mode):
    uid, svc = user
    asyncio.run(svc.search("E4021", uid, k=5, mode=mode))  # build the in-memory BM25 first
    assert asyncio.run(svc.delete_document(uid, _doc_id(svc, uid, "redis.txt"))) == "deleted"
    assert [f["filename"] for f in asyncio.run(svc.get_ingested_files(uid))] == ["kafka.txt"]
    res = asyncio.run(svc.search("E4021 redis eviction", uid, k=5, mode=mode))
    assert all(h.chunk.meta["source"] != "redis.txt" for h in res.hits)
    with db._conn() as conn:
        assert conn.execute("SELECT count(*) FROM chunks WHERE user_id = %s AND source = 'redis.txt'",
                            (uid,)).fetchone()[0] == 0


def test_upload_after_delete_is_searchable(user, tmp_path):
    uid, svc = user
    asyncio.run(svc.delete_document(uid, _doc_id(svc, uid, "redis.txt")))
    # The same file can come back, and the rebuilt BM25 finds it
    assert _ingest(svc, uid, tmp_path, "redis.txt", DOCS["redis.txt"])["status"] == "success"
    res = asyncio.run(svc.search("E4021", uid, k=1, mode="bm25"))
    assert res.hits[0].chunk.meta["source"] == "redis.txt"


def test_cannot_delete_another_users_document(user):
    uid, svc = user
    doc = _doc_id(svc, uid, "kafka.txt")
    assert asyncio.run(svc.delete_document(f"someone-else-{uuid.uuid4()}", doc)) == "missing"
    assert len(asyncio.run(svc.get_ingested_files(uid))) == 2


def test_document_still_indexing_is_not_deleted(user):
    uid, svc = user
    doc = db.create_document(uid, embed_model_name(), "pending.txt", "sha-pending")  # status 'processing'
    assert asyncio.run(svc.delete_document(uid, doc)) == "processing"
    assert "pending.txt" in [f["filename"] for f in asyncio.run(svc.get_ingested_files(uid))]


def test_delete_endpoint(user, monkeypatch):
    from fastapi.testclient import TestClient
    import app as app_module
    from cache_manager import semantic_cache

    uid, svc = user
    monkeypatch.setattr(app_module, "rag_service", svc)
    client = TestClient(app_module.app)  # no `with`: skip the lifespan, which loads real models
    asyncio.run(semantic_cache.set(f"{uid}:what is redis", {"answer": "x", "citations": []}, owner=uid))
    doc = _doc_id(svc, uid, "redis.txt")

    assert client.delete("/api/documents/not-a-uuid", headers={"X-User-ID": uid}).status_code == 422
    assert client.delete(f"/api/documents/{doc}", headers={"X-User-ID": "intruder"}).status_code == 404
    r = client.delete(f"/api/documents/{doc}", headers={"X-User-ID": uid})
    assert r.status_code == 200 and r.json() == {"status": "deleted", "id": doc}
    assert asyncio.run(semantic_cache.get(f"{uid}:what is redis", owner=uid)) is None
    assert client.delete(f"/api/documents/{doc}", headers={"X-User-ID": uid}).status_code == 404


def test_rate_limit_errors_get_a_friendly_message():
    from rag_service import _friendly_error
    raw = Exception("429 Too Many Requests. {'error': {'status': 'RESOURCE_EXHAUSTED'}}")
    assert "busy" in _friendly_error(raw) and "429" not in _friendly_error(raw)
    assert "try again" in _friendly_error(Exception("boom")).lower()


def test_concurrent_first_use_builds_one_retriever():
    """An upload and a search arriving together must share one retriever; a second copy
    would overwrite the first and its BM25 would never see the upload's chunks."""
    svc = _new_service()
    uid = f"test-{uuid.uuid4()}"

    async def both():
        return await asyncio.gather(svc.get_retriever(uid), svc.get_retriever(uid))

    first, second = asyncio.run(both())
    assert first is second is svc.retrievers[uid]


def test_failed_ingest_leaves_nothing_behind(tmp_path):
    uid = f"test-{uuid.uuid4()}"
    svc = _new_service()
    svc.embeddings = DeterministicFakeEmbedding(size=EMBED_DIM + 1)  # wrong size → insert rejected
    out = _ingest(svc, uid, tmp_path, "kafka.txt", DOCS["kafka.txt"])
    assert out["status"] == "error"
    assert asyncio.run(svc.get_ingested_files(uid)) == []
