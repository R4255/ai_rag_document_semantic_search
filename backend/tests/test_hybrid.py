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
from providers import EMBED_DIM  # noqa: E402
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


def test_failed_ingest_leaves_nothing_behind(tmp_path):
    uid = f"test-{uuid.uuid4()}"
    svc = _new_service()
    svc.embeddings = DeterministicFakeEmbedding(size=EMBED_DIM + 1)  # wrong size → insert rejected
    out = _ingest(svc, uid, tmp_path, "kafka.txt", DOCS["kafka.txt"])
    assert out["status"] == "error"
    assert asyncio.run(svc.get_ingested_files(uid)) == []
