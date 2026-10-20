import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from ragv2.api import create_app  # noqa: E402
from ragv2.factory import build_retriever  # noqa: E402


def _client():
    return TestClient(create_app(build_retriever("hashing", "lexical")))


def test_ingest_then_search():
    c = _client()
    r = c.post(
        "/ingest",
        json={
            "documents": [
                {"id": "a", "text": "raft elects a leader and replicates a log"},
                {"id": "b", "text": "bloom filters test set membership"},
            ]
        },
    )
    assert r.status_code == 200 and r.json()["chunks"] == 2

    r = c.post("/search", json={"query": "raft leader log", "k": 2, "mode": "hybrid"})
    body = r.json()
    assert r.status_code == 200
    assert body["hits"][0]["doc_id"] == "a"
    assert "total" in body["timings_ms"]


def test_bad_mode_is_400():
    r = _client().post("/search", json={"query": "x", "mode": "nope"})
    assert r.status_code == 400


def test_health():
    assert _client().get("/health").json()["status"] == "ok"
