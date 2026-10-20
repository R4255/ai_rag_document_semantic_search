import math

import pytest

from ragv2.chunking import chunk_text
from ragv2.eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from ragv2.factory import build_retriever
from ragv2.fusion import reciprocal_rank_fusion
from ragv2.index_bm25 import BM25Index
from ragv2.types import Chunk, Hit


def _hit(cid: str, score: float = 1.0) -> Hit:
    return Hit(Chunk(id=cid, doc_id=cid.split("::")[0], text=cid), score, "x")


# ---- chunking ----
def test_chunking_overlap_and_ids():
    text = " ".join(f"w{i}" for i in range(25))
    chunks = chunk_text("d", text, max_words=10, overlap=2)
    assert [c.id for c in chunks] == ["d::0", "d::1", "d::2"]
    assert chunks[0].text.split()[-2:] == chunks[1].text.split()[:2]
    assert chunks[-1].text.split()[-1] == "w24"


def test_chunking_short_and_empty():
    assert len(chunk_text("d", "a b c", max_words=10, overlap=2)) == 1
    assert chunk_text("d", "   ") == []


def test_chunking_validates_params():
    with pytest.raises(ValueError):
        chunk_text("d", "a b", max_words=5, overlap=5)


# ---- bm25 ----
def test_bm25_ranks_exact_term_first():
    idx = BM25Index()
    idx.add(
        [
            Chunk("a::0", "a", "redis cache for hot reads"),
            Chunk("b::0", "b", "kafka partitions and consumer groups"),
            Chunk("c::0", "c", "postgres index scans"),
        ]
    )
    hits = idx.search("kafka consumer", k=3)
    assert hits[0].chunk.doc_id == "b"
    assert all(h.chunk.doc_id != "a" for h in hits)


def test_bm25_empty_index_and_unknown_terms():
    idx = BM25Index()
    assert idx.search("anything") == []
    idx.add([Chunk("a::0", "a", "hello world")])
    assert idx.search("zzzz") == []


# ---- fusion ----
def test_rrf_rewards_agreement():
    a = [_hit("x::0"), _hit("y::0"), _hit("z::0")]
    b = [_hit("y::0"), _hit("x::0"), _hit("w::0")]
    fused = reciprocal_rank_fusion([a, b], k=60)
    assert {fused[0].chunk.id, fused[1].chunk.id} == {"x::0", "y::0"}
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_weights():
    a = [_hit("x::0")]
    b = [_hit("y::0")]
    fused = reciprocal_rank_fusion([a, b], k=60, weights=[1.0, 3.0])
    assert fused[0].chunk.id == "y::0"
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([a, b], weights=[1.0])


# ---- metrics ----
def test_metrics_known_values():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, {"b", "z"}, 2) == 0.5
    assert mrr_at_k(ranked, {"c"}, 10) == pytest.approx(1 / 3)
    assert mrr_at_k(ranked, {"zz"}, 10) == 0.0
    assert ndcg_at_k(["a", "b"], {"a": 1.0, "b": 1.0}, 10) == pytest.approx(1.0)
    # relevant doc at rank 2 only: dcg = 1/log2(3), idcg = 1
    assert ndcg_at_k(["x", "a"], {"a": 1.0}, 10) == pytest.approx(1 / math.log2(3))


# ---- retriever ----
def _retriever(reranker="lexical"):
    r = build_retriever("hashing", reranker)
    r.add_documents(
        [
            {"id": "kafka", "text": "kafka partitions limit consumer parallelism in a group"},
            {"id": "redis", "text": "redis caches hot reads and reduces database round trips"},
            {"id": "pg", "text": "postgres b-tree index avoids full table scans"},
        ]
    )
    return r


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "hybrid_rerank"])
def test_all_modes_find_the_right_doc(mode):
    res = _retriever().search("kafka consumer parallelism", k=3, mode=mode)
    assert res.hits[0].chunk.doc_id == "kafka"
    assert "total" in res.timings_ms


def test_rerank_mode_requires_reranker():
    r = _retriever(reranker="none")
    assert r.available_modes == ("bm25", "dense", "hybrid")
    with pytest.raises(ValueError):
        r.search("x", mode="hybrid_rerank")


def test_invalid_mode():
    with pytest.raises(ValueError):
        _retriever().search("x", mode="nope")


def test_timings_have_stages_for_hybrid_rerank():
    res = _retriever().search("kafka", k=2, mode="hybrid_rerank")
    assert {"bm25", "dense", "fuse", "rerank", "total"} <= set(res.timings_ms)
