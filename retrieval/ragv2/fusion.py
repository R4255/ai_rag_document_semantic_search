from __future__ import annotations

from .types import Chunk, Hit


def reciprocal_rank_fusion(
    rankings: list[list[Hit]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[Hit]:
    """Fuse ranked lists using only ranks: score = sum(weight / (k + rank)).

    Rank-only fusion avoids calibrating BM25 scores against cosine similarities.
    """
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("weights must match number of rankings")

    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for i, hits in enumerate(rankings):
        w = weights[i] if weights else 1.0
        for rank, h in enumerate(hits, start=1):
            cid = h.chunk.id
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
            chunks[cid] = h.chunk
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    return [Hit(chunks[cid], s, "fused") for cid, s in ordered]
