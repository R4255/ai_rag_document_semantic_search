from __future__ import annotations

import math


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    for i, doc in enumerate(ranked[:k], start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], rels: dict[str, float], k: int) -> float:
    """rels maps doc_id -> graded relevance (use 1 for binary)."""
    dcg = sum(rels.get(d, 0.0) / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1))
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 1) for i, r in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0
