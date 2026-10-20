from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..retriever import HybridRetriever
from .metrics import mrr_at_k, ndcg_at_k, recall_at_k


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _dedupe_docs(hits) -> list[str]:
    """Collapse chunk hits to doc ids, keeping the best-ranked chunk per doc."""
    seen: set[str] = set()
    out: list[str] = []
    for h in hits:
        if h.chunk.doc_id not in seen:
            seen.add(h.chunk.doc_id)
            out.append(h.chunk.doc_id)
    return out


def run_eval(
    retriever: HybridRetriever,
    queries: list[dict],
    modes: tuple[str, ...] | None = None,
    recall_ks: tuple[int, ...] = (5, 10),
    cutoff: int = 10,
    fetch_k: int = 50,
) -> dict[str, dict[str, float]]:
    """Evaluate retrieval quality and total query latency by search mode.

    ``queries`` contains ``{"id", "text", "relevant": [doc_id, ...]}`` rows.
    Retrieval happens at chunk level, then hits are deduplicated to document ids
    before scoring because relevance labels refer to source documents.
    """
    if not recall_ks or any(k <= 0 for k in recall_ks):
        raise ValueError("recall_ks must contain positive cutoffs")
    if cutoff <= 0 or fetch_k <= 0:
        raise ValueError("cutoff and fetch_k must be positive")

    modes = modes or retriever.available_modes
    report: dict[str, dict[str, float]] = {}
    for mode in modes:
        recalls = {k: [] for k in recall_ks}
        mrrs, ndcgs, lat = [], [], []
        for q in queries:
            relevant = set(map(str, q["relevant"]))
            res = retriever.search(q["text"], k=fetch_k, mode=mode)
            ranked = _dedupe_docs(res.hits)
            for k in recall_ks:
                recalls[k].append(recall_at_k(ranked, relevant, k))
            mrrs.append(mrr_at_k(ranked, relevant, cutoff))
            ndcgs.append(ndcg_at_k(ranked, {d: 1.0 for d in relevant}, cutoff))
            lat.append(res.timings_ms["total"])
        report[mode] = {
            **{f"recall@{k}": float(np.mean(recalls[k])) for k in recall_ks},
            f"mrr@{cutoff}": float(np.mean(mrrs)),
            f"ndcg@{cutoff}": float(np.mean(ndcgs)),
            "p50_ms": float(np.percentile(lat, 50)),
            "p95_ms": float(np.percentile(lat, 95)),
        }
    return report


def format_table(report: dict[str, dict[str, float]]) -> str:
    if not report:
        return ""
    cols = list(next(iter(report.values())).keys())
    lines = ["| mode | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for mode, row in report.items():
        cells = [f"{row[c]:.3f}" if "ms" not in c else f"{row[c]:.2f}" for c in cols]
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
