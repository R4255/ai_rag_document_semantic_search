from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    meta: dict = field(default_factory=dict)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    source: str  # "bm25" | "dense" | "fused" | "rerank"
