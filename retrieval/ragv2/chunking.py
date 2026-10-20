from __future__ import annotations

from .types import Chunk


def chunk_text(
    doc_id: str,
    text: str,
    max_words: int = 120,
    overlap: int = 20,
    meta: dict | None = None,
) -> list[Chunk]:
    """Split text into overlapping word windows. Chunk ids are ``doc_id::n``."""
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if not 0 <= overlap < max_words:
        raise ValueError("overlap must be >= 0 and < max_words")

    words = text.split()
    if not words:
        return []

    step = max_words - overlap
    chunks: list[Chunk] = []
    for n, start in enumerate(range(0, len(words), step)):
        window = words[start : start + max_words]
        chunks.append(
            Chunk(id=f"{doc_id}::{n}", doc_id=doc_id, text=" ".join(window), meta=dict(meta or {}))
        )
        if start + max_words >= len(words):
            break
    return chunks
