from __future__ import annotations

import heapq
import math
import re
from collections import Counter, defaultdict

from .types import Chunk, Hit

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were will with".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


class BM25Index:
    """Inverted-index BM25 (Robertson/Sparck Jones idf, +1 smoothed so it never goes negative)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len: list[int] = []
        self.avgdl: float = 0.0

    def add(self, chunks: list[Chunk]) -> None:
        for ch in chunks:
            idx = len(self.chunks)
            self.chunks.append(ch)
            toks = tokenize(ch.text)
            self.doc_len.append(len(toks))
            for term, tf in Counter(toks).items():
                self.postings[term].append((idx, tf))
        total = sum(self.doc_len)
        self.avgdl = total / len(self.doc_len) if self.doc_len else 0.0

    def search(self, query: str, k: int = 10) -> list[Hit]:
        n = len(self.chunks)
        if n == 0:
            return []
        avgdl = self.avgdl or 1.0
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            df = len(plist)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for idx, tf in plist:
                norm = 1 - self.b + self.b * self.doc_len[idx] / avgdl
                scores[idx] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        top = heapq.nlargest(k, scores.items(), key=lambda kv: kv[1])
        return [Hit(self.chunks[i], s, "bm25") for i, s in top]
