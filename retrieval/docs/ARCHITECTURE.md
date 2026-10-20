# RAG v2: Architecture

```mermaid
flowchart LR
    subgraph Ingest
        D[Documents] --> C[Chunker<br/>sliding window]
        C --> B[(BM25<br/>inverted index)]
        C --> E[Embedder]
        E --> V[(Dense index<br/>flat cosine)]
    end
    subgraph Query
        Q[Query] --> B2[BM25 top-N]
        Q --> V2[Dense top-N]
        B2 --> F[RRF fusion]
        V2 --> F
        F --> R[Reranker<br/>cross-encoder]
        R --> K[Top-k chunks<br/>+ stage timings]
    end
    subgraph Eval
        L[Labelled queries] --> H[Harness]
        H -->|bm25 / dense / hybrid / hybrid+rerank| M[Recall@k, MRR, nDCG<br/>p50/p95 latency]
    end
```

## Components

| Module | Responsibility |
|---|---|
| `chunking.py` | Overlapping word windows; chunk ids are `doc_id::n` so hits map back to documents |
| `index_bm25.py` | Inverted index, BM25 (k1=1.5, b=0.75), heap-based top-k |
| `embedders.py` | `Embedder` protocol; `HashingEmbedder` (offline, tests) and `SentenceTransformerEmbedder` (real) |
| `index_dense.py` | Normalised matrix, exact cosine via matmul (swap for FAISS/HNSW later) |
| `fusion.py` | Reciprocal Rank Fusion with optional per-retriever weights |
| `rerankers.py` | `Reranker` protocol; cross-encoder, plus a lexical stand-in for offline tests |
| `retriever.py` | `HybridRetriever.search(query, k, mode)` returning hits plus per-stage timings |
| `eval/` | Metrics, ablation harness, table output |
| `api.py` | FastAPI `/ingest`, `/search` |

## Key design decisions

- **RRF over score blending.** BM25 scores and cosine similarities live on different scales. RRF uses ranks only, so it needs no calibration.
- **Search modes as a first-class parameter.** `bm25`, `dense`, `hybrid` and `hybrid_rerank` share one code path so ablations are apples-to-apples.
- **Doc-level evaluation.** Retrieval is per chunk; eval dedupes to the best chunk per document, matching how benchmarks label relevance.
- **Pluggable embedder/reranker.** Tests run offline and deterministically; real models are one flag away.
- **Per-stage timings in every result.** Latency is a metric, not an afterthought.

## Known limits (v1 of v2)

- The dense index is exact and in-memory: fine to roughly 1M chunks, not beyond.
- The offline hashing embedder is lexical, so offline eval numbers validate the pipeline, not semantic quality. Use a real embedder for real numbers.
