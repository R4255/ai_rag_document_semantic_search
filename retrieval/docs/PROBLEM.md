# RAG v2: Problem Statement

## The problem

Naive RAG (embed the query, take top-k by cosine similarity, stuff into the prompt) fails in predictable ways:

1. **Dense-only retrieval misses exact terms.** Error codes, IDs, product names and rare jargon have weak embeddings.
2. **BM25-only retrieval misses paraphrases.** "stop duplicate messages" never matches a passage about "idempotent consumers".
3. **Top-k contains near-misses.** Passages that look relevant but are not mislead the generator.
4. **Nobody measures retrieval.** Chunk size, embedder, fusion weights and rerankers get changed by feel, and regressions ship unnoticed.

## Goal

Build a retrieval layer that is **hybrid, reranked, and measured**:

- Lexical (BM25) and semantic (dense) retrieval run in parallel and are fused.
- A reranker re-scores the fused candidates for precision.
- Every change is scored against a labelled eval set, and per-stage latency is visible, so quality/latency trade-offs are explicit.

## Success criteria (targets, to be verified on a real benchmark)

| Metric | Target |
|---|---|
| nDCG@10 (BEIR SciFact) | hybrid > max(bm25, dense) |
| MRR@10 | hybrid+rerank > hybrid |
| p95 latency, rerank depth 50 | stated budget, reported per stage |
| Reproducibility | one command re-runs the full ablation |

## Non-goals (v2)

- Training embedding models.
- Multi-tenant auth.
- Answer generation (planned as phase 3, with faithfulness evals).

## Phases

1. **Retrieval core + eval harness** (this repo, current state)
2. Real embedder/reranker on SciFact, tuned chunking and fusion weights, FAISS/HNSW
3. Generation with citations, answer-faithfulness eval, query rewriting
4. Persistence (pgvector/Postgres), async ingestion over Kafka/SQS, tracing into ClickHouse
