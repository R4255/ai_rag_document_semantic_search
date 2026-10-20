# RAG v2: Hybrid, Reranked, Measured Retrieval

BM25 + dense retrieval, fused with RRF, reranked, and evaluated with Recall@k / MRR / nDCG plus per-stage latency.

This package is the retrieval core behind the Nexus RAG backend ([`../backend/hybrid.py`](../backend/hybrid.py)),
which pairs its BM25 index and RRF fusion with the Gemini-embedding FAISS store. It also runs standalone.

- Problem statement: [docs/PROBLEM.md](docs/PROBLEM.md)
- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Quickstart (offline, no model downloads)

```bash
pip install -e ".[dev]"
pytest
python scripts/run_eval.py
```

The offline `HashingEmbedder` is lexical, so these numbers only prove the pipeline works.
They are not semantic-retrieval results and should not be used as portfolio claims.

## Real numbers (SciFact)

```bash
pip install -e ".[ml]"
python scripts/fetch_scifact.py
python scripts/run_eval.py \
  --corpus data/scifact/corpus.jsonl --queries data/scifact/queries.jsonl \
  --embedder st --reranker cross --output results/scifact-st-cross.json
```

## Evaluation protocol

Run the same corpus and query set through `bm25`, `dense`, `hybrid`, and
`hybrid_rerank`. The command reports document-level `recall@5`, `recall@10`,
`mrr@10`, `ndcg@10`, and p50/p95 total latency. Retrieval is chunk-level, but
metrics deduplicate results to their original document ids because benchmark
labels refer to documents.

The JSON output makes each run reproducible and easy to compare. Record the
model names, chunking settings, candidate depth, hardware, and date next to a
published result. Do not compare latency across different machines or model
providers as if the numbers were directly equivalent.

### Portfolio results table

Fill this table only with measurements from the same corpus and evaluation
settings. The sample-data command is a correctness check, not a portfolio run.

| Configuration | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | p95 latency |
|---|---:|---:|---:|---:|---:|
| BM25 | — | — | — | — | — |
| Dense | — | — | — | — | — |
| Hybrid RRF | — | — | — | — | — |
| Hybrid + cross-encoder rerank | — | — | — | — | — |

## Serve

```bash
pip install -e ".[api]"
RAGV2_EMBEDDER=st RAGV2_RERANKER=cross uvicorn ragv2.api:app
```

`POST /ingest {"documents":[{"id","text"}]}` and `POST /search {"query","k","mode"}` where mode is one of `bm25 | dense | hybrid | hybrid_rerank`.

## Roadmap

1. Real embedder + reranker on SciFact; tune chunk size, RRF weights, rerank depth
2. FAISS/HNSW dense index; persistence
3. Generation with citations and answer-faithfulness eval
4. Async ingestion (Kafka/SQS), tracing into ClickHouse
