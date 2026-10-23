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

### Results: SciFact (BEIR test split)

300 queries over 5,183 abstracts (12,647 chunks). Embedder `BAAI/bge-small-en-v1.5`,
120-word chunks with 20-word overlap, 50 candidates per retriever, RRF k=60,
rerank depth 50. Apple M2 (16 GB), CPU/MPS, single process. Measured 2026-10-21.

| Configuration | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | p50 latency | p95 latency |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.719 | 0.774 | 0.612 | 0.646 | 1.3 ms | 3.1 ms |
| Dense (bge-small) | 0.771 | 0.820 | 0.696 | 0.721 | 9.5 ms | 10.5 ms |
| Hybrid RRF (1 : 1) | 0.771 | **0.846** | 0.671 | 0.709 | 11.4 ms | 13.5 ms |
| Hybrid RRF, BM25 down-weighted (0.3 : 1) | **0.788** | 0.826 | 0.694 | 0.722 | 11.2 ms | 13.3 ms |
| Hybrid + rerank, `ms-marco-MiniLM-L-6-v2` | 0.795 | 0.840 | 0.689 | 0.719 | 225 ms | 277 ms |
| **Hybrid + rerank, `ms-marco-MiniLM-L-12-v2`** | 0.775 | 0.840 | **0.701** | **0.728** | 587 ms | 735 ms |

**Best post-reranking result:** hybrid RRF + MiniLM-L12 cross-encoder at depth 50,
**nDCG@10 0.728 / MRR@10 0.701 at 735 ms p95**: +12.7% nDCG over BM25 and +2.7% over
equal-weight hybrid. **Best quality per millisecond:** hybrid with BM25 down-weighted to 0.3,
**nDCG@10 0.722 at 13 ms p95**, within 0.006 of the reranked result at ~55x lower latency.

What the sweep showed (rerank depth 20 / 50 / 100, three cross-encoders, three RRF weightings).
Sweep runs used `--fetch-k 10` so `--candidates` sets the rerank depth exactly; their scores sit
~0.004 below the table above, so compare them with each other rather than with the table:

- **Depth 50 is the sweet spot.** Depth 100 was slower *and* slightly worse for both MiniLM models
  (L12: 0.719 at 988 ms p50); depth 20 gave up ~0.005 nDCG.
- **Bigger is not better.** `BAAI/bge-reranker-base` (8x the parameters of MiniLM-L12) scored
  0.692 at depth 20 and 0.660 at depth 50, at 0.8-2.1 s p50. Domain fit beat model size.
- **Equal-weight RRF is not optimal here.** SciFact is paraphrase-heavy scientific text, so BM25
  adds noise; down-weighting it lifts nDCG from 0.709 to 0.722. Combining that weighting with
  reranking did *not* stack (0.717): the reranker does better with BM25's broader candidate pool.
- On keyword-heavy corpora (error codes, IDs, product names) BM25 carries more weight; re-run
  the sweep on your own labelled queries before choosing weights.

Reproduce (each command re-embeds the corpus, ~2 min, then runs all four modes):

```bash
pip install -e ".[ml]"
python scripts/fetch_scifact.py
# best post-reranking result
python scripts/run_eval.py --corpus data/scifact/corpus.jsonl --queries data/scifact/queries.jsonl \
  --embedder st --reranker cross --cross-model cross-encoder/ms-marco-MiniLM-L-12-v2
# best quality per millisecond
python scripts/run_eval.py --corpus data/scifact/corpus.jsonl --queries data/scifact/queries.jsonl \
  --embedder st --reranker none --weights 0.3 1
```

Latency is per query, end to end through retrieval (embedding the query, both indexes, fusion,
rerank), on one machine. Compare latency only across runs on the same hardware.

## Serve

```bash
pip install -e ".[api]"
RAGV2_EMBEDDER=st RAGV2_RERANKER=cross uvicorn ragv2.api:app
```

`POST /ingest {"documents":[{"id","text"}]}` and `POST /search {"query","k","mode"}` where mode is one of `bm25 | dense | hybrid | hybrid_rerank`.

## Roadmap

1. ~~Real embedder + reranker on SciFact; tune RRF weights, rerank depth~~ (done, see results); tune chunk size
2. FAISS/HNSW dense index; persistence
3. Generation with citations and answer-faithfulness eval
4. Async ingestion (Kafka/SQS), tracing into ClickHouse
