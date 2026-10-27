# 🧠 Nexus RAG — Intelligent Document QA Engine

A **production-grade** Retrieval-Augmented Generation (RAG) micro-service powered by **Google Gemini 3 Flash** or **local Ollama models**, **hybrid BM25 + pgvector retrieval** fused with RRF, **Postgres storage**, **LRU semantic caching**, asynchronous background ingestion, and **real-time SSE streaming** — all wrapped in a premium React dashboard.

> Built to demonstrate full-stack AI engineering: from async pipeline architecture to live citation-backed inference.

---

## ✨ Key Features

| # | Feature | Details |
|---|---------|---------|
| 0 | **Hybrid Retrieval (RAG v2)** | BM25 and embedding-based dense search (pgvector) run side by side and are fused with Reciprocal Rank Fusion, so exact terms (error codes, IDs, names) and paraphrases are both found. Optional cross-encoder reranking. Per-stage latency is streamed with every answer. Core lives in [`retrieval/`](retrieval/) with its own eval harness. On BEIR SciFact: **nDCG@10 0.728** with cross-encoder reranking (735 ms p95), or 0.722 at 13 ms p95 with tuned RRF alone, vs 0.646 for BM25. [Full results](retrieval/README.md#results-scifact-beir-test-split). |
| 1 | **Source Citations** | Every answer includes verifiable `[Source, Page]` references with context snippets so you can trust every response. |
| 2 | **Semantic LRU Cache** | Thread-safe `AsyncLRUCache` with TTL eviction — identical queries bypass the LLM entirely, returning in **< 10 ms** with live per-user hit/miss metrics visible in the UI. |
| 3 | **Async Background Ingestion** | Inspired by CDC / EventBridge patterns — uploads return immediately (HTTP 200), while chunking + embedding + the Postgres insert run as a FastAPI `BackgroundTask`. Re-uploading the same file is detected by its SHA-256 and skipped; files with no extractable text (e.g. scanned PDFs) are flagged in the sidebar. |
| 4 | **Real-Time SSE Streaming** | LLM tokens stream to the frontend via Server-Sent Events with zero buffering. |
| 5 | **Ingested Document Tracker** | Live sidebar widget shows every uploaded document, its chunk count, vectorisation time, and success/error state. |
| 6 | **Health + Monitoring API** | `/api/health` and `/api/cache/stats` endpoints for operational observability. |

---

## 📸 Screenshots

**Grounded answer with source citations** (running fully local on Ollama `qwen2.5:7b-instruct`, hybrid BM25 + dense retrieval):

![Answer with citations](docs/screenshots/02-answer-with-citations.png)

**Repeat question served from the semantic cache**, with per-user hit/miss metrics and live backend status in the sidebar:

![Cache hit](docs/screenshots/03-cache-hit.png)

**API** (`/docs`), including the retrieval-only `/api/search` endpoint for comparing `bm25` / `dense` / `hybrid` / `hybrid_rerank`:

![API docs](docs/screenshots/04-api-docs.png)

---

## 🏗️ Architecture

```
┌──────────────┐         ┌──────────────────────────────────────────────┐
│  React + TS  │  HTTP   │            FastAPI (AsyncIO)                │
│  Vite  SPA   │◀───────▶│                                            │
│  TailwindCSS │   SSE   │  ┌────────┐  ┌──────────┐  ┌───────────┐  │
│  Framer Mot. │         │  │ Upload │──│ Chunking │──│ Embedding │  │
└──────────────┘         │  │ (BG)   │  │ (Recurs) │  │ (Gemini)  │  │
                         │  └────────┘  └──────────┘  └─────┬─────┘  │
                         │                                  │        │
                         │  ┌──────────┐  ┌───────────┐  ┌──▼─────┐  │
                         │  │ LRU     │  │  Hybrid   │──│Postgres│  │
                         │  │ Cache   │  │ Retriever │  │pgvector│  │
                         │  │         │  │ RRF top-5 │──│ BM25   │  │
                         │  └────┬────┘  └─────┬─────┘  └────────┘  │
                         │       │             │                     │
                         │  ┌────▼─────────────▼───────┐             │
                         │  │  Gemini 3 Flash  (LLM)  │             │
                         │  │  + Citation Extraction    │             │
                         │  └──────────────────────────┘             │
                         └──────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Google Gemini 3 Flash  (`gemini-3-flash`) **or** any local Ollama model (default `qwen2.5:7b-instruct`) |
| **Embeddings** | Google `gemini-embedding-001` **or** Ollama `nomic-embed-text` |
| **Framework** | FastAPI, Python AsyncIO |
| **Storage** | PostgreSQL + pgvector: documents and chunks tables, vector and text in the same row |
| **Retrieval** | `ragv2`: BM25 + dense, RRF fusion, optional cross-encoder rerank, eval harness |
| **Caching** | Custom `AsyncLRUCache` — TTL + LRU eviction |
| **Frontend** | React 19, TypeScript, Vite, TailwindCSS, Framer Motion |
| **Streaming** | Server-Sent Events (SSE) |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- PostgreSQL with the pgvector extension (a local container is easiest, see below)
- **Either** [Ollama](https://ollama.com/) running locally (fully offline, no key), **or** a [Google AI Studio](https://aistudio.google.com/) API key

### 1. Database

```bash
# Docker or Podman; tables are created automatically when the backend starts
docker run -d --name nexus_rag_pg -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=nexus_rag \
  -p 5434:5432 pgvector/pgvector:pg16
```

In `backend/.env`:
```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5434/nexus_rag
```

In production, use any hosted Postgres with pgvector (Neon, Supabase, Render Postgres) and set `DATABASE_URL` in the host's environment.

### 2. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Option A: local, with Ollama (no API key)**
```bash
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
```
With no `GOOGLE_API_KEY` set, the backend uses Ollama automatically. Override in `backend/.env` if you like:
```env
LLM_PROVIDER=ollama
OLLAMA_CHAT_MODEL=qwen2.5:7b-instruct
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://localhost:11434
```

**Option B: Gemini.** Create **`backend/.env`**:
```env
GOOGLE_API_KEY=AIzaSy...your_key_here

# Optional: retrieval tuning
RAG_RETRIEVAL_MODE=hybrid   # bm25 | dense | hybrid | hybrid_rerank
RAG_RERANKER=none           # none | cross (cross needs: pip install sentence-transformers)
RAG_CANDIDATES=30           # candidates per retriever before fusion
```

Start the server:
```bash
uvicorn app:app --reload --port 8000
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## 📡 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/upload` | Upload PDF/TXT → async background vectorisation |
| `POST` | `/api/chat` | SSE stream — RAG query with citations (plus a `retrieval` event with mode + stage timings) |
| `POST` | `/api/search` | Retrieval only — `{"query", "k", "mode"}` to compare `bm25` / `dense` / `hybrid` / `hybrid_rerank` |
| `GET`  | `/api/cache/stats` | Live cache telemetry: server-wide hits, misses, size, plus the caller's own under `user` |
| `GET`  | `/api/documents` | List all ingested documents + metadata |
| `GET`  | `/api/health` | Service health check |

---

## 💡 How It Works

1. **Upload** a PDF/TXT via the sidebar drag-and-drop zone.
2. A **background worker** chunks the document (1000 chars, 200 overlap), generates a 768-number embedding per chunk (Gemini or Ollama), and inserts each chunk, with its vector, as one row in Postgres.
3. **Ask a question** — BM25 and pgvector each return candidates, RRF fuses them (and an optional cross-encoder reranks), and the system takes the top-5 chunks, constructs a grounded prompt, and **streams** the LLM response token-by-token.
4. **Citations** appear at the end of each answer with source filename, page number, and a text snippet.
5. Repeated queries hit the **LRU semantic cache** — served in under 10 ms without touching the LLM.

---

## 🧪 Tests & Retrieval Evaluation

```bash
# Backend tests (fake embeddings, no API key; the retrieval tests need a Postgres with pgvector)
cd backend && TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5434/nexus_rag pytest tests

# Retrieval core: unit tests + bm25/dense/hybrid/rerank ablation
cd retrieval
pip install -e ".[dev]"
pytest
python scripts/run_eval.py          # sample data; see retrieval/README.md for SciFact
```

---

## 📁 Project Structure

```
rag_document/
├── backend/
│   ├── app.py              # FastAPI app — routes, lifespan, CORS
│   ├── rag_service.py      # RAG engine — ingestion, retrieval, streaming
│   ├── hybrid.py           # Glue: pgvector ⇄ ragv2 HybridRetriever (BM25 + RRF)
│   ├── db.py               # Postgres pool and all SQL
│   ├── schema.sql          # documents + chunks tables (applied at startup)
│   ├── providers.py        # Gemini or Ollama models, chosen by LLM_PROVIDER
│   ├── cache_manager.py    # Async LRU cache with TTL
│   ├── tests/              # Hybrid-retrieval tests (need TEST_DATABASE_URL)
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   └── components/
│   │       ├── ChatInterface.tsx
│   │       ├── DocumentUpload.tsx
│   │       ├── DocumentList.tsx
│   │       └── CacheStats.tsx
│   ├── tailwind.config.js
│   └── package.json
├── retrieval/              # RAG v2 core: BM25, dense, RRF, rerankers, eval harness
│   ├── ragv2/
│   ├── tests/
│   ├── scripts/            # run_eval.py, fetch_scifact.py
│   └── docs/               # PROBLEM.md, ARCHITECTURE.md
└── README.md
```
