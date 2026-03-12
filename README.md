# 🧠 Nexus RAG — Intelligent Document QA Engine

A **production-grade** Retrieval-Augmented Generation (RAG) micro-service powered by **Google Gemini 2.0 Flash**, FAISS vector search, **LRU semantic caching**, asynchronous background ingestion, and **real-time SSE streaming** — all wrapped in a premium React dashboard.

> Built to demonstrate full-stack AI engineering: from async pipeline architecture to live citation-backed inference.

---

## ✨ Key Features

| # | Feature | Details |
|---|---------|---------|
| 1 | **Source Citations** | Every answer includes verifiable `[Source, Page]` references with context snippets so you can trust every response. |
| 2 | **Semantic LRU Cache** | Thread-safe `AsyncLRUCache` with TTL eviction — identical queries bypass the LLM entirely, returning in **< 10 ms** with live hit/miss metrics visible in the UI. |
| 3 | **Async Background Ingestion** | Inspired by CDC / EventBridge patterns — uploads return immediately (HTTP 200), while chunking + embedding + FAISS upsert runs as a FastAPI `BackgroundTask`. |
| 4 | **Real-Time SSE Streaming** | LLM tokens stream to the frontend via Server-Sent Events with zero buffering. |
| 5 | **Ingested Document Tracker** | Live sidebar widget shows every uploaded document, its chunk count, vectorisation time, and success/error state. |
| 6 | **Health + Monitoring API** | `/api/health` and `/api/cache/stats` endpoints for operational observability. |

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
                         │  ┌──────────┐  ┌───────────┐  ┌──▼────┐   │
                         │  │ LRU     │  │ Retriever │──│ FAISS │   │
                         │  │ Cache   │  │  (top-5)  │  │ Index │   │
                         │  └────┬────┘  └─────┬─────┘  └───────┘   │
                         │       │             │                     │
                         │  ┌────▼─────────────▼───────┐             │
                         │  │  Gemini 2.0 Flash (LLM)  │             │
                         │  │  + Citation Extraction    │             │
                         │  └──────────────────────────┘             │
                         └──────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Google Gemini 2.0 Flash (`gemini-2.0-flash`) |
| **Embeddings** | Google Generative AI Embeddings (`embedding-001`) |
| **Framework** | FastAPI, Python AsyncIO |
| **Vector DB** | FAISS (local, zero-infra) |
| **Caching** | Custom `AsyncLRUCache` — TTL + LRU eviction |
| **Frontend** | React 19, TypeScript, Vite, TailwindCSS, Framer Motion |
| **Streaming** | Server-Sent Events (SSE) |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- A [Google AI Studio](https://aistudio.google.com/) API key

### 1. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create **`backend/.env`**:
```env
GOOGLE_API_KEY=AIzaSy...your_key_here
```

Start the server:
```bash
uvicorn app:app --reload --port 8000
```

### 2. Frontend

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
| `POST` | `/api/chat` | SSE stream — RAG query with citations |
| `GET`  | `/api/cache/stats` | Live cache telemetry (hits, misses, size) |
| `GET`  | `/api/documents` | List all ingested documents + metadata |
| `GET`  | `/api/health` | Service health check |

---

## 💡 How It Works

1. **Upload** a PDF/TXT via the sidebar drag-and-drop zone.
2. A **background worker** chunks the document (1000 chars, 200 overlap), generates Gemini embeddings, and upserts into the local FAISS index.
3. **Ask a question** — the system retrieves the top-5 relevant chunks, constructs a grounded prompt, and **streams** the Gemini response token-by-token.
4. **Citations** appear at the end of each answer with source filename, page number, and a text snippet.
5. Repeated queries hit the **LRU semantic cache** — served in under 10 ms without touching the LLM.

---

## 📁 Project Structure

```
rag_document/
├── backend/
│   ├── app.py              # FastAPI app — routes, lifespan, CORS
│   ├── rag_service.py      # RAG engine — ingestion, retrieval, streaming
│   ├── cache_manager.py    # Async LRU cache with TTL
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
└── README.md
```
