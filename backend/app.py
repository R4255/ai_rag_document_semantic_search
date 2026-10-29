import os
import shutil
import tempfile
import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()  # before importing hybrid, which reads RAG_* settings at import time

import db
from cache_manager import semantic_cache
from hybrid import RERANKER, RETRIEVAL_MODE
from providers import LLM_PROVIDER, engine_label, model_status, warm_models
from ragv2 import MODES
from rag_service import rag_service


# ──────────────────────────── Lifespan ────────────────────────────

_warmup_task: asyncio.Task | None = None


def _warming() -> bool:
    return _warmup_task is not None and not _warmup_task.done()


def _start_warmup() -> None:
    """Load the Ollama models in the background (one warm-up at a time; no-op for Gemini)."""
    global _warmup_task
    if LLM_PROVIDER != "ollama" or _warming():
        return

    def run():
        t0 = time.perf_counter()
        try:
            warm_models()
            print(f"🔥 Ollama models warm in {time.perf_counter() - t0:.1f}s")
        except Exception as e:
            print(f"⚠️  Could not warm Ollama models: {e}")

    _warmup_task = asyncio.create_task(asyncio.to_thread(run))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan handler replacing deprecated on_event."""
    try:
        rag_service.initialize()
        print(f"🧠 Provider: {engine_label()}")
        _start_warmup()
    except Exception as e:
        print(f"⚠️  Could not initialise {LLM_PROVIDER} models – LLM / embedding features disabled: {e}")
    yield  # app runs here
    db.close_db()
    print("👋 Shutting down Nexus RAG backend.")


app = FastAPI(
    title="Nexus RAG — Intelligent Document QA",
    description=(
        "Production‑grade RAG micro‑service powered by Google Gemini or local Ollama models, "
        "hybrid BM25 + pgvector retrieval (RRF), LRU semantic caching, async background ingestion, "
        "and real‑time SSE streaming."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────── Models ────────────────────────────

class ChatRequest(BaseModel):
    query: str


class SearchRequest(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=50)
    mode: str = RETRIEVAL_MODE


@app.post("/api/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    x_user_id: Annotated[str | None, Header()] = None
):
    """
    Non-blocking document upload.
    The file is immediately saved to a temp path and
    vectorisation happens as a **background task** (CDC-style).
    """
    user_id = x_user_id or "unknown"
    allowed = {"application/pdf", "text/plain"}
    # Some browsers send different MIME types for .txt
    if file.content_type not in allowed and not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    if db.is_ready() and await rag_service.is_duplicate(tmp_path, user_id):
        os.remove(tmp_path)
        return {
            "status": "duplicate",
            "filename": file.filename,
            "message": "This file is already in your library, so it was skipped.",
        }

    background_tasks.add_task(_process_upload, tmp_path, file.filename, user_id)

    return {
        "status": "accepted",
        "filename": file.filename,
        "message": "File enqueued for async vectorisation via background worker.",
    }


async def _process_upload(file_path: str, filename: str, user_id: str):
    try:
        await rag_service.ingest_file(file_path, filename, user_id)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# ──────────────────────────── Chat / SSE Endpoint ────────────────────────────

@app.post("/api/chat")
async def chat_stream(
    request: ChatRequest,
    x_user_id: Annotated[str | None, Header()] = None
):
    """
    SSE streaming endpoint.
    1. Check LRU cache → instant replay on cache hit.
    2. On miss → run full RAG pipeline and cache result.
    """
    user_id = x_user_id or "unknown"
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # ── Cache hit path ──
    # Cache key must now include user_id to prevent data leaks!
    cache_key = f"{user_id}:{query}"
    cached = await semantic_cache.get(cache_key, owner=user_id)
    if cached:
        async def _cached_stream():
            yield f"data: {json.dumps({'type': 'cache_hit'})}\n\n"
            chunks = [cached["answer"][i:i + 6] for i in range(0, len(cached["answer"]), 6)]
            for c in chunks:
                yield f"data: {json.dumps({'type': 'content', 'content': c})}\n\n"
                await asyncio.sleep(0.008)
            yield f"data: {json.dumps({'type': 'citations', 'citations': cached['citations']})}\n\n"

        return StreamingResponse(_cached_stream(), media_type="text/event-stream")

    # ── Cache miss — live RAG execution ──
    async def _live_stream():
        full_answer = ""
        citations = []
        failed = False

        # Pass user_id to RAG service
        async for chunk in rag_service.stream_rag_response(query, user_id):
            yield chunk
            try:
                if chunk.strip().startswith("data: "):
                    payload = json.loads(chunk.strip()[6:])
                else:
                    payload = json.loads(chunk)

                if payload.get("type") == "content":
                    full_answer += payload.get("content", "")
                elif payload.get("type") == "citations":
                    citations = payload.get("citations", [])
                elif payload.get("type") == "error":
                    failed = True
            except Exception:
                pass

        if full_answer and not failed:  # never replay a failed answer from the cache
            await semantic_cache.set(
                cache_key, {"answer": full_answer, "citations": citations}, ttl=3600, owner=user_id,
            )

    return StreamingResponse(_live_stream(), media_type="text/event-stream")


# ──────────────────────────── Retrieval-only Endpoint ────────────────────────────

@app.post("/api/search")
async def search(
    request: SearchRequest,
    x_user_id: Annotated[str | None, Header()] = None
):
    """Retrieval without generation, for comparing bm25 / dense / hybrid / hybrid_rerank."""
    user_id = x_user_id or "unknown"
    if request.mode not in MODES:
        raise HTTPException(status_code=400, detail=f"mode must be one of {list(MODES)}")
    if not rag_service.embeddings:
        raise HTTPException(status_code=503, detail="Service not initialized.")
    try:
        result = await rag_service.search(request.query, user_id, k=request.k, mode=request.mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "mode": request.mode,
        "hits": [
            {
                "source": h.chunk.meta.get("source"),
                "page": h.chunk.meta.get("page"),
                "score": h.score,
                "retriever": h.source,
                "text": h.chunk.text,
            }
            for h in result.hits
        ],
        "timings_ms": result.timings_ms,
    }


# ──────────────────────────── Auxiliary Endpoints ────────────────────────────

@app.get("/api/cache/stats")
async def get_cache_stats(x_user_id: Annotated[str | None, Header()] = None):
    """Live cache telemetry: server-wide totals, plus this user's under "user"."""
    return await semantic_cache.get_stats(owner=x_user_id or "unknown")


@app.get("/api/documents")
async def list_documents(x_user_id: Annotated[str | None, Header()] = None):
    """Return metadata for every document ingested so far."""
    user_id = x_user_id or "unknown"
    if not db.is_ready():
        return {"documents": []}
    return {"documents": await rag_service.get_ingested_files(user_id)}



@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: uuid.UUID, x_user_id: Annotated[str | None, Header()] = None):
    """Delete one of the caller's documents, its chunks and vectors, and their cached answers."""
    user_id = x_user_id or "unknown"
    if not db.is_ready():
        raise HTTPException(status_code=503, detail="Database unavailable.")
    outcome = await rag_service.delete_document(user_id, str(doc_id))
    if outcome == "missing":
        raise HTTPException(status_code=404, detail="Document not found.")
    if outcome == "processing":
        raise HTTPException(status_code=409, detail="This file is still being indexed. Try again when it's ready.")
    await semantic_cache.clear_owner(user_id)  # cached answers may cite the deleted file
    return {"status": "deleted", "id": str(doc_id)}


@app.get("/api/document")
async def document_text(name: str, x_user_id: Annotated[str | None, Header()] = None):
    """One document's passages in reading order, for the document reader."""
    user_id = x_user_id or "unknown"
    if not db.is_ready():
        raise HTTPException(status_code=503, detail="Database unavailable.")
    chunks = await rag_service.get_document_chunks(user_id, name)
    if not chunks:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"name": name, "chunks": chunks}


@app.get("/api/health")
async def health():
    return {
        "status": "healthy" if db.is_ready() and rag_service.embeddings else "degraded",
        "provider": LLM_PROVIDER,
        "engine": engine_label(),
        "vector_store": "Postgres + pgvector",
        "database": "connected" if db.is_ready() else "unavailable",
        "retrieval": RETRIEVAL_MODE,
        "reranker": RERANKER,
        "cache": "AsyncLRU",
        # Ollama only: {"chat": bool, "embed": bool}, True = in memory; None for Gemini or Ollama down
        "models": await asyncio.to_thread(model_status),
        "warming": _warming(),
    }


@app.post("/api/warmup")
async def warmup():
    """Load the models now, e.g. when someone opens the app, so their first question is fast."""
    _start_warmup()
    return {"warming": _warming(), "models": await asyncio.to_thread(model_status)}
