import os
import shutil
import tempfile
import asyncio
import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from cache_manager import semantic_cache
from rag_service import rag_service

load_dotenv()


# ──────────────────────────── Lifespan ────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan handler replacing deprecated on_event."""
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if api_key:
        rag_service.initialize(api_key)
    else:
        print("⚠️  GOOGLE_API_KEY not found in .env – LLM / embedding features disabled.")
    yield  # app runs here
    print("👋 Shutting down Nexus RAG backend.")


app = FastAPI(
    title="Nexus RAG — Intelligent Document QA",
    description=(
        "Production‑grade RAG micro‑service powered by **Google Gemini 3 Flash **, "
        "FAISS vector search, LRU semantic caching, async background ingestion, "
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


# ──────────────────────────── Upload Endpoint ────────────────────────────

@app.post("/api/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Non-blocking document upload.
    The file is immediately saved to a temp path and
    vectorisation happens as a **background task** (CDC‑style).
    """
    allowed = {"application/pdf", "text/plain"}
    # Some browsers send different MIME types for .txt
    if file.content_type not in allowed and not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    background_tasks.add_task(_process_upload, tmp_path, file.filename)

    return {
        "status": "accepted",
        "filename": file.filename,
        "message": "File enqueued for async vectorisation via background worker.",
    }


async def _process_upload(file_path: str, filename: str):
    try:
        await rag_service.ingest_file(file_path, filename)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# ──────────────────────────── Chat / SSE Endpoint ────────────────────────────

@app.post("/api/chat")
async def chat_stream(request: ChatRequest):
    """
    SSE streaming endpoint.
    1. Check LRU cache → instant replay on cache hit.
    2. On miss → run full RAG pipeline and cache result.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # ── Cache hit path ──
    cached = await semantic_cache.get(query)
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

        async for chunk in rag_service.stream_rag_response(query):
            yield chunk
            try:
                payload = json.loads(chunk.removeprefix("data: ").strip())
                if payload.get("type") == "content":
                    full_answer += payload.get("content", "")
                elif payload.get("type") == "citations":
                    citations = payload.get("citations", [])
            except Exception:
                pass

        if full_answer:
            await semantic_cache.set(
                query, {"answer": full_answer, "citations": citations}, ttl=3600,
            )

    return StreamingResponse(_live_stream(), media_type="text/event-stream")


# ──────────────────────────── Auxiliary Endpoints ────────────────────────────

@app.get("/api/cache/stats")
async def get_cache_stats():
    """Live cache telemetry (hits, misses, utilisation)."""
    return await semantic_cache.get_stats()


@app.get("/api/documents")
async def list_documents():
    """Return metadata for every document ingested so far."""
    return {"documents": rag_service.get_ingested_files()}


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "engine": "Gemini 3 Flash ",
        "vector_store": "FAISS (local)",
        "cache": "AsyncLRU",
    }
