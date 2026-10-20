from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .factory import build_retriever
from .retriever import HybridRetriever, MODES


class Doc(BaseModel):
    id: str
    text: str
    meta: dict = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[Doc]


class SearchRequest(BaseModel):
    query: str
    k: int = Field(default=10, ge=1, le=100)
    mode: str = "hybrid_rerank"


def create_app(retriever: HybridRetriever | None = None) -> FastAPI:
    app = FastAPI(title="RAG v2 retrieval service")
    r = retriever or build_retriever(
        os.getenv("RAGV2_EMBEDDER", "hashing"), os.getenv("RAGV2_RERANKER", "lexical")
    )

    @app.get("/health")
    def health():
        return {"status": "ok", "chunks": len(r.bm25.chunks), "modes": list(r.available_modes)}

    @app.post("/ingest")
    def ingest(req: IngestRequest):
        n = r.add_documents([d.model_dump() for d in req.documents])
        return {"documents": len(req.documents), "chunks": n}

    @app.post("/search")
    def search(req: SearchRequest):
        if req.mode not in MODES:
            raise HTTPException(400, f"mode must be one of {list(MODES)}")
        try:
            res = r.search(req.query, k=req.k, mode=req.mode)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {
            "hits": [
                {
                    "chunk_id": h.chunk.id,
                    "doc_id": h.chunk.doc_id,
                    "score": h.score,
                    "source": h.source,
                    "text": h.chunk.text,
                }
                for h in res.hits
            ],
            "timings_ms": res.timings_ms,
        }

    return app


app = create_app()
