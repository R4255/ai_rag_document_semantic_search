import os
import json
import asyncio
import hashlib
import time
import uuid
from typing import AsyncGenerator

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import db
from ragv2 import Chunk, HybridRetriever
from hybrid import RETRIEVAL_MODE, add_chunks, build_user_retriever
from providers import build_models, embed_model_name

NO_TEXT_ERROR = "No text found. Is this a scanned PDF?"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class RAGService:
    """
    Multi-tenant RAG Service.
    - Stores every user's chunks and vectors in Postgres + pgvector (see db.py).
    - Pairs dense search with an in-memory BM25 index per user; queries fuse both with RRF (see hybrid.py).
    - Caches each user's retriever in memory after first use (no eviction yet).
    """

    def __init__(self):
        self.lock = asyncio.Lock()
        self.embeddings = None
        self.llm = None
        # Map user_id -> hybrid (BM25 + pgvector) retriever
        self.retrievers: dict[str, HybridRetriever] = {}
        # Guards first-time builds: two builds for one user would leave a stale BM25 in place
        self._build_lock = asyncio.Lock()

    def initialize(self):
        db.init_db()
        self.embeddings, self.llm = build_models()
        print("✅ Database, embeddings and chat model ready.")

    async def get_retriever(self, user_id: str) -> HybridRetriever:
        """First call per user builds BM25 from their rows; later calls reuse it."""
        if user_id in self.retrievers:
            return self.retrievers[user_id]
        async with self._build_lock:
            # Re-check: another request may have finished the build while we waited for the lock
            if user_id not in self.retrievers:
                self.retrievers[user_id] = await asyncio.to_thread(
                    build_user_retriever, self.embeddings, user_id, embed_model_name()
                )
        return self.retrievers[user_id]

    # ──────────────────────────── Ingestion Pipeline ────────────────────────────

    async def ingest_file(self, file_path: str, filename: str, user_id: str) -> dict:
        """
        Ingests a file into a specific USER'S rows.
        """
        if not self.embeddings:
            raise RuntimeError("Service not initialized.")

        t0 = time.perf_counter()
        print(f"📥 Ingestion started for {user_id}: {filename}")

        sha = await asyncio.to_thread(_sha256, file_path)
        doc_id = await asyncio.to_thread(db.create_document, user_id, embed_model_name(), filename, sha)
        if doc_id is None:
            print(f"↩️  {filename} is already uploaded for {user_id}; skipped")
            return {"filename": filename, "status": "duplicate"}

        try:
            docs = await asyncio.to_thread(self._load_document, file_path, filename)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ". ", " ", ""]
            )
            splits = await asyncio.to_thread(text_splitter.split_documents, docs)
            if not splits:
                await asyncio.to_thread(db.finish_document, doc_id, 0, "error", NO_TEXT_ERROR)
                print(f"⚠️  {filename}: {NO_TEXT_ERROR}")
                return {"filename": filename, "status": "error", "error": NO_TEXT_ERROR}

            chunks = []
            for split in splits:
                chunk_id = str(uuid.uuid4())  # hyphenated, the same form Postgres returns
                chunks.append(Chunk(
                    id=chunk_id,
                    doc_id=filename,
                    text=split.page_content,
                    meta={**split.metadata, "chunk_id": chunk_id, "document_id": doc_id},
                ))

            retriever = await self.get_retriever(user_id)
            async with self.lock:
                await asyncio.to_thread(add_chunks, retriever, chunks)
            await asyncio.to_thread(db.finish_document, doc_id, len(chunks), "success")

            elapsed = round(time.perf_counter() - t0, 2)
            print(f"✅ Ingested {filename} for {user_id} in {elapsed}s")
            return {"filename": filename, "chunks": len(chunks), "status": "success"}

        except Exception as e:
            print(f"❌ Ingestion error: {e}")
            # Remove the record and any rows already written, so the same file can be retried.
            await asyncio.to_thread(db.delete_document, doc_id)
            return {"filename": filename, "status": "error", "error": str(e)}

    # ──────────────────────────── Retrieval ────────────────────────────

    async def stream_rag_response(self, query: str, user_id: str) -> AsyncGenerator[str, None]:
        if not self.embeddings:
            yield f"data: {json.dumps({'type': 'content', 'content': 'System initializing...'})}\n\n"
            return
            
        try:
             # 1. Retrieve (BM25 + dense, fused with RRF)
             result = await self.search(query, user_id, k=5)
             hits = result.hits
             yield f"data: {json.dumps({'type': 'retrieval', 'mode': RETRIEVAL_MODE, 'timings_ms': result.timings_ms})}\n\n"
             
             if not hits:
                 yield f"data: {json.dumps({'type': 'content', 'content': 'I do not have any documents related to this query yet. Please upload some.'})}\n\n"
                 return
                 
             context = "\n\n".join([h.chunk.text for h in hits])
             
             # 2. Construct Prompt
             system_prompt = f"""
             You are a helpful and precise RAG assistant. Use only the provided context to answer the user's question.
             If the answer is not in the context, say "I cannot find the answer in the provided documents."
             
             Context:
             {context}
             
             Question: {query}
             """
             
             # 3. Stream Response
             async for chunk in self.llm.astream(system_prompt):
                 if chunk.content:
                     # --- FIX 1: Guarantee string content ---
                     text_content = chunk.content
                     if isinstance(text_content, list):
                         text_content = "".join([str(t.get("text", t)) if isinstance(t, dict) else str(t) for t in text_content])
                     elif not isinstance(text_content, str):
                         text_content = str(text_content)
                         
                     yield f"data: {json.dumps({'type': 'content', 'content': text_content})}\n\n"
                     
             # 4. Send Citations
             sources_dict = {}
             for h in hits:
                 meta = h.chunk.meta
                 # Extract base filename to hide ugly temp paths like /var/folders/.../
                 raw_source = meta.get('source', 'Unknown')
                 clean_source = os.path.basename(raw_source) if raw_source != "Unknown" else "Unknown"
                 
                 key = f"{clean_source}_{meta.get('page', 0)}"
                 if key not in sources_dict:
                     sources_dict[key] = {
                         "source": clean_source,
                         "page": meta.get("page", 0),
                         "snippet": h.chunk.text[:150].replace("\n", " ") + "..."
                     }
                     
             yield f"data: {json.dumps({'type': 'citations', 'citations': list(sources_dict.values())})}\n\n"
             
        except Exception as e:
            print(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'content', 'content': f'Error generating response: {str(e)}'})}\n\n"

    async def search(self, query: str, user_id: str, k: int = 5, mode: str = RETRIEVAL_MODE):
        """Run retrieval only; returns a ragv2 SearchResult (hits + per-stage timings_ms)."""
        retriever = await self.get_retriever(user_id)
        return await asyncio.to_thread(retriever.search, query, k, mode)

    async def get_ingested_files(self, user_id: str) -> list[dict]:
        return await asyncio.to_thread(db.list_documents, user_id, embed_model_name())

    def _load_document(self, file_path: str, filename: str):
        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, autodetect_encoding=True)
        docs = loader.load()

        for d in docs:
            d.metadata["source"] = filename
            d.metadata.setdefault("page", 1)

        return docs



rag_service = RAGService()
