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


def _friendly_error(e: Exception) -> str:
    """A message a user can act on, instead of the provider's raw error."""
    msg = str(e)
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "Too Many Requests" in msg:
        return "The AI service is busy right now (free-tier limit). Please try again in about a minute."
    return "Something went wrong while writing the answer. Please try again."


def _tidy_text(text: str) -> str:
    """
    Some PDFs extract one word per line ("Build\\n \\na\\n \\nsimple"), with two or more
    blank lines marking a real paragraph. Rejoin those words; leave normal text alone.
    The reader applies the same rule (frontend lib/api.ts tidyText) to text stored earlier.
    """
    lines = text.split("\n")
    words = [line.strip() for line in lines if line.strip()]
    if len(words) < 8 or sum(" " not in w for w in words) / len(words) < 0.8:
        return text
    out, blank = "", 0
    for line in lines:
        w = line.strip()
        if not w:
            blank += 1
            continue
        if out:
            out += "\n\n" if blank >= 2 else "\n" if w in ("-", "•", "*") else " "
        out += w
        blank = 0
    return out


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
            for seq, split in enumerate(splits):
                chunk_id = str(uuid.uuid4())  # hyphenated, the same form Postgres returns
                chunks.append(Chunk(
                    id=chunk_id,
                    doc_id=filename,
                    text=split.page_content,
                    meta={**split.metadata, "chunk_id": chunk_id, "document_id": doc_id, "seq": seq},
                ))

            # Fetch the retriever under the lock, so a delete can't swap it out between fetch and add
            async with self.lock:
                retriever = await self.get_retriever(user_id)
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
                 
             # 2. Construct Prompt: passages are numbered so the answer can cite them as [1], [2]...
             passages = "\n\n".join(
                 f"[{n}] {h.chunk.meta.get('source', 'Unknown')}, page {h.chunk.meta.get('page', 1)}:\n{h.chunk.text}"
                 for n, h in enumerate(hits, start=1)
             )
             system_prompt = f"""
             You are a helpful and precise assistant. Answer the user's question using only the numbered passages below.
             After each sentence that uses a passage, cite it with its number in square brackets, like [1] or [2][3].
             If the passages do not contain the answer, say "I cannot find the answer in the provided documents." and cite nothing.

             Passages:
             {passages}

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
                     
             # 4. Send Citations: one per numbered passage, in prompt order, so [n] maps to citations[n-1]
             citations = []
             for n, h in enumerate(hits, start=1):
                 meta = h.chunk.meta
                 # Extract base filename to hide ugly temp paths like /var/folders/.../
                 raw_source = meta.get('source', 'Unknown')
                 citations.append({
                     "n": n,
                     "chunk_id": h.chunk.id,
                     "source": os.path.basename(raw_source) if raw_source != "Unknown" else "Unknown",
                     "page": meta.get("page", 1),
                     "snippet": h.chunk.text[:150].replace("\n", " ") + "...",
                 })

             yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"

        except Exception as e:
            print(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': _friendly_error(e)})}\n\n"

    async def search(self, query: str, user_id: str, k: int = 5, mode: str = RETRIEVAL_MODE):
        """Run retrieval only; returns a ragv2 SearchResult (hits + per-stage timings_ms)."""
        retriever = await self.get_retriever(user_id)
        return await asyncio.to_thread(retriever.search, query, k, mode)

    async def delete_document(self, user_id: str, doc_id: str) -> str:
        """Returns 'deleted', 'processing' or 'missing' (see db.delete_user_document)."""
        # Same lock order as ingest_file (self.lock, then _build_lock inside get_retriever)
        async with self.lock:
            outcome = await asyncio.to_thread(db.delete_user_document, user_id, embed_model_name(), doc_id)
            if outcome == "deleted":
                # BM25 can't remove chunks, so drop this user's index; the next request rebuilds it
                # from Postgres. Dense search needs nothing: the rows are gone (CASCADE).
                async with self._build_lock:
                    self.retrievers.pop(user_id, None)
        return outcome

    async def get_ingested_files(self, user_id: str) -> list[dict]:
        return await asyncio.to_thread(db.list_documents, user_id, embed_model_name())

    async def is_duplicate(self, file_path: str, user_id: str) -> bool:
        """Same file already uploaded by this user? (ingest_file's unique key is the final guard.)"""
        sha = await asyncio.to_thread(_sha256, file_path)
        return await asyncio.to_thread(db.document_exists, user_id, embed_model_name(), sha)

    async def get_document_chunks(self, user_id: str, filename: str) -> list[dict]:
        rows = await asyncio.to_thread(db.document_chunks, user_id, embed_model_name(), filename)
        return [{"chunk_id": str(cid), "page": page, "text": text} for cid, page, text in rows]

    def _load_document(self, file_path: str, filename: str):
        is_pdf = filename.lower().endswith(".pdf")
        loader = PyPDFLoader(file_path) if is_pdf else TextLoader(file_path, autodetect_encoding=True)
        docs = loader.load()

        for d in docs:
            d.page_content = _tidy_text(d.page_content)
            d.metadata["source"] = filename
            if is_pdf:
                d.metadata["page"] = int(d.metadata.get("page", 0)) + 1  # PyPDFLoader counts from 0
            else:
                d.metadata.setdefault("page", 1)

        return docs



rag_service = RAGService()
