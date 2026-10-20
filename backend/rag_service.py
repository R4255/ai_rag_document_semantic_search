import os
import json
import asyncio
import time
import uuid
from typing import AsyncGenerator

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ragv2 import Chunk, HybridRetriever
from hybrid import RETRIEVAL_MODE, SYSTEM_SOURCE, add_chunks, build_user_retriever
from providers import build_models, store_namespace

VECTOR_STORE_PATH = "./faiss_store"


class RAGService:
    """
    Multi-tenant RAG Service.
    - Manages separate FAISS indices per user_id (session ID).
    - Pairs each index with a BM25 index; queries fuse both with RRF (see hybrid.py).
    - Caches active indices in memory (LRU style not implemented, simpler dict cache).
    """

    def __init__(self):
        self.lock = asyncio.Lock()
        self.embeddings = None
        self.llm = None
        # Map user_id -> FAISS index
        self.vector_stores: dict[str, FAISS] = {}
        # Map user_id -> hybrid (BM25 + FAISS) retriever over the same chunks
        self.retrievers: dict[str, HybridRetriever] = {}
        # Map user_id -> list of ingested files metadata
        self._user_files: dict[str, list[dict]] = {}

    def initialize(self):
        self.embeddings, self.llm = build_models()
        print("✅ Embeddings + chat models ready.")

    def _get_user_store_path(self, user_id: str) -> str:
        safe_id = "".join([c for c in user_id if c.isalnum() or c in "-_"])
        return os.path.join(VECTOR_STORE_PATH, store_namespace(), safe_id)

    async def get_vector_store(self, user_id: str):
        """Lazy loads or creates a vector store for a specific user ID."""
        if user_id in self.vector_stores:
            return self.vector_stores[user_id]

        path = self._get_user_store_path(user_id)
        
        # Try loading existing
        if os.path.exists(path) and os.path.exists(os.path.join(path, "index.faiss")):
            try:
                # Running load_local in thread to avoid blocking
                vs = await asyncio.to_thread(
                    FAISS.load_local, 
                    path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                self.vector_stores[user_id] = vs
                self.retrievers[user_id] = await asyncio.to_thread(build_user_retriever, vs)
                
                # Load metadata if exists
                meta_path = os.path.join(path, "files.json")
                if os.path.exists(meta_path):
                     with open(meta_path, "r") as f:
                         self._user_files[user_id] = json.load(f)
                else:
                     self._user_files[user_id] = []
                     
                print(f"✅ Loaded FAISS index for user: {user_id}")
                return vs
            except Exception as e:
                print(f"⚠️ Error loading store for {user_id}, initializing fresh: {e}")

        # Initialize fresh store
        vs = await asyncio.to_thread(
            FAISS.from_texts,
            ["User space initialization."], 
            self.embeddings,
            metadatas=[{"source": SYSTEM_SOURCE, "page": 0, "chunk_id": SYSTEM_SOURCE}],
            ids=[SYSTEM_SOURCE],
        )
        self.vector_stores[user_id] = vs
        self.retrievers[user_id] = build_user_retriever(vs)
        self._user_files[user_id] = []
        
        # Create dir if not exists
        os.makedirs(path, exist_ok=True)
        await asyncio.to_thread(vs.save_local, path)
        return vs

    # ──────────────────────────── Ingestion Pipeline ────────────────────────────

    async def ingest_file(self, file_path: str, filename: str, user_id: str) -> dict:
        """
        Ingests a file into a specific USER'S vector store.
        """
        if not self.embeddings:
            raise RuntimeError("Service not initialized.")

        t0 = time.perf_counter()
        print(f"📥 Ingestion started for {user_id}: {filename}")

        try:
            docs = await asyncio.to_thread(self._load_document, file_path, filename)
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ". ", " ", ""]
            )
            splits = await asyncio.to_thread(text_splitter.split_documents, docs)
            chunks = []
            for split in splits:
                chunk_id = uuid.uuid4().hex
                chunks.append(Chunk(
                    id=chunk_id,
                    doc_id=filename,
                    text=split.page_content,
                    meta={**split.metadata, "chunk_id": chunk_id},
                ))

            async with self.lock:
                vs = await self.get_vector_store(user_id)
                await asyncio.to_thread(add_chunks, self.retrievers[user_id], chunks)
                
                # Save to user specific path
                store_path = self._get_user_store_path(user_id)
                await asyncio.to_thread(vs.save_local, store_path)

            elapsed = round(time.perf_counter() - t0, 2)
            record = {
                "filename": filename,
                "chunks": len(splits),
                "timestamp": time.time(),
                "status": "success",
            }
            
            # Update user file list
            if user_id not in self._user_files: self._user_files[user_id] = []
            self._user_files[user_id].append(record)
            
            # Persist metadata
            try:
                with open(os.path.join(self._get_user_store_path(user_id), "files.json"), "w") as f:
                    json.dump(self._user_files[user_id], f)
            except Exception as e:
                print(f"Error saving metadata: {e}")

            print(f"✅ Ingested {filename} for {user_id} in {elapsed}s")
            return record

        except Exception as e:
            print(f"❌ Ingestion error: {e}")
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
        await self.get_vector_store(user_id)
        return await asyncio.to_thread(self.retrievers[user_id].search, query, k, mode)

    def get_ingested_files(self, user_id: str = None) -> list[dict]:
        return self._user_files.get(user_id, [])

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
