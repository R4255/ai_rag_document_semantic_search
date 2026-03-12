import os
import json
import asyncio
import time
from typing import AsyncGenerator

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

VECTOR_STORE_PATH = "./faiss_store"


class RAGService:
    """
    Core RAG Service Engine.
    - Uses Google Gemini (gemini-3-flash) for both embeddings and LLM inference.
    - FAISS as the local vector store (no external infra needed).
    - Full async pipeline: file I/O, chunking, embedding, retrieval, and streaming
      are all run via asyncio.to_thread to keep the event loop non-blocking.
    """

    def __init__(self):
        self.lock = asyncio.Lock()
        self.embeddings = None
        self.vector_store = None
        self._ingested_files: list[dict] = []  # Track ingested documents for the UI

    def initialize(self, api_key: str):
        if not api_key:
            print("WARNING: GOOGLE_API_KEY NOT SET.")
            return

        os.environ["GOOGLE_API_KEY"] = api_key
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=api_key,
        )

        if os.path.exists(VECTOR_STORE_PATH):
            try:
                self.vector_store = FAISS.load_local(
                    VECTOR_STORE_PATH,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                print("✅ Loaded existing FAISS index from disk.")
            except Exception as e:
                print(f"⚠️  Error loading FAISS index, creating fresh: {e}")
                self._init_empty_store()
        else:
            self._init_empty_store()

    def _init_empty_store(self):
        self.vector_store = FAISS.from_texts(
            ["System baseline document for vector store initialisation."],
            self.embeddings,
            metadatas=[{"source": "system", "page": 0}],
        )
        self.vector_store.save_local(VECTOR_STORE_PATH)
        print("✅ Created fresh FAISS index.")

    # ──────────────────────────── Ingestion Pipeline ────────────────────────────

    async def ingest_file(self, file_path: str, filename: str) -> dict:
        """
        Background-task compatible ingestion.
        1. PDF / TXT parsing  (blocking → executor)
        2. Recursive character chunking  (blocking → executor)
        3. Embedding + FAISS upsert  (lock-guarded, executor)
        """
        if not self.embeddings:
            raise RuntimeError("Service not initialised – missing GOOGLE_API_KEY.")

        t0 = time.perf_counter()
        print(f"📥 Ingestion started: {filename}")

        try:
            docs = await asyncio.to_thread(self._load_document, file_path, filename)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            splits = await asyncio.to_thread(text_splitter.split_documents, docs)

            print(f"   ↳ {filename}: {len(splits)} chunks created")

            async with self.lock:
                await asyncio.to_thread(self.vector_store.add_documents, splits)
                await asyncio.to_thread(self.vector_store.save_local, VECTOR_STORE_PATH)

            elapsed = round(time.perf_counter() - t0, 2)
            record = {
                "filename": filename,
                "chunks": len(splits),
                "elapsed_s": elapsed,
                "status": "success",
            }
            self._ingested_files.append(record)
            print(f"✅ Ingested {filename} ({len(splits)} chunks) in {elapsed}s")
            return record

        except Exception as e:
            print(f"❌ Ingestion error for {filename}: {e}")
            record = {"filename": filename, "status": "error", "error": str(e)}
            self._ingested_files.append(record)
            return record

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

    # ──────────────────────────── RAG Query Pipeline ────────────────────────────

    async def stream_rag_response(self, query: str) -> AsyncGenerator[str, None]:
        """
        Full RAG pipeline with SSE-formatted streaming output.
        Yields JSON lines: { type: "content" | "citations" | "error", ... }
        """
        if not self.vector_store:
            yield f'data: {json.dumps({"type": "error", "content": "Vector DB not initialised. Upload a document first."})}\n\n'
            return

        # 1. Retrieval
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        docs = await asyncio.to_thread(retriever.invoke, query)

        # 2. Build context with source markers
        context_parts = []
        citations = []
        seen = set()
        for d in docs:
            src = d.metadata.get("source", "Unknown")
            pg = d.metadata.get("page", "?")
            key = f"{src}::{pg}::{d.page_content[:80]}"
            if key in seen:
                continue
            seen.add(key)
            context_parts.append(
                f"[Source: {src}, Page {pg}]\n{d.page_content}"
            )
            citations.append({
                "source": src,
                "page": pg,
                "snippet": d.page_content[:250] + ("…" if len(d.page_content) > 250 else ""),
            })

        context_text = "\n\n---\n\n".join(context_parts)

        # 3. LLM Chain  –  Gemini 3 Flash 
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            streaming=True,
            temperature=0.3,
            max_output_tokens=2048,
        )
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a precise AI research assistant. "
                "Answer the user's question using ONLY the context below. "
                "Cite sources like [Source: filename, Page X] in your answer where applicable. "
                "If the answer is not in the context, say so clearly.\n\n"
                "Context:\n{context}",
            ),
            ("human", "{question}"),
        ])

        chain = prompt | llm | StrOutputParser()

        # 4. Stream chunks as SSE
        try:
            async for chunk in chain.astream({"context": context_text, "question": query}):
                yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"

            # 5. Append citations payload
            yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    # ──────────────────────────── Metadata helpers ──────────────────────────────

    def get_ingested_files(self) -> list[dict]:
        return list(self._ingested_files)


rag_service = RAGService()
