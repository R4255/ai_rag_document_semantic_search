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
    Multi-tenant RAG Service.
    - Manages separate FAISS indices per user_id (session ID).
    - Caches active indices in memory (LRU style not implemented, simpler dict cache).
    """

    def __init__(self):
        self.lock = asyncio.Lock()
        self.embeddings = None
        # Map user_id -> FAISS index
        self.vector_stores: dict[str, FAISS] = {}
        # Map user_id -> list of ingested files metadata
        self._user_files: dict[str, list[dict]] = {}

    def initialize(self, api_key: str):
        if not api_key:
            print("WARNING: GOOGLE_API_KEY NOT SET.")
            return

        os.environ["GOOGLE_API_KEY"] = api_key
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=api_key,
        )
        print("✅ Embeddings model ready.")

    def _get_user_store_path(self, user_id: str) -> str:
        safe_id = "".join([c for c in user_id if c.isalnum() or c in "-_"])
        return os.path.join(VECTOR_STORE_PATH, safe_id)

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
            metadatas=[{"source": "system", "page": 0}]
        )
        self.vector_stores[user_id] = vs
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

            async with self.lock:
                vs = await self.get_vector_store(user_id)
                await asyncio.to_thread(vs.add_documents, splits)
                
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
             # Ensure user index exists
             vs = await self.get_vector_store(user_id)
             retriever = vs.as_retriever(search_kwargs={"k": 5})
             
             # 1. Retrieve
             docs = await asyncio.to_thread(retriever.get_relevant_documents, query)
             
             if not docs:
                 yield f"data: {json.dumps({'type': 'content', 'content': 'I do not have any documents related to this query yet. Please upload some.'})}\n\n"
                 return
                 
             context = "\n\n".join([d.page_content for d in docs])
             
             # 2. Setup Gemini Model
             model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=os.environ["GOOGLE_API_KEY"])
             
             # 3. Construct Prompt
             system_prompt = f"""
             You are a helpful and precise RAG assistant. Use only the provided context to answer the user's question.
             If the answer is not in the context, say "I cannot find the answer in the provided documents."
             
             Context:
             {context}
             
             Question: {query}
             """
             
             # 4. Stream Response
             async for chunk in model.astream(system_prompt):
                 if chunk.content:
                     yield f"data: {json.dumps({'type': 'content', 'content': chunk.content})}\n\n"
                     
             # 5. Send Citations
             sources = list(set([f"{d.metadata.get('source', 'Unknown')} (Pg {d.metadata.get('page', 0)})" for d in docs]))
             yield f"data: {json.dumps({'type': 'citations', 'citations': sources})}\n\n"
             
        except Exception as e:
            print(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'content', 'content': f'Error generating response: {str(e)}'})}\n\n"

    def get_ingested_files(self, user_id: str = None) -> list[dict]:
        return self._user_files.get(user_id, [])

    def _load_document(self, file_path: str, filename: str):
         # ... existing implementation ...
        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, autodetect_encoding=True)
        return loader.load()

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
