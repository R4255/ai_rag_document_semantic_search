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
             docs = await asyncio.to_thread(retriever.invoke, query)
             
             if not docs:
                 yield f"data: {json.dumps({'type': 'content', 'content': 'I do not have any documents related to this query yet. Please upload some.'})}\n\n"
                 return
                 
             context = "\n\n".join([d.page_content for d in docs])
             
             # 2. Setup Gemini Model
             model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.3, google_api_key=os.environ["GOOGLE_API_KEY"])
             
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
                     # --- FIX 1: Guarantee string content ---
                     text_content = chunk.content
                     if isinstance(text_content, list):
                         text_content = "".join([str(t.get("text", t)) if isinstance(t, dict) else str(t) for t in text_content])
                     elif not isinstance(text_content, str):
                         text_content = str(text_content)
                         
                     yield f"data: {json.dumps({'type': 'content', 'content': text_content})}\n\n"
                     
             # 5. Send Citations
             sources_dict = {}
             for d in docs:
                 # Extract base filename to hide ugly temp paths like /var/folders/.../
                 raw_source = d.metadata.get('source', 'Unknown')
                 clean_source = os.path.basename(raw_source) if raw_source != "Unknown" else "Unknown"
                 
                 key = f"{clean_source}_{d.metadata.get('page', 0)}"
                 if key not in sources_dict:
                     sources_dict[key] = {
                         "source": clean_source,
                         "page": d.metadata.get("page", 0),
                         "snippet": d.page_content[:150].replace("\n", " ") + "..."
                     }
                     
             yield f"data: {json.dumps({'type': 'citations', 'citations': list(sources_dict.values())})}\n\n"
             
        except Exception as e:
            print(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'content', 'content': f'Error generating response: {str(e)}'})}\n\n"

    async def chat(self, query: str, user_id: str):
        # ...existing code...
            
            # Check Semantic Cache
            cached_response = self._check_cache(query_embedding)
            if cached_response:
                yield f"data: {json.dumps({'type': 'content', 'content': cached_response['answer']})}\n\n"
                yield f"data: {json.dumps({'type': 'citations', 'citations': cached_response['citations']})}\n\n"
                yield "data: [DONE]\n\n"
                return

            # --- FIX: Use .invoke() instead of .get_relevant_documents() ---
            print(f"🔍 [RAG] Searching vector store for: {query}")
            retriever = self.vector_stores[user_id].as_retriever(search_kwargs={"k": 5})
            
            # docs = retriever.get_relevant_documents(query) # <--- DELETE THIS
            docs = retriever.invoke(query)                   # <--- ADD THIS
            
            # Prepare context
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
            # Stream response
            # ...existing code...

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
