"""
Model provider selection: Google Gemini (cloud) or Ollama (local, no API key).

LLM_PROVIDER defaults to "gemini" when GOOGLE_API_KEY is set, otherwise "ollama".
"""
import os

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LLM_PROVIDER = os.getenv("LLM_PROVIDER") or ("gemini" if GOOGLE_API_KEY else "ollama")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b-instruct")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

GEMINI_CHAT_MODEL = "gemini-3-flash-preview"
GEMINI_EMBED_MODEL = "gemini-embedding-001"

if LLM_PROVIDER not in ("gemini", "ollama"):
    raise ValueError("LLM_PROVIDER must be 'gemini' or 'ollama'")


def engine_label() -> str:
    if LLM_PROVIDER == "ollama":
        return f"Ollama ({OLLAMA_CHAT_MODEL})"
    return "Gemini 3 Flash"


def store_namespace() -> str:
    """Vectors from different embedding models can't share an index, so each gets its own folder.
    Gemini keeps the root folder so stores created before this switch still load."""
    if LLM_PROVIDER == "ollama":
        return "ollama-" + "".join(c if c.isalnum() else "-" for c in OLLAMA_EMBED_MODEL)
    return ""


def build_models():
    """Returns (embeddings, chat_model) for the configured provider."""
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama, OllamaEmbeddings

        return (
            OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL),
            ChatOllama(model=OLLAMA_CHAT_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.3),
        )

    if not GOOGLE_API_KEY:
        raise RuntimeError("LLM_PROVIDER=gemini requires GOOGLE_API_KEY")
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

    return (
        GoogleGenerativeAIEmbeddings(model=GEMINI_EMBED_MODEL, google_api_key=GOOGLE_API_KEY),
        ChatGoogleGenerativeAI(model=GEMINI_CHAT_MODEL, temperature=0.3, google_api_key=GOOGLE_API_KEY),
    )
