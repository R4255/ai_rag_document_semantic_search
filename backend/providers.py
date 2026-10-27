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

# Numbers per vector; must match vector(768) in schema.sql. nomic-embed-text produces 768
# natively; Gemini is asked for 768 (it defaults to 3072).
EMBED_DIM = 768

if LLM_PROVIDER not in ("gemini", "ollama"):
    raise ValueError("LLM_PROVIDER must be 'gemini' or 'ollama'")


def engine_label() -> str:
    if LLM_PROVIDER == "ollama":
        return f"Ollama ({OLLAMA_CHAT_MODEL})"
    return "Gemini 3 Flash"


def embed_model_name() -> str:
    """Vectors from different embedding models can't be compared, so every row records its model."""
    return OLLAMA_EMBED_MODEL if LLM_PROVIDER == "ollama" else GEMINI_EMBED_MODEL


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
        GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBED_MODEL, google_api_key=GOOGLE_API_KEY, output_dimensionality=EMBED_DIM
        ),
        ChatGoogleGenerativeAI(model=GEMINI_CHAT_MODEL, temperature=0.3, google_api_key=GOOGLE_API_KEY),
    )
