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


def _seconds(value: str) -> int:
    """'30m', '1h', '90s' or '600' → seconds; '-1' keeps a model loaded forever."""
    value = value.strip().lower()
    units = {"s": 1, "m": 60, "h": 3600}
    if value[-1:] in units:
        return int(float(value[:-1]) * units[value[-1]])
    return int(value)


# How long Ollama keeps a model in memory after its last use (its own default is 5m,
# after which the next question waits seconds for the model to load again). In seconds,
# because OllamaEmbeddings only accepts a number.
OLLAMA_KEEP_ALIVE = _seconds(os.getenv("OLLAMA_KEEP_ALIVE", "30m"))

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
            OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL, keep_alive=OLLAMA_KEEP_ALIVE),
            ChatOllama(model=OLLAMA_CHAT_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.3, keep_alive=OLLAMA_KEEP_ALIVE),
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


# ──────────────────────────── Ollama model warmth ────────────────────────────

def _tagged(name: str) -> str:
    """Ollama reports untagged models as name:latest."""
    return name if ":" in name else f"{name}:latest"


def model_status() -> dict[str, bool] | None:
    """Which Ollama models are in memory right now ({"chat": .., "embed": ..}); None for Gemini
    or when Ollama can't be reached. A model that isn't loaded makes the next request wait."""
    if LLM_PROVIDER != "ollama":
        return None
    import httpx

    try:
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/ps", timeout=2)
        r.raise_for_status()
    except httpx.HTTPError:
        return None
    loaded = {_tagged(m.get("name", "")) for m in r.json().get("models", [])}
    return {"chat": _tagged(OLLAMA_CHAT_MODEL) in loaded, "embed": _tagged(OLLAMA_EMBED_MODEL) in loaded}


def warm_models() -> None:
    """Load both Ollama models into memory now, so the first question doesn't pay for it.
    An empty /api/generate request loads a chat model without generating anything."""
    if LLM_PROVIDER != "ollama":
        return
    import httpx

    with httpx.Client(base_url=OLLAMA_BASE_URL, timeout=120) as client:
        client.post("/api/embed", json={"model": OLLAMA_EMBED_MODEL, "input": "warm up", "keep_alive": OLLAMA_KEEP_ALIVE}).raise_for_status()
        client.post("/api/generate", json={"model": OLLAMA_CHAT_MODEL, "keep_alive": OLLAMA_KEEP_ALIVE}).raise_for_status()
