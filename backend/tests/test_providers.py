import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import providers  # noqa: E402


class _Resp:
    def __init__(self, models):
        self._models = models

    def raise_for_status(self):
        pass

    def json(self):
        return {"models": [{"name": n} for n in self._models]}


def _ps_returns(monkeypatch, *names):
    monkeypatch.setattr(providers, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(providers, "OLLAMA_CHAT_MODEL", "qwen2.5:7b-instruct")
    monkeypatch.setattr(providers, "OLLAMA_EMBED_MODEL", "nomic-embed-text")
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp(names))


def test_both_loaded_untagged_name_matches_latest(monkeypatch):
    _ps_returns(monkeypatch, "qwen2.5:7b-instruct", "nomic-embed-text:latest")
    assert providers.model_status() == {"chat": True, "embed": True}


def test_embed_model_unloaded(monkeypatch):
    _ps_returns(monkeypatch, "qwen2.5:7b-instruct")
    assert providers.model_status() == {"chat": True, "embed": False}


def test_ollama_unreachable_is_unknown(monkeypatch):
    _ps_returns(monkeypatch)

    def down(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", down)
    assert providers.model_status() is None


def test_gemini_has_no_local_models(monkeypatch):
    monkeypatch.setattr(providers, "LLM_PROVIDER", "gemini")
    assert providers.model_status() is None
    providers.warm_models()  # no-op, no network


def test_keep_alive_durations_become_seconds():
    assert [providers._seconds(v) for v in ("30m", "1h", "90s", "600", "-1", " 2H ")] == [1800, 3600, 90, 600, -1, 7200]


def test_ollama_models_build_with_keep_alive(monkeypatch):
    """Regression: OllamaEmbeddings rejects keep_alive='30m' (it wants seconds), which disabled the app."""
    monkeypatch.setattr(providers, "LLM_PROVIDER", "ollama")
    embeddings, llm = providers.build_models()
    assert isinstance(providers.OLLAMA_KEEP_ALIVE, int)
    assert embeddings.keep_alive == llm.keep_alive == providers.OLLAMA_KEEP_ALIVE
