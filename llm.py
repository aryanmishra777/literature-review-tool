"""Chat-client factory: one ``.chat()`` interface over Ollama or Groq.

Both ``QueryUnderstanding`` and ``Synthesizer`` talk to an LLM the same way — they call
``client.chat(model=, messages=, options={"temperature": ...})`` and read
``response.message.content``. This module returns a client matching that shape regardless
of provider, so the call sites don't care which backend is in use.

- **ollama**: the official ``ollama.Client`` (used as-is; covers local + Ollama cloud).
- **groq**: a thin wrapper over Groq's OpenAI-compatible REST endpoint via ``requests``
  (no extra dependency), shimmed to expose the same ``response.message.content`` shape.
"""
import requests
import ollama

from config import USER_AGENT


class ChatError(Exception):
    """Provider-agnostic chat failure. Mirrors the role of ``ollama.ResponseError``."""


class _Message:
    def __init__(self, content: str):
        self.content = content


class _Response:
    """Minimal stand-in for an ollama chat response (only ``.message.content`` is used)."""
    def __init__(self, content: str):
        self.message = _Message(content)


class _GroqClient:
    """Calls Groq's OpenAI-compatible ``/chat/completions`` and adapts the response shape."""

    _TIMEOUT = 120

    def __init__(self, host: str, api_key: str):
        self._url = f"{host.rstrip('/')}/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }

    def chat(self, model: str, messages: list[dict], options: dict | None = None) -> _Response:
        body: dict = {"model": model, "messages": messages}
        temperature = (options or {}).get("temperature")
        if temperature is not None:
            body["temperature"] = temperature
        try:
            resp = requests.post(self._url, headers=self._headers, json=body,
                                 timeout=self._TIMEOUT)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        except requests.HTTPError as exc:
            # Surface Groq's error body — it explains bad model ids, rate limits, etc.
            # NB: a Response with a 4xx/5xx status is falsy, so test `is not None`.
            resp = exc.response
            status = resp.status_code if resp is not None else "?"
            detail = resp.text[:300] if resp is not None else str(exc)
            raise ChatError(f"Groq HTTP {status}: {detail}") from exc
        except (requests.RequestException, KeyError, ValueError) as exc:
            raise ChatError(f"Groq request failed: {exc}") from exc
        return _Response(content)


def make_client(provider: str, host: str, api_key: str):
    """Build a chat client for ``provider`` ('ollama' or 'groq').

    The returned object exposes ``chat(model, messages, options)`` → response with
    ``.message.content``, so callers are provider-agnostic.
    """
    if provider == "groq":
        if not api_key:
            raise ChatError("Groq selected but GROQ_API_KEY is not set (add it to .env).")
        return _GroqClient(host, api_key)

    kwargs: dict = {"host": host}
    if api_key:
        kwargs["headers"] = {"Authorization": f"Bearer {api_key}"}
    return ollama.Client(**kwargs)
