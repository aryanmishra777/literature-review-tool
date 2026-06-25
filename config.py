"""Loads .env from the project root and exposes Ollama cloud config."""
import os
from pathlib import Path


def _load_env() -> dict[str, str]:
    env_path = Path(__file__).resolve().parent / ".env"
    env: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env


_env = _load_env()

OLLAMA_API_KEY: str = _env.get("OLLAMA_API_KEY") or os.environ.get("OLLAMA_API_KEY", "")
OLLAMA_HOST: str = "https://ollama.com" if OLLAMA_API_KEY else "http://localhost:11434"
DEFAULT_MODEL: str = "gpt-oss:120b-cloud"

# ── Groq provider (OpenAI-compatible chat API) ─────────────────────────────── #
# Selected with `--provider groq`. Groq is cloud-only and requires a key (no local
# fallback like Ollama has). Set GROQ_API_KEY in .env. The default model is Groq's
# hosted GPT-OSS 120B; override with --model (e.g. "llama-3.3-70b-versatile").
GROQ_API_KEY: str = _env.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY", "")
GROQ_HOST: str = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL: str = "openai/gpt-oss-120b"


def provider_defaults(provider: str) -> tuple[str, str, str]:
    """Return (host, api_key, default_model) for the chosen LLM provider."""
    if provider == "groq":
        return GROQ_HOST, GROQ_API_KEY, GROQ_DEFAULT_MODEL
    return OLLAMA_HOST, OLLAMA_API_KEY, DEFAULT_MODEL

# Contact address sent to scholarly APIs (Crossref/OpenAlex `mailto`, Unpaywall `email`).
# Identifying yourself opts you into the faster, more reliable "polite pool" and is
# required by Unpaywall. Override via .env CONTACT_EMAIL.
CONTACT_EMAIL: str = (
    _env.get("CONTACT_EMAIL")
    or os.environ.get("CONTACT_EMAIL")
    or "lr-tool@example.com"
)

# Shared User-Agent for all outbound API requests (polite-pool best practice).
USER_AGENT: str = f"lr_tool/0.1 (mailto:{CONTACT_EMAIL})"
