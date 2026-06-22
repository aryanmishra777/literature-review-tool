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
DEFAULT_MODEL: str = "gemma4:31b-cloud"

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
