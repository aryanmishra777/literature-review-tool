"""Tiny shared HTTP helpers for the enrichment sources."""
import requests

from config import USER_AGENT

# One timeout for every enrichment request. Generous, because these are third-party
# APIs that occasionally take a few seconds, and a slow enricher must never wedge a run.
TIMEOUT = 25


def new_session() -> requests.Session:
    """A fresh session carrying our polite-pool User-Agent.

    Note: callers that fan out across threads should give *each* thread its own session
    (``requests.Session`` is not guaranteed thread-safe), so this returns a new object
    every call rather than a shared singleton.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session
