"""HTTP layer for the Crossref REST API.

Everything network-related lives here so the parsing and paging code stays pure and
easy to read. We follow Crossref's "polite pool" etiquette: a descriptive
``User-Agent`` plus a ``mailto`` on every request buys faster, more reliable service.
"""
import sys
import time

import requests

from config import USER_AGENT

# The single Crossref endpoint we use. `query.bibliographic` does a relevance search
# across title/author/container, which is exactly what we want for a topic query.
WORKS_URL = "https://api.crossref.org/works"

_TIMEOUT = 40            # seconds to wait on any single request (abstract-heavy pages are slow)
_MAX_RETRIES = 3         # total attempts before giving up on a request
_BASE_RETRY_DELAY = 2.0  # seconds; doubles each retry (exponential backoff)

# Trim the response to just the fields we map — smaller payloads, faster requests.
SELECT = ",".join([
    "DOI", "title", "author", "issued", "type",
    "container-title", "abstract", "is-referenced-by-count", "URL",
])

# Status codes worth retrying: rate-limit (429) and transient server errors (5xx).
# Anything else (e.g. a 400 from a bad query) won't fix itself, so we stop early.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def make_session() -> requests.Session:
    """A pre-configured session carrying our polite-pool User-Agent."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def fetch_message(session: requests.Session, params: dict) -> dict | None:
    """GET the works endpoint with bounded exponential backoff.

    Returns Crossref's ``message`` object on success, or ``None`` if the request
    ultimately failed — callers treat ``None`` as "stop paging, return what we have".
    """
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = session.get(WORKS_URL, params=params, timeout=_TIMEOUT)
            if resp.status_code == 200:
                return resp.json().get("message", {})
            if resp.status_code not in _RETRYABLE_STATUS:
                print(f"[crossref] HTTP {resp.status_code} — giving up on this request", file=sys.stderr)
                return None
            print(f"[crossref] HTTP {resp.status_code} (attempt {attempt}/{_MAX_RETRIES})", file=sys.stderr)
        except requests.RequestException as exc:
            print(f"[crossref] request error (attempt {attempt}/{_MAX_RETRIES}): {exc}", file=sys.stderr)

        # Back off before the next try, but not after the final attempt.
        if attempt < _MAX_RETRIES:
            time.sleep(_BASE_RETRY_DELAY * (2 ** (attempt - 1)))
    return None
