"""Endpoints, timeouts, and the browser fingerprint used by the ACM scraper.

The context options below mimic a real Chrome browser as closely as practical — a
plausible User-Agent, locale, timezone, and the ``Sec-Fetch-*`` headers a genuine
navigation sends. This is fingerprint *consistency*, not an attempt to defeat the
Cloudflare challenge (which we simply wait out, and otherwise give up on).
"""
import os
from urllib.parse import urlencode

_SEARCH_URL = "https://dl.acm.org/action/doSearch"
PAGE_SIZE = 20

# Timeouts in milliseconds (Playwright's unit). CF gets extra budget on a challenge page.
NAV_TIMEOUT = 60_000   # full page navigation
WAIT_TIMEOUT = 45_000  # waiting for results / empty-results state
CF_TIMEOUT = 75_000    # extra budget while a Cloudflare challenge resolves

MAX_ATTEMPTS = 3
BASE_RETRY_DELAY = 5.0  # seconds before retrying a transient failure (then doubles)
MIN_PAGE_JITTER = 2.0   # human-like random pause between page fetches
MAX_PAGE_JITTER = 7.0

# Optional residential proxy — set ACM_PROXY=http://user:pass@host:port in the env.
PROXY_URL: str | None = os.environ.get("ACM_PROXY") or None

# CSS that marks a real results page, and text that marks a genuine "no results" page.
RESULTS_SELECTOR = "li.search__item, li.issue-item-container"
EMPTY_RESULTS_TEXT = (
    "no results",
    "did not match any results",
    "your search returned no results",
)

LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--no-first-run",
    "--no-default-browser-check",
]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

CONTEXT_OPTIONS = {
    "user_agent": USER_AGENT,
    "viewport": {"width": 1280, "height": 800},
    "locale": "en-US",
    "timezone_id": "America/New_York",
    "extra_http_headers": {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "DNT": "1",
    },
}


def build_url(query: str, page: int) -> str:
    """Build an ACM search URL for ``query`` at zero-based ``page``."""
    return f"{_SEARCH_URL}?{urlencode({'AllField': query, 'startPage': page, 'pageSize': PAGE_SIZE})}"
