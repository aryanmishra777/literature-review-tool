"""Page navigation, the Cloudflare-challenge wait, and stable content reads.

The tricky part of scraping ACM is timing: a navigation may land on a Cloudflare
interstitial, the results may load asynchronously, and Playwright's ``content()`` throws
if called mid-redirect. The helpers here wait for the right state and distinguish a real
Cloudflare wall (worth waiting out, then giving up on) from an ordinary empty result.
"""
import asyncio
import sys

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeout

from retrieval.acm.config import (
    CF_TIMEOUT, EMPTY_RESULTS_TEXT, NAV_TIMEOUT, RESULTS_SELECTOR, WAIT_TIMEOUT,
)

# Signals that we're looking at a Cloudflare challenge rather than search results.
_CF_TITLES = {"just a moment", "checking your browser", "please wait"}
_CF_MARKERS = (
    "cloudflare", "checking your browser", "verify you are human",
    "just a moment", "cf-challenge", "turnstile",
)

# Transient error fragments that a brief wait + retry may clear.
_TRANSIENT_MARKERS = (
    "navigating", "err_name_not_resolved", "err_internet_disconnected",
    "err_connection", "err_timed_out", "target closed", "browser has been closed",
)


def is_retryable(exc: Exception) -> bool:
    """True for errors a short backoff may fix (CF timeout, mid-redirect grab, network blip)."""
    msg = str(exc).lower()
    return isinstance(exc, (PlaywrightTimeout, PlaywrightError, TimeoutError)) or any(
        marker in msg for marker in _TRANSIENT_MARKERS
    )


async def _is_cloudflare_page(page) -> bool:
    """Heuristically detect a Cloudflare interstitial from the title, URL, and body text."""
    try:
        title = (await page.title()).lower()
    except PlaywrightError:
        title = ""
    try:
        async with asyncio.timeout(2):
            body = (await page.locator("body").inner_text()).lower()
    except (TimeoutError, PlaywrightError):
        body = ""
    haystack = f"{title}\n{page.url.lower()}\n{body[:2000]}"
    return any(m in haystack for m in _CF_MARKERS) or any(t in title for t in _CF_TITLES)


async def _wait_for_results(page) -> None:
    """Resolve once ACM shows result items OR an explicit empty-results message.

    The caller wraps this in ``asyncio.timeout`` to bound how long we wait.
    """
    await page.wait_for_function(
        """
        ({ resultsSelector, emptyTexts }) => {
            if (document.querySelector(resultsSelector)) return true;
            const body = (document.body && document.body.innerText || '').toLowerCase();
            return emptyTexts.some((text) => body.includes(text));
        }
        """,
        arg={"resultsSelector": RESULTS_SELECTOR, "emptyTexts": list(EMPTY_RESULTS_TEXT)},
    )


async def read_content(page) -> str:
    """Read the page HTML once navigation has settled, retrying through redirects."""
    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            async with asyncio.timeout(10):
                await page.wait_for_load_state("domcontentloaded")
            return await page.content()
        except Exception as exc:
            last_exc = exc
            still_navigating = "navigating" in str(exc).lower() or isinstance(exc, (PlaywrightTimeout, TimeoutError))
            if not still_navigating:
                raise
            await asyncio.sleep(1.5 * attempt)
    assert last_exc is not None
    raise last_exc


async def navigate(page, url: str) -> None:
    """Go to ``url`` and wait for results, sitting through a Cloudflare challenge if present.

    Raises ``TimeoutError`` only when a Cloudflare wall fails to clear. An ordinary
    timeout (0 results / changed markup) is swallowed so the caller just parses an
    empty page instead of seeing an exception.
    """
    async with asyncio.timeout(NAV_TIMEOUT / 1000):
        await page.goto(url, wait_until="domcontentloaded")
    try:
        async with asyncio.timeout(WAIT_TIMEOUT / 1000):
            await _wait_for_results(page)
    except TimeoutError:
        if await _is_cloudflare_page(page):
            print("[acm] Cloudflare challenge detected — waiting for auto-resolution...", file=sys.stderr)
            try:
                async with asyncio.timeout(CF_TIMEOUT / 1000):
                    await _wait_for_results(page)
            except TimeoutError:
                raise TimeoutError(f"Cloudflare challenge did not resolve within {CF_TIMEOUT // 1000}s")
        # Non-CF timeout: results simply absent — fall through to an empty parse.

    # Make sure no navigation is in flight before the caller reads content().
    try:
        async with asyncio.timeout(10):
            await page.wait_for_load_state("domcontentloaded")
    except TimeoutError:
        pass  # read_content() will retry the grab anyway.
