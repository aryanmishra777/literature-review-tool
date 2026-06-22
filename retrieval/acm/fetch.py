"""Per-page fetching: open a page, navigate, parse — with retries and a concurrency cap.

Page 0 is special (it also tells us the total hit count, and a hard failure there means
the whole search failed), so it has its own helper. The remaining pages are fetched
concurrently under a semaphore.
"""
import asyncio
import random
import sys

from models import CSLRecord
from retrieval.acm.browser import navigate, is_retryable, read_content
from retrieval.acm.config import (
    BASE_RETRY_DELAY, MAX_ATTEMPTS, MAX_PAGE_JITTER, MIN_PAGE_JITTER, build_url,
)
from retrieval.acm.parse import parse_html


async def _polite_delay(page_num: int) -> None:
    """Pause a random human-like interval before a fetch (skipped for page 0)."""
    if page_num > 0:
        await asyncio.sleep(random.uniform(MIN_PAGE_JITTER, MAX_PAGE_JITTER))


def _backoff(attempt: int) -> float:
    """Exponential backoff with jitter, in seconds, for retry ``attempt`` (1-based)."""
    return BASE_RETRY_DELAY * (2 ** (attempt - 1)) + random.uniform(0.0, 3.0)


async def _attempt_fetch(context, query: str, page_num: int):
    """Open a fresh page, navigate, and parse it. Returns ``(records, total, page)``."""
    page = await context.new_page()
    await _polite_delay(page_num)
    await navigate(page, build_url(query, page_num))
    records, total = parse_html(await read_content(page))
    return records, total, page


async def fetch_page_zero(context, query: str) -> tuple[list[CSLRecord], int]:
    """Fetch page 0 with retries. Raises if it never succeeds (caller treats as fatal)."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        page = None
        try:
            records, total, page = await _attempt_fetch(context, query, 0)
            return records, total
        except Exception as exc:
            print(f"[acm] page 0 failed (attempt {attempt}/{MAX_ATTEMPTS}): {exc}", file=sys.stderr)
            if not is_retryable(exc) or attempt == MAX_ATTEMPTS:
                raise
            await asyncio.sleep(_backoff(attempt))
        finally:
            if page and not page.is_closed():
                await page.close()
    raise RuntimeError("unreachable")


async def fetch_page(context, sem: asyncio.Semaphore, query: str, page_num: int, total_pages: int) -> list[CSLRecord]:
    """Fetch one results page under the concurrency semaphore; never raises.

    Retries transient failures, then gives up on that page and returns ``[]`` so one bad
    page can't sink the whole search.
    """
    async with sem:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            page = None
            try:
                records, _, page = await _attempt_fetch(context, query, page_num)
                print(f"[acm] page {page_num}/{total_pages} — {len(records)} records", file=sys.stderr)
                return records
            except Exception as exc:
                label = "timeout" if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) else "error"
                print(f"[acm] page {page_num} {label} (attempt {attempt}/{MAX_ATTEMPTS}): {exc}", file=sys.stderr)
                if is_retryable(exc) and attempt < MAX_ATTEMPTS:
                    await asyncio.sleep(_backoff(attempt))
                    continue
                break
            finally:
                if page and not page.is_closed():
                    await page.close()
        print(f"[acm] page {page_num}/{total_pages} — skipped after failed attempts", file=sys.stderr)
        return []


async def fetch_remaining(context, query: str, remaining_pages: int, workers: int):
    """Fetch pages 1..N concurrently. Returns ``(records, pages_ok, pages_failed)``."""
    sem = asyncio.Semaphore(workers)
    tasks = [fetch_page(context, sem, query, p, remaining_pages) for p in range(1, remaining_pages + 1)]
    results = await asyncio.gather(*tasks)
    records = [rec for page_records in results for rec in page_records]
    pages_ok = sum(1 for r in results if r)
    pages_failed = sum(1 for r in results if not r)
    return records, pages_ok, pages_failed
