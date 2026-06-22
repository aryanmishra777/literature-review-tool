"""Top-level ACM search: launch a stealth browser, fetch pages, dedupe, return records."""
import asyncio
import sys

from playwright.async_api import async_playwright
from playwright_stealth import Stealth

from models import CSLRecord
from retrieval.base import BaseTranslator
from retrieval.acm.config import CONTEXT_OPTIONS, LAUNCH_ARGS, PAGE_SIZE, PROXY_URL
from retrieval.acm.fetch import fetch_page_zero, fetch_remaining

# Applied to the whole browser context, so every page it opens is patched once.
_STEALTH = Stealth()


def _dedupe(records: list[CSLRecord]) -> list[CSLRecord]:
    """Drop duplicate works (same DOI, or same title-hash id), preserving order."""
    seen: set[str] = set()
    out: list[CSLRecord] = []
    for r in records:
        key = r.DOI or r.id
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _pages_needed(total: int, limit: int | None) -> int:
    """How many pages *after* page 0 we still need to reach ``limit`` (capped by ``total``)."""
    n_target = min(limit, total) if limit is not None else total
    return max(0, (n_target + PAGE_SIZE - 1) // PAGE_SIZE - 1)


async def _async_search(query: str, limit: int | None, workers: int) -> list[CSLRecord]:
    workers = max(1, workers or 1)
    launch_kwargs: dict = {"headless": True, "channel": "chrome", "args": LAUNCH_ARGS}
    if PROXY_URL:
        launch_kwargs["proxy"] = {"server": PROXY_URL}
        print(f"[acm] routing through proxy: {PROXY_URL.split('@')[-1]}", file=sys.stderr)

    all_records: list[CSLRecord] = []
    pages_ok = pages_failed = 0

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(**launch_kwargs)
        context = await browser.new_context(**CONTEXT_OPTIONS)
        await _STEALTH.apply_stealth_async(context)
        try:
            first_records, total = await fetch_page_zero(context, query)
            all_records.extend(first_records)
            pages_ok += 1
            print(f"[acm] page 0 — {len(first_records)} records (total reported: {total})", file=sys.stderr)

            remaining = _pages_needed(total, limit)
            if remaining:
                more, ok, failed = await fetch_remaining(context, query, remaining, workers)
                all_records.extend(more)
                pages_ok += ok
                pages_failed += failed
        except Exception as exc:
            if pages_ok == 0:
                print(f"[acm] retrieval failed entirely: {exc}", file=sys.stderr)
        finally:
            if pages_ok or pages_failed:
                print(f"[acm] done — {pages_ok} page(s) succeeded, {pages_failed} skipped", file=sys.stderr)
            await browser.close()

    deduped = _dedupe(all_records)
    return deduped[:limit] if limit is not None else deduped


class ACMScraper(BaseTranslator):
    """Best-effort ACM Digital Library scraper. See the package docstring for caveats."""

    def search(self, query: str, limit: int | None = None, workers: int = 2) -> list[CSLRecord]:
        return asyncio.run(_async_search(query, limit, workers))
