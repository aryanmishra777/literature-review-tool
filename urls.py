"""Best-effort URL shortening via is.gd.

Paper / DOI / open-access links are long, so we shorten them for the Markdown table.
This is purely cosmetic: if is.gd is slow, down, or rate-limiting, every helper falls
back to the original URL rather than failing the run.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

_CREATE_URL = "https://is.gd/create.php"
_TIMEOUT = 6
_WORKERS = 6


def shorten_url(url: str) -> str:
    """Shorten one URL, returning the original on empty input or any failure."""
    if not url:
        return url
    try:
        resp = requests.get(_CREATE_URL, params={"format": "simple", "url": url}, timeout=_TIMEOUT)
        short = resp.text.strip()
        if resp.ok and short.startswith("http"):
            return short
    except Exception:
        pass
    return url


def shorten_urls(urls: list[str]) -> list[str]:
    """Shorten a list of URLs in parallel, preserving input order."""
    results: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        futures = {pool.submit(shorten_url, url): i for i, url in enumerate(urls)}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    return [results[i] for i in range(len(urls))]
