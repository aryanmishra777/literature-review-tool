"""ACM Digital Library scraper — a best-effort *fallback* source (``--source acm``).

This exists for completeness, but in practice ``dl.acm.org`` sits behind a Cloudflare
managed challenge that a headless browser usually cannot clear, so it frequently returns
nothing. Crossref is the real default. The scraper is kept here, well-isolated, in case a
run happens from an environment (or proxy) that can reach ACM directly.

Layout:
  * ``config``  — endpoints, timeouts, and browser fingerprint constants.
  * ``parse``   — BeautifulSoup extraction of records from a results page.
  * ``browser`` — navigation, the Cloudflare-challenge dance, and stable content reads.
  * ``fetch``   — per-page fetching with retries and a concurrency limit.
  * ``search``  — the top-level async search loop and the public ``ACMScraper``.
"""
from retrieval.acm.search import ACMScraper

__all__ = ["ACMScraper"]
