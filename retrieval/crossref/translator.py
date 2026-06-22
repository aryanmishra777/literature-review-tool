"""Cursor-paged search loop and the public ``CrossrefTranslator``."""
import sys

from config import CONTACT_EMAIL
from models import CSLRecord
from retrieval.base import BaseTranslator
from retrieval.crossref.client import SELECT, fetch_message, make_session
from retrieval.crossref.parse import parse_item

_MAX_ROWS_PER_REQUEST = 100  # Crossref allows up to 1000; 100 keeps payloads sane.
_CURSOR_HARD_CAP = 1000      # safety ceiling when the caller asks for "unlimited".


def _search(query: str, limit: int | None) -> list[CSLRecord]:
    """Page through Crossref results until we have ``limit`` records (or run out).

    Paging uses Crossref's deep-paging cursor rather than offsets. We dedupe by DOI as
    we go, because the same work can surface more than once across page boundaries.
    """
    session = make_session()
    # A missing/zero limit means "as many as is reasonable" — capped for safety.
    target = limit if (limit is not None and limit > 0) else _CURSOR_HARD_CAP

    records: list[CSLRecord] = []
    seen: set[str] = set()
    cursor = "*"  # "*" asks Crossref to start a fresh cursor.

    while len(records) < target:
        rows = min(_MAX_ROWS_PER_REQUEST, target - len(records))
        message = fetch_message(session, {
            "query.bibliographic": query,
            "rows": rows,
            "cursor": cursor,
            "select": SELECT,
            "mailto": CONTACT_EMAIL,
        })
        if not message:
            break  # request failed — return whatever we've gathered so far.

        items = message.get("items", [])
        if not items:
            break

        for item in items:
            rec = parse_item(item)
            if rec and rec.DOI not in seen:
                seen.add(rec.DOI)
                records.append(rec)

        cursor = message.get("next-cursor")
        # No cursor, or a short page, means the result set is exhausted.
        if not cursor or len(items) < rows:
            break

    print(f"[crossref] retrieved {len(records)} records", file=sys.stderr)
    return records[:limit] if (limit is not None and limit > 0) else records


class CrossrefTranslator(BaseTranslator):
    """Default retrieval source — see the package docstring for the rationale."""

    def search(self, query: str, limit: int | None = None, workers: int = 2) -> list[CSLRecord]:
        # `workers` exists only for interface parity with the ACM scraper; Crossref
        # paging is inherently sequential (each cursor depends on the previous page),
        # so the parameter is intentionally unused here.
        return _search(query, limit)
