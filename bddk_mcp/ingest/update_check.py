"""What the live BDDK site lists and serves that a corpus file does not hold (`bddk-mcp check-updates`), and the
entries `bddk-mcp sync --add` takes from the site.

The check reads the corpus file and the site; it writes nothing. It compares three things:

- **new**: catalog entries the site lists and the corpus catalog does not hold. Pages 49, 50, 51, 54, 58 and 63
  are held in full apart from out-of-scope entries, so every entry they list that the catalog lacks is new. Pages
  56 and 55 list about 780 older decisions that were deliberately not selected, so there only an entry whose
  BDDK id is above the highest numeric id of the catalog is new (BDDK numbers documents in increasing order);
  page 55 (Resmi Gazetede Yayımlanan Kurul Kararları) is a group of its own because it is not in the default
  selection. Adding any document raises that bar, so an older entry left out is not reported again.
- **not listed**: catalog entries that no page lists. A document moved to another site section, one added by hand
  and one the site removed look alike; when a page failed, the list is incomplete.
- **changed**: stored documents whose source differs from what was downloaded, by one cheap signal per source.
  A BDDK document: the size the site declares for it (HEAD, no download) against the stored ``file_size``; a
  same-size change is not seen. A mevzuat.gov.tr text: the dates (DD.MM.YYYY or DD/MM/YYYY) of the official
  consolidated text that the stored text does not contain. An amendment brings the date of its Resmî Gazete into
  the text (an inline "(Değişik:RG-...)" note, a footnote or the closing amendment table); a change that brings no
  new date (a correction, a second change on a date already cited, an annex outside the official HTML) is not seen.
  An official text that shares no date with the stored one, or is far shorter, is reported as an error, not as
  unchanged, because it is probably not the document. The stored text is never compared with a fresh extraction,
  because a repaired text never equals its source.
"""

from __future__ import annotations

import asyncio
import logging
import re

import httpx
from bs4 import BeautifulSoup

from bddk_mcp.core.exceptions import BddkUpstreamError
from bddk_mcp.core.models import BddkDecisionSummary
from bddk_mcp.core.outbound_http import OutboundHttpPolicyError
from bddk_mcp.ingest.client import (
    _ALL_PAGE_IDS,
    _DECISION_PAGE_IDS,
    RG_DECISION_PAGE_ID,
    BddkApiClient,
    _canonicalize_catalog,
    _is_in_scope,
)
from bddk_mcp.ingest.doc_sync import _decode_html, _http_charset, _is_error_page, _parse_mevzuat_params
from bddk_mcp.store.doc_store import DocumentStore

logger = logging.getLogger(__name__)

EXIT_NOTHING_TO_REPORT = 0
EXIT_ERROR = 1
EXIT_SOMETHING_TO_REPORT = 3

CHECK_PAGE_IDS = [*_ALL_PAGE_IDS, RG_DECISION_PAGE_ID]
_THRESHOLD_PAGE_IDS = frozenset({*_DECISION_PAGE_IDS, RG_DECISION_PAGE_ID})
# The consolidated texts of mevzuat.gov.tr reach about 5 MB (the financial-statement annexes of mevzuat_16290).
_MAX_MEVZUAT_TEXT_BYTES = 16 * 1024 * 1024
_NUMBER_RE = re.compile(r"[0-9]{1,20}")
_DATE_RE = re.compile(r"(?<!\d)(\d{1,2})[./](\d{1,2})[./](\d{4})(?!\d)")
# On 2026-11-15 the shortest official text was 39% of its stored text (mevzuat_21192 stores annexes from Word).
_MIN_OFFICIAL_SHARE = 0.25
_CHECK_ERRORS = (httpx.HTTPError, OutboundHttpPolicyError, ValueError)

Pages = list[tuple[int, list[BddkDecisionSummary], str]]


def _summary(page_id: int, entry: BddkDecisionSummary) -> dict:
    return {
        "document_id": entry.document_id,
        "page": page_id,
        "category": entry.category,
        "decision_date": entry.decision_date,
        "decision_number": entry.decision_number,
        "title": entry.title,
        "source_url": entry.source_url,
    }


def compare_catalog(pages: Pages, catalog: list[dict]) -> dict:
    """The new and the unlisted entries of ``pages`` (page id, entries, error) against the catalog rows."""
    held = {row["document_id"]: row for row in catalog}
    newest = max((int(document_id) for document_id in held if _NUMBER_RE.fullmatch(document_id)), default=0)
    order = {page_id: position for position, page_id in enumerate(CHECK_PAGE_IDS)}
    new: dict[str, dict] = {}
    older: dict[int, set[str]] = {page_id: set() for page_id in sorted(_THRESHOLD_PAGE_IDS)}
    for page_id, entries, _error in sorted(pages, key=lambda page: order.get(page[0], len(order))):
        for entry in entries:
            document_id = entry.document_id
            if document_id in held or document_id in new or not _is_in_scope(entry):
                continue
            if page_id in _THRESHOLD_PAGE_IDS and not (_NUMBER_RE.fullmatch(document_id) and int(document_id) > newest):
                older[page_id].add(document_id)
                continue
            new[document_id] = _summary(page_id, entry)
    listed = {entry.document_id for _, entries, _ in pages for entry in entries}
    # Page order, then id (numbers by value), so a reordered page gives the same report.
    found = sorted(new.values(), key=lambda item: (order.get(item["page"]), item["document_id"].zfill(30)))
    return {
        "newest_catalogued_bddk_id": newest,
        "new": [item for item in found if item["page"] != RG_DECISION_PAGE_ID],
        "new_on_page_55": [item for item in found if item["page"] == RG_DECISION_PAGE_ID],
        "older_not_catalogued": {str(page_id): len(ids - set(new)) for page_id, ids in older.items()},
        "not_listed": [
            {"document_id": document_id, "category": row.get("category", ""), "title": row.get("title", "")}
            for document_id, row in sorted(held.items())
            if document_id not in listed
        ],
        "not_listed_complete": all(not error for _, _, error in pages),
    }


def text_dates(text: str) -> set[str]:
    """The ISO form of every DD.MM.YYYY or DD/MM/YYYY date of a text."""
    return {f"{year}-{int(month):02d}-{int(day):02d}" for day, month, year in _DATE_RE.findall(text)}


def _error_text(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return type(exc).__name__


async def _check_bddk(client: BddkApiClient, row: dict) -> tuple[str, str]:
    """("unchanged" | "changed" | "not_checked" | "error", reason) from the size BDDK declares for the file."""
    try:
        response = await client._fetch_with_retry(
            f"https://www.bddk.org.tr/Mevzuat/DokumanGetir/{row['document_id']}", method="HEAD"
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in (404, 410):
            return "changed", f"source answers HTTP {exc.response.status_code}"
        return "error", _error_text(exc)
    except _CHECK_ERRORS as exc:
        return "error", _error_text(exc)
    size = response.headers.get("content-length", "")
    if not _NUMBER_RE.fullmatch(size):
        return "not_checked", "the source declares no size"
    if int(size) != row["file_size"]:
        return "changed", f"source file is {size} bytes, {row['file_size']} when it was downloaded"
    return "unchanged", ""


async def _check_mevzuat(client: BddkApiClient, store: DocumentStore, row: dict) -> tuple[str, str]:
    """Amendment-note dates of the official consolidated text that the stored text does not have."""
    number, kind, series = _parse_mevzuat_params(row["source_url"])
    number = number or row["document_id"].removeprefix("mevzuat_")
    if not all(_NUMBER_RE.fullmatch(part) for part in (number, kind, series)):
        return "not_checked", "the source address names no mevzuat number, type and series"
    url = (
        "https://www.mevzuat.gov.tr/anasayfa/MevzuatFihristDetayIframe"
        f"?MevzuatTur={kind}&MevzuatNo={number}&MevzuatTertip={series}"
    )
    try:
        response = await client._fetch_with_retry(url, max_bytes=_MAX_MEVZUAT_TEXT_BYTES)
    except _CHECK_ERRORS as exc:
        return "error", _error_text(exc)
    page = _decode_html(response.content, _http_charset(response.headers))
    official = BeautifulSoup(page, "html.parser").get_text(" ")
    if _is_error_page(page) or not official.strip():
        return "error", "the source answered with an error page"
    stored = await store.get_document(row["document_id"])
    stored_text = stored.markdown_content if stored else ""
    official_dates, stored_dates = text_dates(official), text_dates(stored_text)
    if (stored_dates and not official_dates & stored_dates) or len(official) < _MIN_OFFICIAL_SHARE * len(stored_text):
        return "error", "the official text does not look like the stored document (no shared date, or far shorter)"
    added = official_dates - stored_dates
    if added:
        return "changed", f"official text has dates the stored text lacks: {', '.join(sorted(added))}"
    return "unchanged", ""


async def check_documents(store: DocumentStore, client: BddkApiClient) -> dict:
    """Compare every stored document with its source by the signal of its host (see the module docstring)."""
    rows = await store.list_document_sources()

    async def one(row: dict) -> tuple[str, str, str, str]:
        document_id = row["document_id"]
        if _NUMBER_RE.fullmatch(document_id):
            kind, check = "bddk", _check_bddk(client, row)
        elif re.fullmatch(r"mevzuat_[0-9]{1,20}", document_id):
            kind, check = "mevzuat", _check_mevzuat(client, store, row)
        else:
            return document_id, "other", "not_checked", "source host is not on the allowlist"
        async with client._scrape_sem:
            status, reason = await check
        if status == "error":
            logger.warning("Source of a stored document could not be checked", extra={"error_type": reason})
        return document_id, kind, status, reason

    results = await asyncio.gather(*(one(row) for row in rows))
    by_status: dict[str, list[dict]] = {"changed": [], "not_checked": [], "error": []}
    unchanged = {"bddk": 0, "mevzuat": 0}
    for document_id, kind, status, reason in results:
        if status == "unchanged":
            unchanged[kind] += 1
        else:
            by_status[status].append({"document_id": document_id, "reason": reason})
    return {
        "changed": by_status["changed"],
        "not_checked": by_status["not_checked"],
        "check_errors": by_status["error"],
        "unchanged": unchanged,
    }


async def check_updates(store: DocumentStore, client: BddkApiClient) -> dict:
    """The whole report: per page, new, not listed, changed and what could not be checked; plus a status."""
    pages = await client.read_catalog_pages(CHECK_PAGE_IDS)
    report = {
        "pages": [{"page": page_id, "entries": len(entries), "error": error} for page_id, entries, error in pages],
        **compare_catalog(pages, await store.load_catalog()),
        **await check_documents(store, client),
    }
    findings = report["new"] or report["new_on_page_55"] or report["not_listed"] or report["changed"]
    if not report["not_listed_complete"] or report["check_errors"]:
        report["status"] = "error"
    else:
        report["status"] = "something_to_report" if findings else "nothing_to_report"
    return report


def exit_code(report: dict) -> int:
    return {"nothing_to_report": EXIT_NOTHING_TO_REPORT, "something_to_report": EXIT_SOMETHING_TO_REPORT}.get(
        report["status"], EXIT_ERROR
    )


def _entry_line(item: dict) -> str:
    dated = " ".join(part for part in (item["decision_date"], item["decision_number"]) if part)
    return f"  {item['document_id']} [{item['category']}] page {item['page']}{' ' + dated if dated else ''}: {item['title']}"


def format_report(report: dict) -> str:
    lines = ["Catalog pages:"]
    for page in report["pages"]:
        if page["error"]:
            failed = "empty (no entry parsed)" if page["error"] == "empty_page" else f"failed ({page['error']})"
            lines.append(f"  page {page['page']}: {failed}")
        else:
            lines.append(f"  page {page['page']}: {page['entries']} entries")
    lines.append(f"New (listed on the site, not in the catalog): {len(report['new'])}")
    lines += [_entry_line(item) for item in report["new"]]
    lines.append(
        f"New on page 55 (Resmi Gazetede Yayımlanan Kurul Kararları, not in the default selection): "
        f"{len(report['new_on_page_55'])}"
    )
    lines += [_entry_line(item) for item in report["new_on_page_55"]]
    older = report["older_not_catalogued"]
    lines.append(
        f"Not reported: {older.get('56', 0)} entries of page 56 and {older.get('55', 0)} of page 55 that the catalog "
        f"does not hold, numbered at or below its newest BDDK id {report['newest_catalogued_bddk_id']}"
    )
    incomplete = "" if report["not_listed_complete"] else " (incomplete: a page could not be read)"
    lines.append(
        f"Not listed on any catalog page (moved, added by hand or removed; the site does not say which){incomplete}: "
        f"{len(report['not_listed'])}"
    )
    lines += [f"  {item['document_id']} [{item['category']}]: {item['title']}" for item in report["not_listed"]]
    lines.append(f"Changed at the source: {len(report['changed'])}")
    lines += [f"  {item['document_id']}: {item['reason']}" for item in report["changed"]]
    unchanged = report["unchanged"]
    lines.append(f"Unchanged by the signal of their source: {unchanged['bddk']} BDDK, {unchanged['mevzuat']} mevzuat")
    lines.append(f"Not checked: {len(report['not_checked'])}")
    lines += [f"  {item['document_id']}: {item['reason']}" for item in report["not_checked"]]
    if report["check_errors"]:
        lines.append(f"Could not be checked (error): {len(report['check_errors'])}")
        lines += [f"  {item['document_id']}: {item['reason']}" for item in report["check_errors"]]
    lines.append(f"Status: {report['status'].replace('_', ' ')}")
    return "\n".join(lines)


async def entries_to_add(
    store: DocumentStore, client: BddkApiClient, document_ids: list[str]
) -> list[BddkDecisionSummary]:
    """The live catalog entries of ``document_ids``; refuses, before any catalog row is written, an id the catalog
    already holds, one no page lists, one listed with conflicting records and one out of scope."""
    wanted = list(dict.fromkeys(document_ids))
    held = sorted(set(wanted) & {row["document_id"] for row in await store.load_catalog()})
    if held:
        raise RuntimeError(f"already in the catalog: {', '.join(held)} (use --doc-id to sync a catalogued document)")
    pages = await client.read_catalog_pages(CHECK_PAGE_IDS)
    try:
        found = {
            entry.document_id: entry
            for entry in _canonicalize_catalog(
                [entry for _, entries, _ in pages for entry in entries if entry.document_id in wanted]
            )
        }
    except BddkUpstreamError:
        raise RuntimeError("the catalog pages list the document with conflicting records; nothing was added") from None
    missing = [document_id for document_id in wanted if document_id not in found]
    if missing:
        failed = [str(page_id) for page_id, _, error in pages if error]
        unread = f" (pages that could not be read: {', '.join(failed)})" if failed else ""
        raise RuntimeError(f"not listed on the catalog pages: {', '.join(missing)}{unread}")
    out_of_scope = [document_id for document_id in wanted if not _is_in_scope(found[document_id])]
    if out_of_scope:
        raise RuntimeError(f"out of the corpus scope: {', '.join(out_of_scope)}")
    return [found[document_id] for document_id in wanted]
