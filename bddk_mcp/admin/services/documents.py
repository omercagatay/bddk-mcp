"""Document queries and edits for the admin console."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from bddk_mcp.admin.drafts import DraftConflict, DraftStore, EditForm, RevisionForm, fingerprint

MAX_PAGE_SIZE = 200
STORE_FAILURE = "Veri katmani kullanilamiyor."


@dataclass(frozen=True, slots=True)
class DocumentPage:
    """One page of document rows plus the flag a template needs for paging."""

    items: list[dict[str, Any]]
    page: int
    page_size: int
    has_next: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    """Search results, or the reason there are none."""

    query: str
    hits: list[Any]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class DocumentOutcome:
    """A single document lookup, or the reason it could not be loaded.

    Distinct from "not found": a store failure must never be rendered as a
    missing document, so callers branch on `error` before `doc is None`.
    """

    doc: Any | None
    error: str | None = None
    draft: dict | None = None
    base_fingerprint: str = ""


@dataclass(frozen=True, slots=True)
class SaveOutcome:
    """Document save result."""

    doc: Any | None = None
    error: str | None = None
    status: int = 200


class DocumentService:
    """Wraps DocumentStore so views never touch SQL or pagination arithmetic."""

    def __init__(self, store: Any, drafts: DraftStore | None = None) -> None:
        self._store = store
        self.drafts = drafts

    async def list_page(self, page: int = 1, page_size: int = 50, category: str | None = None) -> DocumentPage:
        page = max(1, page)
        page_size = max(1, min(page_size, MAX_PAGE_SIZE))
        offset = (page - 1) * page_size
        try:
            # Fetch one extra row: cheaper than a COUNT(*) and enough to know
            # whether a Next control should render.
            rows = await self._store.list_documents(category=category, limit=page_size + 1, offset=offset)
        except Exception:  # never rendered as an empty list or as the exception text
            return DocumentPage(items=[], page=page, page_size=page_size, has_next=False, error=STORE_FAILURE)
        has_next = len(rows) > page_size
        return DocumentPage(items=list(rows[:page_size]), page=page, page_size=page_size, has_next=has_next)

    async def get(self, doc_id: str) -> DocumentOutcome:
        try:
            doc = await self._store.get_document(doc_id)
            if doc is None:
                return DocumentOutcome(doc=None)
            base = fingerprint(doc)
            draft = await asyncio.to_thread(self.drafts.get, doc_id) if self.drafts else None
            if draft:
                doc = doc.model_copy(update=draft["payload"]["edited"])
            return DocumentOutcome(doc=doc, draft=draft, base_fingerprint=base)
        except Exception:  # never rendered as "not found" or as the exception text
            return DocumentOutcome(doc=None, error=STORE_FAILURE)

    async def save(self, doc_id: str, fields: dict[str, str], *, sign: bool = False) -> SaveOutcome:
        # Validate before reading canonical data; never accept provenance fields from a form.
        try:
            form = (RevisionForm if sign else EditForm).model_validate(fields)
        except ValueError:
            return SaveOutcome(error="Invalid or oversized document fields.", status=422)
        if self.drafts is None:
            return SaveOutcome(error="Draft editing disabled: configure BDDK_ADMIN_DRAFT_DB.", status=503)
        try:
            doc = await self._store.get_document(doc_id)
            if doc is None:
                return SaveOutcome(status=404)
            await asyncio.to_thread(self.drafts.change, doc, form, sign=sign)
            return SaveOutcome(doc=doc)
        except DraftConflict as exc:
            return SaveOutcome(error=str(exc), status=409)
        except ValueError:
            return SaveOutcome(error="Signing unavailable; configure matching editorial keys.", status=503)
        except Exception:
            return SaveOutcome(error=STORE_FAILURE, status=503)

    async def search(self, query: str, limit: int = 20) -> SearchOutcome:
        query = query.strip()
        if not query:
            return SearchOutcome(query="", hits=[])
        try:
            hits = list(await self._store.search_content(query, limit=limit))
        except Exception:  # never rendered as "no results" or as the exception text
            return SearchOutcome(query=query, hits=[], error=STORE_FAILURE)
        return SearchOutcome(query=query, hits=hits)

    async def categories(self) -> dict[str, int]:
        stats = await self._store.stats()
        return dict(stats.categories)
