"""Document browser routes."""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route
from starlette.templating import Jinja2Templates

from bddk_mcp.admin.services.documents import DocumentService


def _int_param(request: Request, name: str, default: int) -> int:
    raw = request.query_params.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def register(routes: list, templates: Jinja2Templates, service: DocumentService) -> None:
    async def list_documents(request: Request) -> Response:
        category = request.query_params.get("category") or None
        page = await service.list_page(page=_int_param(request, "page", 1), category=category)
        # Don't hit an already-failing store a second time for the filter
        # dropdown; an empty category list is harmless once page.error says why.
        categories = {} if page.error else await service.categories()
        return templates.TemplateResponse(
            request,
            "documents/list.html",
            {"page": page, "categories": categories, "selected_category": category},
        )

    async def document_detail(request: Request) -> Response:
        document_id = request.path_params["document_id"]
        outcome = await service.get(document_id)
        if outcome.error:
            return templates.TemplateResponse(
                request,
                "documents/error.html",
                {"error": outcome.error, "document_id": document_id},
            )
        if outcome.doc is None:
            return templates.TemplateResponse(
                request,
                "not_found.html",
                {"document_id": document_id},
                status_code=404,
            )
        return templates.TemplateResponse(request, "documents/detail.html", {"doc": outcome.doc})

    async def search(request: Request) -> Response:
        outcome = await service.search(request.query_params.get("q", ""))
        return templates.TemplateResponse(request, "documents/search.html", {"outcome": outcome})

    routes.append(Route("/documents", list_documents, methods=["GET"], name="documents"))
    routes.append(Route("/documents/{document_id}", document_detail, methods=["GET"], name="document_detail"))
    routes.append(Route("/search", search, methods=["GET"], name="search"))
