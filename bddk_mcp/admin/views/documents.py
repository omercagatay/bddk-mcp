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
        categories = await service.categories()
        return templates.TemplateResponse(
            request,
            "documents/list.html",
            {"page": page, "categories": categories, "selected_category": category},
        )

    routes.append(Route("/documents", list_documents, methods=["GET"], name="documents"))
