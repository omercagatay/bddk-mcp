"""Document browser routes."""

from __future__ import annotations

from urllib.parse import quote

from starlette.requests import Request
from starlette.responses import RedirectResponse, Response
from starlette.routing import Route
from starlette.templating import Jinja2Templates

from bddk_mcp.admin.csrf import FormSecurity
from bddk_mcp.admin.drafts import canonical_json
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
    security = FormSecurity()

    def render_document(request, template, outcome):
        nonce, token = security.issue(request)
        draft = outcome.draft
        payload = draft["payload"] if draft else {}
        response = templates.TemplateResponse(
            request,
            template,
            {
                "doc": outcome.doc,
                "draft": draft,
                "revision": payload.get("revision", 0),
                "base_fingerprint": payload.get("base_fingerprint", outcome.base_fingerprint),
                "stale_base": bool(draft and payload["base_fingerprint"] != outcome.base_fingerprint),
                "editing_enabled": service.drafts is not None,
                "signing_enabled": bool(service.drafts and service.drafts.signer),
                "csrf_token": token,
            },
        )
        return security.attach(request, response, nonce)

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
        return render_document(request, "documents/detail.html", outcome)

    async def edit_document(request: Request) -> Response:
        document_id = request.path_params["document_id"]
        outcome = await service.get(document_id)
        if outcome.error:
            return templates.TemplateResponse(
                request,
                "documents/error.html",
                {"error": outcome.error, "document_id": document_id},
            )
        if outcome.doc is None:
            return templates.TemplateResponse(request, "not_found.html", {"document_id": document_id}, status_code=404)
        return render_document(request, "documents/edit.html", outcome)

    async def save_document(request: Request) -> Response:
        document_id = request.path_params["document_id"]
        fields = await security.read(request)
        outcome = await service.save(document_id, fields, sign=request.url.path.endswith("/sign"))
        if outcome.error:
            return templates.TemplateResponse(
                request,
                "documents/error.html",
                {"error": outcome.error, "document_id": document_id},
                status_code=outcome.status,
            )
        if outcome.doc is None:
            return templates.TemplateResponse(request, "not_found.html", {"document_id": document_id}, status_code=404)
        return RedirectResponse(f"/documents/{quote(document_id, safe='')}", status_code=303)

    async def download_document(request: Request) -> Response:
        outcome = await service.get(request.path_params["document_id"])
        if outcome.error:
            return Response("Draft unavailable", status_code=503)
        if not outcome.draft:
            return Response("Save a draft first", status_code=404)
        return Response(
            canonical_json(outcome.draft),
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="editorial-draft.json"', "Cache-Control": "no-store"},
        )

    async def search(request: Request) -> Response:
        outcome = await service.search(request.query_params.get("q", ""))
        return templates.TemplateResponse(request, "documents/search.html", {"outcome": outcome})

    routes.append(Route("/documents", list_documents, methods=["GET"], name="documents"))
    routes.append(Route("/documents/{document_id}", document_detail, methods=["GET"], name="document_detail"))
    routes.append(Route("/documents/{document_id}/edit", edit_document, methods=["GET"], name="document_edit"))
    routes.append(Route("/documents/{document_id}/edit", save_document, methods=["POST"], name="document_save"))
    routes.append(Route("/documents/{document_id}/sign", save_document, methods=["POST"], name="document_sign"))
    routes.append(
        Route("/documents/{document_id}/draft.json", download_document, methods=["GET"], name="document_download")
    )
    routes.append(Route("/search", search, methods=["GET"], name="search"))
