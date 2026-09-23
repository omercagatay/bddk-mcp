"""Upload, correction and admission-request pages; never a public-corpus write."""

from __future__ import annotations

import asyncio
from urllib.parse import quote

from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response
from starlette.routing import Route
from starlette.templating import Jinja2Templates

from bddk_mcp.admin.csrf import FormSecurity
from bddk_mcp.admin.uploads import UploadStore

_REJECTION_TEXT = {"unsupported_type": "Sadece PDF veya DOCX yuklenebilir."}
_FORM_ERROR = "Dosya okunamadi veya cok buyuk."
_NOT_CORRECTED = "Bu belge henuz duzeltilmis metin icermiyor."
_ALREADY_WAITING = "Bu belge icin bekleyen bir istek var."


def register(routes: list, templates: Jinja2Templates, store: UploadStore) -> None:
    security = FormSecurity()

    def render(request: Request, template: str, context: dict, *, status_code: int = 200) -> Response:
        nonce, token = security.issue(request)
        response = templates.TemplateResponse(
            request, template, {**context, "csrf_token": token}, status_code=status_code
        )
        return security.attach(request, response, nonce)

    def not_found(request: Request, upload_id: str) -> Response:
        return templates.TemplateResponse(request, "not_found.html", {"document_id": upload_id}, status_code=404)

    def upload_url(upload_id: str, suffix: str) -> str:
        return f"/uploads/{quote(upload_id, safe='')}{suffix}"

    async def new_upload(request: Request) -> Response:
        return render(request, "uploads/new.html", {"error": None})

    async def post_upload(request: Request) -> Response:
        try:
            form = await security.read_multipart(request)
        except HTTPException as exc:
            # Starlette surfaces malformed/oversized multipart bodies as 400
            # with an English detail; the operator page stays Turkish.
            message = _FORM_ERROR if exc.status_code == 400 else exc.detail
            return render(request, "uploads/new.html", {"error": message}, status_code=exc.status_code)
        file = form.get("file")
        if not isinstance(file, UploadFile):
            return render(request, "uploads/new.html", {"error": "Dosya secilmedi."}, status_code=422)
        data = await file.read()
        reason = store.reject_reason(file.filename or "", data)
        if reason is not None:
            return render(request, "uploads/new.html", {"error": _REJECTION_TEXT.get(reason, reason)}, status_code=415)
        upload_id = await asyncio.to_thread(store.save, file.filename or "", data)
        try:
            await asyncio.to_thread(store.extract, upload_id)
        except Exception:
            # The bytes stay outside the corpus; the operator can re-upload.
            return render(request, "uploads/new.html", {"error": "Metin cikarilamadi."}, status_code=422)
        return RedirectResponse(upload_url(upload_id, "/edit"), status_code=303)

    async def edit_upload(request: Request) -> Response:
        upload_id = request.path_params["upload_id"]
        try:
            text = await asyncio.to_thread(store.extract, upload_id)
        except FileNotFoundError:
            return not_found(request, upload_id)
        except Exception:
            return render(
                request,
                "uploads/edit.html",
                {"upload_id": upload_id, "text": "", "error": "Metin cikarilamadi."},
                status_code=422,
            )
        return render(request, "uploads/edit.html", {"upload_id": upload_id, "text": text, "error": None})

    async def save_upload_edit(request: Request) -> Response:
        upload_id = request.path_params["upload_id"]
        try:
            fields = await security.read(request)
        except HTTPException as exc:
            return render(
                request,
                "uploads/edit.html",
                {"upload_id": upload_id, "text": "", "error": exc.detail},
                status_code=exc.status_code,
            )
        text = fields.get("text", "")
        try:
            await asyncio.to_thread(store.save_correction, upload_id, text)
        except FileNotFoundError:
            return not_found(request, upload_id)
        except ValueError:
            return render(
                request,
                "uploads/edit.html",
                {"upload_id": upload_id, "text": text, "error": "Kaydedilemedi: metin bos veya gecersiz."},
                status_code=422,
            )
        return RedirectResponse(upload_url(upload_id, "/edit"), status_code=303)

    async def admit_page(request: Request) -> Response:
        upload_id = request.path_params["upload_id"]
        try:
            await asyncio.to_thread(store.path_for, upload_id)
        except FileNotFoundError:
            return not_found(request, upload_id)
        request_id = await asyncio.to_thread(store.latest_request, upload_id)
        state = await asyncio.to_thread(store.request_state, request_id) if request_id else None
        return render(request, "uploads/admit.html", {"upload_id": upload_id, "state": state, "error": None})

    async def post_admit(request: Request) -> Response:
        upload_id = request.path_params["upload_id"]
        try:
            await security.read(request)
        except HTTPException as exc:
            # A rejected form must still render the live request state, so a
            # waiting request never re-offers the admit button.
            request_id = await asyncio.to_thread(store.latest_request, upload_id)
            state = await asyncio.to_thread(store.request_state, request_id) if request_id else None
            return render(
                request,
                "uploads/admit.html",
                {"upload_id": upload_id, "state": state, "error": exc.detail},
                status_code=exc.status_code,
            )
        try:
            await asyncio.to_thread(store.admit, upload_id)
        except FileNotFoundError:
            return not_found(request, upload_id)
        except ValueError as exc:
            reason = str(exc)
            message = _NOT_CORRECTED if reason == "not_corrected" else _ALREADY_WAITING
            status_code = 409 if reason == "already_waiting" else 422
            request_id = await asyncio.to_thread(store.latest_request, upload_id)
            state = await asyncio.to_thread(store.request_state, request_id) if request_id else None
            return render(
                request,
                "uploads/admit.html",
                {"upload_id": upload_id, "state": state, "error": message},
                status_code=status_code,
            )
        return RedirectResponse(upload_url(upload_id, "/admit"), status_code=303)

    routes.extend(
        [
            Route("/uploads/new", new_upload, methods=["GET"], name="upload_new"),
            Route("/uploads", post_upload, methods=["POST"], name="upload_post"),
            Route("/uploads/{upload_id}/edit", edit_upload, methods=["GET"], name="upload_edit"),
            Route("/uploads/{upload_id}/edit", save_upload_edit, methods=["POST"], name="upload_edit_save"),
            Route("/uploads/{upload_id}/admit", admit_page, methods=["GET"], name="upload_admit"),
            Route("/uploads/{upload_id}/admit", post_admit, methods=["POST"], name="upload_admit_post"),
        ]
    )
