"""Login and logout for a remote admin bind."""

from __future__ import annotations

from typing import Any

from starlette.requests import Request
from starlette.responses import RedirectResponse, Response
from starlette.routing import Route
from starlette.templating import Jinja2Templates

from bddk_mcp.admin.auth import COOKIE_NAME, verify_admin_token

_MAX_COOKIE_TOKEN_CHARS = 3500


def register(
    routes: list, templates: Jinja2Templates, verifier: Any, *, required_scopes: frozenset[str], secure_cookie: bool
) -> None:
    async def login(request: Request) -> Response:
        if request.method == "GET":
            return templates.TemplateResponse(request, "login.html", {"error": None})
        form = await request.form()
        token = str(form.get("token") or "").strip()
        if not await verify_admin_token(verifier, token, required_scopes):
            return templates.TemplateResponse(
                request, "login.html", {"error": "Gecersiz erisim anahtari."}, status_code=401
            )
        if len(token) > _MAX_COOKIE_TOKEN_CHARS:
            return templates.TemplateResponse(
                request,
                "login.html",
                {"error": "Anahtar cerez icin cok buyuk; Authorization: Bearer kullanin."},
                status_code=401,
            )
        response = RedirectResponse("/documents", status_code=303)
        response.set_cookie(
            COOKIE_NAME,
            token,
            httponly=True,
            samesite="strict",
            secure=secure_cookie,
            path="/",
        )
        return response

    async def logout(_request: Request) -> Response:
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(COOKIE_NAME, path="/")
        return response

    routes.append(Route("/login", login, methods=["GET", "POST"], name="login"))
    routes.append(Route("/logout", logout, methods=["POST"], name="logout"))
