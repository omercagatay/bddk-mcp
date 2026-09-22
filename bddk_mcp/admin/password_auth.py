"""One shared operator password for a remote admin console without an IdP."""

from __future__ import annotations

import hmac
import secrets

from starlette.requests import Request
from starlette.responses import PlainTextResponse, RedirectResponse, Response
from starlette.routing import Route
from starlette.templating import Jinja2Templates
from starlette.types import ASGIApp, Receive, Scope, Send

from bddk_mcp.admin.auth import COOKIE_NAME
from bddk_mcp.admin.config import AdminConfig


class PasswordSessions:
    """Process-local login cookies. A restart signs everyone out."""

    def __init__(self, password: str) -> None:
        self._password = password
        self._valid: set[str] = set()

    def accepts(self, presented: str) -> bool:
        return hmac.compare_digest(presented, self._password)

    def issue(self) -> str:
        token = secrets.token_urlsafe(32)
        self._valid.add(token)
        return token

    def drop(self, token: str) -> None:
        self._valid.discard(token)

    def valid(self, token: str) -> bool:
        return bool(token) and token in self._valid


def register(routes: list, templates: Jinja2Templates, sessions: PasswordSessions, *, secure_cookie: bool) -> None:
    async def login(request: Request) -> Response:
        if request.method == "GET":
            return templates.TemplateResponse(request, "login.html", {"error": None, "label": "Sifre"})
        form = await request.form()
        presented = str(form.get("token") or "")
        if not sessions.accepts(presented):
            return templates.TemplateResponse(
                request, "login.html", {"error": "Sifre yanlis.", "label": "Sifre"}, status_code=401
            )
        response = RedirectResponse("/documents", status_code=303)
        response.set_cookie(
            COOKIE_NAME,
            sessions.issue(),
            httponly=True,
            samesite="strict",
            secure=secure_cookie,
            path="/",
        )
        return response

    async def logout(request: Request) -> Response:
        sessions.drop((request.cookies.get(COOKIE_NAME) or "").strip())
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(COOKIE_NAME, path="/")
        return response

    routes.append(Route("/login", login, methods=["GET", "POST"], name="login"))
    routes.append(Route("/logout", logout, methods=["POST"], name="logout"))


class PasswordAuthMiddleware:
    """Require the operator password on every non-exempt request."""

    def __init__(self, app: ASGIApp, config: AdminConfig, sessions: PasswordSessions) -> None:
        self._app = app
        self._hosts = frozenset(config.allowed_hosts)
        self._origins = frozenset(config.allowed_origins)
        self._sessions = sessions

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        request = Request(scope, receive)
        if request.url.path in {"/health/live", "/health/ready"}:
            await self._app(scope, receive, send)
            return
        host = request.headers.get("host", "")
        if host not in self._hosts:
            await PlainTextResponse("Invalid Host header", status_code=421)(scope, receive, send)
            return
        origin = request.headers.get("origin")
        if origin and origin not in self._origins:
            await PlainTextResponse("Invalid Origin header", status_code=403)(scope, receive, send)
            return
        if request.url.path in {"/login", "/logout"}:
            await self._app(scope, receive, send)
            return
        if self._sessions.valid((request.cookies.get(COOKIE_NAME) or "").strip()):
            await self._app(scope, receive, send)
            return
        await _reject(request)(scope, receive, send)


def _reject(request: Request) -> Response:
    accept = request.headers.get("accept", "")
    if request.method == "GET" and "text/html" in accept:
        return RedirectResponse("/login", status_code=303)
    return PlainTextResponse("Unauthorized", status_code=401)
