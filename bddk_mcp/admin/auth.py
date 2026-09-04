"""Bearer or cookie JWT gate for a non-loopback admin bind."""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import PlainTextResponse, RedirectResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from bddk_mcp.http_security import HttpSecurityConfig, JwtTokenVerifier

COOKIE_NAME = "bddk_admin"


def token_from_request(request: Request) -> str:
    """Return the presented bearer or cookie token, or an empty string."""
    authorization = request.headers.get("authorization", "")
    scheme, _, remainder = authorization.partition(" ")
    if scheme.lower() == "bearer":
        return remainder.strip()
    return (request.cookies.get(COOKIE_NAME) or "").strip()


class AdminAuthMiddleware:
    """Require a verified operator JWT on every non-exempt request."""

    def __init__(self, app: ASGIApp, config: HttpSecurityConfig, verifier: JwtTokenVerifier) -> None:
        self._app = app
        self._allowed_hosts = frozenset(config.allowed_hosts)
        self._allowed_origins = frozenset(config.allowed_origins)
        self._verifier = verifier

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        request = Request(scope, receive)
        if request.url.path in {"/health/live", "/health/ready"}:
            await self._app(scope, receive, send)
            return
        host = request.headers.get("host", "")
        if host not in self._allowed_hosts:
            await PlainTextResponse("Invalid Host header", status_code=421)(scope, receive, send)
            return
        origin = request.headers.get("origin")
        if origin and origin not in self._allowed_origins:
            await PlainTextResponse("Invalid Origin header", status_code=403)(scope, receive, send)
            return
        if request.url.path in {"/login", "/logout"}:
            await self._app(scope, receive, send)
            return
        token = token_from_request(request)
        if token and await self._verifier.verify_token(token) is not None:
            await self._app(scope, receive, send)
            return
        await _reject(request)(scope, receive, send)


def _reject(request: Request) -> Response:
    accept = request.headers.get("accept", "")
    if request.method == "GET" and "text/html" in accept:
        return RedirectResponse("/login", status_code=303)
    return PlainTextResponse("Unauthorized", status_code=401)
