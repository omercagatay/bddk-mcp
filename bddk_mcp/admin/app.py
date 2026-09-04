"""Starlette application factory for the admin console."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse, RedirectResponse, Response
from starlette.routing import Route
from starlette.templating import Jinja2Templates
from starlette.types import ASGIApp

from bddk_mcp.admin.auth import AdminAuthMiddleware
from bddk_mcp.admin.config import AdminConfig
from bddk_mcp.admin.services.documents import DocumentService
from bddk_mcp.admin.services.governance import GovernanceService
from bddk_mcp.admin.views import documents as documents_view
from bddk_mcp.admin.views import governance as governance_view
from bddk_mcp.admin.views import session as session_view

_PACKAGE_ROOT = Path(__file__).resolve().parent


def create_app(
    config: AdminConfig,
    document_service: DocumentService,
    governance_service: GovernanceService,
    *,
    token_verifier: Any | None = None,
) -> ASGIApp:
    """Build the admin console app from already-resolved collaborators."""

    templates = Jinja2Templates(directory=str(_PACKAGE_ROOT / "templates"))

    async def root(_request: Request) -> Response:
        return RedirectResponse("/documents")

    async def live(_request: Request) -> Response:
        return PlainTextResponse("ok")

    routes: list = [
        Route("/", root, methods=["GET"]),
        Route("/health/live", live, methods=["GET"]),
        Route("/health/ready", live, methods=["GET"]),
    ]
    documents_view.register(routes, templates, document_service)
    governance_view.register(routes, templates, governance_service)
    if token_verifier is not None:
        session_view.register(routes, templates, token_verifier, secure_cookie=not config.loopback_only)

    middleware = []
    if config.loopback_only:
        middleware = [
            Middleware(
                TrustedHostMiddleware,
                allowed_hosts=("127.0.0.1", "localhost", "[::1]"),
            )
        ]
    app = Starlette(routes=routes, middleware=middleware)
    app.state.config = config
    if token_verifier is not None and config.http_security is not None:
        return AdminAuthMiddleware(app, config.http_security, token_verifier)
    return app
