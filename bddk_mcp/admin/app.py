"""Starlette application factory for the admin console."""

from __future__ import annotations

from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response
from starlette.routing import Route
from starlette.templating import Jinja2Templates

from bddk_mcp.admin.config import AdminConfig
from bddk_mcp.admin.services.documents import DocumentService
from bddk_mcp.admin.services.governance import GovernanceService
from bddk_mcp.admin.views import documents as documents_view
from bddk_mcp.admin.views import governance as governance_view

_PACKAGE_ROOT = Path(__file__).resolve().parent


def create_app(
    config: AdminConfig,
    document_service: DocumentService,
    governance_service: GovernanceService,
) -> Starlette:
    """Build the admin console app from already-resolved collaborators."""

    templates = Jinja2Templates(directory=str(_PACKAGE_ROOT / "templates"))

    async def root(_request: Request) -> Response:
        return RedirectResponse("/documents")

    routes: list = [Route("/", root, methods=["GET"])]
    documents_view.register(routes, templates, document_service)
    governance_view.register(routes, templates, governance_service)

    app = Starlette(
        routes=routes,
        middleware=[
            Middleware(
                TrustedHostMiddleware,
                allowed_hosts=("127.0.0.1", "localhost", "[::1]"),
            )
        ],
    )
    app.state.config = config
    return app
