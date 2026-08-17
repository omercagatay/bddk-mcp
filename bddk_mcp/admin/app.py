"""Starlette application factory for the admin console."""

from __future__ import annotations

from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response
from starlette.routing import Route
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from bddk_mcp.admin.config import AdminConfig
from bddk_mcp.admin.services.documents import DocumentService
from bddk_mcp.admin.views import documents as documents_view

_PACKAGE_ROOT = Path(__file__).resolve().parent


def create_app(config: AdminConfig, document_service: DocumentService) -> Starlette:
    """Build the admin console app from already-resolved collaborators."""

    templates = Jinja2Templates(directory=str(_PACKAGE_ROOT / "templates"))

    async def root(_request: Request) -> Response:
        return RedirectResponse("/documents")

    routes: list = [Route("/", root, methods=["GET"])]
    documents_view.register(routes, templates, document_service)

    app = Starlette(routes=routes)
    app.mount("/static", StaticFiles(directory=str(_PACKAGE_ROOT / "static")), name="static")
    app.state.config = config
    return app
