"""Read-only signature status route. GET only by design: signing is not a console action."""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route
from starlette.templating import Jinja2Templates

from bddk_mcp.admin.services.governance import GovernanceService


def register(routes: list, templates: Jinja2Templates, service: GovernanceService) -> None:
    async def signature_status(request: Request) -> Response:
        status = await service.status()
        return templates.TemplateResponse(request, "governance/status.html", {"status": status})

    routes.append(Route("/governance", signature_status, methods=["GET"], name="governance"))
