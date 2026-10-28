"""Link-proposal review pages; decisions go to the sidecar review store, never to PostgreSQL."""

from __future__ import annotations

import asyncio
import json
from urllib.parse import quote

from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response
from starlette.routing import Route
from starlette.templating import Jinja2Templates

from bddk_mcp.admin.csrf import FormSecurity
from bddk_mcp.admin.services.proposals import FILTERS, ProposalService
from bddk_mcp.ontology.review_store import DECISIONS, ReviewStoreError

_INVALID_DECISION = "Karar 'approved' veya 'rejected' olmalı."


def register(routes: list, templates: Jinja2Templates, service: ProposalService) -> None:
    security = FormSecurity()

    def card_url(proposal_id: str) -> str:
        return f"/proposals/{quote(proposal_id, safe='')}"

    def not_found(request: Request, proposal_id: str) -> Response:
        return templates.TemplateResponse(request, "not_found.html", {"document_id": proposal_id}, status_code=404)

    async def render_card(request: Request, proposal_id: str, *, error: str | None = None, status_code: int = 200):
        item = await asyncio.to_thread(service.item, proposal_id)
        if item is None:
            return not_found(request, proposal_id)
        queue = await asyncio.to_thread(service.queue)
        next_id = await asyncio.to_thread(service.next_pending_after, proposal_id)
        history = await asyncio.to_thread(service.history, proposal_id)
        nonce, token = security.issue(request)
        response = templates.TemplateResponse(
            request,
            "proposals/card.html",
            {
                "item": item,
                "queue": queue,
                "history": history,
                "next_url": card_url(next_id) if next_id else None,
                "decide_url": card_url(proposal_id) + "/decide",
                "document_id": service.file.document_id,
                "title": service.file.title,
                "reviewer": service.reviewer,
                "csrf_token": token,
                "error": error,
            },
            status_code=status_code,
        )
        return security.attach(request, response, nonce)

    async def queue_page(request: Request) -> Response:
        filter_name = request.query_params.get("durum", "hepsi")
        if filter_name not in FILTERS:
            filter_name = "hepsi"
        queue = await asyncio.to_thread(service.queue, filter_name)
        return templates.TemplateResponse(
            request,
            "proposals/queue.html",
            {
                "queue": queue,
                "filters": FILTERS,
                "document_id": service.file.document_id,
                "title": service.file.title,
                "proposals_path": str(service.path),
                "file_sha256": service.file_sha256,
                "card_url": card_url,
            },
        )

    async def card_page(request: Request) -> Response:
        return await render_card(request, request.path_params["proposal_id"])

    async def decide(request: Request) -> Response:
        proposal_id = request.path_params["proposal_id"]
        try:
            fields = await security.read(request)
        except HTTPException as exc:
            if await asyncio.to_thread(service.item, proposal_id) is None:
                return not_found(request, proposal_id)
            return await render_card(request, proposal_id, error=exc.detail, status_code=exc.status_code)
        decision = fields.get("decision", "")
        reason = fields.get("reason", "")
        if decision not in DECISIONS:
            return await render_card(request, proposal_id, error=_INVALID_DECISION, status_code=422)
        try:
            await asyncio.to_thread(service.decide, proposal_id, decision, reason)
        except KeyError:
            return not_found(request, proposal_id)
        except ReviewStoreError as exc:
            return await render_card(request, proposal_id, error=str(exc), status_code=422)
        next_id = await asyncio.to_thread(service.next_pending_after, proposal_id)
        return RedirectResponse(card_url(next_id) if next_id else "/proposals", status_code=303)

    async def download_decisions(_request: Request) -> Response:
        payload = await asyncio.to_thread(service.export)
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1)
        return Response(
            body,
            media_type="application/json",
            headers={
                "Content-Disposition": 'attachment; filename="22599.decisions.json"',
                "Cache-Control": "no-store",
            },
        )

    routes.extend(
        [
            Route("/proposals", queue_page, methods=["GET"], name="proposals"),
            Route("/proposals/decisions.json", download_decisions, methods=["GET"], name="proposal_decisions"),
            Route("/proposals/{proposal_id}", card_page, methods=["GET"], name="proposal_card"),
            Route("/proposals/{proposal_id}/decide", decide, methods=["POST"], name="proposal_decide"),
        ]
    )
