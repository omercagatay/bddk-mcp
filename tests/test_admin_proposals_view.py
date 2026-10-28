"""Link-proposal review pages: queue, card, CSRF-protected decisions, download; never a PostgreSQL read."""

from __future__ import annotations

from pathlib import Path

import pytest
from bs4 import BeautifulSoup
from starlette.testclient import TestClient

from bddk_mcp.admin.app import create_app
from bddk_mcp.admin.config import AdminConfig
from bddk_mcp.admin.services.documents import DocumentService
from bddk_mcp.admin.services.proposals import ProposalService
from bddk_mcp.ontology.review_store import ReviewStore

PROPOSALS = Path(__file__).resolve().parents[1] / "docs" / "evidence" / "legal-versions" / "22599.proposals.json"
ORIGIN = {"Origin": "http://127.0.0.1"}


class FakeStore:
    """The proposal pages must not consult the public document store."""

    async def list_documents(self, **_kwargs):
        raise AssertionError("proposal pages must not read public documents")

    async def stats(self):
        raise AssertionError("proposal pages must not read public documents")

    async def get_document(self, _id):
        raise AssertionError("proposal pages must not read public documents")


class StubGovernance:
    async def status(self):
        raise AssertionError("governance must not be consulted by proposal pages")


def make_app(tmp_path: Path):
    config = AdminConfig(
        bind_host="127.0.0.1",
        port=8100,
        database_url="postgresql://x",
        loopback_only=True,
        ontology_proposals=PROPOSALS,
        ontology_review_db=tmp_path / "review.sqlite",
        ontology_reviewer="test-reviewer",
    )
    service = ProposalService(PROPOSALS, ReviewStore(tmp_path / "review.sqlite"), reviewer="test-reviewer")
    app = create_app(config, DocumentService(FakeStore()), StubGovernance(), proposal_service=service)
    return app, service


def http_client(app) -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1", follow_redirects=False)


def csrf_token(response) -> str:
    tag = BeautifulSoup(response.text, "html.parser").select_one('form#decide-form input[name="csrf_token"]')
    assert tag is not None, "card page must carry a CSRF token"
    return tag["value"]


@pytest.fixture
def client_and_service(tmp_path):
    app, service = make_app(tmp_path)
    return http_client(app), service


def test_queue_shows_counter_auto_acceptance_and_navigation(client_and_service):
    client, service = client_and_service
    page = client.get("/proposals")
    assert page.status_code == 200
    queue = service.queue()
    assert f"{queue.reviewed} / 34" in page.text
    assert queue.auto_accepted == service.auto_accepted_now > 20
    assert queue.pending == 34 - queue.auto_accepted
    assert "Mevzuat ilişkileri" in page.text  # navigation link
    assert "İncelemeye başla" in page.text
    assert "PostgreSQL'e dokunulmaz" in page.text


def test_queue_filters_select_subsets(client_and_service):
    client, service = client_and_service
    pending = client.get("/proposals?durum=bekleyen")
    assert pending.status_code == 200
    assert pending.text.count("bekliyor</span>") == service.queue().pending
    approved = client.get("/proposals?durum=onayli")
    assert "onaylı (oto)" in approved.text
    nonsense = client.get("/proposals?durum=garip")
    assert nonsense.status_code == 200


def test_card_shows_quote_flags_and_what_approving_adds(client_and_service):
    client, service = client_and_service
    queue = service.queue()
    page = client.get(f"/proposals/{queue.next_pending_id}")
    assert page.status_code == 200
    item = service.item(queue.next_pending_id)
    rendered = BeautifulSoup(page.text, "html.parser").get_text(" ")
    assert item.proposal.evidence.quote[:40] in rendered
    assert item.adds_tr in rendered  # compared on text, not HTML, because apostrophes are escaped
    assert "Onayla" in page.text and "Reddet" in page.text
    assert 'name="csrf_token"' in page.text
    assert "Content-Security-Policy" in page.headers


def test_approve_records_the_reviewer_and_moves_to_the_next_pending(client_and_service):
    client, service = client_and_service
    first = service.queue().next_pending_id
    page = client.get(f"/proposals/{first}")
    response = client.post(
        f"/proposals/{first}/decide",
        data={"csrf_token": csrf_token(page), "decision": "approved", "reason": "tablo ile uyumlu"},
        headers=ORIGIN,
    )
    assert response.status_code == 303, response.text
    decision = service.store.latest(service.file_sha256)[first]
    assert decision.decision == "approved"
    assert decision.decided_by == "test-reviewer"
    assert decision.method == "human"
    assert decision.reason == "tablo ile uyumlu"
    second = service.queue().next_pending_id
    assert second != first
    assert response.headers["location"] == f"/proposals/{second}"


def test_reject_then_approve_keeps_history_and_latest_wins(client_and_service):
    client, service = client_and_service
    first = service.queue().next_pending_id
    for decision in ("rejected", "approved"):
        page = client.get(f"/proposals/{first}")
        assert (
            client.post(
                f"/proposals/{first}/decide",
                data={"csrf_token": csrf_token(page), "decision": decision, "reason": decision},
                headers=ORIGIN,
            ).status_code
            == 303
        )
    history = service.history(first)
    assert [item.decision for item in history] == ["rejected", "approved"]
    assert service.item(first).status == "approved"
    card = client.get(f"/proposals/{first}")
    assert "Karar geçmişi" in card.text


def test_decision_forms_require_same_origin_and_a_valid_token(client_and_service):
    client, service = client_and_service
    first = service.queue().next_pending_id
    page = client.get(f"/proposals/{first}")
    token = csrf_token(page)
    assert (
        client.post(f"/proposals/{first}/decide", data={"csrf_token": token, "decision": "approved"}).status_code == 403
    )
    assert (
        client.post(
            f"/proposals/{first}/decide",
            data={"csrf_token": token, "decision": "approved"},
            headers={"Origin": "https://evil.example"},
        ).status_code
        == 403
    )
    assert (
        client.post(
            f"/proposals/{first}/decide", data={"csrf_token": "bad", "decision": "approved"}, headers=ORIGIN
        ).status_code
        == 403
    )
    # A token issued for one proposal does not decide another one.
    other = service.next_pending_after(first)
    assert (
        client.post(
            f"/proposals/{other}/decide", data={"csrf_token": token, "decision": "approved"}, headers=ORIGIN
        ).status_code
        == 403
    )
    assert service.store.latest(service.file_sha256).get(first) is None


def test_invalid_decision_value_is_rejected_on_the_card(client_and_service):
    client, service = client_and_service
    first = service.queue().next_pending_id
    page = client.get(f"/proposals/{first}")
    response = client.post(
        f"/proposals/{first}/decide",
        data={"csrf_token": csrf_token(page), "decision": "maybe"},
        headers=ORIGIN,
    )
    assert response.status_code == 422
    assert "approved" in response.text
    assert service.item(first).status == "pending"


def test_unknown_proposal_is_not_found(client_and_service):
    client, _service = client_and_service
    missing = "prop_sha256_" + "0" * 64
    assert client.get(f"/proposals/{missing}").status_code == 404
    assert client.post(f"/proposals/{missing}/decide", data={"decision": "approved"}, headers=ORIGIN).status_code == 404


def test_decisions_download_is_a_json_attachment(client_and_service):
    client, service = client_and_service
    response = client.get("/proposals/decisions.json")
    assert response.status_code == 200
    assert response.headers["content-disposition"] == 'attachment; filename="22599.decisions.json"'
    assert response.headers["cache-control"] == "no-store"
    payload = response.json()
    assert payload["artifact_type"] == "bddk-link-decisions-v1"
    assert payload["proposals_file_sha256"] == service.file_sha256
    assert payload["counts"]["approved"] == service.auto_accepted_now
    assert payload["counts"]["pending"] == 34 - service.auto_accepted_now


def test_whole_queue_can_be_reviewed_in_one_sitting(client_and_service):
    client, service = client_and_service
    seen = 0
    while (pending := service.queue().next_pending_id) is not None:
        page = client.get(f"/proposals/{pending}")
        assert (
            client.post(
                f"/proposals/{pending}/decide",
                data={"csrf_token": csrf_token(page), "decision": "approved"},
                headers=ORIGIN,
            ).status_code
            == 303
        )
        seen += 1
        assert seen <= 34
    queue = service.queue()
    assert queue.pending == 0 and queue.reviewed == 34
    assert seen == 34 - service.auto_accepted_now
    assert "Bekleyen öneri kalmadı" in client.get("/proposals").text


def test_proposals_navigation_is_hidden_without_configuration(tmp_path):
    config = AdminConfig(bind_host="127.0.0.1", port=8100, database_url="postgresql://x", loopback_only=True)
    app = create_app(config, DocumentService(FakeStore()), StubGovernance())
    client = http_client(app)
    assert client.get("/proposals").status_code == 404
