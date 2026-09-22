"""HTTP pages for operator uploads; never a public-corpus write."""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup
from starlette.testclient import TestClient

from bddk_mcp.admin.app import create_app
from bddk_mcp.admin.config import AdminConfig
from bddk_mcp.admin.services.documents import DocumentService
from bddk_mcp.admin.uploads import UploadStore

CONFIG = AdminConfig(bind_host="127.0.0.1", port=8100, database_url="postgresql://x", loopback_only=True)
ORIGIN = {"Origin": "http://127.0.0.1"}


class FakeStore:
    """The upload pages must not consult the public document store."""

    async def list_documents(self, **_kwargs):
        raise AssertionError("upload pages must not read public documents")

    async def stats(self):
        raise AssertionError("upload pages must not read public documents")


class StubGovernance:
    async def status(self):
        raise AssertionError("governance must not be consulted by upload pages")


def upload_app(store: UploadStore):
    return create_app(CONFIG, DocumentService(FakeStore()), StubGovernance(), upload_store=store)


def http_client(app) -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1", follow_redirects=False)


def csrf_token(response) -> str:
    tag = BeautifulSoup(response.text, "html.parser").select_one('form input[name="csrf_token"]')
    assert tag is not None, "form page must carry a CSRF token"
    return tag["value"]


def upload_pdf(client: TestClient, name="note.pdf", data=b"%PDF-1.4\n"):
    page = client.get("/uploads/new")
    return client.post(
        "/uploads",
        data={"csrf_token": csrf_token(page)},
        files={"file": (name, data, "application/pdf")},
        headers=ORIGIN,
    )


def corrected_upload(client: TestClient, store: UploadStore, monkeypatch, text="duzeltilmis metin"):
    monkeypatch.setattr(store, "_extract_bytes", lambda *_: "ilk metin")
    response = upload_pdf(client)
    assert response.status_code == 303, response.text
    upload_id = Path(response.headers["location"]).parts[-2]
    edit = client.get(f"/uploads/{upload_id}/edit")
    assert "ilk metin" in edit.text
    saved = client.post(
        f"/uploads/{upload_id}/edit",
        data={"csrf_token": csrf_token(edit), "text": text},
        headers=ORIGIN,
    )
    assert saved.status_code == 303, saved.text
    assert store.corrected_text(upload_id) == text
    return upload_id


def test_upload_page_does_not_publish_and_admit_shows_waiting(tmp_path, monkeypatch):
    store = UploadStore(tmp_path / "drafts.sqlite")
    client = http_client(upload_app(store))

    upload_id = corrected_upload(client, store, monkeypatch)

    admit_page = client.get(f"/uploads/{upload_id}/admit")
    admitted = client.post(
        f"/uploads/{upload_id}/admit",
        data={"csrf_token": csrf_token(admit_page)},
        headers=ORIGIN,
    )
    assert admitted.status_code == 303, admitted.text
    state_page = client.get(f"/uploads/{upload_id}/admit")
    assert state_page.status_code == 200
    assert "waiting" in state_page.text


def test_duplicate_admit_request_is_rejected_on_the_page(tmp_path, monkeypatch):
    store = UploadStore(tmp_path / "drafts.sqlite")
    client = http_client(upload_app(store))
    upload_id = corrected_upload(client, store, monkeypatch)
    admitted = client.post(
        f"/uploads/{upload_id}/admit",
        data={"csrf_token": csrf_token(client.get(f"/uploads/{upload_id}/admit"))},
        headers=ORIGIN,
    )
    assert admitted.status_code == 303

    duplicate = client.post(
        f"/uploads/{upload_id}/admit",
        data={"csrf_token": csrf_token(client.get(f"/uploads/{upload_id}/edit"))},
        headers=ORIGIN,
    )
    assert duplicate.status_code == 409
    assert "bekleyen" in duplicate.text


def test_unsupported_upload_type_is_rejected_not_500(tmp_path):
    store = UploadStore(tmp_path / "drafts.sqlite")
    client = http_client(upload_app(store))

    response = upload_pdf(client, name="note.txt", data=b"hello")

    assert response.status_code == 415
    assert "PDF" in response.text and "DOCX" in response.text


def test_extraction_failure_is_shown_on_the_upload_page(tmp_path, monkeypatch):
    store = UploadStore(tmp_path / "drafts.sqlite")

    def boom(*_args):
        raise ValueError("extraction_failed")

    monkeypatch.setattr(store, "_extract_bytes", boom)
    client = http_client(upload_app(store))

    response = upload_pdf(client)

    assert response.status_code == 422
    assert "Metin cikarilamadi" in response.text


def test_unknown_upload_is_not_found(tmp_path):
    store = UploadStore(tmp_path / "drafts.sqlite")
    client = http_client(upload_app(store))

    assert client.get("/uploads/deadbeef/edit").status_code == 404
    assert client.get("/uploads/deadbeef/admit").status_code == 404


def test_upload_requires_same_origin_and_token(tmp_path, monkeypatch):
    store = UploadStore(tmp_path / "drafts.sqlite")
    monkeypatch.setattr(store, "_extract_bytes", lambda *_: "ilk metin")
    client = http_client(upload_app(store))
    page = client.get("/uploads/new")
    token = csrf_token(page)

    no_origin = client.post(
        "/uploads", data={"csrf_token": token}, files={"file": ("note.pdf", b"%PDF-1.4\n", "application/pdf")}
    )
    assert no_origin.status_code == 403

    bad_token = client.post(
        "/uploads",
        data={"csrf_token": "bogus"},
        files={"file": ("note.pdf", b"%PDF-1.4\n", "application/pdf")},
        headers=ORIGIN,
    )
    assert bad_token.status_code == 403


def test_upload_pages_absent_without_store():
    client = http_client(create_app(CONFIG, DocumentService(FakeStore()), StubGovernance()))

    assert client.get("/uploads/new").status_code == 404
