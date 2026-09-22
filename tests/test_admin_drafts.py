"""Real SQLite, HTTP forms and Ed25519; no mock draft persistence or signing."""

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from bs4 import BeautifulSoup
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from starlette.testclient import TestClient

from bddk_mcp.admin.app import create_app
from bddk_mcp.admin.config import AdminConfig, AdminConfigError
from bddk_mcp.admin.drafts import DraftConflict, DraftSigner, DraftStore, EditForm, canonical_json, fingerprint
from bddk_mcp.admin.services.documents import DocumentService
from bddk_mcp.store.doc_store import StoredDocument, StoreStats
from tests.test_admin_auth import _remote_config, _Verifier


class ReadOnlyStore:
    def __init__(self):
        self.doc = StoredDocument(
            document_id="doc",
            title="Original",
            markdown_content="Original text",
            category="mevzuat",
            source_url="https://bddk.org.tr/1",
            pdf_bytes=b"original pdf",
            content_hash="original hash",
            extraction_method="markitdown_degraded",
            total_pages=4,
            file_size=12,
        )

    async def get_document(self, _id):
        return self.doc.model_copy(deep=True)

    async def store_document(self, _doc):
        pytest.fail("Public corpus write attempted")

    async def list_documents(self, **_kwargs):
        return [self.doc.model_dump()]

    async def stats(self):
        return StoreStats(categories={"mevzuat": 1})

    async def search_content(self, query, **_kwargs):
        return []


def key_files(tmp_path):
    directory = tmp_path / "keys"
    directory.mkdir(exist_ok=True)
    key = Ed25519PrivateKey.generate()
    private, public = directory / "private.pem", directory / "public.pem"
    private.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    private.chmod(0o600)
    public.write_bytes(
        key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    )
    return key, private, public


def edit_fields(doc, revision=0):
    return {
        **doc.model_dump(include=set(EditForm.model_fields) - {"revision", "base_fingerprint"}),
        "revision": str(revision),
        "base_fingerprint": fingerprint(doc),
    }


def form_fields(response):
    soup = BeautifulSoup(response.text, "html.parser")
    fields = {tag["name"]: tag.get("value", "") for tag in soup.select("form input[name]")}
    fields.update({tag["name"]: tag.text for tag in soup.select("form textarea[name]")})
    return fields


def test_real_persistence_revision_race_sign_and_invalidate(tmp_path):
    key, private, public = key_files(tmp_path)
    drafts = DraftStore(tmp_path / "drafts.sqlite", DraftSigner(private, public))
    doc = ReadOnlyStore().doc
    form = EditForm.model_validate(edit_fields(doc))

    def save_once(_):
        try:
            return drafts.change(doc, form)
        except DraftConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(save_once, range(2)))
    assert results.count("conflict") == 1
    restarted = DraftStore(drafts.path, DraftSigner(private, public))
    assert restarted.get("doc")["payload"]["revision"] == 1
    form = EditForm.model_validate(edit_fields(doc, 1))
    signed = restarted.change(doc, form, sign=True)
    key.public_key().verify(
        base64.b64decode(signed["signature"]["signature_base64"]), canonical_json(signed["payload"])
    )
    assert signed["payload"]["original"]["extraction_method"] == "markitdown_degraded"
    for field, value in signed["payload"]["edited"].items():
        tampered = {**signed["payload"], "edited": {**signed["payload"]["edited"], field: value + "changed"}}
        with pytest.raises(InvalidSignature):
            key.public_key().verify(base64.b64decode(signed["signature"]["signature_base64"]), canonical_json(tampered))
    for field in ("revision", "base_fingerprint", "document_id", "manual_edit_verification"):
        with pytest.raises(InvalidSignature):
            key.public_key().verify(
                base64.b64decode(signed["signature"]["signature_base64"]),
                canonical_json({**signed["payload"], field: "changed"}),
            )
    edited = restarted.change(doc, form.model_copy(update={"title": "New"}))
    assert edited["signature"] is None
    assert edited["payload"]["revision"] == 2
    assert DraftStore(drafts.path).get("doc") == edited
    with pytest.raises(DraftConflict):
        restarted.change(doc, form, sign=True)
    with pytest.raises(DraftConflict):
        restarted.change(doc.model_copy(update={"category": "changed"}), EditForm.model_validate(edit_fields(doc, 2)))


def test_key_mismatch_and_permissions_fail_closed(tmp_path):
    _, private, public = key_files(tmp_path)
    public.write_bytes(
        Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    )
    with pytest.raises(ValueError, match="mismatched"):
        DraftSigner(private, public)
    private.chmod(0o644)
    with pytest.raises(ValueError, match="unsafe"):
        DraftSigner(private, public)


@pytest.mark.parametrize("remote", [False, True])
def test_http_edit_sign_download_csrf_and_restart(tmp_path, monkeypatch, remote):
    monkeypatch.setenv("BDDK_ALLOW_INSECURE_DATABASE", "true")
    key, private, public = key_files(tmp_path)
    store = ReadOnlyStore()
    before = store.doc.model_dump()
    drafts = DraftStore(tmp_path / "draft.sqlite", DraftSigner(private, public))
    origin = "https://admin.bank.example:8443" if remote else "http://127.0.0.1"
    config = _remote_config() if remote else AdminConfig("127.0.0.1", 8100, "postgresql://x", True)
    if remote:
        config = replace(config, http_security=replace(config.http_security, allowed_origins=(origin,)))
    service = DocumentService(store, drafts)
    client = TestClient(
        create_app(config, service, None, token_verifier=_Verifier() if remote else None), base_url=origin
    )
    if remote:
        client.cookies.set("bddk_admin", "good")
    fields = form_fields(client.get("/documents/doc/edit"))
    assert fields["revision"] == "0"
    for endpoint in ("edit", "sign"):
        for bad_origin in (None, "https://evil.example", "null"):
            headers = {"origin": bad_origin} if bad_origin else {}
            assert client.post(f"/documents/doc/{endpoint}", data=fields, headers=headers).status_code == 403
        assert (
            client.post(
                f"/documents/doc/{endpoint}", data={**fields, "csrf_token": "bad"}, headers={"origin": origin}
            ).status_code
            == 403
        )
    fields["title"] = "Edited"
    assert (
        client.post("/documents/doc/edit", data=fields, headers={"origin": origin}, follow_redirects=False).status_code
        == 303
    )
    assert client.post("/documents/doc/edit", data=fields, headers={"origin": origin}).status_code == 409
    detail = client.get("/documents/doc")
    assert "Unpublished editorial draft" in detail.text
    assert "Original" in client.get("/documents").text  # list never overlays
    sign_fields = form_fields(detail)
    assert (
        client.post(
            "/documents/doc/sign", data=sign_fields, headers={"origin": origin}, follow_redirects=False
        ).status_code
        == 303
    )
    artifact = client.get("/documents/doc/draft.json")
    assert artifact.headers["content-disposition"].startswith("attachment")
    signed = artifact.json()
    key.public_key().verify(
        base64.b64decode(signed["signature"]["signature_base64"]), canonical_json(signed["payload"])
    )
    fields = form_fields(client.get("/documents/doc/edit"))
    assert (
        client.post("/documents/doc/edit", data=fields, headers={"origin": origin}, follow_redirects=False).status_code
        == 303
    )
    assert client.get("/documents/doc/draft.json").json()["signature"] is None
    assert store.doc.model_dump() == before
    assert asyncio.run(DocumentService(store, DraftStore(drafts.path)).get("doc")).doc.title == "Edited"
    store.doc.category = "new canonical metadata"
    fields = form_fields(client.get("/documents/doc/edit"))
    assert client.post("/documents/doc/edit", data=fields, headers={"origin": origin}).status_code == 409
    assert client.get("/documents/doc/draft.json").status_code == 200


@pytest.mark.parametrize(
    "change",
    [
        {"title": " "},
        {"title": "x" * 501},
        {"markdown_content": "\n "},
        {"markdown_content": "x" * 1_000_001},
        {"source_url": "javascript:alert(1)"},
        {"source_url": "https://bddk.org.tr.evil.test/"},
        {"source_url": "https://u@bddk.org.tr/"},
        {"source_url": "http://bddk.org.tr/"},
        {"source_url": "/relative"},
        {"source_url": "https://bddk.org.tr:444/"},
        {"extraction_method": "manual_latex"},
        {"revision": "-1"},
        {"category": "x" * 101},
    ],
)
def test_form_validation_never_mutates(tmp_path, change):
    store = ReadOnlyStore()
    drafts = DraftStore(tmp_path / "draft.sqlite")
    outcome = asyncio.run(DocumentService(store, drafts).save("doc", {**edit_fields(store.doc), **change}))
    assert outcome.status == 422
    assert drafts.get("doc") is None


def test_csrf_cookie_document_expiry_and_bounded_reads(tmp_path, monkeypatch):
    import time

    import bddk_mcp.admin.csrf as csrf

    store = ReadOnlyStore()
    drafts = DraftStore(tmp_path / "draft.sqlite")
    config = AdminConfig("127.0.0.1", 8100, "postgresql://x", True)
    client = TestClient(create_app(config, DocumentService(store, drafts), None), base_url="http://127.0.0.1")
    headers = {"origin": "http://127.0.0.1"}
    fields = form_fields(client.get("/documents/doc/edit"))
    assert client.post("/documents/other/edit", data=fields, headers=headers).status_code == 403
    nonce = client.cookies.get(csrf.COOKIE)
    client.cookies.clear()
    assert client.post("/documents/doc/edit", data=fields, headers=headers).status_code == 403
    client.cookies.set(csrf.COOKIE, nonce)
    assert (
        client.post(
            "/documents/doc/edit", data={**fields, "csrf_token": "1234567890." + "ü" * 64}, headers=headers
        ).status_code
        == 403
    )
    assert client.post("/documents/doc/edit", data=list(), headers=headers).status_code == 415
    assert (
        client.post(
            "/documents/doc/edit",
            content="revision=1&revision=2",
            headers={**headers, "content-type": "application/x-www-form-urlencoded"},
        ).status_code
        == 422
    )
    with monkeypatch.context() as patch:
        patch.setattr(csrf.time, "time", lambda: 4_000_000_000)
        assert client.post("/documents/doc/edit", data=fields, headers=headers).status_code == 403
    assert time.time() < 4_000_000_000
    monkeypatch.setattr(csrf, "MAX_BODY", 10)
    assert client.post("/documents/doc/edit", data=fields, headers=headers).status_code == 413
    assert drafts.get("doc") is None


def test_remote_form_token_is_bound_to_operator_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("BDDK_ALLOW_INSECURE_DATABASE", "true")
    config = _remote_config()
    origin = "https://admin.bank.example:8443"
    config = replace(config, http_security=replace(config.http_security, allowed_origins=(origin,)))
    store = ReadOnlyStore()
    client = TestClient(
        create_app(
            config, DocumentService(store, DraftStore(tmp_path / "draft.sqlite")), None, token_verifier=_Verifier()
        ),
        base_url=origin,
    )
    fields = form_fields(client.get("/documents/doc/edit", headers={"authorization": "Bearer good"}))
    assert client.post("/documents/doc/edit", data=fields, headers={"origin": origin}).status_code == 401

    # A replacement valid operator JWT has the same scopes, but is a different identity binding.
    class OtherVerifier(_Verifier):
        async def verify_token(self, token):
            return await super().verify_token("good" if token == "other" else token)

    second = TestClient(
        create_app(
            config, DocumentService(store, DraftStore(tmp_path / "other.sqlite")), None, token_verifier=OtherVerifier()
        ),
        base_url=origin,
    )
    fields = form_fields(second.get("/documents/doc/edit", headers={"authorization": "Bearer good"}))
    assert (
        second.post(
            "/documents/doc/edit", data=fields, headers={"origin": origin, "authorization": "Bearer other"}
        ).status_code
        == 403
    )


def test_storage_and_keys_outside_corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("BDDK_ALLOW_INSECURE_DATABASE", "true")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    env = {"BDDK_DATABASE_URL": "postgresql://x", "BDDK_SEED_DIR": str(corpus)}
    with pytest.raises(AdminConfigError, match="outside"):
        AdminConfig.from_env({**env, "BDDK_ADMIN_DRAFT_DB": str(corpus / "draft.sqlite")})
    link = tmp_path / "link"
    link.symlink_to(corpus, target_is_directory=True)
    with pytest.raises(AdminConfigError, match="outside"):
        AdminConfig.from_env({**env, "BDDK_ADMIN_DRAFT_DB": str(link / "draft.sqlite")})
    with pytest.raises(AdminConfigError, match="both"):
        AdminConfig.from_env({**env, "BDDK_ADMIN_SIGNING_KEY": str(tmp_path / "key")})
