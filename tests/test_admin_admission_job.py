from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from bddk_mcp.admin.uploads import UploadStore


def _store_with_two_waiting_uploads(tmp_path: Path) -> tuple[UploadStore, str, str, str, str]:
    store = UploadStore(tmp_path / "drafts.sqlite")
    first = store.save("a.pdf", b"%PDF-1.4\n")
    second = store.save("b.pdf", b"%PDF-1.4\n")
    store.save_correction(first, "bir")
    store.save_correction(second, "iki")
    first_request = store.admit(first)
    second_request = store.admit(second)
    return store, first_request, second_request, first, second


def test_failed_activation_leaves_request_in_error_and_does_not_combine_requests(tmp_path):
    from bddk_mcp.admin.admission_job import admit_next

    store, first_request, second_request, first_upload, _second_upload = _store_with_two_waiting_uploads(tmp_path)

    def fail_publish(text, upload_id):
        fail_publish.calls.append((text, upload_id))
        raise RuntimeError("activate failed")

    fail_publish.calls = []

    assert admit_next(store, fail_publish) == "error"
    assert store.request_state(first_request) == "error"
    assert store.request_state(second_request) == "waiting"
    assert fail_publish.calls == [("bir", first_upload)]


def test_successful_admission_publishes_the_corrected_text(tmp_path):
    from bddk_mcp.admin.admission_job import admit_next

    store = UploadStore(tmp_path / "drafts.sqlite")
    upload_id = store.save("a.pdf", b"%PDF-1.4\n")
    store.save_correction(upload_id, "tek metin")
    request_id = store.admit(upload_id)

    calls: list[tuple[str, str]] = []

    def publish(text, upload_id):
        calls.append((text, upload_id))

    assert admit_next(store, publish) == "published"
    assert calls == [("tek metin", upload_id)]
    assert store.request_state(request_id) == "published"


def test_admit_next_with_no_waiting_request_is_idle(tmp_path):
    from bddk_mcp.admin.admission_job import admit_next

    store = UploadStore(tmp_path / "drafts.sqlite")
    assert admit_next(store, lambda text, upload_id: None) == "idle"


def test_admit_next_publishes_one_request_per_call(tmp_path):
    from bddk_mcp.admin.admission_job import admit_next

    store, _first_request, second_request, first_upload, _second_upload = _store_with_two_waiting_uploads(tmp_path)

    calls: list[tuple[str, str]] = []

    def publish(text, upload_id):
        calls.append((text, upload_id))

    assert admit_next(store, publish) == "published"
    assert calls == [("bir", first_upload)]
    assert store.request_state(second_request) == "waiting"


def test_admit_next_refuses_an_unwired_publisher_without_marking(tmp_path):
    """An unwired publisher refuses; that refusal is never a request error."""

    from bddk_mcp.admin.admission_job import admit_next

    store, first_request, second_request, first_upload, _second_upload = _store_with_two_waiting_uploads(tmp_path)

    def unwired_publish(_text, _upload_id):
        raise NotImplementedError("gates not wired")

    with pytest.raises(NotImplementedError, match="gates not wired"):
        admit_next(store, unwired_publish)
    assert store.request_state(first_request) == "waiting"
    assert store.request_state(second_request) == "waiting"


def test_publisher_from_env_requires_admission_signing_keys(monkeypatch):
    """Both release credentials without the job-held signing key refuse at construction."""

    from bddk_mcp.admin.admission_job import publisher_from_env

    with pytest.raises(RuntimeError) as exc_info:
        publisher_from_env(
            {
                "BDDK_RELEASE_VERIFIER_DATABASE_URL": "postgresql://verifier@example/db",
                "BDDK_RELEASE_PUBLISHER_DATABASE_URL": "postgresql://publisher@example/db",
            }
        )

    message = str(exc_info.value)
    assert "BDDK_ADMISSION_SIGNING_KEY" in message
    assert "BDDK_ADMISSION_SIGNING_PUBLIC_KEY" in message


def test_cli_admit_next_upload_refuses_before_marking_when_signing_keys_are_missing(tmp_path, monkeypatch):
    """A configured admit-next-upload run refuses before touching any request state."""

    from bddk_mcp.cli import _run_admit_next_upload

    store, first_request, _second_request, _first_upload, _second_upload = _store_with_two_waiting_uploads(tmp_path)
    monkeypatch.setenv("BDDK_RELEASE_VERIFIER_DATABASE_URL", "postgresql://verifier@example/db")
    monkeypatch.setenv("BDDK_RELEASE_PUBLISHER_DATABASE_URL", "postgresql://publisher@example/db")
    monkeypatch.delenv("BDDK_ADMISSION_SIGNING_KEY", raising=False)
    monkeypatch.delenv("BDDK_ADMISSION_SIGNING_PUBLIC_KEY", raising=False)
    args = argparse.Namespace(draft_db=tmp_path / "drafts.sqlite")

    with pytest.raises(RuntimeError, match="BDDK_ADMISSION_SIGNING_KEY"):
        _run_admit_next_upload(args)
    assert store.request_state(first_request) == "waiting"


def test_publisher_from_env_requires_both_release_credentials(monkeypatch):
    from bddk_mcp.admin.admission_job import publisher_from_env

    monkeypatch.delenv("BDDK_RELEASE_VERIFIER_DATABASE_URL", raising=False)
    monkeypatch.delenv("BDDK_RELEASE_PUBLISHER_DATABASE_URL", raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        publisher_from_env({})

    message = str(exc_info.value)
    assert "BDDK_RELEASE_VERIFIER_DATABASE_URL" in message
    assert "BDDK_RELEASE_PUBLISHER_DATABASE_URL" in message


def test_cli_registers_admit_next_upload_command():
    from bddk_mcp.cli import build_parser

    args = build_parser().parse_args(["admit-next-upload"])
    assert args.command == "admit-next-upload"


def test_cli_admit_next_upload_requires_release_variables(tmp_path, monkeypatch):
    from bddk_mcp.cli import _run_admit_next_upload

    draft_db = tmp_path / "drafts.sqlite"
    args = argparse.Namespace(draft_db=draft_db)
    monkeypatch.delenv("BDDK_RELEASE_VERIFIER_DATABASE_URL", raising=False)
    monkeypatch.delenv("BDDK_RELEASE_PUBLISHER_DATABASE_URL", raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        _run_admit_next_upload(args)

    message = str(exc_info.value)
    assert "BDDK_RELEASE_VERIFIER_DATABASE_URL" in message
    assert "BDDK_RELEASE_PUBLISHER_DATABASE_URL" in message


def test_cli_admit_next_upload_requires_a_draft_database(monkeypatch):
    from bddk_mcp.cli import _run_admit_next_upload

    monkeypatch.delenv("BDDK_ADMIN_DRAFT_DB", raising=False)
    args = argparse.Namespace(draft_db=None)

    with pytest.raises(RuntimeError, match="BDDK_ADMIN_DRAFT_DB"):
        _run_admit_next_upload(args)


def test_lost_correction_marks_the_request_error_and_keeps_the_queue_draining(tmp_path):
    """A waiting request whose admissible text vanished must not block later ones."""

    import sqlite3
    from contextlib import closing

    from bddk_mcp.admin.admission_job import admit_next

    store = UploadStore(tmp_path / "drafts.sqlite")
    upload_id = store.save("a.pdf", b"%PDF-1.4\n")
    store.save_correction(upload_id, "kayip metin")
    request_id = store.admit(upload_id)
    with closing(sqlite3.connect(store.draft_db)) as db, db:
        db.execute("DELETE FROM upload_drafts WHERE upload_id = ?", (upload_id,))

    assert admit_next(store, lambda text, upload_id: None) == "error"
    assert store.request_state(request_id) == "error"
    assert admit_next(store, lambda text, upload_id: None) == "idle"


# ── Task 5b: the wired publisher ──────────────────────────────────────────────

_REQUEST_ID = "corpus_release_request_sha256_" + "a" * 64
_VERIFIER_DSN = "postgresql://verifier@example/db?sslmode=verify-full&sslrootcert=%2Fca.pem"
_PUBLISHER_DSN = "postgresql://publisher@example/db?sslmode=verify-full&sslrootcert=%2Fca.pem"


def _write_keypair(directory: Path) -> tuple[Path, Path]:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    directory.mkdir(parents=True, exist_ok=True)
    private = Ed25519PrivateKey.generate()
    private_path = directory / "admission-private.pem"
    private_path.write_bytes(
        private.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )
    )
    private_path.chmod(0o600)
    public_path = directory / "admission-public.pem"
    public_path.write_bytes(
        private.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    )
    return private_path, public_path


def _fixture_document() -> dict:
    content = "MADDE 1 - Fixture body text."
    return {
        "document_id": "mevzuat_fixture",
        "title": "Fixture Regulation",
        "category": "Yonetmelik",
        "decision_date": "",
        "decision_number": "",
        "source_url": "https://www.bddk.org.tr/fixture",
        "markdown_content": content,
        "content_hash": hashlib.sha256(content.encode()).hexdigest(),
        "downloaded_at": 1767225600.0,
        "extracted_at": 1767225600.0,
        "extraction_method": "html_parser",
        "total_pages": 1,
        "file_size": len(content.encode()),
    }


def _write_fixture_seed(root: Path) -> None:
    import yaml

    root.mkdir(parents=True, exist_ok=True)
    document = _fixture_document()
    (root / "documents.json").write_text(json.dumps([document], ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "chunks.json").write_text(
        json.dumps(
            [{"doc_id": "mevzuat_fixture", "chunk_index": 0, "chunk_text": "MADDE 1 - Fixture body text."}], indent=2
        ),
        encoding="utf-8",
    )
    (root / "decision_cache.json").write_text(
        json.dumps(
            [
                {
                    "document_id": "mevzuat_fixture",
                    "title": "Fixture Regulation",
                    "content": "Fixture Regulation",
                    "decision_date": "",
                    "decision_number": "",
                    "category": "Yonetmelik",
                    "source_url": "https://www.bddk.org.tr/fixture",
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    placeholder = "0" * 64
    manifest = {
        "schema_version": 1,
        "manifest_id": "bddk-job-corpus-fixture",
        "selection_owner": "project_owner",
        "purpose": "Fixture corpus for admission wiring tests.",
        "exhaustive": False,
        "included_source_classes": ["Selected public BDDK regulations"],
        "excluded_source_classes": ["Exhaustive historical BDDK coverage"],
        "known_gaps": ["Fixture corpus for tests"],
        "freshness": {
            "source_observed_start": "2026-01-01T00:00:00Z",
            "source_observed_end": "2026-01-02T00:00:00Z",
            "corpus_built_at": "2026-01-02T00:00:00Z",
            "scope_reviewed_at": "2026-01-02T00:00:00Z",
            "business_expectation": "quarterly-reviewed batch corpus",
            "source_detection_slo_seconds": 604800,
            "publication_slo_seconds": 1209600,
            "max_manifest_age_seconds": 15552000,
            "slo_evidence_status": "not_measured",
        },
        "artifacts": [
            {"role": "documents", "path": "documents.json", "sha256": placeholder, "bytes": 0, "records": 1},
            {"role": "chunks", "path": "chunks.json", "sha256": placeholder, "bytes": 0, "records": 1},
            {"role": "decision_cache", "path": "decision_cache.json", "sha256": placeholder, "bytes": 0, "records": 1},
        ],
        "integrity": {"manifest_sha256": placeholder, "signature_status": "not_configured"},
    }
    (root / "corpus_scope.yml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _fake_chunk_generator(docs: list[dict]) -> list[dict]:
    return [
        {"doc_id": doc["document_id"], "chunk_index": 0, "chunk_text": doc["markdown_content"][:40]} for doc in docs
    ]


def test_wired_publisher_builds_staging_corpus_and_calls_gates_in_order(tmp_path):
    from bddk_mcp.admin.admission_job import run_admission

    seed_root = tmp_path / "corpus"
    _write_fixture_seed(seed_root)
    keys_dir = tmp_path / "keys"
    private_path, public_path = _write_keypair(keys_dir)
    (tmp_path / "drafts").mkdir()
    store = UploadStore(tmp_path / "drafts" / "drafts.sqlite")
    upload_id = store.save("rapor.pdf", b"%PDF-1.4\n")
    corrected = "MADDE 1 - Operator corrected text."
    store.save_correction(upload_id, corrected)

    order: list[str] = []
    captured: dict = {}

    def fake_verify_stage(**kwargs):
        order.append("verify")
        staging = kwargs["seed_dir"]
        captured["kwargs"] = kwargs
        captured["documents"] = json.loads((staging / "documents.json").read_text(encoding="utf-8"))
        captured["chunks"] = json.loads((staging / "chunks.json").read_text(encoding="utf-8"))
        captured["cache"] = json.loads((staging / "decision_cache.json").read_text(encoding="utf-8"))
        import yaml

        captured["manifest"] = yaml.safe_load((staging / "corpus_scope.yml").read_text(encoding="utf-8"))
        captured["signature_exists"] = (staging / "corpus_scope.sig").is_file()
        return {"corpus_release_request": {"request_id": _REQUEST_ID}}

    def fake_activate(**kwargs):
        order.append("activate")
        captured["activate_kwargs"] = kwargs

    returned = run_admission(
        corrected,
        upload_id,
        upload_store=store,
        seed_root=seed_root,
        signing_key=private_path,
        trusted_public_key=public_path,
        verifier_dsn=_VERIFIER_DSN,
        publisher_dsn=_PUBLISHER_DSN,
        verify_stage=fake_verify_stage,
        activate=fake_activate,
        chunk_generator=_fake_chunk_generator,
    )

    assert returned == _REQUEST_ID
    assert order == ["verify", "activate"]

    documents = captured["documents"]
    assert len(documents) == 2
    new_document = next(doc for doc in documents if doc["document_id"].startswith("admin_upload_"))
    fixture_document = _fixture_document()
    assert set(new_document) == set(fixture_document)
    assert new_document["document_id"] == "admin_upload_" + hashlib.sha256(corrected.encode()).hexdigest()[:12]
    assert new_document["markdown_content"] == corrected
    assert new_document["content_hash"] == hashlib.sha256(corrected.encode()).hexdigest()
    assert new_document["title"] == "rapor"
    assert new_document["category"] == "editorial"
    assert new_document["decision_date"] == ""
    assert new_document["decision_number"] == ""
    assert new_document["extraction_method"] == "editorial_upload_corrected"
    assert new_document["total_pages"] >= 1
    assert new_document["file_size"] == 9
    assert isinstance(new_document["downloaded_at"], float) and isinstance(new_document["extracted_at"], float)

    assert len(captured["chunks"]) == 2
    assert len(captured["cache"]) == 2
    new_cache = next(item for item in captured["cache"] if item["document_id"] == new_document["document_id"])
    assert set(new_cache) == {
        "document_id",
        "title",
        "content",
        "decision_date",
        "decision_number",
        "category",
        "source_url",
    }

    manifest = captured["manifest"]
    assert manifest["manifest_id"].startswith("bddk-job-corpus-admission-")
    artifact_records = {entry["role"]: entry["records"] for entry in manifest["artifacts"]}
    assert artifact_records == {"documents": 2, "chunks": 2, "decision_cache": 2}
    assert captured["signature_exists"] is True

    verify_kwargs = captured["kwargs"]
    assert verify_kwargs["dsn"] == _VERIFIER_DSN
    assert verify_kwargs["trusted_signing_key"] == public_path
    assert verify_kwargs["accept_unmeasured_freshness"] is True
    assert verify_kwargs["seed_dir"].name.startswith("bddk_admission_")
    assert captured["activate_kwargs"] == {"dsn": _PUBLISHER_DSN, "request_id": _REQUEST_ID}

    # The staging copy never touches the source corpus directory.
    source_documents = json.loads((seed_root / "documents.json").read_text(encoding="utf-8"))
    assert len(source_documents) == 1


def test_wired_publisher_gate_failure_marks_exactly_one_request_error(tmp_path):
    from bddk_mcp.admin.admission_job import admit_next, run_admission

    seed_root = tmp_path / "corpus"
    _write_fixture_seed(seed_root)
    keys_dir = tmp_path / "keys"
    private_path, public_path = _write_keypair(keys_dir)
    store, first_request, second_request, first_upload, _second_upload = _store_with_two_waiting_uploads(tmp_path)

    activate_calls: list[dict] = []

    def failing_verify_stage(**_kwargs):
        raise RuntimeError("verify gate failed")

    def fake_activate(**kwargs):
        activate_calls.append(kwargs)

    publisher = lambda text, upload_id: run_admission(  # noqa: E731
        text,
        upload_id,
        upload_store=store,
        seed_root=seed_root,
        signing_key=private_path,
        trusted_public_key=public_path,
        verifier_dsn=_VERIFIER_DSN,
        publisher_dsn=_PUBLISHER_DSN,
        verify_stage=failing_verify_stage,
        activate=fake_activate,
        chunk_generator=_fake_chunk_generator,
    )

    assert admit_next(store, publisher) == "error"
    assert store.request_state(first_request) == "error"
    assert store.request_state(second_request) == "waiting"
    assert activate_calls == []


def test_wired_publisher_refuses_a_mismatched_key_pair_before_any_gate(tmp_path):
    from bddk_mcp.admin.admission_job import run_admission

    seed_root = tmp_path / "corpus"
    _write_fixture_seed(seed_root)
    keys_dir = tmp_path / "keys"
    _private_path, trusted_public_path = _write_keypair(keys_dir)
    other_private_path, _other_public_path = _write_keypair(tmp_path / "other-keys")
    (tmp_path / "drafts").mkdir()
    store = UploadStore(tmp_path / "drafts" / "drafts.sqlite")
    upload_id = store.save("rapor.pdf", b"%PDF-1.4\n")
    store.save_correction(upload_id, "metin")

    gate_calls: list[str] = []

    def fake_verify_stage(**_kwargs):
        gate_calls.append("verify")
        return {"corpus_release_request": {"request_id": _REQUEST_ID}}

    def fake_activate(**_kwargs):
        gate_calls.append("activate")

    with pytest.raises(RuntimeError, match="correspond"):
        run_admission(
            "metin",
            upload_id,
            upload_store=store,
            seed_root=seed_root,
            signing_key=other_private_path,
            trusted_public_key=trusted_public_path,
            verifier_dsn=_VERIFIER_DSN,
            publisher_dsn=_PUBLISHER_DSN,
            verify_stage=fake_verify_stage,
            activate=fake_activate,
            chunk_generator=_fake_chunk_generator,
        )
    assert gate_calls == []


def test_admin_modules_do_not_import_admission_job():
    """The admin service never gains release-gate or publisher wiring."""

    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[1] / "bddk_mcp" / "admin"
    for module in ("app.py", "uploads.py"):
        source = (root / module).read_text(encoding="utf-8")
        assert "admission_job" not in source, module


def test_publisher_from_env_rejects_a_signing_key_inside_the_corpus(tmp_path, monkeypatch):
    from bddk_mcp.admin.admission_job import publisher_from_env

    seed_root = tmp_path / "corpus"
    _write_fixture_seed(seed_root)
    keys_dir = seed_root / "keys"
    private_path, public_path = _write_keypair(keys_dir)

    monkeypatch.setenv("BDDK_RELEASE_VERIFIER_DATABASE_URL", _VERIFIER_DSN)
    monkeypatch.setenv("BDDK_RELEASE_PUBLISHER_DATABASE_URL", _PUBLISHER_DSN)
    monkeypatch.setenv("BDDK_SEED_DIR", str(seed_root))
    monkeypatch.setenv("BDDK_ADMISSION_SIGNING_KEY", str(private_path))
    monkeypatch.setenv("BDDK_ADMISSION_SIGNING_PUBLIC_KEY", str(public_path))

    with pytest.raises(RuntimeError, match="outside"):
        publisher_from_env()


def test_publisher_from_env_builds_a_wired_publisher(tmp_path, monkeypatch):
    from bddk_mcp.admin.admission_job import publisher_from_env

    seed_root = tmp_path / "corpus"
    _write_fixture_seed(seed_root)
    keys_dir = tmp_path / "keys"
    private_path, public_path = _write_keypair(keys_dir)
    (tmp_path / "drafts").mkdir()
    store = UploadStore(tmp_path / "drafts" / "drafts.sqlite")

    monkeypatch.setenv("BDDK_RELEASE_VERIFIER_DATABASE_URL", _VERIFIER_DSN)
    monkeypatch.setenv("BDDK_RELEASE_PUBLISHER_DATABASE_URL", _PUBLISHER_DSN)
    monkeypatch.setenv("BDDK_SEED_DIR", str(seed_root))
    monkeypatch.setenv("BDDK_ADMISSION_SIGNING_KEY", str(private_path))
    monkeypatch.setenv("BDDK_ADMISSION_SIGNING_PUBLIC_KEY", str(public_path))

    publisher = publisher_from_env(store=store)
    assert callable(publisher)
