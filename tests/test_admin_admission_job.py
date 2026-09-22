from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from bddk_mcp.admin.uploads import UploadStore


def _store_with_two_waiting_uploads(tmp_path: Path) -> tuple[UploadStore, str, str]:
    store = UploadStore(tmp_path / "drafts.sqlite")
    first = store.save("a.pdf", b"%PDF-1.4\n")
    second = store.save("b.pdf", b"%PDF-1.4\n")
    store.save_correction(first, "bir")
    store.save_correction(second, "iki")
    first_request = store.admit(first)
    second_request = store.admit(second)
    return store, first_request, second_request


def test_failed_activation_leaves_request_in_error_and_does_not_combine_requests(tmp_path):
    from bddk_mcp.admin.admission_job import admit_next

    store, first_request, second_request = _store_with_two_waiting_uploads(tmp_path)

    def fail_publish(text):
        fail_publish.calls.append(text)
        raise RuntimeError("activate failed")

    fail_publish.calls = []

    assert admit_next(store, fail_publish) == "error"
    assert store.request_state(first_request) == "error"
    assert store.request_state(second_request) == "waiting"
    assert fail_publish.calls == ["bir"]


def test_successful_admission_publishes_the_corrected_text(tmp_path):
    from bddk_mcp.admin.admission_job import admit_next

    store = UploadStore(tmp_path / "drafts.sqlite")
    upload_id = store.save("a.pdf", b"%PDF-1.4\n")
    store.save_correction(upload_id, "tek metin")
    request_id = store.admit(upload_id)

    calls: list[str] = []

    def publish(text):
        calls.append(text)

    assert admit_next(store, publish) == "published"
    assert calls == ["tek metin"]
    assert store.request_state(request_id) == "published"


def test_admit_next_with_no_waiting_request_is_idle(tmp_path):
    from bddk_mcp.admin.admission_job import admit_next

    store = UploadStore(tmp_path / "drafts.sqlite")
    assert admit_next(store, lambda text: None) == "idle"


def test_admit_next_publishes_one_request_per_call(tmp_path):
    from bddk_mcp.admin.admission_job import admit_next

    store, _first_request, second_request = _store_with_two_waiting_uploads(tmp_path)

    calls: list[str] = []

    def publish(text):
        calls.append(text)

    assert admit_next(store, publish) == "published"
    assert calls == ["bir"]
    assert store.request_state(second_request) == "waiting"


def test_admit_next_refuses_an_unwired_publisher_without_marking(tmp_path):
    """An unwired publisher refuses; that refusal is never a request error."""

    from bddk_mcp.admin.admission_job import admit_next

    store, first_request, second_request = _store_with_two_waiting_uploads(tmp_path)

    def unwired_publish(_text):
        raise NotImplementedError("gates not wired")

    with pytest.raises(NotImplementedError, match="gates not wired"):
        admit_next(store, unwired_publish)
    assert store.request_state(first_request) == "waiting"
    assert store.request_state(second_request) == "waiting"


def test_publisher_from_env_with_credentials_fails_loudly_while_gates_are_unwired(monkeypatch):
    """The refusal happens at construction, before any admission request is marked."""

    from bddk_mcp.admin.admission_job import publisher_from_env

    monkeypatch.setenv("BDDK_RELEASE_VERIFIER_DATABASE_URL", "postgresql://verifier@example/db")
    monkeypatch.setenv("BDDK_RELEASE_PUBLISHER_DATABASE_URL", "postgresql://publisher@example/db")

    with pytest.raises(NotImplementedError, match="not wired"):
        publisher_from_env()


def test_cli_admit_next_upload_refuses_before_marking_when_gates_are_unwired(tmp_path, monkeypatch):
    """A configured admit-next-upload run refuses before touching any request state."""

    from bddk_mcp.admin.admission_job import admit_next  # noqa: F401  (store helper import clarity)
    from bddk_mcp.cli import _run_admit_next_upload

    store, first_request, _second_request = _store_with_two_waiting_uploads(tmp_path)
    monkeypatch.setenv("BDDK_RELEASE_VERIFIER_DATABASE_URL", "postgresql://verifier@example/db")
    monkeypatch.setenv("BDDK_RELEASE_PUBLISHER_DATABASE_URL", "postgresql://publisher@example/db")
    args = argparse.Namespace(draft_db=tmp_path / "drafts.sqlite")

    with pytest.raises(NotImplementedError, match="not wired"):
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

    assert admit_next(store, lambda text: None) == "error"
    assert store.request_state(request_id) == "error"
    assert admit_next(store, lambda text: None) == "idle"
