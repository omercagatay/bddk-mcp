import pytest

from bddk_mcp.admin.uploads import UploadStore


def test_upload_store_accepts_pdf_and_rejects_other_types(tmp_path):
    store = UploadStore(tmp_path / "drafts.sqlite")
    pdf = b"%PDF-1.4\n"
    upload_id = store.save("note.pdf", pdf)
    assert store.path_for(upload_id).read_bytes() == pdf
    assert store.reject_reason("note.txt", b"hello") == "unsupported_type"
    assert store.reject_reason("note.doc", b"hello") == "unsupported_type"


def test_saved_correction_survives_repeat_extraction(tmp_path, monkeypatch):
    store = UploadStore(tmp_path / "drafts.sqlite")
    upload_id = store.save("note.docx", b"PK\x03\x04fake")
    monkeypatch.setattr(store, "_extract_bytes", lambda *_: "ilk metin")
    assert store.extract(upload_id) == "ilk metin"
    store.save_correction(upload_id, "duzeltilmis metin")
    monkeypatch.setattr(store, "_extract_bytes", lambda *_: "yeniden okunan")
    assert store.extract(upload_id) == "duzeltilmis metin"
    assert store.corrected_text(upload_id) == "duzeltilmis metin"


def test_corrected_text_never_returns_the_raw_extraction(tmp_path, monkeypatch):
    store = UploadStore(tmp_path / "drafts.sqlite")
    upload_id = store.save("note.pdf", b"%PDF-1.4\n")
    monkeypatch.setattr(store, "_extract_bytes", lambda *_: "ilk metin")
    store.extract(upload_id)
    with pytest.raises(FileNotFoundError):
        store.corrected_text(upload_id)


def test_correction_rejects_null_byte(tmp_path):
    store = UploadStore(tmp_path / "drafts.sqlite")
    upload_id = store.save("note.pdf", b"%PDF-1.4\n")
    with pytest.raises(ValueError, match="null"):
        store.save_correction(upload_id, "satir\x00devam")


def test_admit_is_one_waiting_request_and_rejects_a_second(tmp_path):
    store = UploadStore(tmp_path / "drafts.sqlite")
    upload_id = store.save("note.pdf", b"%PDF-1.4\n")
    with pytest.raises(ValueError, match="not_corrected"):
        store.admit(upload_id)
    store.save_correction(upload_id, "yayinlanacak metin")
    request_id = store.admit(upload_id)
    assert store.request_state(request_id) == "waiting"
    with pytest.raises(ValueError, match="already_waiting"):
        store.admit(upload_id)


def test_concurrent_admits_cannot_both_create_waiting_rows(tmp_path):
    """Two overlapping admits in one process yield exactly one waiting row."""
    import sqlite3
    import threading

    store = UploadStore(tmp_path / "drafts.sqlite")
    upload_id = store.save("note.pdf", b"%PDF-1.4\n")
    store.save_correction(upload_id, "yayinlanacak metin")

    barrier = threading.Barrier(2)
    request_ids: list[str] = []
    rejected: list[str] = []

    def admit():
        barrier.wait()
        try:
            request_ids.append(store.admit(upload_id))
        except ValueError as exc:
            rejected.append(str(exc))

    threads = [threading.Thread(target=admit) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(request_ids) == 1
    assert rejected == ["already_waiting"]
    with sqlite3.connect(store.draft_db) as db:
        waiting = db.execute(
            "SELECT COUNT(*) FROM admission_requests WHERE upload_id = ? AND state = 'waiting'",
            (upload_id,),
        ).fetchone()[0]
    assert waiting == 1
