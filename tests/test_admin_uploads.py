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
    store.extract(upload_id)
    assert store.corrected_text(upload_id) == "duzeltilmis metin"


def test_correction_rejects_null_byte(tmp_path):
    store = UploadStore(tmp_path / "drafts.sqlite")
    upload_id = store.save("note.pdf", b"%PDF-1.4\n")
    with pytest.raises(ValueError, match="null"):
        store.save_correction(upload_id, "satir\x00devam")
