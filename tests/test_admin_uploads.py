from bddk_mcp.admin.uploads import UploadStore


def test_upload_store_accepts_pdf_and_rejects_other_types(tmp_path):
    store = UploadStore(tmp_path / "drafts.sqlite")
    pdf = b"%PDF-1.4\n"
    upload_id = store.save("note.pdf", pdf)
    assert store.path_for(upload_id).read_bytes() == pdf
    assert store.reject_reason("note.txt", b"hello") == "unsupported_type"
    assert store.reject_reason("note.doc", b"hello") == "unsupported_type"
