import json

from scripts.sanitize_seed_data import sanitize_seed_documents


def test_sanitize_seed_documents_updates_markdown_hash_and_page_count(tmp_path):
    path = tmp_path / "documents.json"
    path.write_text(
        json.dumps(
            [
                {
                    "document_id": "doc1",
                    "markdown_content": "Başlık\f\n\n\n" + "_" * 20,
                    "content_hash": "old",
                    "total_pages": 99,
                    "extracted_at": 1,
                },
                {
                    "document_id": "doc2",
                    "markdown_content": "Temiz içerik",
                    "content_hash": "same",
                    "total_pages": 1,
                    "extracted_at": 1,
                },
            ]
        ),
        encoding="utf-8",
    )

    result = sanitize_seed_documents(path, write=True)
    documents = json.loads(path.read_text(encoding="utf-8"))

    assert result["changed"] == 1
    assert result["changed_doc_ids"] == ["doc1"]
    assert "\f" not in documents[0]["markdown_content"]
    assert "_" * 20 not in documents[0]["markdown_content"]
    assert documents[0]["content_hash"] != "old"
    assert documents[0]["total_pages"] == 1
    assert documents[1]["content_hash"] == "same"
