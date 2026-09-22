"""Uploaded PDF and DOCX bytes, stored outside the corpus."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from uuid import uuid4


class UploadStore:
    def __init__(self, draft_db: Path) -> None:
        self.draft_db = draft_db
        self.root = draft_db.parent / "uploads"
        self.root.mkdir(mode=0o700, exist_ok=True)
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS upload_drafts ("
                "upload_id TEXT PRIMARY KEY, extracted_text TEXT, corrected_text TEXT, corrected INTEGER NOT NULL)"
            )

    def reject_reason(self, filename: str, data: bytes) -> str | None:
        suffix = Path(filename).suffix.lower()
        if suffix not in {".pdf", ".docx"} or not data:
            return "unsupported_type"
        return None

    def save(self, filename: str, data: bytes) -> str:
        reason = self.reject_reason(filename, data)
        if reason:
            raise ValueError(reason)
        upload_id = uuid4().hex
        target = self.root / f"{upload_id}{Path(filename).suffix.lower()}"
        target.write_bytes(data)
        os.chmod(target, 0o600)
        return upload_id

    def path_for(self, upload_id: str) -> Path:
        matches = list(self.root.glob(f"{upload_id}.*"))
        if len(matches) != 1:
            raise FileNotFoundError(upload_id)
        return matches[0]

    def _connect(self) -> sqlite3.Connection:
        self.draft_db.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.draft_db)
        db.row_factory = sqlite3.Row
        return db

    def _extract_bytes(self, data: bytes, suffix: str) -> str:
        from bddk_mcp.ingest.doc_sync import DocumentSyncer

        syncer = DocumentSyncer.__new__(DocumentSyncer)
        syncer._ocr_backends = []
        result = DocumentSyncer._extract_structured(syncer, data, suffix)
        if not result.content:
            raise ValueError(result.error or "extraction_failed")
        return result.content

    def extract(self, upload_id: str) -> str:
        path = self.path_for(upload_id)
        text = self._extract_bytes(path.read_bytes(), path.suffix.lower())
        with self._connect() as db:
            row = db.execute(
                "SELECT corrected, corrected_text, extracted_text FROM upload_drafts WHERE upload_id = ?",
                (upload_id,),
            ).fetchone()
            if row is not None and row["corrected"]:
                return row["corrected_text"]
            db.execute(
                "INSERT INTO upload_drafts (upload_id, extracted_text, corrected_text, corrected) "
                "VALUES (?, ?, ?, 0) "
                "ON CONFLICT(upload_id) DO UPDATE SET extracted_text = excluded.extracted_text "
                "WHERE corrected = 0",
                (upload_id, text, text),
            )
        return text

    def save_correction(self, upload_id: str, text: str) -> None:
        if "\x00" in text:
            raise ValueError("null byte")
        if not text.strip():
            raise ValueError("empty")
        self.path_for(upload_id)
        with self._connect() as db:
            db.execute(
                "INSERT INTO upload_drafts (upload_id, extracted_text, corrected_text, corrected) "
                "VALUES (?, ?, ?, 1) "
                "ON CONFLICT(upload_id) DO UPDATE SET corrected_text = excluded.corrected_text, corrected = 1",
                (upload_id, text, text),
            )

    def corrected_text(self, upload_id: str) -> str:
        with self._connect() as db:
            row = db.execute(
                "SELECT corrected_text FROM upload_drafts WHERE upload_id = ?",
                (upload_id,),
            ).fetchone()
        if row is None or not row["corrected_text"]:
            raise FileNotFoundError(upload_id)
        return row["corrected_text"]
