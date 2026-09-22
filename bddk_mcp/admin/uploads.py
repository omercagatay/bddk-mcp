"""Uploaded PDF and DOCX bytes, stored outside the corpus."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4


class UploadStore:
    def __init__(self, draft_db: Path) -> None:
        self.root = draft_db.parent / "uploads"
        self.root.mkdir(mode=0o700, exist_ok=True)

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
