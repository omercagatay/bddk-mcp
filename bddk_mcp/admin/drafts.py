"""Private SQLite editorial drafts; never a corpus publication or provenance claim."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import stat
from contextlib import closing
from pathlib import Path

from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from pydantic import BaseModel, ConfigDict, Field, field_validator

from bddk_mcp.core.outbound_http import (
    BDDK_HTTPS_HOSTS,
    MEVZUAT_HTTPS_HOSTS,
    OutboundHttpPolicyError,
    normalize_approved_https_url,
)
from bddk_mcp.store.doc_store import StoredDocument

MAX_CONTENT = 1_000_000


class DraftConflict(ValueError):
    """The submitted revision or canonical base is stale."""


class RevisionForm(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    revision: int = Field(ge=0, le=2**63 - 2)
    base_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator("revision", mode="before")
    @classmethod
    def form_integer(cls, value):
        if isinstance(value, str) and value.isascii() and value.isdecimal() and len(value) <= 19:
            return int(value)
        return value


class EditForm(RevisionForm):
    title: str = Field(min_length=1, max_length=500)
    category: str = Field(max_length=100)
    decision_date: str = Field(max_length=32)
    decision_number: str = Field(max_length=100)
    source_url: str = Field(max_length=2048)
    markdown_content: str = Field(min_length=1, max_length=MAX_CONTENT)

    @field_validator("title", "markdown_content")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("Nonblank text required")
        return value

    @field_validator("source_url")
    @classmethod
    def approved_source(cls, value: str) -> str:
        try:
            return normalize_approved_https_url(
                value, base_url="", allowed_hosts=BDDK_HTTPS_HOSTS | MEVZUAT_HTTPS_HOSTS, boundary_name="regulatory"
            )
        except OutboundHttpPolicyError:
            raise ValueError("Approved absolute HTTPS source required") from None


def canonical_json(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def snapshot(doc: StoredDocument) -> dict:
    data = doc.model_dump(exclude={"pdf_bytes"})
    data["pdf_sha256"] = hashlib.sha256(doc.pdf_bytes).hexdigest() if doc.pdf_bytes is not None else None
    return data


def fingerprint(doc: StoredDocument) -> str:
    return hashlib.sha256(canonical_json(snapshot(doc))).hexdigest()


class DraftSigner:
    """Server-held editorial key paired with an independently supplied trust anchor."""

    def __init__(self, private_path: Path, public_path: Path) -> None:
        try:
            if private_path.stat().st_mode & 0o077:
                raise ValueError("Private signing key must be owner-only")
            for path in (private_path, public_path):
                if not stat.S_ISREG(path.stat().st_mode) or path.stat().st_size > 8192:
                    raise ValueError("Bounded regular key files required")
            with private_path.open("rb") as stream:
                private_bytes = stream.read(8193)
            with public_path.open("rb") as stream:
                public_bytes = stream.read(8193)
            if max(len(private_bytes), len(public_bytes)) > 8192:
                raise ValueError("Bounded key files required")
            private = serialization.load_pem_private_key(private_bytes, password=None)
            public = serialization.load_pem_public_key(public_bytes)
            if not isinstance(private, Ed25519PrivateKey) or not isinstance(public, Ed25519PublicKey):
                raise ValueError
            if private.public_key().public_bytes_raw() != public.public_bytes_raw():
                raise ValueError
        except (OSError, ValueError, TypeError, UnsupportedAlgorithm):
            raise ValueError("Admin signing keys unavailable, unsafe, or mismatched") from None
        self._private = private
        self.public = public

    def sign(self, payload: dict) -> dict:
        signature = self._private.sign(canonical_json(payload))
        self.public.verify(signature, canonical_json(payload))
        return {
            "algorithm": "Ed25519",
            "signature_base64": base64.b64encode(signature).decode("ascii"),
            "public_key_pem": self.public.public_bytes(
                serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode("ascii"),
            "key_fingerprint_sha256": hashlib.sha256(self.public.public_bytes_raw()).hexdigest(),
        }


class DraftStore:
    """One short SQLite transaction per operation, connections never shared across threads."""

    def __init__(self, path: Path, signer: DraftSigner | None = None) -> None:
        self.path = path
        self.signer = signer
        # The operator supplies an existing private parent directory (not a corpus mount).
        if path.exists() and not stat.S_ISREG(path.stat().st_mode):
            raise ValueError("Admin draft database must be a regular file")
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        os.close(descriptor)
        if path.stat().st_mode & 0o077:
            raise ValueError("Admin draft database must be owner-only")
        with closing(self._connect()) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS drafts (document_id TEXT PRIMARY KEY, revision INTEGER NOT NULL, "
                "base_fingerprint TEXT NOT NULL, payload TEXT NOT NULL, signature TEXT)"
            )

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    def get(self, document_id: str) -> dict | None:
        with closing(self._connect()) as db:
            row = db.execute("SELECT * FROM drafts WHERE document_id = ?", (document_id,)).fetchone()
            if row is None:
                return None
            return {"payload": json.loads(row["payload"]), "signature": json.loads(row["signature"] or "null")}

    def change(self, doc: StoredDocument, form: RevisionForm, *, sign: bool = False) -> dict:
        base = fingerprint(doc)
        if form.base_fingerprint != base:
            raise DraftConflict("Canonical base changed; retain the draft and reconcile offline.")
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM drafts WHERE document_id = ?", (doc.document_id,)).fetchone()
            if (row["revision"] if row else 0) != form.revision or (row and row["base_fingerprint"] != base):
                raise DraftConflict("Draft revision changed; reload before editing or signing.")
            if sign:
                if not row or self.signer is None:
                    raise ValueError("Signing requires a saved draft and configured keys")
                payload = json.loads(row["payload"])
                signature = self.signer.sign(payload)
                db.execute(
                    "UPDATE drafts SET signature = ? WHERE document_id = ?",
                    (canonical_json(signature).decode(), doc.document_id),
                )
            else:
                if not isinstance(form, EditForm):
                    raise ValueError("Validated edit required")
                original = json.loads(row["payload"])["original"] if row else snapshot(doc)
                payload = {
                    "artifact_type": "bddk-editorial-draft-v1",
                    "publication_status": "unpublished",
                    "manual_edit_verification": "unverified",
                    "document_id": doc.document_id,
                    "revision": form.revision + 1,
                    "base_fingerprint": base,
                    "original": original,
                    "edited": form.model_dump(exclude={"revision", "base_fingerprint"}),
                }
                signature = None
                db.execute(
                    "INSERT INTO drafts VALUES (?, ?, ?, ?, NULL) ON CONFLICT(document_id) DO UPDATE SET "
                    "revision=excluded.revision, payload=excluded.payload, signature=NULL",
                    (doc.document_id, payload["revision"], base, canonical_json(payload).decode()),
                )
            return {"payload": payload, "signature": signature}
