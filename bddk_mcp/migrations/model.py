"""Immutable migration definitions and checksum calculation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Migration:
    """One ordered, immutable PostgreSQL schema migration."""

    version: int
    name: str
    statements: tuple[str, ...]

    @property
    def checksum(self) -> str:
        """Return the stable SHA-256 checksum persisted in migration history."""

        payload = json.dumps(
            {
                "version": self.version,
                "name": self.name,
                "statements": self.statements,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
