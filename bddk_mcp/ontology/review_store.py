"""Append-only decision store for link proposals: a sidecar SQLite file outside the corpus.

Decisions are the owner's review record for one proposals file: who decided
what, when, why, and by which method (a person on the review screen, or the
cross-confirmation rule).  The store never updates or deletes a row; a change
of mind is a new row and the latest row per proposal wins.  Nothing here
reaches PostgreSQL.  The bundle builder (Step 4) reads the export and takes
only approved decisions.
"""

from __future__ import annotations

import os
import re
import sqlite3
import stat
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from bddk_mcp.ontology.proposals import Proposal, ProposalFile, ProposalFlag, ProposalSource

DECISION_APPROVED = "approved"
DECISION_REJECTED = "rejected"
DECISIONS = (DECISION_APPROVED, DECISION_REJECTED)
METHOD_HUMAN = "human"
METHOD_AUTO = "auto:cross_confirmed"
AUTO_REVIEWER = "auto:cross_confirmed"
AUTO_REASON = "Resmî Gazete tablosu ile çapraz doğrulandı; otomatik kabul."
SPOT_CHECK_EVERY = 5
MAX_REASON_LENGTH = 2000
_REVIEWER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,199}$")
_PROPOSAL_ID_RE = re.compile(r"^prop_sha256_[0-9a-f]{64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ReviewStoreError(ValueError):
    """A decision could not be recorded; nothing was written."""


@dataclass(frozen=True, slots=True)
class Decision:
    decision_id: int
    proposals_file_sha256: str
    proposal_id: str
    decision: str
    decided_by: str
    decided_at: str
    reason: str
    method: str
    spot_check: bool

    def as_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "proposal_id": self.proposal_id,
            "decision": self.decision,
            "decided_by": self.decided_by,
            "decided_at": self.decided_at,
            "reason": self.reason,
            "method": self.method,
            "spot_check": self.spot_check,
        }


def is_spot_check(proposal_id: str) -> bool:
    """Deterministic one-in-five sample drawn from the proposal identity.

    The same proposals file always yields the same spot checks, so a review
    can be reproduced and audited; a fresh random draw on every page load
    could not.
    """

    return int(proposal_id[-8:], 16) % SPOT_CHECK_EVERY == 0


def is_auto_acceptable(proposal: Proposal) -> bool:
    """Rule proposals the table confirms, unless flagged for review or sampled as a spot check."""

    return (
        proposal.source is ProposalSource.RULE
        and ProposalFlag.CROSS_CONFIRMED in proposal.flags
        and ProposalFlag.NEEDS_REVIEW not in proposal.flags
        and ProposalFlag.RG_NUMBER_MALFORMED not in proposal.flags
        and not is_spot_check(proposal.proposal_id)
    )


def _validate_reason(reason: str) -> str:
    cleaned = " ".join(reason.split())
    if "\x00" in reason or len(cleaned) > MAX_REASON_LENGTH:
        raise ReviewStoreError("Decision reason is too long or contains control characters.")
    return cleaned


class ReviewStore:
    """One short SQLite transaction per operation; connections are never shared across threads."""

    def __init__(self, path: Path) -> None:
        self.path = path
        if path.exists() and not stat.S_ISREG(path.stat().st_mode):
            raise ValueError("Ontology review database must be a regular file")
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        os.close(descriptor)
        if path.stat().st_mode & 0o077:
            raise ValueError("Ontology review database must be owner-only")
        with closing(self._connect()) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS proposal_decisions ("
                "decision_id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "proposals_file_sha256 TEXT NOT NULL, "
                "proposal_id TEXT NOT NULL, "
                "decision TEXT NOT NULL CHECK (decision IN ('approved', 'rejected')), "
                "decided_by TEXT NOT NULL, "
                "decided_at TEXT NOT NULL, "
                "reason TEXT NOT NULL DEFAULT '', "
                "method TEXT NOT NULL CHECK (method IN ('human', 'auto:cross_confirmed')), "
                "spot_check INTEGER NOT NULL DEFAULT 0 CHECK (spot_check IN (0, 1)))"
            )
            # The audit trail is append-only at the database level, not only by convention.
            db.execute(
                "CREATE TRIGGER IF NOT EXISTS proposal_decisions_no_update BEFORE UPDATE ON proposal_decisions "
                "BEGIN SELECT RAISE(ABORT, 'decisions are append-only'); END"
            )
            db.execute(
                "CREATE TRIGGER IF NOT EXISTS proposal_decisions_no_delete BEFORE DELETE ON proposal_decisions "
                "BEGIN SELECT RAISE(ABORT, 'decisions are append-only'); END"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_proposal_decisions_lookup "
                "ON proposal_decisions (proposals_file_sha256, proposal_id, decision_id)"
            )

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def _row(row: sqlite3.Row) -> Decision:
        return Decision(
            decision_id=int(row["decision_id"]),
            proposals_file_sha256=row["proposals_file_sha256"],
            proposal_id=row["proposal_id"],
            decision=row["decision"],
            decided_by=row["decided_by"],
            decided_at=row["decided_at"],
            reason=row["reason"],
            method=row["method"],
            spot_check=bool(row["spot_check"]),
        )

    def record(
        self,
        *,
        proposals_file_sha256: str,
        proposal_id: str,
        decision: str,
        decided_by: str,
        reason: str = "",
        method: str = METHOD_HUMAN,
        spot_check: bool = False,
        now: datetime | None = None,
    ) -> Decision:
        """Append one decision; the previous decisions on the same proposal stay in the history."""

        if decision not in DECISIONS:
            raise ReviewStoreError("Decision must be approved or rejected.")
        if method not in (METHOD_HUMAN, METHOD_AUTO):
            raise ReviewStoreError("Decision method is not reviewed.")
        if not _REVIEWER_RE.fullmatch(decided_by):
            raise ReviewStoreError("Reviewer identity is invalid.")
        if not _PROPOSAL_ID_RE.fullmatch(proposal_id) or not _SHA256_RE.fullmatch(proposals_file_sha256):
            raise ReviewStoreError("Proposal or file identity is invalid.")
        cleaned = _validate_reason(reason)
        decided_at = (now or datetime.now(UTC)).astimezone(UTC).isoformat(timespec="seconds")
        with closing(self._connect()) as db, db:
            cursor = db.execute(
                "INSERT INTO proposal_decisions (proposals_file_sha256, proposal_id, decision, decided_by, decided_at, "
                "reason, method, spot_check) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    proposals_file_sha256,
                    proposal_id,
                    decision,
                    decided_by,
                    decided_at,
                    cleaned,
                    method,
                    int(spot_check),
                ),
            )
            row = db.execute("SELECT * FROM proposal_decisions WHERE decision_id = ?", (cursor.lastrowid,)).fetchone()
        return self._row(row)

    def latest(self, proposals_file_sha256: str) -> dict[str, Decision]:
        """The current decision per proposal: the most recently appended row wins."""

        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT * FROM proposal_decisions WHERE proposals_file_sha256 = ? ORDER BY decision_id",
                (proposals_file_sha256,),
            ).fetchall()
        current: dict[str, Decision] = {}
        for row in rows:
            decision = self._row(row)
            current[decision.proposal_id] = decision
        return current

    def history(self, proposals_file_sha256: str, proposal_id: str) -> list[Decision]:
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT * FROM proposal_decisions WHERE proposals_file_sha256 = ? AND proposal_id = ? "
                "ORDER BY decision_id",
                (proposals_file_sha256, proposal_id),
            ).fetchall()
        return [self._row(row) for row in rows]

    def auto_accept(self, file: ProposalFile, *, proposals_file_sha256: str, now: datetime | None = None) -> int:
        """Approve the cross-confirmed rule proposals that carry no decision yet; idempotent."""

        current = self.latest(proposals_file_sha256)
        added = 0
        for proposal in file.proposals:
            if proposal.proposal_id in current or not is_auto_acceptable(proposal):
                continue
            self.record(
                proposals_file_sha256=proposals_file_sha256,
                proposal_id=proposal.proposal_id,
                decision=DECISION_APPROVED,
                decided_by=AUTO_REVIEWER,
                reason=AUTO_REASON,
                method=METHOD_AUTO,
                spot_check=False,
                now=now,
            )
            added += 1
        return added

    def export(self, file: ProposalFile, *, proposals_file_sha256: str, now: datetime | None = None) -> dict:
        """The current decisions in proposal order, ready for the bundle builder."""

        current = self.latest(proposals_file_sha256)
        decisions = [current[item.proposal_id].as_dict() for item in file.proposals if item.proposal_id in current]
        approved = sum(1 for item in decisions if item["decision"] == DECISION_APPROVED)
        rejected = sum(1 for item in decisions if item["decision"] == DECISION_REJECTED)
        return {
            "artifact_type": "bddk-link-decisions-v1",
            "schema_version": 1,
            "publication_status": "unpublished",
            "document_id": file.document_id,
            "content_hash": file.content_hash,
            "proposals_file_sha256": proposals_file_sha256,
            "exported_at": (now or datetime.now(UTC)).astimezone(UTC).isoformat(timespec="seconds"),
            "counts": {
                "total": len(file.proposals),
                "approved": approved,
                "rejected": rejected,
                "pending": len(file.proposals) - approved - rejected,
            },
            "decisions": decisions,
        }
