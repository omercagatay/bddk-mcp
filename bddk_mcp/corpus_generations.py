"""Atomic, privacy-safe retention of an active v5 corpus release.

This module is an administrative persistence boundary.  It is not registered
as an MCP tool and does not activate, reactivate, or serve a retained
generation.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RELEASE_ID_RE = re.compile(r"^corpus_release_sha256_[0-9a-f]{64}$")
_GENERATION_ID_RE = re.compile(r"^corpus_generation_sha256_[0-9a-f]{64}$")
_SEAL_ID_RE = re.compile(r"^corpus_generation_seal_sha256_[0-9a-f]{64}$")

_RETAIN_SQL = """
SELECT generation_id,
       seal_id,
       release_id,
       source_activation_sequence,
       corpus_state_sha256,
       retrieval_profile_sha256,
       inventory_sha256,
       relation_count,
       row_count,
       retained_at
FROM bddk_meta.retain_active_corpus_generation($1)
"""
_STORAGE_SQL = """
SELECT generation_id,
       relation_count,
       row_count,
       generation_logical_bytes,
       retained_store_heap_main_bytes,
       retained_store_heap_auxiliary_bytes,
       retained_store_toast_bytes,
       retained_store_index_bytes,
       retained_store_total_bytes
FROM bddk_meta.inspect_retained_generation_storage($1)
"""
_RETENTION_STATUS_SQL = """
SELECT release_id,
       retention_status,
       generation_id,
       seal_id,
       corpus_state_sha256,
       retrieval_profile_sha256,
       retained_at
FROM bddk_meta.corpus_release_retention_status
WHERE release_id = $1
"""


class CorpusGenerationError(RuntimeError):
    """Raised when retained-generation evidence cannot be safely produced or read."""


def _value(row: Any, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return default


def _aware_timestamp(value: object) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp is not timezone-aware")
    return value


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("value is not text")
    return value


def _integer(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("value is not an integer")
    return value


def _optional_non_negative_integer(value: object) -> int | None:
    if value is None:
        return None
    parsed = _integer(value)
    if parsed < 0:
        raise ValueError("value is negative")
    return parsed


def _fingerprint_frame(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return b"\x01" + len(encoded).to_bytes(8, byteorder="big", signed=True) + encoded


def _generation_id(corpus_state_sha256: str, retrieval_profile_sha256: str) -> str:
    digest = hashlib.sha256(
        _fingerprint_frame("1") + _fingerprint_frame(corpus_state_sha256) + _fingerprint_frame(retrieval_profile_sha256)
    ).hexdigest()
    return "corpus_generation_sha256_" + digest


def _seal_id(
    generation_id: str,
    corpus_state_sha256: str,
    retrieval_profile_sha256: str,
    inventory_sha256: str,
) -> str:
    digest = hashlib.sha256(
        _fingerprint_frame("1")
        + _fingerprint_frame(generation_id)
        + _fingerprint_frame(corpus_state_sha256)
        + _fingerprint_frame(retrieval_profile_sha256)
        + _fingerprint_frame(inventory_sha256)
    ).hexdigest()
    return "corpus_generation_seal_sha256_" + digest


@dataclass(frozen=True, slots=True)
class CorpusGenerationReceipt:
    """Content-free receipt for one atomic typed snapshot and seal."""

    generation_id: str
    seal_id: str
    release_id: str
    source_activation_sequence: int
    corpus_state_sha256: str
    retrieval_profile_sha256: str
    inventory_sha256: str
    relation_count: int
    row_count: int
    retained_at: datetime

    def safe_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["retained_at"] = self.retained_at.isoformat()
        return value


@dataclass(frozen=True, slots=True)
class CorpusGenerationStorageEvidence:
    """Catalog-reconciled store sizes; not a measured backup or exact WAL cost."""

    generation_id: str
    relation_count: int
    row_count: int
    generation_logical_bytes: int
    retained_store_heap_main_bytes: int
    retained_store_heap_auxiliary_bytes: int
    retained_store_toast_bytes: int
    retained_store_index_bytes: int
    retained_store_total_bytes: int
    observed_cluster_wal_bytes: int | None = None
    wal_attribution: str = "not_measured"
    backup_growth_bytes: int | None = None
    backup_growth_status: str = "not_measured"

    def safe_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CorpusReleaseRetentionStatus:
    release_id: str
    retention_status: str
    generation_id: str | None
    seal_id: str | None
    corpus_state_sha256: str
    retrieval_profile_sha256: str
    retained_at: datetime | None

    def safe_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["retained_at"] = self.retained_at.isoformat() if self.retained_at is not None else None
        return value


def _receipt(row: Any) -> CorpusGenerationReceipt:
    try:
        receipt = CorpusGenerationReceipt(
            generation_id=_text(_value(row, "generation_id")),
            seal_id=_text(_value(row, "seal_id")),
            release_id=_text(_value(row, "release_id")),
            source_activation_sequence=_integer(_value(row, "source_activation_sequence")),
            corpus_state_sha256=_text(_value(row, "corpus_state_sha256")),
            retrieval_profile_sha256=_text(_value(row, "retrieval_profile_sha256")),
            inventory_sha256=_text(_value(row, "inventory_sha256")),
            relation_count=_integer(_value(row, "relation_count")),
            row_count=_integer(_value(row, "row_count")),
            retained_at=_aware_timestamp(_value(row, "retained_at")),
        )
    except (TypeError, ValueError):
        raise CorpusGenerationError("Retained corpus generation receipt is invalid.") from None
    if (
        _GENERATION_ID_RE.fullmatch(receipt.generation_id) is None
        or _SEAL_ID_RE.fullmatch(receipt.seal_id) is None
        or _RELEASE_ID_RE.fullmatch(receipt.release_id) is None
        or any(
            _SHA256_RE.fullmatch(value) is None
            for value in (
                receipt.corpus_state_sha256,
                receipt.retrieval_profile_sha256,
                receipt.inventory_sha256,
            )
        )
        or receipt.source_activation_sequence < 1
        or receipt.relation_count != 17
        or receipt.row_count < 0
        or receipt.generation_id
        != _generation_id(
            receipt.corpus_state_sha256,
            receipt.retrieval_profile_sha256,
        )
        or receipt.seal_id
        != _seal_id(
            receipt.generation_id,
            receipt.corpus_state_sha256,
            receipt.retrieval_profile_sha256,
            receipt.inventory_sha256,
        )
    ):
        raise CorpusGenerationError("Retained corpus generation receipt is invalid.")
    return receipt


def _storage(
    row: Any,
    *,
    observed_cluster_wal_bytes: object = None,
    backup_growth_bytes: object = None,
) -> CorpusGenerationStorageEvidence:
    try:
        wal_bytes = _optional_non_negative_integer(observed_cluster_wal_bytes)
        backup_bytes = _optional_non_negative_integer(backup_growth_bytes)
        evidence = CorpusGenerationStorageEvidence(
            generation_id=_text(_value(row, "generation_id")),
            relation_count=_integer(_value(row, "relation_count")),
            row_count=_integer(_value(row, "row_count")),
            generation_logical_bytes=_integer(_value(row, "generation_logical_bytes")),
            retained_store_heap_main_bytes=_integer(_value(row, "retained_store_heap_main_bytes")),
            retained_store_heap_auxiliary_bytes=_integer(_value(row, "retained_store_heap_auxiliary_bytes")),
            retained_store_toast_bytes=_integer(_value(row, "retained_store_toast_bytes")),
            retained_store_index_bytes=_integer(_value(row, "retained_store_index_bytes")),
            retained_store_total_bytes=_integer(_value(row, "retained_store_total_bytes")),
            observed_cluster_wal_bytes=wal_bytes,
            wal_attribution=("observed_cluster_interval_not_exclusive" if wal_bytes is not None else "not_measured"),
            backup_growth_bytes=backup_bytes,
            backup_growth_status=("measured_controlled_backup" if backup_bytes is not None else "not_measured"),
        )
    except (TypeError, ValueError):
        raise CorpusGenerationError("Retained corpus generation storage evidence is invalid.") from None
    components = (
        evidence.retained_store_heap_main_bytes,
        evidence.retained_store_heap_auxiliary_bytes,
        evidence.retained_store_toast_bytes,
        evidence.retained_store_index_bytes,
    )
    if (
        _GENERATION_ID_RE.fullmatch(evidence.generation_id) is None
        or evidence.relation_count != 17
        or min(evidence.row_count, evidence.generation_logical_bytes, *components) < 0
        or sum(components) != evidence.retained_store_total_bytes
    ):
        raise CorpusGenerationError("Retained corpus generation storage evidence is invalid.")
    return evidence


async def retain_active_corpus_generation(connection: Any, *, expected_release_id: str) -> CorpusGenerationReceipt:
    """Atomically retain and seal the exact expected active release."""

    if not isinstance(expected_release_id, str) or _RELEASE_ID_RE.fullmatch(expected_release_id) is None:
        raise CorpusGenerationError("Expected active corpus release identity is invalid.")
    try:
        row = await connection.fetchrow(_RETAIN_SQL, expected_release_id)
    except Exception:
        raise CorpusGenerationError("Active corpus release could not be retained.") from None
    if row is None:
        raise CorpusGenerationError("Active corpus release could not be retained.")
    receipt = _receipt(row)
    if receipt.release_id != expected_release_id:
        raise CorpusGenerationError("Retained corpus generation receipt is invalid.")
    return receipt


async def collect_generation_storage_evidence(
    connection: Any,
    *,
    generation_id: str,
    observed_cluster_wal_bytes: int | None = None,
    backup_growth_bytes: int | None = None,
) -> CorpusGenerationStorageEvidence:
    """Collect fixed numeric catalog sizes and explicitly qualified measurements.

    WAL is an observed cluster-wide interval and is never presented as exact
    generation attribution.  Backup growth may be supplied only by a separate,
    controlled backup workflow; catalog sizes are not treated as a substitute.
    """

    if not isinstance(generation_id, str) or _GENERATION_ID_RE.fullmatch(generation_id) is None:
        raise CorpusGenerationError("Retained corpus generation identity is invalid.")
    try:
        row = await connection.fetchrow(_STORAGE_SQL, generation_id)
    except Exception:
        raise CorpusGenerationError("Retained corpus generation storage evidence is unavailable.") from None
    if row is None:
        raise CorpusGenerationError("Retained corpus generation storage evidence is unavailable.")
    try:
        evidence = _storage(
            row,
            observed_cluster_wal_bytes=observed_cluster_wal_bytes,
            backup_growth_bytes=backup_growth_bytes,
        )
    except CorpusGenerationError:
        raise
    except (TypeError, ValueError):
        raise CorpusGenerationError("Retained corpus generation storage evidence is invalid.") from None
    if evidence.generation_id != generation_id:
        raise CorpusGenerationError("Retained corpus generation storage evidence is invalid.")
    return evidence


async def inspect_release_retention(
    connection: Any,
    *,
    release_id: str,
) -> CorpusReleaseRetentionStatus | None:
    """Return the explicit retained or legacy-unretained state of one v5 release."""

    if not isinstance(release_id, str) or _RELEASE_ID_RE.fullmatch(release_id) is None:
        raise CorpusGenerationError("Corpus release identity is invalid.")
    try:
        row = await connection.fetchrow(_RETENTION_STATUS_SQL, release_id)
    except Exception:
        raise CorpusGenerationError("Corpus release retention status is unavailable.") from None
    if row is None:
        return None
    retained_at = _value(row, "retained_at")
    try:
        status = CorpusReleaseRetentionStatus(
            release_id=_text(_value(row, "release_id")),
            retention_status=_text(_value(row, "retention_status")),
            generation_id=(_text(_value(row, "generation_id")) if _value(row, "generation_id") is not None else None),
            seal_id=(_text(_value(row, "seal_id")) if _value(row, "seal_id") is not None else None),
            corpus_state_sha256=_text(_value(row, "corpus_state_sha256")),
            retrieval_profile_sha256=_text(_value(row, "retrieval_profile_sha256")),
            retained_at=(_aware_timestamp(retained_at) if retained_at is not None else None),
        )
    except (TypeError, ValueError):
        raise CorpusGenerationError("Corpus release retention status is invalid.") from None
    retained = status.retention_status == "retained"
    complete_retained_binding = (
        status.generation_id is not None
        and status.seal_id is not None
        and status.retained_at is not None
        and _GENERATION_ID_RE.fullmatch(status.generation_id) is not None
        and _SEAL_ID_RE.fullmatch(status.seal_id) is not None
    )
    empty_legacy_binding = status.generation_id is None and status.seal_id is None and status.retained_at is None
    if (
        status.release_id != release_id
        or status.retention_status not in {"legacy_v5_unretained", "retained"}
        or _SHA256_RE.fullmatch(status.corpus_state_sha256) is None
        or _SHA256_RE.fullmatch(status.retrieval_profile_sha256) is None
        or (retained and not complete_retained_binding)
        or (not retained and not empty_legacy_binding)
    ):
        raise CorpusGenerationError("Corpus release retention status is invalid.")
    return status


__all__ = (
    "CorpusGenerationError",
    "CorpusGenerationReceipt",
    "CorpusGenerationStorageEvidence",
    "CorpusReleaseRetentionStatus",
    "collect_generation_storage_evidence",
    "inspect_release_retention",
    "retain_active_corpus_generation",
)
