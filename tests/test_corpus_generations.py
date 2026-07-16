"""Unit contracts for the administrative retained-generation facade."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

import pytest

from bddk_mcp.corpus_generations import (
    CorpusGenerationError,
    collect_generation_storage_evidence,
    inspect_release_retention,
    retain_active_corpus_generation,
)

RELEASE_ID = "corpus_release_sha256_" + "a" * 64
STATE_HASH = "d" * 64
PROFILE_HASH = "e" * 64
INVENTORY_HASH = "f" * 64
RETAINED_AT = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)


def _frame(value: str) -> bytes:
    encoded = value.encode()
    return b"\x01" + len(encoded).to_bytes(8, "big", signed=True) + encoded


GENERATION_ID = (
    "corpus_generation_sha256_" + hashlib.sha256(_frame("1") + _frame(STATE_HASH) + _frame(PROFILE_HASH)).hexdigest()
)
SEAL_ID = (
    "corpus_generation_seal_sha256_"
    + hashlib.sha256(
        _frame("1") + _frame(GENERATION_ID) + _frame(STATE_HASH) + _frame(PROFILE_HASH) + _frame(INVENTORY_HASH)
    ).hexdigest()
)


class FakeConnection:
    def __init__(self, row: Any = None, *, error: Exception | None = None) -> None:
        self.row = row
        self.error = error
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetchrow(self, query: str, *args: Any) -> Any:
        self.calls.append((query, args))
        if self.error is not None:
            raise self.error
        return self.row


def _receipt_row(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "generation_id": GENERATION_ID,
        "seal_id": SEAL_ID,
        "release_id": RELEASE_ID,
        "source_activation_sequence": 7,
        "corpus_state_sha256": STATE_HASH,
        "retrieval_profile_sha256": PROFILE_HASH,
        "inventory_sha256": INVENTORY_HASH,
        "relation_count": 17,
        "row_count": 34,
        "retained_at": RETAINED_AT,
    }
    value.update(overrides)
    return value


def _storage_row(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "generation_id": GENERATION_ID,
        "relation_count": 17,
        "row_count": 34,
        "generation_logical_bytes": 1_000,
        "retained_store_heap_main_bytes": 2_000,
        "retained_store_heap_auxiliary_bytes": 300,
        "retained_store_toast_bytes": 400,
        "retained_store_index_bytes": 500,
        "retained_store_total_bytes": 3_200,
    }
    value.update(overrides)
    return value


def _status_row(*, retained: bool, **overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "release_id": RELEASE_ID,
        "retention_status": "retained" if retained else "legacy_v5_unretained",
        "generation_id": GENERATION_ID if retained else None,
        "seal_id": SEAL_ID if retained else None,
        "corpus_state_sha256": STATE_HASH,
        "retrieval_profile_sha256": PROFILE_HASH,
        "retained_at": RETAINED_AT if retained else None,
    }
    value.update(overrides)
    return value


@pytest.mark.asyncio
async def test_retain_returns_a_content_free_strict_receipt() -> None:
    connection = FakeConnection(_receipt_row())

    receipt = await retain_active_corpus_generation(connection, expected_release_id=RELEASE_ID)

    assert receipt.release_id == RELEASE_ID
    assert receipt.relation_count == 17
    assert receipt.safe_dict() == {
        **_receipt_row(),
        "retained_at": RETAINED_AT.isoformat(),
    }
    assert connection.calls[0][1] == (RELEASE_ID,)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("generation_id", GENERATION_ID.encode()),
        ("source_activation_sequence", True),
        ("relation_count", "17"),
        ("row_count", 1.5),
        ("retained_at", datetime(2026, 7, 16, 12, 0)),
    ),
)
async def test_retain_rejects_type_coercion_and_naive_timestamps(field: str, value: object) -> None:
    connection = FakeConnection(_receipt_row(**{field: value}))

    with pytest.raises(CorpusGenerationError, match="receipt is invalid") as captured:
        await retain_active_corpus_generation(connection, expected_release_id=RELEASE_ID)

    assert captured.value.__cause__ is None


@pytest.mark.asyncio
async def test_retain_validates_the_expected_release_before_database_access() -> None:
    connection = FakeConnection(_receipt_row())

    with pytest.raises(CorpusGenerationError, match="identity is invalid"):
        await retain_active_corpus_generation(connection, expected_release_id="wrong")

    assert connection.calls == []


@pytest.mark.asyncio
async def test_retain_sanitizes_database_failures_and_suppresses_the_cause() -> None:
    connection = FakeConnection(error=RuntimeError("postgresql://secret corpus text /private/path"))

    with pytest.raises(CorpusGenerationError) as captured:
        await retain_active_corpus_generation(connection, expected_release_id=RELEASE_ID)

    message = str(captured.value)
    assert "postgresql://" not in message
    assert "corpus text" not in message
    assert "/private/path" not in message
    assert captured.value.__cause__ is None


@pytest.mark.asyncio
async def test_storage_evidence_reconciles_and_qualifies_optional_measurements() -> None:
    connection = FakeConnection(_storage_row())

    evidence = await collect_generation_storage_evidence(
        connection,
        generation_id=GENERATION_ID,
        observed_cluster_wal_bytes=123,
        backup_growth_bytes=456,
    )

    assert evidence.observed_cluster_wal_bytes == 123
    assert evidence.wal_attribution == "observed_cluster_interval_not_exclusive"
    assert evidence.backup_growth_bytes == 456
    assert evidence.backup_growth_status == "measured_controlled_backup"
    assert evidence.retained_store_total_bytes == 3_200
    assert (
        sum(
            (
                evidence.retained_store_heap_main_bytes,
                evidence.retained_store_heap_auxiliary_bytes,
                evidence.retained_store_toast_bytes,
                evidence.retained_store_index_bytes,
            )
        )
        == evidence.retained_store_total_bytes
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("row_override", "wal_bytes", "backup_bytes"),
    (
        ({"relation_count": True}, None, None),
        ({"retained_store_total_bytes": 3_201}, None, None),
        ({"generation_logical_bytes": -1}, None, None),
        ({}, True, None),
        ({}, -1, None),
        ({}, None, "456"),
    ),
)
async def test_storage_evidence_rejects_invalid_scalars_or_arithmetic(
    row_override: dict[str, Any],
    wal_bytes: object,
    backup_bytes: object,
) -> None:
    connection = FakeConnection(_storage_row(**row_override))

    with pytest.raises(CorpusGenerationError, match="storage evidence is invalid"):
        await collect_generation_storage_evidence(
            connection,
            generation_id=GENERATION_ID,
            observed_cluster_wal_bytes=wal_bytes,  # type: ignore[arg-type]
            backup_growth_bytes=backup_bytes,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_storage_defaults_do_not_infer_wal_or_backup_from_catalog_sizes() -> None:
    evidence = await collect_generation_storage_evidence(
        FakeConnection(_storage_row()),
        generation_id=GENERATION_ID,
    )

    assert evidence.observed_cluster_wal_bytes is None
    assert evidence.wal_attribution == "not_measured"
    assert evidence.backup_growth_bytes is None
    assert evidence.backup_growth_status == "not_measured"


@pytest.mark.asyncio
async def test_storage_evidence_must_match_the_requested_generation() -> None:
    other_generation = "corpus_generation_sha256_" + "0" * 64

    with pytest.raises(CorpusGenerationError, match="storage evidence is invalid"):
        await collect_generation_storage_evidence(
            FakeConnection(_storage_row(generation_id=other_generation)),
            generation_id=GENERATION_ID,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("retained", (False, True))
async def test_retention_status_preserves_explicit_legacy_and_retained_states(retained: bool) -> None:
    status = await inspect_release_retention(
        FakeConnection(_status_row(retained=retained)),
        release_id=RELEASE_ID,
    )

    assert status is not None
    assert status.retention_status == ("retained" if retained else "legacy_v5_unretained")
    assert (status.generation_id is not None) is retained


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "override",
    (
        {"retention_status": "retained"},
        {"generation_id": GENERATION_ID},
        {"release_id": RELEASE_ID.encode()},
        {"retained_at": datetime(2026, 7, 16, 12, 0)},
    ),
)
async def test_retention_status_rejects_inconsistent_or_coerced_rows(override: dict[str, Any]) -> None:
    connection = FakeConnection(_status_row(retained=False, **override))

    with pytest.raises(CorpusGenerationError, match="retention status is invalid"):
        await inspect_release_retention(connection, release_id=RELEASE_ID)
