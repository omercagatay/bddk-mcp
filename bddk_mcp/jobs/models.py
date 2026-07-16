"""Privacy-safe records shared by operator job repositories and runners."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_METRIC_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

type MetricValue = int | float | bool
type MetricItems = tuple[tuple[str, MetricValue], ...]


class JobKind(StrEnum):
    """Reviewed kinds of long-running operator mutation."""

    CACHE_REFRESH = "cache_refresh"
    DOCUMENT_SYNC = "document_sync"
    CORPUS_RECONCILE = "corpus_reconcile"
    BACKFILL = "backfill"
    VECTOR_RECONCILE = "vector_reconcile"

    @property
    def execution_resource(self) -> str:
        """Return the exclusive resource used by the current mutable corpus.

        Cache refresh currently changes the selection read by document sync,
        while all other kinds write documents or indexes.  Until publication
        uses immutable generations, they intentionally share one lock.
        """

        return "corpus_mutation"


class JobState(StrEnum):
    """Persisted operator job lifecycle states."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"

    @property
    def terminal(self) -> bool:
        return self in {
            JobState.SUCCEEDED,
            JobState.COMPLETED_WITH_ERRORS,
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.INTERRUPTED,
        }


_ALLOWED_TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    JobState.QUEUED: frozenset(
        {
            JobState.RUNNING,
            JobState.CANCEL_REQUESTED,
            JobState.CANCELLED,
            JobState.INTERRUPTED,
        }
    ),
    JobState.RUNNING: frozenset(
        {
            JobState.SUCCEEDED,
            JobState.COMPLETED_WITH_ERRORS,
            JobState.FAILED,
            JobState.CANCEL_REQUESTED,
            JobState.CANCELLED,
            JobState.INTERRUPTED,
        }
    ),
    JobState.CANCEL_REQUESTED: frozenset({JobState.CANCELLED, JobState.INTERRUPTED}),
}


def can_transition(current: JobState, target: JobState) -> bool:
    """Return whether a manager may move between two lifecycle states."""

    return target in _ALLOWED_TRANSITIONS.get(current, frozenset())


@dataclass(frozen=True, slots=True)
class JobProgress:
    """Bounded numeric progress that cannot contain document or query text."""

    total: int = 0
    completed: int = 0
    succeeded: int = 0
    failed: int = 0

    def __post_init__(self) -> None:
        values = (self.total, self.completed, self.succeeded, self.failed)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise ValueError("job progress counters must be non-negative integers")
        if self.total and self.completed > self.total:
            raise ValueError("completed progress cannot exceed total")
        if self.succeeded + self.failed > self.completed:
            raise ValueError("success and failure counters cannot exceed completed")


def normalize_metrics(metrics: Mapping[str, MetricValue] | None) -> MetricItems:
    """Validate and freeze a small numeric result summary.

    Strings and nested values are rejected so upstream content and exception
    text cannot accidentally become durable job metadata.
    """

    if not metrics:
        return ()
    if len(metrics) > 32:
        raise ValueError("job result metrics are limited to 32 entries")

    normalized: list[tuple[str, MetricValue]] = []
    for name, value in sorted(metrics.items()):
        if not isinstance(name, str) or not _METRIC_NAME.fullmatch(name):
            raise ValueError("job metric names must be lowercase identifiers")
        if isinstance(value, bool):
            normalized.append((name, value))
        elif isinstance(value, int):
            normalized.append((name, value))
        elif isinstance(value, float) and math.isfinite(value):
            normalized.append((name, value))
        else:
            raise ValueError("job metric values must be finite numbers or booleans")
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class JobOutcome:
    """A runner's text-free completion result."""

    completed_with_errors: bool = False
    metrics: MetricItems = ()

    def __post_init__(self) -> None:
        if not isinstance(self.completed_with_errors, bool):
            raise ValueError("completed_with_errors must be a boolean")
        normalized = normalize_metrics(dict(self.metrics))
        if len(normalized) != len(self.metrics):
            raise ValueError("job metric names must be unique")
        object.__setattr__(self, "metrics", normalized)

    @classmethod
    def from_metrics(
        cls,
        metrics: Mapping[str, MetricValue] | None = None,
        *,
        completed_with_errors: bool = False,
    ) -> JobOutcome:
        return cls(
            completed_with_errors=completed_with_errors,
            metrics=normalize_metrics(metrics),
        )


@dataclass(frozen=True, slots=True)
class OperatorJob:
    """Immutable, persistence-friendly operator job record.

    Only fingerprints/digests are retained for caller-controlled inputs.  A
    repository must never add raw argument or exception fields to this record.
    """

    job_id: UUID
    kind: JobKind
    state: JobState
    args_fingerprint: str
    idempotency_digest: str | None
    created_at: datetime
    updated_at: datetime
    revision: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress: JobProgress = field(default_factory=JobProgress)
    result_metrics: MetricItems = ()
    error_code: str | None = None

    def __post_init__(self) -> None:
        if not _HEX_DIGEST.fullmatch(self.args_fingerprint):
            raise ValueError("args_fingerprint must be a SHA-256 hex digest")
        if self.idempotency_digest is not None and not _HEX_DIGEST.fullmatch(self.idempotency_digest):
            raise ValueError("idempotency_digest must be a SHA-256 hex digest")
        if self.revision < 0:
            raise ValueError("job revision cannot be negative")
        for value in (self.created_at, self.updated_at, self.started_at, self.finished_at):
            if value is not None and value.tzinfo is None:
                raise ValueError("job timestamps must be timezone-aware")
        if self.state.terminal and self.finished_at is None:
            raise ValueError("terminal jobs require finished_at")
        if not self.state.terminal and self.finished_at is not None:
            raise ValueError("non-terminal jobs cannot have finished_at")
        if self.error_code is not None and not _METRIC_NAME.fullmatch(self.error_code):
            raise ValueError("error_code must be a lowercase identifier")
        normalized_metrics = normalize_metrics(dict(self.result_metrics))
        if len(normalized_metrics) != len(self.result_metrics):
            raise ValueError("job result metric names must be unique")
        object.__setattr__(self, "result_metrics", normalized_metrics)

    @classmethod
    def create(
        cls,
        *,
        kind: JobKind,
        args_fingerprint: str,
        idempotency_digest: str | None,
        now: datetime | None = None,
    ) -> OperatorJob:
        timestamp = now or datetime.now(UTC)
        return cls(
            job_id=uuid4(),
            kind=kind,
            state=JobState.QUEUED,
            args_fingerprint=args_fingerprint,
            idempotency_digest=idempotency_digest,
            created_at=timestamp,
            updated_at=timestamp,
        )


@dataclass(frozen=True, slots=True)
class DrainReport:
    """Text-free result of closing manager admission and draining tasks."""

    observed: int
    completed: int
    cancelled: int
    still_running: int


def _canonical_value(value: Any) -> Any:
    """Convert arbitrary inputs to deterministic, hash-only material."""

    if value is None or isinstance(value, bool | int | str):
        return {"type": type(value).__name__, "value": value}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("job arguments cannot contain non-finite floats")
        return {"type": "float", "value": value}
    if isinstance(value, bytes | bytearray | memoryview):
        return {
            "type": "bytes",
            "sha256": hashlib.sha256(bytes(value)).hexdigest(),
            "length": len(value),
        }
    if isinstance(value, UUID):
        return {"type": "uuid", "value": str(value)}
    if isinstance(value, Path):
        return {"type": "path", "value": str(value)}
    if isinstance(value, Enum):
        return {"type": type(value).__qualname__, "value": _canonical_value(value.value)}
    if isinstance(value, Mapping):
        items = [(_canonical_value(key), _canonical_value(item_value)) for key, item_value in value.items()]
        items.sort(key=lambda item: json.dumps(item[0], sort_keys=True, separators=(",", ":")))
        return {"type": "mapping", "items": items}
    if isinstance(value, set | frozenset):
        items = [_canonical_value(item) for item in value]
        items.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
        return {"type": "set", "items": items}
    if isinstance(value, Sequence):
        return {"type": type(value).__name__, "items": [_canonical_value(item) for item in value]}

    # The representation is used only as input to a one-way digest and is
    # never retained or returned.  Type tagging avoids common string-coercion
    # collisions while allowing adapters to pass path/model-like values.
    rendered = repr(value).encode("utf-8", errors="replace")
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "sha256": hashlib.sha256(rendered).hexdigest(),
    }


def fingerprint_arguments(kind: JobKind, arguments: Mapping[str, Any] | None) -> str:
    """Return a deterministic SHA-256 fingerprint without retaining inputs."""

    material = {
        "kind": kind.value,
        "arguments": _canonical_value(arguments or {}),
    }
    encoded = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def digest_idempotency_key(key: str | None) -> str | None:
    """Validate and hash an optional caller key before persistence."""

    if key is None:
        return None
    if not isinstance(key, str) or not key.strip() or len(key) > 256:
        raise ValueError("idempotency_key must contain 1 to 256 characters")
    return hashlib.sha256(key.encode("utf-8")).hexdigest()
