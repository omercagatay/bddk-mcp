"""Privacy-safe benchmark trace persistence helpers.

Benchmark traces are audit artifacts, but they must never become a convenient
place to persist credentials.  These helpers redact credential-shaped keys and
values before data is written to disk or sent to an optional external grader.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

REDACTED = "[REDACTED]"

_SENSITIVE_KEY = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|authorization|bearer|cookie|database[_-]?url|dsn|password|passwd|secret|token)(?:$|[_-])",
    re.IGNORECASE,
)
_BEARER_VALUE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")
_AUTH_HEADER = re.compile(r"(?i)\bauthorization\s*:\s*(?:basic|bearer)\s+[^\s,;]+")
_PROVIDER_KEY = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{12,}|sk_[A-Za-z0-9_-]{12,}|xai-[A-Za-z0-9_-]{12,}|"
    r"AIza[A-Za-z0-9_-]{20,}|(?:AKIA|ASIA)[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9_]{20,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9_-]{20,}|xox[baprs]-[A-Za-z0-9-]{12,})\b"
)
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|authorization|bearer[_-]?token|password|passwd|secret|token)"
    r"(\s*[:=]\s*)([^\s,;\]\[{}]+)"
)
_CREDENTIAL_URL = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)([^/@\s:]+):([^/@\s]+)@")
_PRIVATE_KEY = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----.*?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    re.DOTALL,
)


def redact_text(value: str) -> str:
    """Redact common credential forms without hiding ordinary source URLs."""

    redacted = _PRIVATE_KEY.sub(REDACTED, value)
    redacted = _AUTH_HEADER.sub(f"Authorization: {REDACTED}", redacted)
    redacted = _BEARER_VALUE.sub(f"Bearer {REDACTED}", redacted)
    redacted = _PROVIDER_KEY.sub(REDACTED, redacted)
    redacted = _JWT.sub(REDACTED, redacted)
    redacted = _ASSIGNMENT.sub(lambda match: f"{match.group(1)}{match.group(2)}{REDACTED}", redacted)
    return _CREDENTIAL_URL.sub(lambda match: f"{match.group(1)}{REDACTED}@", redacted)


def sanitize_for_audit(value: Any) -> Any:
    """Return a JSON-compatible copy with credential-shaped data redacted."""

    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            sanitized[key] = REDACTED if _SENSITIVE_KEY.search(key) else sanitize_for_audit(item)
        return sanitized
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest(), "bytes_length": len(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [sanitize_for_audit(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact_text(str(value))


def canonical_sha256(value: Any) -> str:
    """Hash a sanitized value using deterministic JSON serialization."""

    sanitized = sanitize_for_audit(value)
    payload = json.dumps(
        sanitized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
