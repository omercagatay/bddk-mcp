"""Admit one corrected upload through the separate release gates.

The admin console only records the request; this module is the spine that a
separate operator process uses to process exactly one waiting request per
call. The publisher callable owns the actual verify-and-stage and activate
sequence and is never part of the admin service.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

from bddk_mcp.admin.uploads import UploadStore

_MISSING_RELEASE_CREDENTIALS = (
    "BDDK_RELEASE_VERIFIER_DATABASE_URL and BDDK_RELEASE_PUBLISHER_DATABASE_URL must both be set "
    "for the admission job; run it outside the admin service with the separate "
    "release-verifier and release-publisher identities."
)


def admit_next(store: UploadStore, publisher: Callable[[str], Any]) -> str:
    """Process the oldest waiting admission request; never touch the next one."""

    waiting = store.oldest_waiting_request()
    if waiting is None:
        return "idle"
    request_id, upload_id = waiting
    try:
        text = store.corrected_text(upload_id)
        publisher(text)
    except NotImplementedError:
        # An unwired publisher refuses admission before any gate ran; a refusal
        # is never a request error, so no state is recorded at all.
        raise
    except Exception:
        store.mark(request_id, "error")
        return "error"
    store.mark(request_id, "published")
    return "published"


def publisher_from_env(env: Mapping[str, str] | None = None) -> Callable[[str], Any]:
    """Require the separate verifier and publisher credentials before admitting."""

    source = os.environ if env is None else env
    verifier = source.get("BDDK_RELEASE_VERIFIER_DATABASE_URL", "").strip()
    publisher = source.get("BDDK_RELEASE_PUBLISHER_DATABASE_URL", "").strip()
    if not verifier or not publisher:
        raise RuntimeError(_MISSING_RELEASE_CREDENTIALS)
    # The verify-and-stage/activate wiring for uploaded documents is a separate
    # follow-up task. Refuse here, before admit_next runs, so an unwired command
    # can never mark an admission request as failed.
    raise NotImplementedError("admission publication through the release gates is not wired yet")
