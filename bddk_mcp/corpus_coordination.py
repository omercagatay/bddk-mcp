"""Shared PostgreSQL advisory-lock contract for mutable corpus workflows.

The constants in this module are part of the database coordination protocol.
They must remain stable across processes and releases: changing either value
would allow old and new workers to enter the protected critical section at the
same time.

Corpus writers and the release publisher use the transaction-scoped mutation
key.  The operator-job scheduler uses a *different*, session-scoped admission
key.  Keeping the keys distinct is intentional: an admitted job pins one pool
connection while its implementation performs transaction-scoped writes on
another, and reusing the mutation key would deadlock the job against itself.
"""

from __future__ import annotations

from typing import Any, Final

# First signed int64 of SHA-256("bddk_mcp:execution:v1:corpus_mutation").
CORPUS_MUTATION_ADVISORY_KEY: Final[int] = -6139789007653789941

# First signed int64 of SHA-256("bddk_mcp:execution:v1:corpus_job_execution").
CORPUS_JOB_EXECUTION_ADVISORY_KEY: Final[int] = -6417981786228610200


async def acquire_corpus_mutation_lock(connection: Any) -> None:
    """Acquire the shared transaction-scoped corpus mutation lock.

    The caller must already be inside the transaction that contains every
    protected write.  PostgreSQL releases this lock automatically at the end
    of that transaction, including rollback and cancellation paths.
    """

    await connection.fetchval(
        "SELECT pg_catalog.pg_advisory_xact_lock($1::pg_catalog.int8)",
        CORPUS_MUTATION_ADVISORY_KEY,
    )


__all__ = (
    "CORPUS_JOB_EXECUTION_ADVISORY_KEY",
    "CORPUS_MUTATION_ADVISORY_KEY",
    "acquire_corpus_mutation_lock",
)
