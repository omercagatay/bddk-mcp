"""Migration 0002: privacy-safe durable operator job ledger."""

from bddk_mcp.migrations.model import Migration

V0002_OPERATOR_JOBS = Migration(
    version=2,
    name="durable_operator_jobs",
    statements=(
        "CREATE SCHEMA bddk_operator",
        """
        CREATE TABLE bddk_operator.operator_jobs (
            job_id pg_catalog.uuid NOT NULL,
            kind pg_catalog.text NOT NULL,
            state pg_catalog.text NOT NULL,
            args_fingerprint pg_catalog.text NOT NULL,
            idempotency_digest pg_catalog.text,
            created_at pg_catalog.timestamptz NOT NULL,
            updated_at pg_catalog.timestamptz NOT NULL,
            revision pg_catalog.int8 NOT NULL DEFAULT 0,
            started_at pg_catalog.timestamptz,
            finished_at pg_catalog.timestamptz,
            progress_total pg_catalog.int8 NOT NULL DEFAULT 0,
            progress_completed pg_catalog.int8 NOT NULL DEFAULT 0,
            progress_succeeded pg_catalog.int8 NOT NULL DEFAULT 0,
            progress_failed pg_catalog.int8 NOT NULL DEFAULT 0,
            result_metrics pg_catalog.jsonb NOT NULL DEFAULT '{}'::pg_catalog.jsonb,
            error_code pg_catalog.text,
            CONSTRAINT operator_jobs_pkey PRIMARY KEY (job_id),
            CONSTRAINT operator_jobs_kind_check CHECK (
                kind IN ('cache_refresh', 'document_sync', 'corpus_reconcile', 'backfill', 'vector_reconcile')
            ),
            CONSTRAINT operator_jobs_state_check CHECK (
                state IN (
                    'queued', 'running', 'succeeded', 'completed_with_errors', 'failed',
                    'cancel_requested', 'cancelled', 'interrupted'
                )
            ),
            CONSTRAINT operator_jobs_args_fingerprint_check CHECK (args_fingerprint ~ '^[0-9a-f]{64}$'),
            CONSTRAINT operator_jobs_idempotency_digest_check CHECK (
                idempotency_digest IS NULL OR idempotency_digest ~ '^[0-9a-f]{64}$'
            ),
            CONSTRAINT operator_jobs_revision_check CHECK (revision >= 0),
            CONSTRAINT operator_jobs_progress_total_check CHECK (progress_total >= 0),
            CONSTRAINT operator_jobs_progress_completed_check CHECK (progress_completed >= 0),
            CONSTRAINT operator_jobs_progress_succeeded_check CHECK (progress_succeeded >= 0),
            CONSTRAINT operator_jobs_progress_failed_check CHECK (progress_failed >= 0),
            CONSTRAINT operator_jobs_result_metrics_check CHECK (
                pg_catalog.jsonb_typeof(result_metrics) = 'object'
            ),
            CONSTRAINT operator_jobs_error_code_check CHECK (
                error_code IS NULL OR error_code ~ '^[a-z][a-z0-9_]{0,63}$'
            ),
            CONSTRAINT operator_jobs_progress_within_total CHECK (
                (progress_total = 0 OR progress_completed <= progress_total)
                AND progress_succeeded + progress_failed <= progress_completed
            ),
            CONSTRAINT operator_jobs_terminal_timestamp CHECK (
                (
                    state IN ('succeeded', 'completed_with_errors', 'failed', 'cancelled', 'interrupted')
                    AND finished_at IS NOT NULL
                )
                OR (
                    state NOT IN ('succeeded', 'completed_with_errors', 'failed', 'cancelled', 'interrupted')
                    AND finished_at IS NULL
                )
            )
        )
        """,
        """
        CREATE UNIQUE INDEX operator_jobs_idempotency_digest_uq
        ON bddk_operator.operator_jobs (idempotency_digest)
        WHERE idempotency_digest IS NOT NULL
        """,
        """
        CREATE INDEX operator_jobs_list_idx
        ON bddk_operator.operator_jobs (created_at DESC, job_id DESC)
        """,
        """
        CREATE INDEX operator_jobs_unfinished_idx
        ON bddk_operator.operator_jobs (created_at, job_id)
        WHERE state IN ('queued', 'running', 'cancel_requested')
        """,
        """
        CREATE INDEX operator_jobs_terminal_idx
        ON bddk_operator.operator_jobs (finished_at DESC, job_id DESC)
        WHERE state IN ('succeeded', 'completed_with_errors', 'failed', 'cancelled', 'interrupted')
        """,
    ),
)
