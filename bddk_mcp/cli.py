"""Command-line entry points for serving and explicitly preparing BDDK data."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import secrets
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Final

from bddk_mcp import __version__

_CURRENT_WAL_INSERT_LSN_SQL = "SELECT pg_catalog.pg_current_wal_insert_lsn()::pg_catalog.text"
_OBSERVED_WAL_BYTES_SQL = "SELECT pg_catalog.pg_wal_lsn_diff($1::pg_catalog.pg_lsn, $2::pg_catalog.pg_lsn)"
_SCHEMA_MIGRATION_LOCK_SQL = "SELECT pg_catalog.pg_advisory_xact_lock($1::pg_catalog.int8)"
_CORPUS_RETENTION_LOCK_TIMEOUT: Final[str] = "30s"
_CORPUS_RETENTION_STATEMENT_TIMEOUT: Final[str] = "30min"
_CORPUS_RELEASE_LOCK_TIMEOUT: Final[str] = "30s"
_CORPUS_RELEASE_STATEMENT_TIMEOUT: Final[str] = "2min"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_REQUEST_ID_RE = re.compile(r"^corpus_release_request_sha256_[0-9a-f]{64}$")


def _positive_port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _sha256(value: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("value must be exactly 64 lowercase hexadecimal characters")
    return value


def _image_digest(value: str) -> str:
    if _IMAGE_DIGEST_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("image digest must be sha256: followed by 64 lowercase hexadecimal characters")
    return value


def _release_request_id(value: str) -> str:
    if _REQUEST_ID_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("request id must be a corpus_release_request_sha256_ identity")
    return value


def _verification_validity(value: str) -> int:
    seconds = int(value)
    if not 60 <= seconds <= 3600:
        raise argparse.ArgumentTypeError("verification validity must be between 60 and 3600 seconds")
    return seconds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bddk-mcp",
        description="Serve or explicitly prepare the BDDK MCP regulatory corpus.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="Start the MCP server (default command)")
    serve.add_argument(
        "--profile",
        choices=("public", "operator"),
        help="Expose exactly one reviewed tool profile; defaults to BDDK_TOOL_PROFILE or public",
    )
    serve.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        help="Override MCP_TRANSPORT for this process",
    )
    serve.add_argument("--host", help="Override MCP_HOST for Streamable HTTP")
    serve.add_argument("--port", type=_positive_port, help="Override PORT for Streamable HTTP")

    migrate = subparsers.add_parser("migrate", help="Create or upgrade the PostgreSQL schema explicitly")
    migrate.add_argument("--db", help="PostgreSQL DSN; defaults to BDDK_SCHEMA_OWNER_DATABASE_URL")
    migrate.add_argument(
        "--adopt-legacy",
        action="store_true",
        help="Explicitly verify and adopt the exact supported pre-ledger schema; never use for a clean database",
    )
    migrate.add_argument(
        "--allow-retrieval-publication-backfill",
        action="store_true",
        help=(
            "Permit the blocking v3 backfill only after stopping workloads, proving a backup, and rehearsing a "
            "size-matched restore; never use for a clean database"
        ),
    )

    bootstrap = subparsers.add_parser(
        "bootstrap",
        help="Import the reviewed seed into an already migrated schema and validate readiness",
    )
    bootstrap.add_argument("--db", help="PostgreSQL DSN; defaults to BDDK_INGESTION_DATABASE_URL")
    bootstrap.add_argument(
        "--seed-dir", type=Path, help="Seed directory; defaults to BDDK_SEED_DIR or checkout seed_data"
    )
    bootstrap.add_argument("--force", action="store_true", help="Replace matching seed-owned rows")
    bootstrap.add_argument(
        "--reindex-existing",
        action="store_true",
        help="Rebuild and publish every canonical document under the current retrieval profile",
    )
    bootstrap.add_argument(
        "--require-quantified-freshness",
        action="store_true",
        help="Fail unless the imported corpus declares numeric freshness objectives",
    )
    bootstrap.add_argument(
        "--require-measured-freshness",
        action="store_true",
        help="Fail unless per-document freshness events satisfy the declared objectives",
    )
    bootstrap.add_argument(
        "--require-verified-signature",
        action="store_true",
        help="Fail unless the imported corpus signature verifies against --trusted-signing-key",
    )
    bootstrap.add_argument(
        "--trusted-signing-key",
        type=Path,
        help="Separately mounted PEM Ed25519 public key for the imported corpus manifest",
    )

    publish_release = subparsers.add_parser(
        "publish-corpus-release",
        help="Deprecated fail-closed alias; use independent verification staging and activation commands",
    )
    publish_release.add_argument(
        "--db",
        help="PostgreSQL DSN; defaults to BDDK_RELEASE_PUBLISHER_DATABASE_URL",
    )
    publish_release.add_argument(
        "--seed-dir",
        type=Path,
        help="Deprecated compatibility input; it is never read",
    )
    publish_release.add_argument(
        "--trusted-signing-key",
        type=Path,
        help="Deprecated compatibility input; it is never read",
    )

    verify_and_stage = subparsers.add_parser(
        "verify-and-stage-corpus-release",
        help="Verify signed corpus membership and stage a short-lived request without activating it",
    )
    verify_and_stage.add_argument(
        "--db",
        help="PostgreSQL DSN; defaults to BDDK_RELEASE_VERIFIER_DATABASE_URL",
    )
    verify_and_stage.add_argument(
        "--seed-dir",
        type=Path,
        help="Seed directory; defaults to BDDK_SEED_DIR or checkout seed_data",
    )
    verify_and_stage.add_argument(
        "--trusted-signing-key",
        type=Path,
        required=True,
        help="Separately mounted PEM Ed25519 public key for the imported corpus manifest",
    )
    verify_and_stage.add_argument(
        "--verifier-revision-sha256",
        type=_sha256,
        help="Verifier source revision; defaults to BDDK_RELEASE_VERIFIER_REVISION_SHA256",
    )
    verify_and_stage.add_argument(
        "--verifier-image-digest",
        type=_image_digest,
        help="Immutable verifier image; defaults to BDDK_RELEASE_VERIFIER_IMAGE_DIGEST",
    )
    verify_and_stage.add_argument(
        "--verification-valid-for-seconds",
        type=_verification_validity,
        help="Short request lifetime; defaults to BDDK_RELEASE_VERIFICATION_VALIDITY_SECONDS (900)",
    )

    activate_release = subparsers.add_parser(
        "activate-corpus-release",
        help="Activate one unexpired staged request using only the publisher identity and request id",
    )
    activate_release.add_argument(
        "--db",
        help="PostgreSQL DSN; defaults to BDDK_RELEASE_PUBLISHER_DATABASE_URL",
    )
    activate_release.add_argument(
        "--request-id",
        type=_release_request_id,
        required=True,
        help="Exact corpus_release_request_sha256_... identity produced by independent staging",
    )
    retain_generation = subparsers.add_parser(
        "retain-corpus-generation",
        help="Atomically retain and seal the expected active release using the release-publisher identity",
    )
    retain_generation.add_argument(
        "--db",
        help="PostgreSQL DSN; defaults to BDDK_RELEASE_PUBLISHER_DATABASE_URL",
    )
    retain_generation.add_argument(
        "--expected-release-id",
        required=True,
        help="Exact corpus_release_sha256_... identity that must still be active",
    )

    verify_corpus = subparsers.add_parser(
        "verify-corpus",
        help="Verify the reviewed corpus manifest and every declared seed artifact without database access",
    )
    verify_corpus.add_argument(
        "--seed-dir",
        type=Path,
        help="Corpus directory; defaults to BDDK_SEED_DIR or checkout seed_data",
    )
    verify_corpus.add_argument(
        "--require-quantified-freshness",
        action="store_true",
        help="Fail unless source-detection, publication, and maximum-age objectives are numeric",
    )
    verify_corpus.add_argument(
        "--require-measured-freshness",
        action="store_true",
        help="Fail unless per-document source-detection and retrieval-publication events meet declared SLOs",
    )
    verify_corpus.add_argument(
        "--require-verified-signature",
        action="store_true",
        help="Fail unless a detached Ed25519 signature verifies against --trusted-signing-key",
    )
    verify_corpus.add_argument(
        "--trusted-signing-key",
        type=Path,
        help="Separately provisioned PEM Ed25519 public key used to verify a signed corpus manifest",
    )

    return parser


def _run_serve(args: argparse.Namespace) -> None:
    if getattr(args, "profile", None):
        os.environ["BDDK_TOOL_PROFILE"] = args.profile
    if args.transport:
        os.environ["MCP_TRANSPORT"] = args.transport
    if args.host:
        os.environ["MCP_HOST"] = args.host
    if args.port is not None:
        os.environ["PORT"] = str(args.port)

    # Import after applying CLI overrides because FastMCP captures host/port at
    # construction time. Help and version therefore remain dependency-free.
    from bddk_mcp.server import main as server_main

    server_main()


async def _migrate(
    dsn: str | None,
    *,
    adopt_legacy: bool = False,
    allow_retrieval_publication_backfill: bool = False,
) -> None:
    from bddk_mcp.db_lifecycle import migrate_database

    await migrate_database(
        dsn=dsn,
        adopt_legacy=adopt_legacy,
        allow_retrieval_publication_backfill=allow_retrieval_publication_backfill,
    )


async def _bootstrap(
    dsn: str | None,
    seed_dir: Path | None,
    force: bool,
    *,
    reindex_existing: bool = False,
    require_quantified_freshness: bool = False,
    require_measured_freshness: bool = False,
    require_verified_signature: bool = False,
    trusted_signing_key: Path | None = None,
) -> dict:
    from bddk_mcp.core.config import require_database_url
    from bddk_mcp.db_lifecycle import assert_database_ready
    from bddk_mcp.ingest import seed

    # Resolve the ingestion identity once and use that exact DSN for import
    # and post-import readiness.  Falling back independently would make a
    # bootstrap that used BDDK_INGESTION_DATABASE_URL validate against the
    # unrelated public-serving variable after it had already committed data.
    selected_dsn = dsn or require_database_url("ingestion")

    selected_seed_dir = seed_dir
    if selected_seed_dir is None and os.environ.get("BDDK_SEED_DIR"):
        selected_seed_dir = Path(os.environ["BDDK_SEED_DIR"])
    if selected_seed_dir is not None:
        seed.SEED_DIR = selected_seed_dir.resolve()

    if not seed.SEED_DIR.is_dir():
        raise RuntimeError(
            f"Seed directory is unavailable: {seed.SEED_DIR}. "
            "Mount the reviewed corpus and pass --seed-dir or BDDK_SEED_DIR."
        )

    result = await seed.import_seed(
        dsn=selected_dsn,
        force=force,
        reindex_existing=reindex_existing,
        require_quantified_freshness=require_quantified_freshness,
        require_measured_freshness=require_measured_freshness,
        require_verified_signature=require_verified_signature,
        trusted_signing_key=trusted_signing_key,
    )
    await assert_database_ready(dsn=selected_dsn)
    return result


async def _publish_corpus_release(
    dsn: str | None,
    seed_dir: Path | None,
    *,
    trusted_signing_key: Path | None,
) -> dict:
    del dsn, seed_dir, trusted_signing_key
    raise RuntimeError(
        "publish-corpus-release is disabled because one publisher credential must not both verify and activate. "
        "Run verify-and-stage-corpus-release with the release-verifier identity, then pass only its request id "
        "to activate-corpus-release with the separate release-publisher identity."
    )


def _required_sha256(value: str, *, variable: str, label: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise RuntimeError(f"{variable} must be set to a 64-character lowercase hexadecimal {label}.")
    return value


def _required_image_digest(value: str) -> str:
    if _IMAGE_DIGEST_RE.fullmatch(value) is None:
        raise RuntimeError(
            "BDDK_RELEASE_VERIFIER_IMAGE_DIGEST must be set to sha256: followed by 64 lowercase hexadecimal characters."
        )
    return value


def _detached_signature_sha256(seed_root: Path, signature_reference: str | None) -> str:
    if not signature_reference:
        raise RuntimeError("Verified corpus release staging requires a detached signature reference.")
    path = (seed_root / signature_reference).resolve()
    if not path.is_relative_to(seed_root):
        raise RuntimeError("Verified corpus release staging signature path is invalid.")
    try:
        with path.open("rb") as handle:
            payload = handle.read(1025)
    except OSError:
        raise RuntimeError("Verified corpus release staging signature could not be read.") from None
    if not 1 <= len(payload) <= 1024:
        raise RuntimeError("Verified corpus release staging signature is not a bounded file.")
    return hashlib.sha256(payload).hexdigest()


async def _verify_and_stage_corpus_release(
    dsn: str | None,
    seed_dir: Path | None,
    *,
    trusted_signing_key: Path,
    verifier_revision_sha256: str | None,
    verifier_image_digest: str | None,
    valid_for_seconds: int | None,
) -> dict:
    """Verify signed artifacts against locked DB state and stage, but never activate."""

    from functools import partial

    import asyncpg

    from bddk_mcp.core import config
    from bddk_mcp.corpus_publication import (
        CorpusPublicationError,
        stage_strict_corpus_release,
        strict_verification_evidence_sha256,
    )
    from bddk_mcp.db_identity import (
        DatabaseIdentityError,
        assert_database_connection_identity,
        assert_database_identity,
    )
    from bddk_mcp.db_transport import assert_database_transport
    from bddk_mcp.ingest import seed
    from bddk_mcp.store.vector_store import VectorStore

    selected_dsn = assert_database_transport(dsn) if dsn else config.require_database_url("release-verifier")
    revision = _required_sha256(
        verifier_revision_sha256 or config.RELEASE_VERIFIER_REVISION_SHA256,
        variable="BDDK_RELEASE_VERIFIER_REVISION_SHA256",
        label="verifier revision",
    )
    image_digest = _required_image_digest(verifier_image_digest or config.RELEASE_VERIFIER_IMAGE_DIGEST)
    validity = valid_for_seconds if valid_for_seconds is not None else config.RELEASE_VERIFICATION_VALIDITY_SECONDS
    if isinstance(validity, bool) or not isinstance(validity, int) or not 60 <= validity <= 3600:
        raise RuntimeError("Corpus release verification validity must be between 60 and 3600 seconds.")

    selected_root = seed_dir
    if selected_root is None and os.environ.get("BDDK_SEED_DIR"):
        selected_root = Path(os.environ["BDDK_SEED_DIR"])
    root = (selected_root or seed.SEED_DIR).resolve()
    if not root.is_dir():
        raise RuntimeError(
            f"Seed directory is unavailable: {root}. Mount the reviewed corpus and pass --seed-dir or BDDK_SEED_DIR."
        )

    validation, artifacts_by_role = seed._manifest_seed_artifacts(
        root,
        require_quantified_freshness=True,
        require_measured_freshness=True,
        require_verified_signature=True,
        trusted_signing_key=trusted_signing_key,
    )
    if validation is None:
        raise RuntimeError("Strict corpus release staging requires a verified manifest.")
    missing_roles = {"documents", "chunks", "decision_cache"} - set(artifacts_by_role)
    if missing_roles:
        raise RuntimeError(
            "Strict corpus release requires manifest-bound documents, chunks, and decision-cache artifacts."
        )
    documents = seed._load_manifest_bound_records(root, artifacts_by_role["documents"])
    reviewed_chunks = seed._load_manifest_bound_records(root, artifacts_by_role["chunks"])
    decision_cache = seed._load_manifest_bound_records(root, artifacts_by_role["decision_cache"])
    seed._validate_seed_documents(documents)
    seed._validate_strict_seed_artifact_shapes(documents, decision_cache)
    expected_sections = seed._expected_seed_sections(documents)
    signature_sha256 = _detached_signature_sha256(
        root,
        validation.manifest.integrity.signature_reference,
    )

    try:
        pool = await asyncpg.create_pool(
            selected_dsn,
            min_size=1,
            max_size=3,
            init=partial(assert_database_connection_identity, profile="release-verifier"),
        )
    except Exception:
        raise RuntimeError("Release-verifier database connection could not be established safely.") from None
    operation_committed = False
    try:
        await assert_database_identity(pool, "release-verifier")
        vector_store = VectorStore(pool)
        generated_chunks, _grouped = seed._generate_seed_chunks(vector_store, documents)
        comparison = {
            "chunk_artifact_match": None,
            "corpus_scope_warnings": list(validation.warnings),
        }
        seed._record_chunk_artifact_match(
            comparison,
            reviewed_chunks=reviewed_chunks,
            generated_chunks=generated_chunks,
            strict_release=True,
        )
        expected_embeddings = await seed._regenerate_seed_embedding_vectors(vector_store, generated_chunks)
        verification_evidence_sha256 = strict_verification_evidence_sha256(
            validation,
            signature_sha256=signature_sha256,
            retrieval_profile_sha256=vector_store.retrieval_profile_hash,
            verifier_revision_sha256=revision,
            verifier_image_digest=image_digest,
            verification_run_sha256=secrets.token_hex(32),
        )
        try:
            async with pool.acquire() as connection:
                async with connection.transaction():
                    await connection.execute(f"SET LOCAL lock_timeout = '{_CORPUS_RELEASE_LOCK_TIMEOUT}'")
                    await connection.execute(f"SET LOCAL statement_timeout = '{_CORPUS_RELEASE_STATEMENT_TIMEOUT}'")
                    request = await stage_strict_corpus_release(
                        connection,
                        validation,
                        signature_sha256=signature_sha256,
                        verification_evidence_sha256=verification_evidence_sha256,
                        retrieval_profile_sha256=vector_store.retrieval_profile_hash,
                        verifier_revision_sha256=revision,
                        verifier_image_digest=image_digest,
                        valid_for_seconds=validity,
                    )
                    # The stage routine holds the corpus mutation/table locks
                    # through this transaction. A mismatch rolls the request back.
                    await seed._assert_strict_seed_membership(
                        connection,
                        expected_documents=documents,
                        expected_cache=decision_cache,
                        expected_chunks=generated_chunks,
                        expected_embeddings=expected_embeddings,
                        expected_sections=expected_sections,
                        retrieval_profile_sha256=vector_store.retrieval_profile_hash,
                    )
                operation_committed = True
        except Exception:
            if not operation_committed:
                raise
    except (CorpusPublicationError, DatabaseIdentityError) as exc:
        raise RuntimeError(str(exc)) from None
    except RuntimeError:
        raise
    except Exception:
        raise RuntimeError("Verified corpus release evidence could not be staged.") from None
    finally:
        try:
            await pool.close()
        except Exception:
            if not operation_committed:
                raise RuntimeError("Verified corpus release evidence could not be staged.") from None

    return {
        "schema_version": 1,
        "corpus_release_request": request.safe_dict(),
        "verification_evidence_sha256": verification_evidence_sha256,
        "chunk_artifact_match": comparison["chunk_artifact_match"],
        "corpus_manifest_id": validation.manifest.manifest_id,
        "corpus_manifest_sha256": validation.manifest_sha256,
        "corpus_scope_warnings": comparison["corpus_scope_warnings"],
        "documents": len(documents),
        "chunks": len(generated_chunks),
    }


async def _activate_corpus_release(
    dsn: str | None,
    *,
    request_id: str,
) -> dict:
    """Activate one staged request without access to corpus or trust files."""

    from functools import partial

    import asyncpg

    from bddk_mcp.core.config import require_database_url
    from bddk_mcp.corpus_publication import (
        CorpusPublicationError,
        activate_staged_corpus_release,
        inspect_active_corpus_release,
    )
    from bddk_mcp.db_identity import (
        DatabaseIdentityError,
        assert_database_connection_identity,
        assert_database_identity,
    )
    from bddk_mcp.db_transport import assert_database_transport

    if _REQUEST_ID_RE.fullmatch(request_id) is None:
        raise RuntimeError("Corpus release request identity is invalid.")
    selected_dsn = assert_database_transport(dsn) if dsn else require_database_url("release-publisher")
    try:
        pool = await asyncpg.create_pool(
            selected_dsn,
            min_size=1,
            max_size=1,
            init=partial(assert_database_connection_identity, profile="release-publisher"),
        )
    except Exception:
        raise RuntimeError("Release-publisher database connection could not be established safely.") from None
    operation_committed = False
    try:
        await assert_database_identity(pool, "release-publisher")
        try:
            async with pool.acquire() as connection:
                async with connection.transaction():
                    await connection.execute(f"SET LOCAL lock_timeout = '{_CORPUS_RELEASE_LOCK_TIMEOUT}'")
                    await connection.execute(f"SET LOCAL statement_timeout = '{_CORPUS_RELEASE_STATEMENT_TIMEOUT}'")
                    receipt = await activate_staged_corpus_release(connection, request_id=request_id)
                    active = await inspect_active_corpus_release(connection)
                    if active is None or active.release_id != receipt.release.release_id:
                        raise RuntimeError("Activated corpus release could not be verified.")
                operation_committed = True
        except Exception:
            if not operation_committed:
                raise
    except (CorpusPublicationError, DatabaseIdentityError) as exc:
        raise RuntimeError(str(exc)) from None
    except RuntimeError:
        raise
    except Exception:
        raise RuntimeError("Staged corpus release could not be activated.") from None
    finally:
        try:
            await pool.close()
        except Exception:
            if not operation_committed:
                raise RuntimeError("Staged corpus release could not be activated.") from None
    return {"schema_version": 1, **receipt.safe_dict()}


async def _retain_corpus_generation(
    dsn: str | None,
    *,
    expected_release_id: str,
) -> dict:
    """Retain one active release through the dedicated administrative identity."""

    from functools import partial

    import asyncpg

    from bddk_mcp.core.config import require_database_url
    from bddk_mcp.corpus_coordination import SCHEMA_MIGRATION_ADVISORY_KEY
    from bddk_mcp.corpus_generations import (
        CorpusGenerationError,
        collect_generation_storage_evidence,
        retain_active_corpus_generation,
    )
    from bddk_mcp.db_identity import (
        DatabaseIdentityError,
        assert_database_connection_identity,
        assert_database_identity,
    )
    from bddk_mcp.db_lifecycle import DatabaseLifecycleError, assert_database_ready
    from bddk_mcp.db_transport import assert_database_transport

    selected_dsn = assert_database_transport(dsn) if dsn else require_database_url("release-publisher")
    operation_committed = False
    acquire_cleanup_failed = False
    try:
        pool = await asyncpg.create_pool(
            selected_dsn,
            min_size=1,
            max_size=1,
            init=partial(assert_database_connection_identity, profile="release-publisher"),
        )
        try:
            await assert_database_identity(pool, "release-publisher")
            try:
                async with pool.acquire() as connection:
                    async with connection.transaction():
                        await connection.execute(f"SET LOCAL lock_timeout = '{_CORPUS_RETENTION_LOCK_TIMEOUT}'")
                        await connection.execute(
                            f"SET LOCAL statement_timeout = '{_CORPUS_RETENTION_STATEMENT_TIMEOUT}'"
                        )
                        await connection.fetchval(
                            _SCHEMA_MIGRATION_LOCK_SQL,
                            SCHEMA_MIGRATION_ADVISORY_KEY,
                        )
                        # Retention binds the exact expected active release
                        # inside the SQL routine.  This preflight deliberately
                        # verifies only schema/catalog readiness so a locally
                        # upgraded retrieval profile cannot reject an older,
                        # already governed release before it can be retained.
                        await assert_database_ready(
                            pool=connection,  # type: ignore[arg-type]
                            require_corpus=False,
                        )
                        before_lsn = None
                        try:
                            # This optional observation runs in a savepoint.
                            # A permission or catalog error therefore rolls
                            # back only the measurement attempt instead of
                            # poisoning the durable retention transaction.
                            async with connection.transaction():
                                before_lsn = await connection.fetchval(_CURRENT_WAL_INSERT_LSN_SQL)
                        except Exception:
                            pass
                        receipt = await retain_active_corpus_generation(
                            connection,
                            expected_release_id=expected_release_id,
                        )
                        storage = await collect_generation_storage_evidence(
                            connection,
                            generation_id=receipt.generation_id,
                        )
                        if (
                            storage.generation_id != receipt.generation_id
                            or storage.relation_count != receipt.relation_count
                            or storage.row_count != receipt.row_count
                        ):
                            raise CorpusGenerationError("Retained corpus generation storage evidence is inconsistent.")
                    operation_committed = True

                    # Cost evidence is deliberately best-effort after the
                    # durable seal. A missing/invalid observation must not
                    # report the already committed operation as failed.
                    if before_lsn is not None:
                        try:
                            after_lsn = await connection.fetchval(_CURRENT_WAL_INSERT_LSN_SQL)
                            observed_wal_value = await connection.fetchval(
                                _OBSERVED_WAL_BYTES_SQL,
                                after_lsn,
                                before_lsn,
                            )
                            observed_wal_bytes = int(observed_wal_value)
                            if (
                                isinstance(observed_wal_value, bool)
                                or observed_wal_bytes < 0
                                or observed_wal_value != observed_wal_bytes
                            ):
                                raise ValueError("invalid WAL observation")
                            storage = replace(
                                storage,
                                observed_cluster_wal_bytes=observed_wal_bytes,
                                wal_attribution="observed_cluster_interval_not_exclusive",
                            )
                        except Exception:
                            pass
            except Exception:
                # Releasing an acquired pool connection happens after the
                # transaction has committed. Preserve that durable success if
                # only the pool context cleanup fails.
                if not operation_committed:
                    raise
                acquire_cleanup_failed = True
        finally:
            if acquire_cleanup_failed:
                # Pool.close() waits for every acquired connection. If the
                # acquire context itself could not release this connection,
                # terminate the one-shot CLI pool instead of hanging after a
                # successful commit.
                try:
                    pool.terminate()
                except Exception:
                    pass
            else:
                try:
                    await pool.close()
                except Exception:
                    if not operation_committed:
                        raise
    except CorpusGenerationError as exc:
        raise RuntimeError(str(exc)) from None
    except DatabaseIdentityError as exc:
        raise RuntimeError(str(exc)) from None
    except DatabaseLifecycleError as exc:
        raise RuntimeError(str(exc)) from None
    except Exception:
        raise RuntimeError("Retained corpus generation operation failed.") from None

    return {
        "schema_version": 1,
        "retained_generation": receipt.safe_dict(),
        "storage_evidence": storage.safe_dict(),
    }


def _verify_corpus(
    seed_dir: Path | None,
    *,
    require_quantified_freshness: bool = False,
    require_measured_freshness: bool = False,
    require_verified_signature: bool = False,
    trusted_signing_key: Path | None = None,
) -> dict:
    from bddk_mcp.corpus_manifest import (
        CORPUS_MANIFEST_FILENAME,
        CorpusManifestError,
        load_and_validate_corpus_manifest,
    )
    from bddk_mcp.ingest.seed import SEED_DIR

    selected_dir = seed_dir
    if selected_dir is None and os.environ.get("BDDK_SEED_DIR"):
        selected_dir = Path(os.environ["BDDK_SEED_DIR"])
    root = (selected_dir or SEED_DIR).resolve()
    try:
        validation = load_and_validate_corpus_manifest(
            root / CORPUS_MANIFEST_FILENAME,
            corpus_root=root,
            require_quantified_freshness=require_quantified_freshness,
            require_measured_freshness=require_measured_freshness,
            require_verified_signature=require_verified_signature,
            trusted_signing_key=trusted_signing_key,
        )
    except CorpusManifestError as exc:
        raise RuntimeError(f"Corpus verification failed: {exc}") from exc
    return {
        "manifest_id": validation.manifest.manifest_id,
        "manifest_sha256": validation.manifest_sha256,
        "artifact_count": len(validation.manifest.artifacts),
        "exhaustive": validation.manifest.exhaustive,
        "warnings": list(validation.warnings),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in (None, "serve"):
        # No subcommand remains a compatibility alias for `bddk-mcp serve`.
        if args.command is None:
            args = argparse.Namespace(profile=None, transport=None, host=None, port=None)
        _run_serve(args)
        return

    try:
        if args.command == "migrate":
            asyncio.run(
                _migrate(
                    args.db,
                    adopt_legacy=args.adopt_legacy,
                    allow_retrieval_publication_backfill=args.allow_retrieval_publication_backfill,
                )
            )
            print("Database schema is ready.")
            return
        if args.command == "bootstrap":
            result = asyncio.run(
                _bootstrap(
                    args.db,
                    args.seed_dir,
                    args.force,
                    reindex_existing=args.reindex_existing,
                    require_quantified_freshness=args.require_quantified_freshness,
                    require_measured_freshness=args.require_measured_freshness,
                    require_verified_signature=args.require_verified_signature,
                    trusted_signing_key=args.trusted_signing_key,
                )
            )
            print(
                "Bootstrap complete: "
                f"{result['decision_cache']} cache, {result['documents']} documents, "
                f"{result.get('sections', 0)} sections, {result['chunks']} chunks, "
                f"{result.get('embedded', 0)} embeddings, "
                f"{result.get('reindex_published', 0)} existing documents reindexed."
            )
            if result.get("corpus_manifest_id") and result.get("corpus_manifest_sha256"):
                print(
                    f"Corpus manifest used: id={result['corpus_manifest_id']} sha256={result['corpus_manifest_sha256']}"
                )
            for warning in result.get("corpus_scope_warnings", []):
                print(f"WARNING: {warning}")
            if result.get("release_publication_required"):
                print(
                    "Release publication required: run verify-and-stage-corpus-release with the independent "
                    "release-verifier identity, then activate-corpus-release with the separate publisher identity."
                )
            elif active_release := result.get("active_corpus_release"):
                print(
                    "Active corpus release: "
                    f"id={active_release['release_id']} "
                    f"manifest_sha256={active_release['manifest_sha256']} "
                    f"profile_sha256={active_release['retrieval_profile_sha256']}"
                )
            return
        if args.command == "publish-corpus-release":
            asyncio.run(
                _publish_corpus_release(
                    args.db,
                    args.seed_dir,
                    trusted_signing_key=args.trusted_signing_key,
                )
            )
            return
        if args.command == "verify-and-stage-corpus-release":
            result = asyncio.run(
                _verify_and_stage_corpus_release(
                    args.db,
                    args.seed_dir,
                    trusted_signing_key=args.trusted_signing_key,
                    verifier_revision_sha256=args.verifier_revision_sha256,
                    verifier_image_digest=args.verifier_image_digest,
                    valid_for_seconds=args.verification_valid_for_seconds,
                )
            )
            print(json.dumps(result, sort_keys=True, separators=(",", ":")))
            return
        if args.command == "activate-corpus-release":
            result = asyncio.run(
                _activate_corpus_release(
                    args.db,
                    request_id=args.request_id,
                )
            )
            active_release = result["active_corpus_release"]
            print(
                "Corpus release activated: "
                f"id={active_release['release_id']} "
                f"manifest_sha256={active_release['manifest_sha256']} "
                f"profile_sha256={active_release['retrieval_profile_sha256']}"
            )
            return
        if args.command == "retain-corpus-generation":
            result = asyncio.run(
                _retain_corpus_generation(
                    args.db,
                    expected_release_id=args.expected_release_id,
                )
            )
            print(json.dumps(result, sort_keys=True, separators=(",", ":")))
            return
        if args.command == "verify-corpus":
            result = _verify_corpus(
                args.seed_dir,
                require_quantified_freshness=args.require_quantified_freshness,
                require_measured_freshness=args.require_measured_freshness,
                require_verified_signature=args.require_verified_signature,
                trusted_signing_key=args.trusted_signing_key,
            )
            print(
                "Corpus manifest verified: "
                f"id={result['manifest_id']} sha256={result['manifest_sha256']} "
                f"artifacts={result['artifact_count']} exhaustive={str(result['exhaustive']).lower()}"
            )
            for warning in result["warnings"]:
                print(f"WARNING: {warning}")
            return
    except RuntimeError as exc:
        parser.error(str(exc))

    parser.error(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
