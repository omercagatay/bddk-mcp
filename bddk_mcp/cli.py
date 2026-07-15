"""Command-line entry points for serving and explicitly preparing BDDK data."""

from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import Sequence
from pathlib import Path

from bddk_mcp import __version__


def _positive_port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


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
        help="Verify and activate an imported strict corpus using the dedicated release-publisher identity",
    )
    publish_release.add_argument(
        "--db",
        help="PostgreSQL DSN; defaults to BDDK_RELEASE_PUBLISHER_DATABASE_URL",
    )
    publish_release.add_argument(
        "--seed-dir",
        type=Path,
        help="Seed directory; defaults to BDDK_SEED_DIR or checkout seed_data",
    )
    publish_release.add_argument(
        "--trusted-signing-key",
        type=Path,
        required=True,
        help="Separately mounted PEM Ed25519 public key for the imported corpus manifest",
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
    trusted_signing_key: Path,
) -> dict:
    from bddk_mcp.core.config import require_database_url
    from bddk_mcp.ingest import seed

    selected_dsn = dsn or require_database_url("release-publisher")
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

    return await seed.publish_seed_release(
        dsn=selected_dsn,
        trusted_signing_key=trusted_signing_key,
    )


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
                    "Release publication required: run publish-corpus-release with the separate "
                    "release-publisher database identity after reviewing the imported state."
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
            result = asyncio.run(
                _publish_corpus_release(
                    args.db,
                    args.seed_dir,
                    trusted_signing_key=args.trusted_signing_key,
                )
            )
            active_release = result["active_corpus_release"]
            print(
                "Corpus release published: "
                f"id={active_release['release_id']} "
                f"manifest_sha256={active_release['manifest_sha256']} "
                f"profile_sha256={active_release['retrieval_profile_sha256']}"
            )
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
