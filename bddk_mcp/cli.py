"""Command-line entry points for serving and explicitly preparing BDDK data."""

from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import Sequence

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
        "--transport",
        choices=("stdio", "streamable-http"),
        help="Override MCP_TRANSPORT for this process",
    )
    serve.add_argument("--host", help="Override MCP_HOST for Streamable HTTP")
    serve.add_argument("--port", type=_positive_port, help="Override PORT for Streamable HTTP")

    migrate = subparsers.add_parser("migrate", help="Create or upgrade the PostgreSQL schema explicitly")
    migrate.add_argument("--db", help="PostgreSQL DSN; defaults to BDDK_SCHEMA_OWNER_DATABASE_URL or BDDK_DATABASE_URL")
    migrate.add_argument(
        "--allow-retrieval-publication-backfill",
        action="store_true",
        help=(
            "Permit the blocking v3 backfill only after stopping workloads, proving a backup, and rehearsing a "
            "size-matched restore; never use for a clean database"
        ),
    )

    sync = subparsers.add_parser("sync", help="Download and extract the documents listed in the BDDK catalog")
    sync.add_argument("--db", help="PostgreSQL DSN; defaults to BDDK_INGESTION_DATABASE_URL or BDDK_DATABASE_URL")
    sync.add_argument(
        "--refresh-catalog",
        action="store_true",
        help="Re-scrape the BDDK catalog pages into decision_cache before syncing",
    )
    sync.add_argument("--force", action="store_true", help="Re-download and re-extract every document")
    sync.add_argument("--doc-id", help="Sync a single catalogued document by ID")
    sync.add_argument("--concurrency", type=int, default=5, help="Parallel downloads (default 5)")

    return parser


def _run_serve(args: argparse.Namespace) -> None:
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
    allow_retrieval_publication_backfill: bool = False,
) -> None:
    from bddk_mcp.db_lifecycle import migrate_database

    await migrate_database(
        dsn=dsn,
        allow_retrieval_publication_backfill=allow_retrieval_publication_backfill,
    )


async def _sync(args: argparse.Namespace) -> None:
    """Download and extract the catalogued documents into the local store."""
    import asyncpg

    from bddk_mcp.core.config import require_database_url
    from bddk_mcp.db_lifecycle import assert_database_ready
    from bddk_mcp.db_transport import assert_database_transport
    from bddk_mcp.ingest.client import BddkApiClient
    from bddk_mcp.ingest.doc_sync import DocumentSyncer
    from bddk_mcp.store.doc_store import DocumentStore
    from bddk_mcp.store.vector_store import VectorStore

    dsn = assert_database_transport(args.db) if args.db else require_database_url("ingestion")
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
    try:
        await assert_database_ready(pool=pool, require_corpus=False)
        store = DocumentStore(pool)
        client = BddkApiClient(pool=pool, doc_store=store, allow_live_population=True)
        try:
            if args.refresh_catalog:
                print(f"Catalog refreshed from BDDK: {await client.refresh_cache()} items")
            else:
                await client.ensure_cache()
            catalog = client.get_cache_items()
            if args.doc_id:
                catalog = [item for item in catalog if item.document_id == args.doc_id]
                if not catalog:
                    raise RuntimeError(f"{args.doc_id} is not in the catalog; rerun with --refresh-catalog")
            documents = [
                {
                    "document_id": item.document_id,
                    "title": item.title,
                    "category": item.category,
                    "source_url": item.source_url or "",
                    "decision_date": item.decision_date,
                    "decision_number": item.decision_number,
                }
                for item in catalog
            ]
            async with DocumentSyncer(store, vector_store=VectorStore(pool)) as syncer:
                report = await syncer.sync_all(documents, concurrency=args.concurrency, force=args.force)
        finally:
            await client.close()
    finally:
        await pool.close()

    print(
        f"Sync: {report.total} total, {report.downloaded} downloaded, {report.skipped} skipped, "
        f"{report.failed} failed, {report.elapsed_seconds}s"
    )
    for error in report.errors[:20]:
        print(f"  [{error.document_id}] {error.error}")


def main(argv: Sequence[str] | None = None) -> None:
    from bddk_mcp.db_transport import materialize_postgres_ca_from_env

    materialize_postgres_ca_from_env()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in (None, "serve"):
        # No subcommand remains a compatibility alias for `bddk-mcp serve`.
        if args.command is None:
            args = argparse.Namespace(transport=None, host=None, port=None)
        _run_serve(args)
        return

    try:
        if args.command == "migrate":
            asyncio.run(
                _migrate(
                    args.db,
                    allow_retrieval_publication_backfill=args.allow_retrieval_publication_backfill,
                )
            )
            print("Database schema is ready.")
            return
        if args.command == "sync":
            asyncio.run(_sync(args))
            return
    except RuntimeError as exc:
        parser.error(str(exc))

    parser.error(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
