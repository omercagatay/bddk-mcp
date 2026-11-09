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

    sync = subparsers.add_parser("sync", help="Download and extract documents listed in the decision cache")
    sync.add_argument("--db", help="PostgreSQL DSN; defaults to BDDK_INGESTION_DATABASE_URL or BDDK_DATABASE_URL")
    sync.add_argument("--force", action="store_true", help="Re-download and re-extract every document")
    sync.add_argument("--doc-id", help="Sync a single document by ID")
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
    from bddk_mcp.ingest.doc_sync import _cli_sync

    await _cli_sync(args)


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
