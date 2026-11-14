"""Command-line entry points: serve the corpus, build it from seed data, export it, sync documents, describe it."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Sequence

from bddk_mcp import __version__


def _positive_port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        number = 0
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bddk-mcp",
        description="Serve, build, export, sync or describe the BDDK MCP regulatory corpus (one SQLite file).",
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

    corpus_help = "Corpus file path; defaults to BDDK_CORPUS_PATH (corpus/bddk.sqlite)"
    seed_help = "Seed directory with documents.json and decision_cache.json (default: seed_data/)"

    build = subparsers.add_parser(
        "build", help="Build the corpus file from seed_data/ (documents, sections, embeddings)"
    )
    build.add_argument("--corpus", help=corpus_help)
    build.add_argument("--seed-dir", help=seed_help)
    build.add_argument("--force", action="store_true", help="Rewrite documents and re-embed even when current")

    export = subparsers.add_parser("export", help="Export the corpus file to seed_data/ JSON")
    export.add_argument("--corpus", help=corpus_help)
    export.add_argument("--seed-dir", help=seed_help)

    sync = subparsers.add_parser("sync", help="Download and extract the documents listed in the BDDK catalog")
    sync.add_argument("--corpus", help=corpus_help)
    sync.add_argument(
        "--refresh-catalog",
        action="store_true",
        help="Re-scrape the BDDK catalog pages into the corpus before syncing",
    )
    sync.add_argument(
        "--force",
        action="store_true",
        help="Re-extract every document (a stored mevzuat PDF is reused unless HTML-first is on)",
    )
    sync.add_argument("--doc-id", help="Sync a single catalogued document by ID")
    sync.add_argument("--concurrency", type=_positive_int, default=5, help="Parallel downloads (default 5)")

    info = subparsers.add_parser("info", help="Describe the corpus file read-only (identity, counts, readiness)")
    info.add_argument("--corpus", help=corpus_help)

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


async def _build(args: argparse.Namespace) -> None:
    from bddk_mcp.ingest.seed import build_corpus

    result = await build_corpus(corpus_path=args.corpus, seed_dir=args.seed_dir, force=args.force)
    if result["skipped"]:
        print("Corpus already matches the seed data (use --force to rebuild).")
        return
    if not any(result[key] for key in ("decision_cache", "documents", "chunks")):
        from bddk_mcp.ingest.seed import SEED_DIR

        raise RuntimeError(f"no seed data in {args.seed_dir or SEED_DIR} (documents.json and decision_cache.json)")
    print(
        f"Built corpus: {result['decision_cache']} catalog entries, {result['documents']} documents, "
        f"{result['sections']} sections, {result['chunks']} chunks "
        f"({result['chunks'] - result['chunks_cached']} embedded, {result['chunks_cached']} from the seed cache)."
    )


async def _export(args: argparse.Namespace) -> None:
    from bddk_mcp.ingest.seed import export_corpus

    documents, catalog, embedded = await export_corpus(corpus_path=args.corpus, seed_dir=args.seed_dir)
    print(f"Exported {documents} documents, {catalog} catalog entries and embeddings for {embedded} documents.")


async def _sync(args: argparse.Namespace) -> None:
    """Download and extract the catalogued documents into the corpus file."""
    from bddk_mcp.ingest.client import BddkApiClient
    from bddk_mcp.ingest.doc_sync import DocumentSyncer
    from bddk_mcp.store.corpus_db import CorpusDB
    from bddk_mcp.store.doc_store import DocumentStore
    from bddk_mcp.store.vector_store import VectorStore

    db = CorpusDB.open(args.corpus)
    try:
        store = DocumentStore(db)
        client = BddkApiClient(store=store, allow_live_population=True)
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
            async with DocumentSyncer(store, vector_store=VectorStore(db)) as syncer:
                report = await syncer.sync_all(documents, concurrency=args.concurrency, force=args.force)
        finally:
            await client.close()
    finally:
        db.close()

    print(
        f"Sync: {report.total} total, {report.downloaded} downloaded, {report.reindexed} reindexed, "
        f"{report.skipped} skipped, {report.failed} failed, {report.elapsed_seconds}s"
    )
    for error in report.errors[:20]:
        print(f"  [{error.document_id}] {error.error}")
    if report.failed:
        sys.exit(1)


def _sync_failures_text(failures: dict[str, int]) -> str:
    """``none`` or ``N (category: count, ...)`` for the info line."""
    if not failures:
        return "none"
    by_category = ", ".join(f"{category}: {count}" for category, count in failures.items())
    return f"{sum(failures.values())} ({by_category})"


def _fingerprint_text(stored: dict[str, str] | None, stale: list[str]) -> str:
    """``current``, ``stale (parts) — ...`` or ``absent (...) — ...``; a warning, never a readiness issue."""
    if stored is None:
        return "absent (file predates fingerprinting) — run bddk-mcp build"
    if stale:
        # build refuses a model-part difference (revision, prefix, dtype: the embedding cache cannot detect it).
        remedy = "delete the corpus file and run bddk-mcp build --force" if "model" in stale else "run bddk-mcp build"
        return f"stale ({', '.join(stale)}) — {remedy}"
    return "current"


async def _info(args: argparse.Namespace) -> None:
    """Print the corpus identity and counts; exit 1 when it is not ready to serve."""
    import sqlite3
    from pathlib import Path

    from bddk_mcp.store.corpus_db import CorpusDB, unreadable_corpus_error

    db = CorpusDB.open(args.corpus, readonly=True)
    try:
        try:
            readiness = await asyncio.to_thread(db.readiness_sync, require_corpus=True)
            journal_mode = await asyncio.to_thread(db.journal_mode_sync)
        except sqlite3.DatabaseError as exc:
            raise unreadable_corpus_error(exc) from exc
        size = Path(db.path).stat().st_size
    finally:
        db.close()

    print(f"bddk-mcp version: {__version__}")
    print(f"corpus digest: {readiness.digest or '-'}")
    print(f"documents: {readiness.documents}")
    print(f"chunks: {readiness.chunks}")
    print(f"catalog entries: {readiness.catalog}")
    print(f"embedding model: {readiness.embedding_model or '-'}")
    print(f"schema version: {readiness.schema_version or '-'}")
    print(f"file size: {size} bytes")
    print(f"journal mode: {journal_mode}")
    print(f"sync failures: {_sync_failures_text(readiness.sync_failures)}")
    print(f"build fingerprint: {_fingerprint_text(readiness.build_fingerprint, readiness.stale_fingerprint_parts)}")
    if readiness.issues:
        print("status: not ready")
        for issue in readiness.issues:
            print(f"  issue: {issue}")
        sys.exit(1)
    print("status: ready")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in (None, "serve"):
        # No subcommand remains a compatibility alias for `bddk-mcp serve`.
        if args.command is None:
            args = argparse.Namespace(transport=None, host=None, port=None)
        _run_serve(args)
        return

    commands = {"build": _build, "export": _export, "sync": _sync, "info": _info}
    try:
        asyncio.run(commands[args.command](args))
    except RuntimeError as exc:
        print(f"bddk-mcp {args.command}: error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
