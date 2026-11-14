"""MCP server exposing BDDK decision search, document retrieval, and data tools."""

import asyncio
import inspect
import logging
import os
import sqlite3
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import fields

import anyio
import httpx
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from bddk_mcp import __version__
from bddk_mcp.core.config import (
    HTTP_CONNECT_TIMEOUT,
    HTTP_POOL_TIMEOUT,
    REQUEST_TIMEOUT,
    SEMANTIC_STARTUP_TIMEOUT_SECONDS,
    validate_model_asset_policy,
)
from bddk_mcp.core.deps import Dependencies
from bddk_mcp.core.logging_config import configure_logging
from bddk_mcp.http_security import (
    HttpSecurityConfig,
    HttpSecurityConfigError,
    HttpSecurityMiddleware,
    JwtTokenVerifier,
    is_loopback_host,
    load_http_security_config,
)
from bddk_mcp.ingest.client import BddkApiClient
from bddk_mcp.mcp_server import BddkFastMCP
from bddk_mcp.store.corpus_db import CorpusDB
from bddk_mcp.store.doc_store import DocumentStore
from bddk_mcp.store.vector_store import VectorStore
from bddk_mcp.tools.registry import assert_tool_surface, register_tools
from bddk_mcp.transport_tls import ServerTlsConfig, load_server_tls_config

logger = logging.getLogger(__name__)

MCP_INSTRUCTIONS = """\
Search and retrieve BDDK (Turkish Banking Regulation) decisions, regulations, and statistical data.

GROUNDING RULES — follow these strictly:
1. ONLY use information returned by tool calls. Never supplement with your own knowledge about BDDK decisions.
2. If a search returns no results, say so explicitly. Do NOT guess or invent decisions.
3. In user-facing answers, identify a document by its actual title/name, not by document_id. Include decision_date and decision_number when available. Keep document_id only for tool calls and structured traceability, unless the user explicitly asks for it.
4. If document content is paginated, do NOT speculate about content on pages you have not retrieved.
5. Never fabricate karar numarası (decision numbers), tarih (dates), or legal conclusions.
6. When quoting from a document, quote only text that appears verbatim in the tool output.
7. If relevance scores are below 50%, flag this to the user and recommend refining the query. Hits with match_strength "keyword" (match_type fts_exact) are keyword matches on a legal reference in the query and carry no semantic score: their relevance 0.0 is not a low score, so do not apply the 50% rule to them; treat them as leads and verify the provision with get_document_section.
8. Distinguish clearly between: (a) information from BDDK tools, and (b) your general knowledge.

RESPONSE STYLE AND TOOL-USE DISCIPLINE:
9. Treat tool discovery, tool schemas, function schemas, Request/Response transcripts, and intermediate tool traces as hidden implementation details. Do not paste them unless the user explicitly asks for raw tool output or debug logs.
10. Do not narrate internal reasoning, private planning, or step-by-step tool orchestration. Avoid phrases like "the user wants", "let me load tools", "now I will", or standalone "done" status lines in the final answer.
11. Answer in the user's language. If the user writes in Turkish, answer in Turkish unless they request another language.
12. Cite each regulatory claim with the available document title/name and section/page reference, such as "TFRS 9 Uyarınca Beklenen Kredi Zararı Karşılığı Hesaplamasına İlişkin Rehber, İlke 5" or "Kredilerin Sınıflandırılması ve Bunlar İçin Ayrılacak Karşılıklara İlişkin Usul ve Esaslar Hakkında Yönetmelik, Madde 9". Prefer exact sections over whole-document summaries for audit or compliance questions.
13. If a tool result contains a quality warning or formula/image extraction warning, surface that caveat and recommend verifying critical figures, formulas, and images against the source PDF.
14. Search results and snippets are leads, not final authority. Retrieve exact sections or pages before making detailed legal or audit conclusions.
15. For multi-concept queries, search each substantive issue separately, then narrow by document and retrieve each exact provision. A loose fallback is lexical discovery, not a legal-confidence score; FTS ranks are not percentages.
16. Preserve the source's unit and scope: paragraf is not madde, and an IRB/IDD capital requirement is not automatically a TFRS 9/BKZ accounting requirement. Distinguish explicit duties from supporting context and your interpretation; never infer a test frequency, data window, threshold, or formula that the cited provision does not state.
17. A catalog URL, clean extraction, or source download alone does not establish legal currentness. Before a current-law conclusion, verify the instrument's publication and amendment status against the official source (mevzuat.gov.tr / Resmî Gazete) for the required date; if that evidence is unavailable, state the gap and abstain from that conclusion.
18. For a legal answer, check each proposed quotation with get_document_section(quotation=...). Inspect answer_assessment: an exact match proves only that the characters occur in the stored provision. Keep a claim-to-provision list covering every substantive claim; mark missing support explicitly instead of filling it with inference. A quotation match never validates a paraphrase, omitted exception, application scope, or unstated numeric requirement; scope_and_entailment remains not_assessed and must be reviewed against the full provision and relevant cross-references.
19. NEVER summarize, paraphrase, shorten, reword, correct, or "clean up" regulatory text returned by a document or section tool. Return the exact characters the tool returned, including punctuation, numbering, and abbreviations. Never present your own summary as the provision's text, and never label a summary as a quotation. Explicitly state when text is truncated, when a section is only an excerpt, or when a source is unavailable.
20. An exact-reference match proves the characters occur in the returned stored text. It does NOT prove the extraction equals the official source or that the rule is currently in force; verify against source_url and the official publication/amendment record, and report any fidelity or freshness gap instead of implying it is verified.
21. Catalog search results may include BDDK-catalog summary/excerpt fields. These are catalog metadata, not extracted document text: never quote them as the provision or document body, and retrieve the exact document/section when the wording matters.
22. Never present a document marked MÜLGA (repealed) or TASLAK (draft) in its "Document status" line as current law; the absence of that mark does not establish that a text is in force.
"""

# Tool functions close over this stable object.  Runtime resources are copied
# into it when the FastMCP lifespan starts, which lets import-based launchers
# discover the complete tool surface before the corpus file is opened.
_runtime_deps = Dependencies(db=None, doc_store=None, client=None, http=None)
_runtime_lock: asyncio.Lock | None = None
_runtime_loop: asyncio.AbstractEventLoop | None = None
_runtime_lock_users = 0
_runtime_leases = 0

_TRANSPORTS = frozenset({"stdio", "streamable-http"})
_READINESS_ATTESTATION_TTL_SECONDS = 5.0


def _runtime_host() -> str:
    """Return the configured bind host, defaulting to loopback."""
    return os.environ.get("MCP_HOST", "127.0.0.1").strip() or "127.0.0.1"


def _runtime_transport() -> str:
    """Return a validated MCP transport name."""
    transport = os.environ.get("MCP_TRANSPORT", "stdio").strip().lower()
    if transport not in _TRANSPORTS:
        allowed = ", ".join(sorted(_TRANSPORTS))
        raise RuntimeError(f"Invalid MCP_TRANSPORT {transport!r}; expected one of: {allowed}")
    return transport


def _uvicorn_options(
    http_security: HttpSecurityConfig,
    tls: ServerTlsConfig,
) -> dict[str, object]:
    """Build the reviewed Uvicorn options, including validated TLS paths."""
    return {
        "host": http_security.bind_host,
        "port": http_security.port,
        "proxy_headers": False,
        "server_header": False,
        **tls.uvicorn_options(),
    }


def _local_discovery_http_config() -> HttpSecurityConfig:
    """Return safe loopback settings for import-time MCP discovery objects."""
    host = _runtime_host()
    if not is_loopback_host(host):
        host = "127.0.0.1"
    port = os.environ.get("PORT", "8000")
    if not port.isdigit() or not 1 <= int(port) <= 65535:
        port = "8000"
    return load_http_security_config({"MCP_HOST": host, "PORT": port})


def _validate_http_policy(config: HttpSecurityConfig) -> None:
    """Require the read scope whenever JWT authentication is configured."""
    if config.jwt_issuer is not None and "bddk.read" not in config.jwt_required_scopes:
        raise HttpSecurityConfigError("The HTTP transport requires JWT scope 'bddk.read'")


async def create_deps() -> Dependencies:
    """Open the corpus read-only and build the serving dependencies."""
    http: httpx.AsyncClient | None = None
    db: CorpusDB | None = None
    doc_store: DocumentStore | None = None
    client: BddkApiClient | None = None
    vector_store: VectorStore | None = None
    try:
        http = httpx.AsyncClient(
            headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            },
            timeout=httpx.Timeout(
                REQUEST_TIMEOUT,
                connect=HTTP_CONNECT_TIMEOUT,
                pool=HTTP_POOL_TIMEOUT,
            ),
            follow_redirects=True,
        )

        db = await asyncio.to_thread(CorpusDB.open, None, readonly=True)
        corpus = await db.assert_ready(require_corpus=True)
        logger.info(
            "Corpus ready: bddk-mcp %s, corpus digest %s, %d documents, %d chunks, %d catalog entries, "
            "embedding model %s, schema version %s",
            __version__,
            corpus.digest,
            corpus.documents,
            corpus.chunks,
            corpus.catalog,
            corpus.embedding_model,
            corpus.schema_version,
        )
        if corpus.build_fingerprint is None:
            logger.warning("Corpus has no build fingerprint (file predates fingerprinting); run bddk-mcp build")
        elif corpus.stale_fingerprint_parts:
            # build refuses a model-part difference (revision, prefix, dtype: the embedding cache cannot detect it).
            remedy = (
                "delete the corpus file and run bddk-mcp build --force"
                if "model" in corpus.stale_fingerprint_parts
                else "run bddk-mcp build"
            )
            logger.warning(
                "Corpus build fingerprint is stale (%s); %s", ", ".join(corpus.stale_fingerprint_parts), remedy
            )

        doc_store = DocumentStore(db)
        client = BddkApiClient(store=doc_store, http=http, allow_live_population=False)
        await client.load_cache_read_only()
        validate_model_asset_policy()
        vector_store = VectorStore(db)
        startup_timeout = SEMANTIC_STARTUP_TIMEOUT_SECONDS
        try:
            async with asyncio.timeout(startup_timeout) as deadline:
                await vector_store.assert_semantic_search_ready()
        except TimeoutError as exc:
            if not deadline.expired():
                raise
            raise TimeoutError(
                f"The embedding model load and probe search did not finish within {startup_timeout} seconds; "
                "on a slow or cold host raise BDDK_SEMANTIC_STARTUP_TIMEOUT_SECONDS (1 to 900)."
            ) from exc

        return Dependencies(
            db=db,
            doc_store=doc_store,
            client=client,
            http=http,
            vector_store=vector_store,
        )
    except BaseException as startup_error:
        partial_deps = Dependencies(
            db=db,
            doc_store=doc_store,
            client=client,
            http=http,
            vector_store=vector_store,
        )
        try:
            await _await_cleanup_shielded(teardown_deps(partial_deps))
        except BaseException as cleanup_error:
            raise BaseExceptionGroup(
                "Dependency creation and rollback both failed",
                [startup_error, cleanup_error],
            ) from None
        raise


def register_health_routes(server: FastMCP, deps: Dependencies) -> None:
    """Expose content-free liveness/readiness probes for orchestrators."""

    attestation_lock = asyncio.Lock()
    last_attested_at = 0.0
    last_readiness = None

    async def corpus_attestation():
        """Periodically re-check the corpus file."""

        nonlocal last_attested_at, last_readiness
        now = time.monotonic()
        if last_attested_at and now - last_attested_at < _READINESS_ATTESTATION_TTL_SECONDS:
            return last_readiness

        async with attestation_lock:
            now = time.monotonic()
            if last_attested_at and now - last_attested_at < _READINESS_ATTESTATION_TTL_SECONDS:
                return last_readiness
            try:
                async with asyncio.timeout(5):
                    readiness = await deps.db.assert_ready(require_corpus=True)
            except (TimeoutError, RuntimeError, OSError, sqlite3.Error):
                last_readiness = None
            else:
                last_readiness = readiness
            last_attested_at = time.monotonic()
            return last_readiness

    @server.custom_route("/health/live", methods=["GET"], include_in_schema=False)
    async def liveness(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "alive"})

    @server.custom_route("/health/ready", methods=["GET"], include_in_schema=False)
    async def readiness(_request: Request) -> JSONResponse:
        if (
            deps.db is None
            or deps.doc_store is None
            or deps.client is None
            or deps.vector_store is None
            or deps.vector_store.semantic_search_ready is not True
        ):
            return JSONResponse({"status": "not_ready"}, status_code=503)
        if await corpus_attestation() is None:
            return JSONResponse({"status": "not_ready"}, status_code=503)
        return JSONResponse({"status": "ready"})


async def teardown_deps(deps: Dependencies) -> None:
    """Attempt to close every dependency and aggregate independent failures."""
    logger.info("Graceful shutdown initiated...")
    errors: list[BaseException] = []

    def record_failure(resource: str, error: BaseException) -> None:
        error.add_note(f"while closing BDDK dependency: {resource}")
        errors.append(error)

    if deps.client:
        try:
            await deps.client.close()
        except BaseException as error:
            record_failure("client", error)
    if deps.http:
        try:
            await deps.http.aclose()
        except BaseException as error:
            record_failure("http", error)
    if deps.db:
        try:
            deps.db.close()
            logger.info("Corpus closed")
        except BaseException as error:
            record_failure("db", error)

    if errors:
        logger.error("Graceful shutdown completed with %d error(s)", len(errors))
        raise BaseExceptionGroup("One or more BDDK dependencies failed to close", errors)

    logger.info("Graceful shutdown complete")


async def _await_cleanup_shielded(awaitable):
    """Finish cleanup before propagating cancellation to the caller.

    AnyIO cancellation is level-triggered while raw asyncio task cancellation
    is edge-triggered.  The cancel scope covers the former; the shielded task
    loop handles the latter without abandoning live resource handles.
    """
    cleanup_task = asyncio.create_task(awaitable)
    pending_cancellation: asyncio.CancelledError | None = None

    with anyio.CancelScope(shield=True):
        while True:
            try:
                result = await asyncio.shield(cleanup_task)
                break
            except asyncio.CancelledError as error:
                if cleanup_task.cancelled():
                    raise
                pending_cancellation = pending_cancellation or error
                if cleanup_task.done():
                    try:
                        result = cleanup_task.result()
                    except BaseException as cleanup_error:
                        raise BaseExceptionGroup(
                            "Cleanup failed while its caller was cancelled",
                            [pending_cancellation, cleanup_error],
                        ) from None
                    break
            except BaseException as cleanup_error:
                if pending_cancellation is not None:
                    raise BaseExceptionGroup(
                        "Cleanup failed while its caller was cancelled",
                        [pending_cancellation, cleanup_error],
                    ) from None
                raise

    if pending_cancellation is not None:
        raise pending_cancellation
    return result


def _copy_deps(source: Dependencies, target: Dependencies = _runtime_deps) -> Dependencies:
    """Copy dependency state while retaining the object captured by tools."""
    for field_info in fields(Dependencies):
        setattr(target, field_info.name, getattr(source, field_info.name))
    return target


def _empty_runtime_deps() -> None:
    """Remove references to closed resources after the final runtime lease."""
    _copy_deps(Dependencies(db=None, doc_store=None, client=None, http=None))


def _get_runtime_lock() -> asyncio.Lock:
    global _runtime_lock, _runtime_loop
    loop = asyncio.get_running_loop()
    if _runtime_lock is None:
        _runtime_lock = asyncio.Lock()
        _runtime_loop = loop
    elif _runtime_loop is not loop:
        raise RuntimeError("The BDDK dependency runtime cannot be shared across event loops")
    return _runtime_lock


@asynccontextmanager
async def _runtime_guard() -> AsyncIterator[None]:
    """Serialize runtime state and retire an idle loop-bound lock safely."""
    global _runtime_lock, _runtime_lock_users, _runtime_loop

    lock = _get_runtime_lock()
    _runtime_lock_users += 1
    try:
        async with lock:
            yield
    finally:
        _runtime_lock_users -= 1
        if _runtime_lock_users == 0 and _runtime_leases == 0:
            _runtime_lock = None
            _runtime_loop = None


async def startup() -> Dependencies:
    """Acquire the process-wide dependency runtime.

    FastMCP invokes its low-level lifespan once per session (and once per
    request in stateless HTTP mode).  Reference counting keeps the dependencies
    alive when the HTTP process holds an outer lease while still supporting the
    import-based stdio launcher used by ``mcp run server.py``.
    """
    global _runtime_leases

    async with _runtime_guard():
        if _runtime_leases == 0:
            deps = await create_deps()
            _copy_deps(deps)
        _runtime_leases += 1
        return _runtime_deps


async def _shutdown_unshielded() -> None:
    """Release one runtime lease; callers must protect this from cancellation."""
    global _runtime_leases

    async with _runtime_guard():
        if _runtime_leases == 0:
            logger.warning("Runtime shutdown requested without an active lease")
            return

        _runtime_leases -= 1
        if _runtime_leases == 0:
            try:
                await teardown_deps(_runtime_deps)
            finally:
                _empty_runtime_deps()


async def shutdown() -> None:
    """Release a runtime lease without abandoning cleanup on cancellation."""
    await _await_cleanup_shielded(_shutdown_unshielded())


@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[Dependencies]:
    """FastMCP lifecycle hook shared by SDK and direct launch paths."""
    deps = await startup()
    try:
        yield deps
    except BaseException as runtime_error:
        try:
            await shutdown()
        except BaseException as cleanup_error:
            raise BaseExceptionGroup(
                "MCP runtime and dependency cleanup both failed",
                [runtime_error, cleanup_error],
            ) from None
        raise
    else:
        await shutdown()


def _sdk_body_limit_kwargs(security: HttpSecurityConfig) -> dict[str, int]:
    """Hand the configured body limit to an SDK that enforces its own.

    The locked mcp release (1.30) wraps /mcp in a body-size check of its own
    (4 MiB by default), which would silently cap ``BDDK_HTTP_MAX_BODY_BYTES``.
    Releases 1.27 and 1.28 have no such check and no such argument; there the
    outer ``HttpSecurityMiddleware`` limit is the only one.
    """
    if "max_request_body_size" in inspect.signature(FastMCP.__init__).parameters:
        return {"max_request_body_size": security.max_body_bytes}
    return {}


def create_mcp(
    deps: Dependencies,
    *,
    lifespan=None,
    http_security: HttpSecurityConfig | None = None,
) -> FastMCP:
    """Construct a fully registered BDDK FastMCP server.

    Supplying dependencies separately makes the protocol surface testable with
    the official MCP client without opening a real corpus file.
    """
    security = http_security or _local_discovery_http_config()
    _validate_http_policy(security)

    auth = None
    token_verifier = None
    if security.jwt_issuer is not None:
        auth = AuthSettings(
            issuer_url=security.jwt_issuer,
            resource_server_url=security.jwt_resource,
            required_scopes=sorted(security.jwt_required_scopes),
            # JwtTokenVerifier enforces the configured ``aud`` itself, so the SDK resource check stays off.
            validate_token_resource=False,
        )
        token_verifier = JwtTokenVerifier(security)

    server = BddkFastMCP(
        "BDDK",
        instructions=MCP_INSTRUCTIONS,
        host=security.bind_host,
        port=security.port,
        stateless_http=True,
        json_response=True,
        lifespan=lifespan,
        transport_security=security.transport_security_settings(),
        auth=auth,
        token_verifier=token_verifier,
        **_sdk_body_limit_kwargs(security),
    )
    # FastMCP v1.x does not expose the low-level version in its constructor.
    # Set the SDK Server field so initialize returns this project's version,
    # rather than the installed `mcp` library version.
    server._mcp_server.version = __version__
    server._bddk_http_security = security
    register_tools(server, deps)
    assert_tool_surface(server)
    register_health_routes(server, deps)
    return server


# This object is intentionally complete at import time.  The MCP CLI imports
# ``server.py:mcp`` and calls ``mcp.run()``; registering tools only in a custom
# main() startup path therefore exposed an empty server to that supported path.
mcp = create_mcp(_runtime_deps, lifespan=server_lifespan)


def main() -> None:
    """Entry point — selects transport and runs the MCP server."""
    # The importable `mcp` object must not mutate a host application's root
    # logger. Packaged process entry points own logging configuration instead.
    configure_logging()

    _transport = _runtime_transport()
    logger.info("Transport: %s", _transport)
    logger.info("The corpus file is validated during startup")

    if _transport == "streamable-http":
        import uvicorn

        http_security = load_http_security_config()
        tls = load_server_tls_config()
        selected_mcp = create_mcp(
            _runtime_deps,
            lifespan=server_lifespan,
            http_security=http_security,
        )
        app = HttpSecurityMiddleware(selected_mcp.streamable_http_app(), http_security)

        async def _run_server():
            config = uvicorn.Config(app, **_uvicorn_options(http_security, tls))
            server = uvicorn.Server(config)

            async with server_lifespan(selected_mcp):
                await server.serve()

        asyncio.run(_run_server())
    else:
        # Default transport: stdio
        async def _run_stdio():
            # FastMCP enters server_lifespan on this event loop.
            await mcp.run_stdio_async()

        anyio.run(_run_stdio)


if __name__ == "__main__":
    main()
