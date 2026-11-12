# Deployment Guide

This guide describes the deployment paths the repository currently supports: local stdio, a container, and the Railway development/preview profile. No bank deployment path is maintained (see [Bank deployment](#bank-deployment)).

## Runtime

One process serves the 14 public tools. `bddk-mcp serve` opens the corpus file named by `BDDK_CORPUS_PATH` (default `corpus/bddk.sqlite`) read-only; stdio is the default transport and Streamable HTTP is the alternative.

| Mode | Selection | Corpus |
|---|---|---|
| stdio | `bddk-mcp serve` (default) | `BDDK_CORPUS_PATH`, opened read-only |
| Streamable HTTP | Add `--transport streamable-http` or `MCP_TRANSPORT=streamable-http` | `BDDK_CORPUS_PATH`, opened read-only |

The corpus is one SQLite file. Its tables are `documents`, `document_versions`, `document_sections` (with the FTS5 index `sections_fts`), `chunks` with float16 embedding BLOBs (with the FTS5 index `chunks_fts`), `decision_cache` (the BDDK catalog), `sync_failures`, and `meta` (`schema_version`, `embedding_model`, `embedding_dimension`). `bddk_mcp/store/corpus_db.py` creates the schema with `CREATE TABLE IF NOT EXISTS`; there is no migration ledger and no in-place upgrade. A schema change means a rebuilt file.

Serving startup performs no write. The file is opened with SQLite `mode=ro`, which is the airlock: a serving process cannot change the corpus even by mistake. Startup validates the file (present, recognised schema version, documents, chunks and catalog present, every chunk and section hash current against its document, embedding model and dimension in `meta` equal to the configured runtime), loads the catalog, and loads every chunk vector into memory for exact cosine search. A missing, incomplete or differently embedded corpus file fails startup with remediation guidance that names `bddk-mcp build`.

## Preparing the corpus

Three explicit commands write the corpus file; the serving process never runs them:

```bash
uv run --frozen bddk-mcp build          # seed_data/ (documents.json + decision_cache.json + embeddings/) -> corpus file with sections, chunks and embeddings
uv run --frozen bddk-mcp sync           # optional: download and extract the catalogued documents into the corpus file and re-embed them
uv run --frozen bddk-mcp export         # corpus file -> seed_data/ JSON + embeddings/ (overwrites the local seed artifacts)
```

All three default to `BDDK_CORPUS_PATH`; `--corpus PATH` overrides it for one command, and `build`/`export` take `--seed-dir PATH` for another seed directory. `sync` also accepts `--refresh-catalog` (re-scrape the BDDK catalog pages into the corpus first), `--force` (re-extract every document; a mevzuat PDF already stored is reused unless HTML-first is on, BDDK documents are downloaded again), `--doc-id ID` (one catalogued document) and `--concurrency N` (parallel downloads, default 5).

`bddk-mcp build` skips when every seed document is stored with the same content hash, indexed with current chunks, and the catalog count matches. It looks at nothing else, so a metadata-only seed edit (title, category, dates) or a document removed from the seed needs a fresh file: delete the local corpus file and run `bddk-mcp build`. A changed or new document is rewritten (the previous body is kept in `document_versions`) and re-embedded; `--force` rewrites every document and re-embeds it without the shipped vectors. After a change of embedding model, section parser or chunker, delete the local corpus file and run `bddk-mcp build`: `--force` cannot switch an existing file to another model and stops with a message that says to delete the file. The vector cache compares chunk texts and model only, so after a tokenizer or prefix change use `bddk-mcp build --force` on the fresh file. `export` writes `documents.json` and `decision_cache.json` in one canonical format (records sorted by id, two-space indent, UTF-8, LF, trailing newline) and keeps each document's `downloaded_at` and `extracted_at`, so `build` followed by `export` gives back the seed files byte for byte. Chunks are always regenerated from the canonical document text. Their vectors come from `seed_data/embeddings/` when `bddk-mcp export` has written them for exactly the same chunk texts and model: `export` stores one float16 `.npy` per document plus `manifest.json` (model name, dimension and a SHA-256 digest of each document's chunk texts), and `build` reuses a document's vectors only when the digest of its freshly computed chunks and the configured model match the manifest, embedding the document otherwise; `--force` ignores the cache. With the real `intfloat/multilingual-e5-base` model, embedding the whole seed from scratch (319 documents, 13,604 chunks when measured on 2026-11-10) took 816 seconds, about 14 minutes, on a laptop CPU; with the exported vectors the current seed (13,721 chunks) built in 21 seconds on 2026-11-12 without embedding a chunk, and `export` then gave back byte-identical seed files and vectors. The corpus file is about 100 MB; the exported vectors add 21 MB to the checkout.

A corpus file is complete only after `bddk-mcp build` or `bddk-mcp sync` has returned. Build into a new file and then replace or redeploy; never write a file that a serving process has open, and never serve or write a corpus over a network share (NFS/SMB), where SQLite locking is not reliable.

### Retrieval inside the corpus file

Dense retrieval is exact cosine similarity computed with numpy over all chunk vectors, which are held in memory and reloaded after any write to the file. Keyword retrieval is SQLite FTS5 BM25 over normalized text (lowercased with Turkish rules and stripped of diacritics in Python, both when indexing and when matching). The candidates of the two lists are merged with Reciprocal Rank Fusion; the final order uses a heuristic relevance score (cosine similarity adjusted by a keyword gate and lexical/phrase boosts, not a probability). For a query with an exact legal reference (an article number, a decision number or a date), keyword-only matches are kept without a dense score and listed after the hits found by both searches, in BM25 order, ahead of dense-only hits. Section search is FTS5 BM25 as well. Brute-force cosine is exact and takes milliseconds up to a few hundred thousand chunks; an ANN index (hnswlib or sqlite-vec) is the upgrade beyond that scale, not something the current corpus needs.

## Local stdio

```bash
uv sync --frozen
uv run --frozen bddk-mcp build          # about 15 minutes of CPU the first time; later runs skip when the seed is unchanged
uv run --frozen bddk-mcp serve
```

The first `build` downloads the embedding model unless `BDDK_EMBEDDING_MODEL_PATH` points at a pre-downloaded copy. `corpus/` is git-ignored; nothing in the checkout needs a service, an open port or a credential.

Use the repository [`.mcp.json`](../.mcp.json) with clients that support that project configuration format. Codex uses `config.toml`; see the examples in the main [README](../README.md#codex-configuration).

## Streamable HTTP Contract

Select HTTP with `MCP_TRANSPORT=streamable-http` or `--transport streamable-http`. The MCP endpoint is `/mcp`. FastMCP is configured for stateless JSON responses rather than durable server sessions. It offers no server-initiated stream, so `GET` and `DELETE` on `/mcp` return 405 with `Allow: POST` before rate and concurrency admission.

Loopback (`127.0.0.1`) is the default bind and is the appropriate development default. A non-loopback bind, including `0.0.0.0` in a container, fails startup unless all remote security settings are complete (or the explicit unauthenticated opt-in described below replaces the bearer-authentication settings):

| Variable | Remote requirement |
|---|---|
| `MCP_HOST`, `PORT` | Explicit bind address and port |
| `BDDK_HTTP_ALLOWED_HOSTS` | Comma-separated exact authorities; no wildcard, scheme, path, or credentials |
| `BDDK_HTTP_ALLOWED_ORIGINS` | Comma-separated exact HTTPS origins; no wildcard, credentials, path, query, or fragment |
| `BDDK_JWT_ISSUER` | Approved absolute HTTPS issuer URL |
| `BDDK_JWT_RESOURCE` | Approved absolute HTTPS protected-resource URL used for MCP metadata and authorization flow |
| `BDDK_JWT_JWKS_URL` | Approved absolute HTTPS JWKS URL |
| `BDDK_JWT_AUDIENCE` | Exact access-token audience mapped by the IdP to this MCP resource server; do not use the calling client's ID-token audience |
| `BDDK_JWT_REQUIRED_SCOPES` | Must include `bddk.read` |
| `BDDK_JWT_ALGORITHMS` | Approved asymmetric algorithms only; symmetric JWT algorithms are rejected |
| `BDDK_JWT_ACCESS_TOKEN_TYPES` | Allowed JOSE `typ` values; fail-closed default is RFC 9068 `at+jwt`; generic `JWT` requires explicit Keycloak compatibility opt-in |
| `BDDK_TLS_CERT_FILE`, `BDDK_TLS_KEY_FILE` | Optional paired PEM paths for HTTPS at the application socket; one without the other, unreadable material, encrypted keys, or a mismatched pair fails startup |

The issuer, resource, JWKS, audience, token type, and scope values are deployment-specific; do not copy values from repository tests. `BDDK_JWT_RESOURCE` identifies this protected resource to MCP clients; the composed application publishes RFC 9728 metadata at `/.well-known/oauth-protected-resource/mcp`, and its unauthenticated 401 challenge supplies the same URL through `resource_metadata`. Access-token binding is enforced through `BDDK_JWT_AUDIENCE`. Configure a dedicated API/resource-server audience distinct from interactive OAuth client audiences so a Keycloak-style ID token cannot be confused with an access token. Keep the fail-closed `at+jwt` default when the issuer supports RFC 9068. Set `BDDK_JWT_ACCESS_TOKEN_TYPES=at+jwt,JWT` only when generic `JWT` is required by the approved IdP profile.

Host, Origin, content type, duplicate headers, and request-body size are validated before bearer authentication. JWT verification then bounds token length and checks the asymmetric signature, key ID/algorithm, access-token type, issuer, exact resource-server audience, expiry, optional not-before time, and subject. A custom token `resource` claim is neither required nor trusted. Missing or invalid authentication returns 401; a cryptographically valid token without the required scope returns 403.

`BDDK_HTTP_ALLOW_UNAUTHENTICATED` is a supported explicit opt-in that serves a non-loopback bind without bearer authentication, for a frontend that reaches the MCP endpoint inside an isolated network segment and authenticates its own users. Network isolation then carries the access control. The opt-in is unset by default; while it is unset, every requirement in the table above applies unchanged. When set:

- it cannot be combined with any configured `BDDK_JWT_`-prefixed setting — startup refuses and names the offending variables;
- the server advertises no OAuth discovery: there is no `WWW-Authenticate` challenge, and both `/.well-known/oauth-protected-resource/mcp` and `/.well-known/oauth-authorization-server` return 404;
- exact `BDDK_HTTP_ALLOWED_HOSTS`/`BDDK_HTTP_ALLOWED_ORIGINS` values, the body-size and body-deadline limits, the concurrency cap, and the per-minute rate limit all still apply.

Without bearer authentication the per-process rate limiter is the primary abuse control, so its client key must be correct. Behind a reverse proxy, set `BDDK_HTTP_TRUSTED_PROXY_HOPS` (below) to the real, operator-controlled hop count; a wrong value makes the limiter either shared across all callers or spoofable.

The application has coarse per-process overload controls:

| Variable | Default | Behavior |
|---|---:|---|
| `BDDK_HTTP_MAX_BODY_BYTES` | 1,048,576 | Rejects oversized POST bodies with 413 |
| `BDDK_HTTP_BODY_FIRST_BYTE_TIMEOUT_SECONDS` | 5 | Requires the first non-empty request-body byte within `(0, 120]` seconds; otherwise returns 408 |
| `BDDK_HTTP_BODY_CHUNK_TIMEOUT_SECONDS` | 5 | Requires each later non-empty body chunk within `(0, 120]` seconds; otherwise returns 408 |
| `BDDK_HTTP_BODY_TOTAL_TIMEOUT_SECONDS` | 30 | Bounds the complete body read to `(0, 300]` seconds, including continuous drip traffic; otherwise returns 408 |
| `BDDK_HTTP_MAX_CONCURRENCY` | 32 | Immediately rejects excess in-process work with 503 and `Retry-After` |
| `BDDK_HTTP_RATE_LIMIT_PER_MINUTE` | 120 | Fixed-window limit per rate-limit client key, returning 429 and `Retry-After` |
| `BDDK_HTTP_TRUSTED_PROXY_HOPS` | 0 | Trusted reverse proxies in front of the bind; `0` keys the rate limiter on the ASGI socket peer and ignores `X-Forwarded-For` entirely |
| `BDDK_JWT_MAX_TOKEN_LENGTH` | 16,384 | Rejects oversized bearer tokens before verification |

The three body deadlines are independent; an empty ASGI request event does not count as the first byte, and the total deadline still applies when every chunk arrives within the chunk deadline. Align ingress/proxy limits explicitly so the intended layer returns the timeout and releases its capacity slot. The rate and concurrency state is bounded, process-local and non-durable. It is not shared across workers or replicas and does not provide a global ingress limit. The process sets no connection-level cap either (uvicorn's `limit_concurrency` would answer every request with a bare 503 once that many connections were merely open), so bounding idle or slow connections is the ingress proxy's job. Uvicorn proxy-header trust is disabled. At the default `BDDK_HTTP_TRUSTED_PROXY_HOPS=0` the rate limiter keys on the ASGI socket peer and does not accept client-supplied `Forwarded` or `X-Forwarded-For` values, which is correct for a directly exposed bind. Behind `n` operator-controlled reverse proxies, set the real hop count; the limiter then keys on the `n`-th entry from the right of the combined forwarded list, and anything unusable degrades to a shared `unknown` bucket rather than falling back to the socket peer, so a spoofed header cannot merge distinct callers. An IPv6 client is keyed by its /64 network, so one subscriber cannot rotate addresses for fresh buckets; IPv4 and IPv4-mapped addresses are keyed whole.

### Fixed health endpoints

Only these exact, content-free GET routes are provided for orchestration:

| Endpoint | Meaning |
|---|---|
| `GET /health/live` | The application process is alive; returns `{"status":"alive"}` |
| `GET /health/ready` | Dependencies exist, semantic search is ready, and the cached five-second corpus-file check (schema, counts, hashes, embedding model) passes; otherwise returns 503 |

These two routes intentionally bypass MCP bearer authentication and Host/Origin admission so orchestrator probes can use a pod-IP Host header. They return no corpus, configuration, or identity data. The readiness probe bounds each refresh to five seconds and caches only the outcome for five seconds. Liveness deliberately does not contact dependencies.

## Docker and Container Contract

The Dockerfile installs and runs the packaged `bddk-mcp` entry point. Its base and `uv` images are digest-pinned. The image downloads `intfloat/multilingual-e5-base` at full commit `d13f1b27baf31030b7fd040960d60d909913633f` and saves it at `/app/embedding_model` for offline runtime loading, then copies `seed_data/` (so a seed change does not invalidate the model layer) and runs `bddk-mcp build` with `BDDK_CORPUS_PATH=/app/corpus/bddk.sqlite`. The image therefore carries the built corpus file and is the versioned corpus artifact: no database, no pre-deploy step, nothing to prepare when the container starts. Because the image carries `seed_data/embeddings/`, that step takes well under a minute; only documents whose chunk texts or model changed since the last `bddk-mcp export` are embedded during the image build (on the order of 15 minutes of CPU for the whole seed).

Every corpus file records the embedding model and dimension it was built with, and configuration rejects any `BDDK_EMBEDDING_DIM` other than `768`. A remote custom embedding model requires an immutable full commit in `BDDK_EMBEDDING_MODEL_REVISION`. Changing an embedding model, tokenizer, prefixes, section parser or chunker requires a controlled rebuild: serving refuses a corpus whose recorded model or dimension differs from the configured runtime, and readiness fails when a chunk or section no longer matches its document's content hash. The corpus records no parser, chunk or prefix identity, so readiness cannot see such a change. After one, rebuild the image (it starts from an empty file) or delete the local corpus file and run `bddk-mcp build` (with `--force` after a tokenizer or prefix change) before serving.

The runtime wheel/sdist exclude tests, `.env` files, Compose, and `seed_data/`; the code package does not silently distribute the corpus. A source checkout or the provided container supplies `seed_data/`; another wheel installation must either point `bddk-mcp build --seed-dir` at an approved seed directory or receive an already built corpus file.

`docker-compose.yml` defines one service, `bddk`, built from the Dockerfile and published on host loopback only (`127.0.0.1:8000`). It runs with `BDDK_HTTP_ALLOW_UNAUTHENTICATED=true` because the loopback publication stands in for a private network boundary:

```bash
docker compose up --build -d bddk
curl -fsS http://localhost:8000/health/ready
```

The Compose and image definitions bind HTTP on `0.0.0.0`. Under the fail-closed HTTP policy, a container must receive the exact Host/Origin settings above and either the complete JWT settings or the explicit `BDDK_HTTP_ALLOW_UNAUTHENTICATED` opt-in, or startup will be refused. The Compose development settings (loopback publication, the unauthenticated opt-in) must never be reused or exposed outside a developer environment.

The repository produces no SBOM, image-scan, or provenance evidence; any promotion path must produce and verify that evidence itself.

### Outbound regulatory HTTP

Live catalog, document-sync, institution, announcement, and bulletin paths use code-owned outbound controls: exact HTTPS hosts for BDDK/mevzuat, no embedded credentials or non-default ports, public-address DNS preflight, bounded redirect hops with destination/DNS revalidation, streamed decoded-body limits, and privacy-safe retry logs that omit URLs, query strings, and exception messages. Catalog/API responses are capped at 8 MiB; document-sync limits vary by artifact class up to 128 MiB. These are application memory safeguards, not source authenticity, malware-scanning, or network-isolation guarantees.

DNS validation and the subsequent socket connection are not atomic. A restricted deployment must enforce the same destinations with its own egress firewall, proxy, and DNS policy. Public institution, announcement, bulletin, and digest tools can call live BDDK sources, so the serving runtime requires narrow TCP 443 egress to the approved regulatory-source destinations or an enterprise proxy. `bddk-mcp build` needs no network at all once the embedding model is local; `bddk-mcp sync` needs the same regulatory egress as the live tools.

## Bank deployment

The OpenShift starter manifests, the recovery-drill tooling, the operational-objectives contract and the supply-chain evidence lane (removed 2026-11-08, last present at commit `e858590`), the signed corpus-release workflow, multi-identity database role contract, operator job manager, admin console and telemetry (removed 2026-11-09, last present at commit `9ed84a0`, tag `archive/pre-simplify-2026-11-09`), and the PostgreSQL backend itself (replaced 2026-11-10, tag `archive/postgres-2026-11-10`) remain in Git history and must be restored and re-reviewed before any bank deployment that needs them. This repository is therefore not bank acceptance or a production-ready platform configuration. Railway (below) is a development/preview profile only.

The corpus file is distributed like a versioned package. Build it once on a build host (`bddk-mcp build`, or the container image), record its version, copy the file or the image to every serving node, and start `bddk-mcp serve` against the node's local copy. Each node serves its own read-only file; no node writes it. Never serve a corpus from, or write it live over, a network share (NFS/SMB). A corpus update is a new file produced by `bddk-mcp build` or `bddk-mcp sync` on the build host and redistributed in the same way; the sync writer never runs on a serving node.

The repository has not completed a bank-cluster acceptance, backup/restore drill, or a release compatibility matrix for Claude, Codex, GPT-based clients, LM Studio, GPT-OSS, or other local clients. Example configurations demonstrate configuration shape; they are not compatibility certification.

## Railway and Spaces

Railway and Hugging Face Spaces are development/preview profiles. They are not part of a reviewed bank deployment path and must not be used for bank data or bank-facing service.

`railway.toml` builds the standard Dockerfile and declares a `/health/ready` healthcheck. Because the corpus file is built into the image, the service needs no database, no pre-deploy command and no connection variables. Railway must inject the complete non-loopback HTTP policy (`BDDK_HTTP_ALLOWED_HOSTS`, `BDDK_HTTP_ALLOWED_ORIGINS`, and either the full `BDDK_JWT_*` set or the explicit unauthenticated opt-in); without them the server refuses startup. The unauthenticated opt-in on a public Railway URL serves the corpus to the whole Internet; prefer JWT or a private network.

There is no separate Spaces image. Preview hosts that need port `7860` should set `PORT` on the standard image. Neither Railway nor a preview host supplies deployment-specific identity or global ingress controls by itself.

## Secrets and Logs

- Keep tokens, JWT settings, and TLS keys in platform Secrets, never in Git, image layers, committed MCP configuration, or command-line examples with real values.
- The corpus file holds public regulatory text and no secret, but it is the reviewed corpus state: distribute it only through the approved artifact path and never let a serving node write it.
- Keep `BDDK_TOOL_LOG_CONTENT=false` in shared and production environments.
- Treat bank queries as confidential even when the regulatory corpus itself is public.
