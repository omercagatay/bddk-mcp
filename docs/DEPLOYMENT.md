# Deployment Guide

This guide describes the deployment paths the repository currently supports: local stdio, a container, and the Railway development/preview profile. No bank deployment path is maintained (see [Bank deployment](#bank-deployment)).

## Runtime

One process serves the 17 public tools. `bddk-mcp serve` reads `BDDK_DATABASE_URL`; stdio is the default transport and Streamable HTTP is the alternative.

| Mode | Selection | Database identity |
|---|---|---|
| stdio | `bddk-mcp serve` (default) | `BDDK_DATABASE_URL` |
| Streamable HTTP | Add `--transport streamable-http` or `MCP_TRANSPORT=streamable-http` | `BDDK_DATABASE_URL` |

Serving startup does not perform schema, seed, synchronization, or embedding-backfill writes. It validates the database (migration ledger at the current version, corpus present and published for the current retrieval profile), loads the decision cache with `SELECT`, and constructs search clients. An incomplete database fails startup with remediation guidance.

## Preparing the database

Three explicit commands prepare a database; the serving process never runs them:

```bash
uv run --frozen bddk-mcp migrate        # create/upgrade the schema (append-only migration ledger)
uv run --frozen bddk-seed import        # load seed_data/ documents + decision cache, regenerate chunks and embeddings
uv run --frozen bddk-mcp sync           # optional: download and extract the catalogued documents (--refresh-catalog re-scrapes BDDK first)
```

All three default to `BDDK_DATABASE_URL`. A deployment that keeps separate PostgreSQL logins can set `BDDK_SCHEMA_OWNER_DATABASE_URL` (used by `migrate`) and `BDDK_INGESTION_DATABASE_URL` (used by `import` and `sync`); each falls back to `BDDK_DATABASE_URL` when unset. `--db DSN` overrides the environment for one command.

`bddk-seed import` is idempotent: it skips when the stored documents, decision cache, section index, chunks, and retrieval publication already match the seed. Content that differs from the database requires `--force` after review. Chunks and embeddings are always regenerated from the canonical document text under the current retrieval profile; no chunk or embedding artifact is shipped.

### Database target and TLS contract

Remote DSNs must use `sslmode=verify-full` with an absolute `sslrootcert` path; `BDDK_POSTGRES_CA_CERT` can materialize the CA file from an environment value at startup. Only isolated loopback development may set `BDDK_ALLOW_INSECURE_DATABASE=true`. The application admits PostgreSQL 17 only; every connection boundary checks the server version before any other statement.

## Local stdio

The disposable loopback Compose topology runs PostgreSQL with pgvector, migrates, and imports the seed:

```bash
uv sync --frozen
docker compose up --build -d bddk-init
docker compose wait bddk-init
BDDK_ALLOW_INSECURE_DATABASE=true \
BDDK_DATABASE_URL=postgresql://bddk:bddk@localhost:5432/bddk \
  uv run --frozen bddk-mcp serve
```

The fixed local credentials are public test fixtures and the PostgreSQL port is bound to host loopback. Do not copy them into a remote environment. The same container also creates the `bddk_test` database used by the `postgres`-marked tests (`BDDK_TEST_DATABASE_URL=postgresql://bddk:bddk@localhost:5432/bddk_test`).

Use the repository [`.mcp.json`](../.mcp.json) with clients that support that project configuration format. Codex uses `config.toml`; see the examples in the main [README](../README.md#codex-configuration).

## Streamable HTTP Contract

Select HTTP with `MCP_TRANSPORT=streamable-http` or `--transport streamable-http`. The MCP endpoint is `/mcp`. FastMCP is configured for stateless JSON responses rather than durable server sessions.

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

The three body deadlines are independent; an empty ASGI request event does not count as the first byte, and the total deadline still applies when every chunk arrives within the chunk deadline. Align ingress/proxy limits explicitly so the intended layer returns the timeout and releases its capacity slot. The rate and concurrency state is bounded, process-local and non-durable. It is not shared across workers or replicas and does not provide a global ingress limit. Uvicorn proxy-header trust is disabled. At the default `BDDK_HTTP_TRUSTED_PROXY_HOPS=0` the rate limiter keys on the ASGI socket peer and does not accept client-supplied `Forwarded` or `X-Forwarded-For` values, which is correct for a directly exposed bind. Behind `n` operator-controlled reverse proxies, set the real hop count; the limiter then keys on the `n`-th entry from the right of the combined forwarded list, and anything unusable degrades to a shared `unknown` bucket rather than falling back to the socket peer, so a spoofed header cannot merge distinct callers.

### Fixed health endpoints

Only these exact, content-free GET routes are provided for orchestration:

| Endpoint | Meaning |
|---|---|
| `GET /health/live` | The application process is alive; returns `{"status":"alive"}` |
| `GET /health/ready` | Dependencies exist, semantic search is ready, and the cached five-second schema/corpus check passes; otherwise returns 503 |

These two routes intentionally bypass MCP bearer authentication and Host/Origin admission so orchestrator probes can use a pod-IP Host header. They return no corpus, configuration, or identity data. The readiness probe bounds each refresh to five seconds and caches only the outcome for five seconds. Liveness deliberately does not contact dependencies.

## Docker and Container Contract

The Dockerfile installs and runs the packaged `bddk-mcp` entry point. Its base and `uv` images are digest-pinned. The image copies `seed_data/` and downloads `intfloat/multilingual-e5-base` at full commit `d13f1b27baf31030b7fd040960d60d909913633f`, then saves it at `/app/embedding_model` for offline runtime loading.

The immutable schema stores `public.vector(768)`, and configuration rejects any `BDDK_EMBEDDING_DIM` other than `768`. A remote custom embedding model requires an immutable full commit in `BDDK_EMBEDDING_MODEL_REVISION`; the optional reranker `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` defaults to full commit `1427fd652930e4ba29e8149678df786c240d8825`. Changing an embedding model, tokenizer, prefixes, or chunk settings requires a controlled full re-embedding: the v3 migration keeps a retrieval-publication record keyed by document, content hash, and retrieval-profile hash, and retrieval fails closed instead of serving stale or differently profiled chunks. After such a change, run `bddk-seed import --reindex-existing` before serving.

The runtime wheel/sdist exclude tests, `.env` files, Compose, and `seed_data/`; the code package does not silently distribute the corpus. A source checkout or the provided container supplies `seed_data/`; another wheel installation must point `BDDK_SEED_DIR` at an approved corpus directory before `bddk-seed import`.

The Compose and image definitions bind HTTP on `0.0.0.0`. Under the fail-closed HTTP policy, a container must receive the exact Host/Origin settings above and either the complete JWT settings or the explicit `BDDK_HTTP_ALLOW_UNAUTHENTICATED` opt-in, or startup will be refused. Development credentials and database ports from Compose must never be reused or exposed outside a developer environment.

The repository produces no SBOM, image-scan, or provenance evidence; any promotion path must produce and verify that evidence itself.

### Outbound regulatory HTTP

Live catalog, document-sync, institution, announcement, and bulletin paths use code-owned outbound controls: exact HTTPS hosts for BDDK/mevzuat, no embedded credentials or non-default ports, public-address DNS preflight, bounded redirect hops with destination/DNS revalidation, streamed decoded-body limits, and privacy-safe retry logs that omit URLs, query strings, and exception messages. Catalog/API responses are capped at 8 MiB; document-sync limits vary by artifact class up to 128 MiB. These are application memory safeguards, not source authenticity, malware-scanning, or network-isolation guarantees.

DNS validation and the subsequent socket connection are not atomic. A restricted deployment must enforce the same destinations with its own egress firewall, proxy, and DNS policy. Public institution, announcement, bulletin, and digest tools can call live BDDK sources, so the serving runtime requires narrow TCP 443 egress to the approved regulatory-source destinations or an enterprise proxy; `migrate` and `import` require only PostgreSQL reachability.

## Bank deployment

The OpenShift starter manifests, the recovery-drill tooling, the operational-objectives contract and the supply-chain evidence lane (removed 2026-11-08, last present at commit `e858590`), and the signed corpus-release workflow, multi-identity PostgreSQL role contract, operator job manager, admin console and telemetry (removed 2026-11-09, last present at commit `9ed84a0`, tag `archive/pre-simplify-2026-11-09`) remain in Git history and must be restored and re-reviewed before any bank deployment. This repository is therefore not bank acceptance or a production-ready platform configuration. Railway (below) is a development/preview profile only.

The repository has not completed a bank-cluster acceptance, backup/restore drill, or a release compatibility matrix for Claude, Codex, GPT-based clients, LM Studio, GPT-OSS, or other local clients. Example configurations demonstrate configuration shape; they are not compatibility certification.

## Railway and Spaces

Railway and Hugging Face Spaces are development/preview profiles. They are not part of a reviewed bank deployment path and must not be used for bank data or bank-facing service.

`railway.toml` builds the standard Dockerfile, declares a `/health/ready` healthcheck, runs `bddk-mcp migrate` and `bddk-seed import` as the pre-deploy command (both are no-ops when the database already matches), and documents the required service variables inline. Railway must inject a `verify-full` PostgreSQL DSN with an available CA path and the complete non-loopback HTTP policy (`BDDK_HTTP_ALLOWED_HOSTS`, `BDDK_HTTP_ALLOWED_ORIGINS`, and either the full `BDDK_JWT_*` set or the explicit unauthenticated opt-in); without them the server refuses startup. When the public DSN is read-only, also set `BDDK_INGESTION_DATABASE_URL` and `BDDK_SCHEMA_OWNER_DATABASE_URL` for the pre-deploy command.

There is no separate Spaces image. Preview hosts that need port `7860` should set `PORT` on the standard image. Neither Railway nor a preview host supplies deployment-specific identity or global ingress controls by itself.

## Secrets and Logs

- Keep DSNs, tokens, and credentials in platform Secrets, never in Git, image layers, committed MCP configuration, or command-line examples with real values.
- Mount the approved PostgreSQL CA separately from credentials; keep `sslmode=verify-full` and the absolute `sslrootcert` path in every remote DSN. Never enable `BDDK_ALLOW_INSECURE_DATABASE` outside isolated local development.
- Keep `BDDK_TOOL_LOG_CONTENT=false` in shared and production environments.
- Treat bank queries as confidential even when the regulatory corpus itself is public.
