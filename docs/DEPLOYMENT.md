# Deployment Guide

This guide describes the deployment paths that the repository currently supports and the controls that are still required before a bank on-premises deployment.

## Runtime Modes

| Mode | Command or endpoint | Intended boundary | Current tool surface |
|---|---|---|---:|
| Local stdio | `uv run --frozen bddk-mcp` | One local MCP client | 15 public tools by default |
| Streamable HTTP | `MCP_TRANSPORT=streamable-http uv run --frozen bddk-mcp` | Trusted development network only | 15 public tools by default |
| Operator | Add `BDDK_ADMIN_TOOLS=true` | Isolated operator process only | 26 total tools |

`BDDK_DATABASE_URL` is required in every mode. Prepare a checkout/container database explicitly before serving:

```bash
BDDK_DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/DATABASE \
uv run --frozen bddk-mcp bootstrap
```

`bddk-mcp migrate` performs schema work only. `bddk-mcp bootstrap` migrates, imports the reviewed seed corpus, builds sections/embeddings, and validates serving readiness. The legacy `bddk-seed import|export` helper remains available for corpus maintenance.

The checked-in seed corpus is job-specific rather than a claim of exhaustive BDDK coverage.

## Local stdio

From a checkout:

```bash
uv sync --frozen
export BDDK_DATABASE_URL=postgresql://bddk:bddk@localhost:5432/bddk
uv run --frozen bddk-mcp bootstrap
uv run --frozen bddk-mcp serve
```

The default transport is stdio. Use the repository [`.mcp.json`](../.mcp.json) with clients that support that project configuration format. Codex uses `config.toml`; see the examples in the main [README](../README.md#codex-configuration).

## Docker Compose

The Compose file is a local-development stack. A one-shot `bddk-init` service must complete bootstrap before the server starts. It exposes PostgreSQL on host port `5432` and the MCP server on port `8000`:

```bash
docker compose up --build
```

The Streamable HTTP MCP endpoint is:

```text
http://localhost:8000/mcp
```

Do not reuse the Compose credentials or expose its database port outside a developer workstation. The application image no longer contains a default database credential; Compose injects its development DSN explicitly.

## Container Image Contract

Both Dockerfiles install and run the packaged `bddk-mcp` entry point. The standard and Spaces images include `seed_data/`, and the embedding model is downloaded during the image build for offline runtime loading.

Required runtime configuration:

| Variable | Requirement |
|---|---|
| `BDDK_DATABASE_URL` | Required secret-backed PostgreSQL DSN; the database must support pgvector |
| `MCP_TRANSPORT` | `stdio` or `streamable-http`; containers default to `streamable-http` |
| `MCP_HOST` | Defaults to `127.0.0.1`; container images set `0.0.0.0` and therefore require a trusted ingress boundary |
| `PORT` | HTTP listen port; standard image defaults to `8000`, Spaces to `7860` |
| `BDDK_ADMIN_TOOLS` | Leave unset/false for the public workload |
| `BDDK_AUTO_SYNC` | Must remain false in serving mode; use a separate operator workflow for synchronization |

Serving startup is read-only for schema/corpus lifecycle: it validates the database, loads the decision cache with `SELECT`, and constructs the search clients without DDL, seed import, synchronization, or embedding backfill. An incomplete database fails startup with commands for remediation. Telemetry and operator tools can still write when explicitly enabled; a truly read-only database role therefore belongs on the public profile with telemetry disabled.

The runtime wheel deliberately excludes the repository-only benchmark harness and does not embed the 24 MB reviewed corpus. A checkout/container provides `seed_data/`; another installation must mount it and pass `--seed-dir` or `BDDK_SEED_DIR` to bootstrap. Run evaluation workflows from a source checkout with `uv sync --group benchmark`.

## OpenShift AI Status

The Docker image is a starting point for an on-premises OpenShift AI workload, but this repository does **not** yet contain production-ready OpenShift manifests. In particular, it currently lacks:

- Deployment, Service, Route, and separate bootstrap Job manifests;
- application authentication and authorization;
- separate public and operator workloads/service accounts;
- NetworkPolicies and egress allowlists;
- HTTP liveness/readiness endpoints;
- a non-root container user and tested restricted-v2 security context;
- resource requests/limits and model-loading capacity guidance;
- migration, backup, restore, rollback, and disaster-recovery procedures.

Streamable HTTP defaults to loopback. The container profile explicitly binds `0.0.0.0` so a Service can reach it, but there is still no application-level authentication or rate limiting. That container profile must not be exposed through a bank Route or to an untrusted network until an approved identity-aware gateway and the roadmap security controls are in place.

For the target bank deployment, plan these boundaries:

1. An external PostgreSQL/pgvector service with a secret-backed DSN and distinct bootstrap, serving, and operator roles.
2. A public MCP workload with `BDDK_ADMIN_TOOLS=false` and a read-only serving role.
3. A separately deployed operator workload or Job, reachable only by approved administrators.
4. An identity-aware ingress/gateway providing TLS, authentication, authorization, request limits, and audit events before traffic reaches `/mcp`.
5. NetworkPolicy rules that allow only required database and explicitly approved BDDK upstream destinations.
6. A release-specific client compatibility test for every MCP host used by the bank.

Until those controls exist, use stdio locally or Streamable HTTP only inside an isolated development namespace.

## Railway and Spaces

`railway.toml` builds the standard Dockerfile. Railway must inject `BDDK_DATABASE_URL`; the `/app/data` volume does not back up an external PostgreSQL database. The repository does not currently define an HTTP health check.

`Dockerfile.spaces` uses port `7860` and includes the reviewed seed corpus, but the database must be bootstrapped separately before its server command starts. It also requires an injected PostgreSQL DSN. Neither deployment path adds authentication to the MCP endpoint, so both require a trusted/private boundary or an external security gateway.

## Secrets and Logs

- Keep DSNs and credentials in platform Secrets, never in Git, image layers, MCP config committed to the repository, or command-line examples with real values.
- Keep `BDDK_TOOL_LOG_CONTENT=false` in shared and production environments.
- Keep `BDDK_TELEMETRY_STORE_TEXT=false` unless storage of raw queries has been explicitly approved.
- Treat bank queries as potentially confidential even when the regulatory corpus itself is public.
