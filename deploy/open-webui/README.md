# Open WebUI MCP OAuth refresh overlay

This overlay fixes long-running MCP chats without weakening the five-minute access-token
policy or asking users to authorize the BDDK integration repeatedly.

## Authentication model

The BDDK connection uses a confidential static OAuth client, authorization-code flow,
PKCE, and the scopes `bddk.read offline_access`. Open WebUI encrypts each user's offline
refresh token in its database. A user completes interactive authorization once; access
tokens are then refreshed by the backend. With the current Keycloak policy, interactive
authorization is required again only after explicit revocation, loss of the persistent
encryption key, or 30 days without using the offline session.

The upstream Open WebUI 0.11.0 transport captures one bearer token when a chat starts and
reuses it for the entire MCP session. A tool call made after the five-minute token expires
therefore receives 401 even though a valid offline refresh token is stored.

The patch:

- resolves the current per-user access token before every MCP HTTP request;
- refreshes and safely replays a rejected request exactly once;
- serializes refresh-token rotation per user and MCP connection;
- coalesces concurrent 401 responses onto one replacement token;
- coalesces concurrent refresh failures during a short identity-provider backoff;
- refreshes 30 seconds before expiry instead of five minutes before a five-minute token;
- retains an encrypted session after a transient refresh failure;
- removes only permanently rejected refresh credentials so the UI can request reconnect; and
- leaves bearer, session, and unauthenticated MCP connections unchanged.

## Build and test

The base image is pinned to Open WebUI revision
`01f4282f1ffe0d6212f58d3afbeae21fffd0c4be` and registry digest
`sha256:6a773e5c3a246b65cbe74ce942b294292c0e5f81c138f703d111bc162f7d7c3d`.
The immutable digest fixes the exact base content; the build also fails if the patch no
longer applies cleanly. The allowlisted `.dockerignore` prevents `.env`, snapshots, chats,
tokens, and other local data from entering the Docker build context or cache.

```bash
docker build --pull=false \
  -t bddk-open-webui:0.11.0-mcp-oauth-refresh.1 \
  deploy/open-webui

docker run --rm --entrypoint sh \
  -e WEBUI_SECRET_KEY=test-only-not-for-production \
  -e DATA_DIR=/tmp/open-webui-test-data \
  -v "$PWD/deploy/open-webui/tests:/tests:ro" \
  bddk-open-webui:0.11.0-mcp-oauth-refresh.1 \
  -lc 'python -m pytest -q -o cache_dir=/tmp/pytest-cache /tests/test_mcp_oauth_refresh.py'
```

The focused suite covers token injection, a single 401 replay with an identical request
body, retry limits, missing tokens, refresh timing, concurrent refresh coalescing,
transient failure preservation, middleware binding, unchanged non-OAuth behavior, and a
real MCP SDK streamable-HTTP session that remains usable after a token rejection.

## Deployment

The Compose definition deliberately uses the existing named volume as an external
volume. It cannot be deleted by `docker compose down -v`. It also pins one Uvicorn worker
because refresh serialization is process-local and SQLite must have only one application
writer.

Run all commands below from `deploy/open-webui`. First build and test while the live
container is still serving traffic:

```bash
set -euo pipefail
docker build --pull=false \
  -t bddk-open-webui:0.11.0-mcp-oauth-refresh.1 .
docker run --rm --entrypoint sh \
  -e WEBUI_SECRET_KEY=test-only-not-for-production \
  -e DATA_DIR=/tmp/open-webui-test-data \
  -v "$PWD/tests:/tests:ro" \
  bddk-open-webui:0.11.0-mcp-oauth-refresh.1 \
  -lc 'python -m pytest -q -o cache_dir=/tmp/pytest-cache /tests/test_mcp_oauth_refresh.py'
```

Validate that the current provider key can be reused without printing or copying it to a
workspace file. The wrapper first accepts an explicitly exported `OPENAI_API_KEY`, then
checks the production and retained rollback container configurations, and finally accepts
an owned mode-`0600` `.env`. It passes the value only in the Compose process environment
and permits only `up`, `down`, and the non-rendering `config --quiet` command:

```bash
set -euo pipefail
test "$(docker inspect --format '{{.State.Health.Status}}' open-webui)" = healthy
./compose-current-key.sh config --quiet
```

Retain at least one source container until the provider key has been migrated. Before both
are deleted, either export the key through an external secret manager when invoking the
wrapper or create the ignored `.env` from `.env.example`, owned by the invoking user with
mode `0600`.

Disable automatic restart on the retained container, stop it cleanly, rename it to free
the production name, and prove that no process can write the SQLite volume:

```bash
set -euo pipefail
rollback=open-webui-pre-mcp-oauth-refresh
! docker container inspect "$rollback" >/dev/null 2>&1
docker update --restart=no open-webui
docker stop --time 30 open-webui
docker rename open-webui "$rollback"
test "$(docker inspect --format '{{.State.Running}}' "$rollback")" = false
test -z "$(docker ps --quiet --filter volume=open-webui)"
```

Take and validate a full quiesced backup. It includes user files, encrypted refresh
tokens, and `.webui_secret_key`, so retain it as sensitive data and copy it to encrypted
off-host storage under the applicable retention policy:

```bash
set -euo pipefail
umask 077
install -d -m 700 snapshots
snapshot="snapshots/open-webui-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
docker run --rm --network none \
  --mount type=volume,source=open-webui,target=/source,readonly \
  --entrypoint tar bddk-open-webui:0.11.0-mcp-oauth-refresh.1 \
  -C /source --numeric-owner -czf - . > "$snapshot"
chmod 600 "$snapshot"
gzip -t "$snapshot"
tar -tzf "$snapshot" | awk '
  $0 == "./webui.db" { db = 1 }
  $0 == "./.webui_secret_key" { key = 1 }
  END { exit !(db && key) }
'
sha256sum "$snapshot" > "$snapshot.sha256"
chmod 600 "$snapshot.sha256"
```

Start the already-tested image and wait for its configured application health check:

```bash
set -euo pipefail
./compose-current-key.sh up -d --no-build --wait --wait-timeout 180
```

Acceptance checks:

- `/health` is healthy and the existing chats/files remain visible;
- the existing BDDK integration remains connected without another consent screen;
- a normal MCP tool call succeeds;
- after more than five minutes in the same chat, another MCP call succeeds;
- logs show a silent token refresh followed by MCP 200, with no task cancellation; and
- the same OAuth session row remains present after an Open WebUI restart.

Rollback normally reuses the current volume because this overlay uses the same base
revision and introduces no schema migration. Remove the overlay container and its restart
policy before starting the retained container:

```bash
set -euo pipefail
./compose-current-key.sh down --timeout 30
test -z "$(docker ps --quiet --filter volume=open-webui)"
docker rename open-webui-pre-mcp-oauth-refresh open-webui
docker update --restart=always open-webui
docker start open-webui
for rollback_attempt in $(seq 1 90); do
  if test "$(docker inspect --format '{{.State.Health.Status}}' open-webui)" = healthy; then
    break
  fi
  sleep 2
done
if ! test "$(docker inspect --format '{{.State.Health.Status}}' open-webui)" = healthy; then
  docker logs --tail 120 open-webui
  exit 1
fi
```

Restore the archive only if a schema migration is proven and both containers are down;
normal rollback must not overwrite the volume.

## Security notes

Do not replace this with a shared service-account bearer token: that removes per-user
authorization, revocation, and audit attribution. Do not make access tokens effectively
permanent; the offline refresh token is the correct mechanism for durable consent.

The single automatic replay is specific to BDDK: its authorization middleware rejects an
invalid bearer token before dispatching the MCP JSON-RPC request. Do not apply this overlay
unchanged to a server that can return 401 after a mutating tool has already executed.

Keep this SQLite deployment at one Open WebUI container and one Uvicorn worker. Before
adding replicas or enabling strict refresh-token rotation, replace the process-local lock
with a database/Redis singleflight or compare-and-swap token update.

Keep `/app/backend/data/.webui_secret_key` persistent. Losing or changing it makes the
encrypted OAuth sessions unreadable and forces users to reconnect. A future key-separation
migration may set dedicated persistent `OAUTH_SESSION_TOKEN_ENCRYPTION_KEY` and
`OAUTH_CLIENT_INFO_ENCRYPTION_KEY`, but those values must be introduced with an explicit
migration so existing encrypted records are not stranded.
