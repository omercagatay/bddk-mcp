# Open WebUI on the bank servers

Open WebUI is the only user-facing entry point for the BDDK MCP server. The
department (about 25 users) signs in with Microsoft (Active Directory) accounts
over LDAP; the MCP connection itself carries no per-user login.

## Authentication model

- **Users → Open WebUI**: LDAP against the bank Active Directory
  (`ENABLE_LDAP=true`, LDAPS on port 636 with the bank trust root). Local
  self-signup is disabled; every account is a directory account.
- **Open WebUI → MCP server**: a direct streamable-HTTP connection without
  OAuth. The MCP public profile runs with
  `BDDK_HTTP_ALLOW_UNAUTHENTICATED=true` and is reachable only inside the bank
  network segment approved for this deployment. Network isolation — the MCP
  exact Host/Origin allowlists, the bank firewall/ingress rules, and the
  OpenShift NetworkPolicies — replaces the earlier Keycloak bearer-token layer.
- The earlier single-host stack (Keycloak realm, confidential OAuth client,
  and the patched Open WebUI image with per-request MCP token refresh) was
  removed with this migration; the compose file now runs the plain upstream
  image pinned by digest.

Because MCP requests are no longer attributable to an end user by the MCP
server, per-user attribution lives in Open WebUI's own chat history and the
bank's access logs. Do not expose the MCP endpoint outside the approved
segment, and do not reuse this profile for the operator tools: the application
refuses to serve the operator profile unauthenticated on a non-loopback bind.

## Configuration

Resolve every `REPLACE_*` value in `compose.yml` from the bank design:

- `WEBUI_URL` — the HTTPS name users visit (TLS terminates at the bank-approved
  reverse proxy; the container binds loopback only).
- `LDAP_SERVER_HOST`/`LDAP_SERVER_PORT` — the bank AD endpoint (LDAPS 636).
- `LDAP_APP_DN` — a read-only directory service account used only to look up
  users; export its password as `LDAP_APP_PASSWORD` or place it in the ignored
  `.env` (owner-only, mode `0600`).
- `LDAP_SEARCH_BASE` — the OU that contains the department users. Restrict it
  (or add `LDAP_SEARCH_FILTERS`) so only the intended ~25 accounts can sign in.
- `LDAP_CA_CERT_FILE` — copy the bank AD trust root into the data volume as
  `ldap-ca.crt` before first start.
- `OPENAI_API_BASE_URL` — the bank-internal OpenAI-compatible model gateway.

Add the MCP server in Open WebUI as an external tool/MCP connection using the
streamable-HTTP URL (`https://<mcp-host>/mcp`) with authentication set to none.
The MCP server's `BDDK_HTTP_ALLOWED_HOSTS`/`BDDK_HTTP_ALLOWED_ORIGINS` must
list the exact values this deployment presents.

## Deployment

Run all commands from `deploy/open-webui`. The wrapper script passes the model
provider key only through the Compose process environment and permits only
`up`, `down`, and the non-rendering `config --quiet`:

```bash
set -euo pipefail
./compose-current-key.sh config --quiet
./compose-current-key.sh up -d --wait --wait-timeout 180
```

The named volume is external, so `docker compose down -v` cannot delete it.
`UVICORN_WORKERS` stays at 1 because SQLite must have a single application
writer; before adding replicas, move to a database backend.

## Backup

Take quiesced backups of the `open-webui` volume (it contains user files, chat
history, and `.webui_secret_key`) and store them encrypted under the applicable
retention policy:

```bash
set -euo pipefail
umask 077
install -d -m 700 snapshots
snapshot="snapshots/open-webui-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
docker run --rm --network none \
  --mount type=volume,source=open-webui,target=/source,readonly \
  --entrypoint tar ghcr.io/open-webui/open-webui@sha256:6a773e5c3a246b65cbe74ce942b294292c0e5f81c138f703d111bc162f7d7c3d \
  -C /source --numeric-owner -czf - . > "$snapshot"
chmod 600 "$snapshot"
gzip -t "$snapshot"
```

Keep `/app/backend/data/.webui_secret_key` persistent: losing it invalidates
sessions and any encrypted per-user secrets stored by Open WebUI.

## Security notes

- The unauthenticated MCP profile is safe only while the network boundary
  holds. Treat any change that widens reachability of the MCP endpoint (a new
  route, a broader firewall rule, a NetworkPolicy edit) as a security change
  requiring review.
- The MCP application still enforces exact Host/Origin allowlists, body-size
  and body-deadline limits, and per-process rate/concurrency caps; shared
  request limits and audit events belong at the bank ingress.
- Keep the LDAP bind account read-only and scoped to the department OU.
