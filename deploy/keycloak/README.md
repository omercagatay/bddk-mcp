# Keycloak realm for the BDDK MCP connector

The `bddk` realm is the authorization server behind `https://mcp.bankreg.app/mcp`. Its
configuration used to live **only** in Keycloak's Postgres, with no representation in this
repo — so anything that took the realm off its URL took every connector down with no way
to restore it except rediscovering the settings from scratch. That is what
`rebuild_realm.py` and the exported `realm-bddk.json` exist to prevent.

**The realm name is load-bearing.** It is the `/realms/<name>` path segment that
`BDDK_JWT_ISSUER` pins, so renaming the realm breaks every client instantly. Use
`displayName` for labels.

## What binds to what

The running MCP service pins these values (Railway service `bddk-mcp`, environment
`production`) — the realm name cannot change without changing them too:

| Service variable | Value |
| --- | --- |
| `BDDK_JWT_ISSUER` | `https://keycloak-production-8b65.up.railway.app/realms/bddk` |
| `BDDK_JWT_JWKS_URL` | `<issuer>/protocol/openid-connect/certs` |
| `BDDK_JWT_AUDIENCE` | `bddk-mcp` |
| `BDDK_JWT_REQUIRED_SCOPES` | `bddk.read` |
| `BDDK_JWT_RESOURCE` | `https://mcp.bankreg.app/mcp` |
| `BDDK_JWT_ACCESS_TOKEN_TYPES` | `at+jwt,JWT` — Keycloak mints `typ: JWT` |

The server publishes RFC 9728 metadata at `/.well-known/oauth-protected-resource/mcp`
pointing at the realm. Claude follows that pointer, fetches
`<issuer>/.well-known/openid-configuration`, and registers a client via anonymous RFC 7591
dynamic client registration. A missing realm surfaces in Claude as
*"Couldn't register with BDDK-MCP's sign-in service."*

## Rebuilding

```bash
KC_ADMIN_PASSWORD='<keycloak service KC_BOOTSTRAP_ADMIN_PASSWORD>' \
  .venv/bin/python deploy/keycloak/rebuild_realm.py
```

Idempotent — safe to re-run against a healthy realm. Flags:

- `--verify-only` — run the checks, change nothing.
- `--clients claude,chatgpt,local` — which client families may register (default: all
  three). Each group is a set of trusted hosts plus a redirect URI the verifier probes
  with:

  | Group | Trusted hosts | Probed redirect URI |
  | --- | --- | --- |
  | `claude` | `claude.ai`, `*.claude.ai` | `https://claude.ai/api/mcp/auth_callback` |
  | `chatgpt` | `chatgpt.com`, `*.chatgpt.com`, `openai.com`, `*.openai.com` | `https://chatgpt.com/connector_platform_oauth_redirect` |
  | `local` | `localhost`, `127.0.0.1` | `http://localhost:8765/callback` |

The script recreates the realm, the `bddk.read` client scope with its `bddk-mcp` audience
mapper, the two client-registration policies, the `bddk-test` service-account client, and
the human realm user, then verifies and exports.

## The other allowlist: `BDDK_HTTP_ALLOWED_ORIGINS`

Keycloak's trusted hosts are only half the story. The MCP server keeps its own,
independent origin allowlist, and refuses an unlisted `Origin` with **403 before
authentication is considered** — so a perfectly rebuilt realm can still leave a connector
broken. Verified behaviour against the live server:

| Request `Origin` | Response | Meaning |
| --- | --- | --- |
| `https://claude.ai` | 401 | allowed; only the bearer token was missing |
| `https://chatgpt.com` | 403 | **rejected outright** |
| *(absent)* | 401 | allowed — server-side callers such as open-webui's backend send no Origin |

ChatGPT's connector calls originate from OpenAI's backend and may well send no `Origin` at
all, in which case this was never what blocked it — the missing realm was. But
`https://chatgpt.com` is the only origin it could plausibly send, so the variable is
widened alongside trusting the host in Keycloak:

```bash
railway variables --service bddk-mcp --environment production \
  --set 'BDDK_HTTP_ALLOWED_ORIGINS=https://claude.ai,https://chatgpt.com,https://mcp.bankreg.app'
```

Setting a variable triggers a redeploy, which re-runs the `preDeployCommand` bootstrap
(~2 min, a no-op when the corpus already matches).

Locally hosted open-webui needs **no** origin entry as long as it calls the server from its
backend. Do not try to add `http://localhost:3000`: the config validator rejects non-HTTPS
origins for non-loopback binds (`Non-loopback HTTP requires HTTPS allowed Origins`) and the
server will refuse to start.

## The two policies that gate registration

Both are **anonymous**-subtype client registration policies — the subtype matters, since
Claude, ChatGPT and open-webui all register anonymously — and both rejected Claude before
they were tuned:

1. **Trusted Hosts** (`providerId: trusted-hosts`) — needs
   `trusted-hosts=[claude.ai, *.claude.ai]`, `client-uris-must-match=true`, and
   `host-sending-registration-request-must-match=false`. That last one is not optional:
   reverse DNS on Anthropic's egress addresses (seen from `160.79.106.0/24`) can never
   resolve to `claude.ai`, so leaving it on rejects every registration.
2. **Allowed Client Scopes** — note the `providerId` is the legacy
   `allowed-client-templates`, and the config key is `allowed-client-scopes`. `bddk.read`
   must be appended, or registration fails with *"Not permitted to use specified
   clientScope."*

## Connecting Open WebUI

This server speaks **MCP streamable HTTP**. It does not serve an OpenAPI spec, and it emits
**no CORS headers at all** — verified: even a request from an allowed origin comes back with
no `Access-Control-Allow-*`. So it can only be driven from a backend, never from browser
JavaScript.

Open WebUI must therefore be added as a **remote MCP** tool server, not an OpenAPI one:

```json
{
  "type": "mcp",
  "url": "https://mcp.bankreg.app/mcp",
  "auth_type": "oauth_2.1",
  "info": { "name": "BDDK", "description": "Turkish banking regulatory intelligence" }
}
```

Adding it as `"type": "openapi"` makes Open WebUI fetch `<url>/openapi.json` from the
browser, producing a CORS preflight that this server answers `403` (unlisted origin) — or
`404` from an allowed origin, since the path does not exist. In the Railway HTTP logs that
signature is unmistakable:

```
OPTIONS /mcp/openapi.json 403      <- misconfigured as OpenAPI, called from the browser
POST /mcp 401 -> 200 -> 202 -> 200 <- correct: MCP session from a backend
```

Notes that come from Open WebUI's own docs and matter here:

- Native MCP is **streamable HTTP only** and is a recent feature. If the tool-server dialog
  offers no MCP type, upgrade Open WebUI. Do **not** reach for `mcpo`: it bridges *stdio*
  MCP servers to OpenAPI and has nowhere to put a refreshing OAuth token.
- `WEBUI_SECRET_KEY` must be set to a persistent value, or the OAuth session breaks on
  every container restart.
- The OAuth consent has to be completed once interactively in a browser; after that the
  backend holds the token.

### Redirect-URI hosts

Open WebUI registers with a redirect URI on **its own** address, and the Trusted Hosts
policy matches on that host. `localhost` and `127.0.0.1` are trusted by default; an
instance reached over the LAN or a domain needs its host added explicitly:

```bash
KC_ADMIN_PASSWORD='...' .venv/bin/python deploy/keycloak/rebuild_realm.py \
  --extra-hosts 192.168.0.195
```

Use exactly the host you type in the browser. The verifier then probes
`http://<host>:3000/oauth/callback` for each one.

### The `openid` scope

Keycloak's Allowed Client Scopes policy rejects an **entire** registration over a single
unlisted scope, and OIDC-style clients routinely ask for `openid`. This realm rejected
exactly that on 2026-08-12:

```
Requested scope 'openid' not trusted in the list: [bddk.read, role_list, ...]
KC-SERVICES0099: Operation 'before register client' rejected. Policy 'Allowed Client Scopes'
```

So `REGISTRABLE_SCOPES` carries both `bddk.read` and `openid`, and the DCR probes request
the pair rather than `bddk.read` alone.

## Verifying by hand

Any manual DCR probe **must include the `scope` field**. Without it the probe sails through
the Trusted Hosts policy and never touches the Allowed Client Scopes policy — so the probe
passes while Claude still fails:

```bash
curl -s -o /dev/stderr -w '%{http_code}\n' \
  -X POST https://keycloak-production-8b65.up.railway.app/realms/bddk/clients-registrations/openid-connect \
  -H 'Content-Type: application/json' \
  -d '{"client_name":"probe","redirect_uris":["https://claude.ai/api/mcp/auth_callback"],
       "grant_types":["authorization_code","refresh_token"],"response_types":["code"],
       "token_endpoint_auth_method":"client_secret_post","scope":"bddk.read"}'
```

Expect `201` with `"scope":"bddk.read"` in the body. Swapping the redirect URI for a
non-claude host must give `403`. Delete the probe client afterwards using the
`registration_access_token` from the response.

## Renamed, deleted, or lost?

**A realm's name is its URL path segment.** Keycloak's admin console presents that name in
a field labelled **"Realm ID"**, directly above **"Display name"** — so editing the field
that looks like a cosmetic label instead renames the realm, 404s the issuer, and takes
every connector down while all the clients, users and policies sit there perfectly intact.
This is far more likely than a deletion, and the recovery is completely different.

All three failures look identical from outside, so check before changing anything:

```bash
KC_DB_PASSWORD='<Postgres service PGPASSWORD>' .venv/bin/python deploy/keycloak/probe_realm_db.py
```

- A realm carrying id `dc17d66a-c2fe-4065-98a0-801d1237083c` under another name → a
  rename. Nothing was lost; `rebuild_realm.py` renames it back. **Do not build a fresh
  realm** — it would strand the real one's clients and users.
- `master` and its clients intact, no realm with that id → a genuine deletion; rebuild.
- `keycloak` database missing, `realm` table empty, or a freshly-initialised store →
  storage loss. Check the Postgres volume (`RAILWAY_VOLUME_ID`) first, because a rebuild
  will not stop it recurring.

### 2026-08-15 incident — a rename

`https://.../realms/bddk` returned `{"error":"Realm does not exist"}` and Claude reported
*"Couldn't register with BDDK-MCP's sign-in service."* Nothing had been redeployed —
keycloak last deployed 2026-08-10, Postgres 2026-08-06 — and Keycloak, `master`, every
service variable and Postgres were healthy throughout, which is exactly the signature of a
rename rather than an outage.

The database confirmed it: realm id `dc17d66a-c2fe-4065-98a0-801d1237083c` — the same id
serving traffic at 2026-08-12 20:17 UTC — had been renamed to **`Mevzuat MCP`**, with
`bddk-test`, the human users, and three DCR-registered clients all still attached. The fix
was to restore the name to `bddk` and move `Mevzuat MCP` to `displayName`, where it was
presumably meant to go.

Nothing was lost, and no credential had to be reissued. One incidental repair came out of
it: the `bddk.read` audience mapper was missing and had to be re-added.

## `realm-bddk.json`

A Keycloak partial export (clients, groups, roles, scopes, policies). It carries **no
client secrets and no users** — those are recreated by the script. To restore from it
instead, import via the admin console or `POST /admin/realms` with the file as the body,
then re-run `rebuild_realm.py` to recreate the secret-bearing pieces and verify.
