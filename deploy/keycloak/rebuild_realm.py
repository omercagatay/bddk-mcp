#!/usr/bin/env python3
"""Restore the `bddk` Keycloak realm to a known-good state, idempotently.

The realm config lives only in Keycloak's Postgres — there is no IaC — so anything that
takes the realm off its URL takes every MCP connector down with it: clients fetch AS
metadata from `<issuer>/.well-known/openid-configuration` and report "couldn't register"
on a 404. Claude, ChatGPT and open-webui all fail the same way.

The realm is identified by its stable UUID, not its name, so a *renamed* realm is renamed
back rather than rebuilt — a fresh realm would strand the real one's clients and users.

This script converges the realm, verifies it the way the clients actually exercise it, and
writes a partial export to `realm-bddk.json` so the next incident is a one-import fix.
Re-running it patches in place rather than duplicating.

Usage (from the repo root, via `!` so the admin password never hits a tool call):

    KC_ADMIN_PASSWORD='...' .venv/bin/python deploy/keycloak/rebuild_realm.py

Optional flags:
    --verify-only        run the verification suite, change nothing
    --clients a,b,c      which client families may register (claude, chatgpt, local)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import string
import sys
from pathlib import Path
from urllib.parse import quote

import httpx

KC = "https://keycloak-production-8b65.up.railway.app"
REALM = "bddk"
# The realm's stable UUID, unchanged since it was first built. A realm can be renamed out
# from under the issuer URL, so identity is checked by id, never by name.
KNOWN_REALM_ID = "dc17d66a-c2fe-4065-98a0-801d1237083c"
SCOPE = "bddk.read"
# Scopes a client may name in its DCR payload. `openid` is here because OIDC-style
# clients (Open WebUI among them) request it as a matter of course, and Keycloak's
# Allowed Client Scopes policy rejects the whole registration over one unlisted scope:
#   "Requested scope 'openid' not trusted in the list: [...]"  -> 403, "Connection failed"
REGISTRABLE_SCOPES = [SCOPE, "openid"]
AUDIENCE = "bddk-mcp"
TEST_CLIENT = "bddk-test"
HUMAN_USER = "cagatay"
MCP_URL = "https://mcp.bankreg.app/mcp"
EXPORT_PATH = Path(__file__).with_name("realm-bddk.json")

# Every MCP client registers through anonymous DCR, which both policies below gate.
# Each group is a set of trusted hosts plus a representative redirect URI to probe with.
HOST_GROUPS: dict[str, tuple[list[str], str]] = {
    "claude": (["claude.ai", "*.claude.ai"], "https://claude.ai/api/mcp/auth_callback"),
    "chatgpt": (
        ["chatgpt.com", "*.chatgpt.com", "openai.com", "*.openai.com"],
        "https://chatgpt.com/connector_platform_oauth_redirect",
    ),
    # open-webui, mcp-inspector, codex and other locally hosted clients.
    "local": (["localhost", "127.0.0.1"], "http://localhost:8765/callback"),
}

OK = "\033[32m✓\033[0m"
BAD = "\033[31m✗\033[0m"
INFO = "\033[36m·\033[0m"


def say(mark: str, msg: str) -> None:
    print(f" {mark} {msg}", flush=True)


class Admin:
    def __init__(self, password: str) -> None:
        self.c = httpx.Client(base_url=KC, timeout=30.0)
        r = self.c.post(
            "/realms/master/protocol/openid-connect/token",
            data={
                "grant_type": "password",
                "client_id": "admin-cli",
                "username": os.environ.get("KC_ADMIN_USERNAME", "admin"),
                "password": password,
            },
        )
        r.raise_for_status()
        self.token = r.json()["access_token"]
        self.c.headers["Authorization"] = f"Bearer {self.token}"

    def get(self, path: str, **kw) -> httpx.Response:
        return self.c.get(f"/admin/realms{path}", **kw)

    def post(self, path: str, payload: dict, **kw) -> httpx.Response:
        return self.c.post(f"/admin/realms{path}", json=payload, **kw)

    def put(self, path: str, payload: dict) -> httpx.Response:
        return self.c.put(f"/admin/realms{path}", json=payload)


# --------------------------------------------------------------------------- realm


def ensure_realm(a: Admin) -> None:
    r = a.get(f"/{REALM}")
    if r.status_code == 200:
        say(INFO, f"realm '{REALM}' already exists (id {r.json()['id']})")
        return

    # A realm's *name* is its URL path segment, and Keycloak's admin console presents it
    # as the "Realm ID" field right above "Display name" — editing the wrong one renames
    # the realm and 404s the issuer, taking every connector down while all the clients,
    # users and policies sit there intact. Recover by renaming back, never by rebuilding:
    # a fresh realm would strand the existing users and registered clients.
    realms = a.c.get("/admin/realms").json()
    stray = next((x for x in realms if x.get("id") == KNOWN_REALM_ID), None)
    if stray:
        current = stray["realm"]
        rep = a.get(f"/{quote(current, safe='')}").json()
        rep["realm"] = REALM
        if not rep.get("displayName"):
            rep["displayName"] = current  # keep the intended label where it belongs
        a.put(f"/{quote(current, safe='')}", rep).raise_for_status()
        say(
            OK, f"renamed realm '{current}' back to '{REALM}' (id {KNOWN_REALM_ID}); displayName={rep['displayName']!r}"
        )
        return

    say(INFO, f"no realm with id {KNOWN_REALM_ID} found — creating '{REALM}' from scratch")
    r = a.c.post(
        "/admin/realms",
        json={
            "realm": REALM,
            "enabled": True,
            "displayName": "BDDK Regulatory Intelligence",
            "sslRequired": "external",
            # Claude shows a consent screen; keep consent server-side too.
            "loginTheme": None,
            "registrationAllowed": False,
            "resetPasswordAllowed": True,
        },
    )
    r.raise_for_status()
    say(OK, f"created realm '{REALM}'")


def ensure_client_scope(a: Admin) -> str:
    scopes = a.get(f"/{REALM}/client-scopes").json()
    existing = next((s for s in scopes if s["name"] == SCOPE), None)
    if existing:
        say(INFO, f"client scope '{SCOPE}' already exists")
        scope_id = existing["id"]
    else:
        r = a.post(
            f"/{REALM}/client-scopes",
            {
                "name": SCOPE,
                "description": "Read access to the BDDK regulatory corpus",
                "protocol": "openid-connect",
                "attributes": {
                    "include.in.token.scope": "true",
                    "display.on.consent.screen": "true",
                    "consent.screen.text": "Read BDDK regulatory documents",
                },
            },
        )
        r.raise_for_status()
        scope_id = next(s["id"] for s in a.get(f"/{REALM}/client-scopes").json() if s["name"] == SCOPE)
        say(OK, f"created client scope '{SCOPE}'")

    # Audience mapper — the MCP server enforces aud == bddk-mcp exactly.
    mappers = a.get(f"/{REALM}/client-scopes/{scope_id}/protocol-mappers/models").json()
    if any(m["name"] == "bddk-mcp-audience" for m in mappers):
        say(INFO, "audience mapper already present")
    else:
        r = a.post(
            f"/{REALM}/client-scopes/{scope_id}/protocol-mappers/models",
            {
                "name": "bddk-mcp-audience",
                "protocol": "openid-connect",
                "protocolMapper": "oidc-audience-mapper",
                "config": {
                    "included.custom.audience": AUDIENCE,
                    "access.token.claim": "true",
                    "id.token.claim": "false",
                    "introspection.token.claim": "true",
                },
            },
        )
        r.raise_for_status()
        say(OK, f"added audience mapper -> {AUDIENCE}")

    # Make it assignable as an optional scope at realm level.
    r = a.put(f"/{REALM}/default-optional-client-scopes/{scope_id}", {})
    if r.status_code in (204, 200):
        say(OK, f"'{SCOPE}' registered as realm default-optional scope")
    return scope_id


# ------------------------------------------------------- client registration policies


def _policies(a: Admin) -> list[dict]:
    return a.get(
        f"/{REALM}/components",
        params={"type": "org.keycloak.services.clientregistration.policy.ClientRegistrationPolicy"},
    ).json()


def fix_trusted_hosts(a: Admin, groups: list[str], extra_hosts: list[str]) -> None:
    hosts = [h for g in groups for h in HOST_GROUPS[g][0]] + extra_hosts
    for comp in _policies(a):
        # Only the anonymous policy set gates unauthenticated DCR, which is what
        # Claude, ChatGPT and open-webui all use.
        if comp.get("providerId") != "trusted-hosts" or comp.get("subType") != "anonymous":
            continue
        comp["config"] = {
            **comp.get("config", {}),
            "trusted-hosts": hosts,
            "client-uris-must-match": ["true"],
            # Reverse DNS of Anthropic's egress IPs can never resolve to claude.ai.
            "host-sending-registration-request-must-match": ["false"],
        }
        a.put(f"/{REALM}/components/{comp['id']}", comp).raise_for_status()
        say(OK, f"Trusted Hosts policy -> {hosts}")
        return
    say(BAD, "anonymous 'Trusted Hosts' policy not found")


def fix_allowed_scopes(a: Admin) -> None:
    for comp in _policies(a):
        # providerId is the LEGACY name, not 'allowed-client-scopes'.
        if comp.get("providerId") != "allowed-client-templates" or comp.get("subType") != "anonymous":
            continue
        cfg = comp.get("config", {})
        allowed = list(cfg.get("allowed-client-scopes", []))
        added = [s for s in REGISTRABLE_SCOPES if s not in allowed]
        allowed.extend(added)
        cfg["allowed-client-scopes"] = allowed
        cfg["allow-default-scopes"] = ["true"]
        comp["config"] = cfg
        a.put(f"/{REALM}/components/{comp['id']}", comp).raise_for_status()
        say(OK, f"Allowed Client Scopes policy permits {REGISTRABLE_SCOPES}" + (f" (added {added})" if added else ""))
        return
    say(BAD, "anonymous 'Allowed Client Scopes' policy not found")


# ------------------------------------------------------------------ clients and users


def ensure_test_client(a: Admin) -> str | None:
    clients = a.get(f"/{REALM}/clients", params={"clientId": TEST_CLIENT}).json()
    if clients:
        cid = clients[0]["id"]
        say(INFO, f"client '{TEST_CLIENT}' already exists")
    else:
        secret = secrets.token_urlsafe(32)
        r = a.post(
            f"/{REALM}/clients",
            {
                "clientId": TEST_CLIENT,
                "enabled": True,
                "publicClient": False,
                "serviceAccountsEnabled": True,
                "standardFlowEnabled": False,
                "directAccessGrantsEnabled": False,
                "secret": secret,
                "defaultClientScopes": ["basic", "acr", SCOPE],
                "optionalClientScopes": [],
            },
        )
        r.raise_for_status()
        cid = a.get(f"/{REALM}/clients", params={"clientId": TEST_CLIENT}).json()[0]["id"]
        say(OK, f"created service-account client '{TEST_CLIENT}'")

    got = a.get(f"/{REALM}/clients/{cid}/client-secret").json().get("value")
    return got


def ensure_human_user(a: Admin) -> str | None:
    users = a.get(f"/{REALM}/users", params={"username": HUMAN_USER, "exact": "true"}).json()
    if users:
        say(INFO, f"user '{HUMAN_USER}' already exists")
        return None
    alphabet = string.ascii_letters + string.digits
    password = "".join(secrets.choice(alphabet) for _ in range(20))
    r = a.post(
        f"/{REALM}/users",
        {
            "username": HUMAN_USER,
            "enabled": True,
            "emailVerified": True,
            "email": "cagataytasdeviren@gmail.com",
            "credentials": [{"type": "password", "value": password, "temporary": False}],
            "requiredActions": [],
        },
    )
    r.raise_for_status()
    say(OK, f"created realm user '{HUMAN_USER}'")
    return password


# ------------------------------------------------------------------------ verification


def _b64(seg: str) -> dict:
    return json.loads(base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4)))


def verify(secret: str | None, groups: list[str], extra_hosts: list[str]) -> bool:
    ok = True
    c = httpx.Client(timeout=30.0)

    r = c.get(f"{KC}/realms/{REALM}/.well-known/openid-configuration")
    if r.status_code == 200:
        say(OK, "realm discovery document reachable")
    else:
        say(BAD, f"realm discovery {r.status_code} — realm still missing")
        return False

    reg = f"{KC}/realms/{REALM}/clients-registrations/openid-connect"

    # 1. Client-identical anonymous DCR, once per enabled host group. The `scope` field
    #    is essential: without it the probe passes while the real client still fails on
    #    the Allowed Client Scopes policy.
    payload = {
        "client_name": "rebuild-probe",
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "client_secret_post",
        # Deliberately the superset an OIDC-style client sends, not the bare `bddk.read`
        # Claude sends: one unlisted scope fails the whole registration, so probe the
        # widest payload a real client might use.
        "scope": " ".join(REGISTRABLE_SCOPES),
    }
    probes = [(g, HOST_GROUPS[g][1]) for g in groups]
    probes += [(h, f"http://{h}:3000/oauth/callback") for h in extra_hosts]
    for group, redirect in probes:
        r = c.post(reg, json={**payload, "redirect_uris": [redirect]})
        if r.status_code == 201 and SCOPE in (r.json().get("scope") or ""):
            body = r.json()
            say(OK, f"{group:8s} DCR ({redirect}) -> 201 with scope={body['scope']}")
            # Clean up the probe client using its registration access token.
            c.delete(
                f"{reg}/{body['client_id']}",
                headers={"Authorization": f"Bearer {body['registration_access_token']}"},
            )
        else:
            ok = False
            say(BAD, f"{group:8s} DCR ({redirect}) -> {r.status_code}: {r.text[:300]}")

    # 2. Negative control: an untrusted redirect URI must still be rejected.
    r = c.post(reg, json={**payload, "redirect_uris": ["https://evil.example/cb"]})
    if r.status_code == 403:
        say(OK, "untrusted redirect URI still rejected (403)")
    else:
        ok = False
        say(BAD, f"expected 403 for untrusted redirect URI, got {r.status_code}")

    # 3. Real token from the test client, checked against what the server enforces.
    if secret:
        r = c.post(
            f"{KC}/realms/{REALM}/protocol/openid-connect/token",
            data={"grant_type": "client_credentials", "client_id": TEST_CLIENT, "client_secret": secret},
        )
        if r.status_code == 200:
            tok = r.json()["access_token"]
            head, body = _b64(tok.split(".")[0]), _b64(tok.split(".")[1])
            aud = body.get("aud")
            aud_ok = AUDIENCE == aud or (isinstance(aud, list) and AUDIENCE in aud)
            scope_ok = SCOPE in (body.get("scope") or "")
            say(OK if aud_ok else BAD, f"token aud={aud} (need {AUDIENCE})")
            say(OK if scope_ok else BAD, f"token scope={body.get('scope')!r}")
            say(INFO, f"token typ={head.get('typ')} (server accepts at+jwt,JWT)")
            ok = ok and aud_ok and scope_ok

            # 4. End-to-end against the live MCP server.
            unauth = c.post(MCP_URL, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
            auth = c.post(
                MCP_URL,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "rebuild-probe", "version": "1"},
                    },
                },
                headers={
                    "Authorization": f"Bearer {tok}",
                    "Accept": "application/json, text/event-stream",
                },
            )
            say(OK if unauth.status_code == 401 else BAD, f"unauthenticated /mcp -> {unauth.status_code} (want 401)")
            if auth.status_code == 401:
                ok = False
                say(BAD, f"authenticated /mcp -> 401: {auth.text[:200]}")
            else:
                say(OK, f"authenticated /mcp -> {auth.status_code}")
        else:
            ok = False
            say(BAD, f"client_credentials token -> {r.status_code}: {r.text[:200]}")

    # 5. The MCP server has a second, independent allowlist: BDDK_HTTP_ALLOWED_ORIGINS.
    #    A client whose browser sends an unlisted Origin is refused before auth is even
    #    considered, so a perfectly rebuilt realm still yields a broken connector.
    #    401 here means the Origin passed and only the bearer token was missing; 403
    #    means the Origin itself was rejected.
    for group, origin in (("claude", "https://claude.ai"), ("chatgpt", "https://chatgpt.com")):
        if group not in groups:
            continue
        r = c.post(
            MCP_URL,
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            headers={"Origin": origin, "Accept": "application/json, text/event-stream"},
        )
        if r.status_code == 403:
            ok = False
            say(BAD, f"Origin {origin} rejected by the MCP server (403) — add it to BDDK_HTTP_ALLOWED_ORIGINS")
        else:
            say(OK, f"Origin {origin} accepted by the MCP server ({r.status_code})")

    return ok


def export_realm(a: Admin) -> None:
    r = a.post(
        f"/{REALM}/partial-export",
        {},
        params={"exportGroupsAndRoles": "true", "exportClients": "true"},
    )
    if r.status_code != 200:
        say(BAD, f"partial export failed: {r.status_code}")
        return
    EXPORT_PATH.write_text(json.dumps(r.json(), indent=2, ensure_ascii=False) + "\n")
    say(OK, f"realm exported to {EXPORT_PATH.relative_to(Path.cwd())} (no secrets, no users)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument(
        "--clients",
        default="claude,chatgpt,local",
        help=f"comma-separated host groups to trust: {', '.join(HOST_GROUPS)} (default: all)",
    )
    ap.add_argument(
        "--extra-hosts",
        default="",
        help="comma-separated extra redirect-URI hosts to trust, e.g. a LAN IP running open-webui "
        "(192.168.1.50) or a reverse-proxied hostname (webui.example.com)",
    )
    args = ap.parse_args()

    extra_hosts = [h.strip() for h in args.extra_hosts.split(",") if h.strip()]

    groups = [g.strip() for g in args.clients.split(",") if g.strip()]
    unknown = [g for g in groups if g not in HOST_GROUPS]
    if unknown:
        print(f"unknown client group(s): {unknown}; choose from {list(HOST_GROUPS)}")
        return 2

    password = os.environ.get("KC_ADMIN_PASSWORD")
    if not password:
        print("KC_ADMIN_PASSWORD is not set (Railway var KC_BOOTSTRAP_ADMIN_PASSWORD on the keycloak service)")
        return 2

    a = Admin(password)
    say(OK, "authenticated to Keycloak master realm")

    if args.verify_only:
        # A missing realm makes this a 404 error object rather than a list, so guard the
        # lookup and let verify() report the absent realm cleanly.
        clients = a.get(f"/{REALM}/clients", params={"clientId": TEST_CLIENT}).json()
        secret = None
        if isinstance(clients, list) and clients:
            secret = a.get(f"/{REALM}/clients/{clients[0]['id']}/client-secret").json().get("value")
        return 0 if verify(secret, groups, extra_hosts) else 1

    print("\n— rebuilding realm —")
    ensure_realm(a)
    ensure_client_scope(a)
    fix_trusted_hosts(a, groups, extra_hosts)
    fix_allowed_scopes(a)
    secret = ensure_test_client(a)
    new_password = ensure_human_user(a)

    print("\n— verifying —")
    ok = verify(secret, groups, extra_hosts)

    print("\n— exporting —")
    export_realm(a)

    if new_password:
        print(f"\n  NEW LOGIN for realm user '{HUMAN_USER}': {new_password}")
        print("  (save it now — this is the only time it is printed)")
    if secret:
        print(f"  {TEST_CLIENT} client secret: {secret}")

    print(f"\n{'REALM REBUILT AND VERIFIED' if ok else 'REBUILD INCOMPLETE — see failures above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
