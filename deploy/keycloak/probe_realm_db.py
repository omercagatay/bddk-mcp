#!/usr/bin/env python3
"""Inspect Keycloak's own Postgres to tell a realm *rename* from *deletion* from data *loss*.

All three look identical from outside (a 404 on the realm), but they need different
responses — and the rename is both the most likely and the least destructive:

* a realm carrying the known id under a different name -> someone renamed the realm.
  Nothing was lost; rename it back. Building a fresh realm here would strand the real
  one's clients and users.

* `keycloak` database missing, or present but empty, or a `master` realm whose
  `not_before` is minutes old -> the storage was lost (volume reset, DB dropped).
  Rebuilding the realm will not stop it happening again.
* `master` and other rows intact, only the `bddk` realm row gone -> something deleted
  that one realm. The rebuild is sufficient.

Usage (from the repo root — the DSN carries a password, so run it via `!`):

    KC_DB_PASSWORD='<Postgres service PGPASSWORD>' .venv/bin/python deploy/keycloak/probe_realm_db.py
"""

from __future__ import annotations

import asyncio
import os
import sys

import asyncpg

KNOWN_REALM_ID = "dc17d66a-c2fe-4065-98a0-801d1237083c"
HOST = "maglev.proxy.rlwy.net"
PORT = 15756
DSN = "postgresql://postgres:{pw}@{host}:{port}/{db}"


async def main() -> int:
    pw = os.environ.get("KC_DB_PASSWORD")
    if not pw:
        print("KC_DB_PASSWORD is not set (Railway var PGPASSWORD on the Postgres service)")
        return 2

    conn = await asyncpg.connect(DSN.format(pw=pw, host=HOST, port=PORT, db="railway"))
    databases = [r["datname"] for r in await conn.fetch("select datname from pg_database order by 1")]
    print(f"databases: {databases}")
    await conn.close()

    if "keycloak" not in databases:
        print("\n!! the `keycloak` database itself is gone — this is storage loss, not a deletion.")
        print("   Check the Postgres volume (RAILWAY_VOLUME_ID) before rebuilding anything.")
        return 1

    conn = await asyncpg.connect(DSN.format(pw=pw, host=HOST, port=PORT, db="keycloak"))
    try:
        realms = await conn.fetch("select id, name, enabled from realm order by name")
        if not realms:
            print("\n!! `realm` table is empty — storage loss.")
            return 1
        print("\nrealms:")
        for r in realms:
            print(f"  {r['name']:12s} enabled={r['enabled']}  id={r['id']}")

        clients = await conn.fetch(
            "select c.client_id, r.name as realm from client c join realm r on r.id = c.realm_id order by 2, 1"
        )
        print(f"\nclients ({len(clients)}):")
        for c in clients:
            print(f"  {c['realm']:12s} {c['client_id']}")

        users = await conn.fetch(
            "select u.username, r.name as realm from user_entity u join realm r on r.id = u.realm_id order by 2, 1"
        )
        print(f"\nusers ({len(users)}):")
        for u in users:
            print(f"  {u['realm']:12s} {u['username']}")

        names = {r["name"] for r in realms}
        renamed = next((r for r in realms if r["id"] == KNOWN_REALM_ID and r["name"] != "bddk"), None)
        print()
        if "bddk" in names:
            print("=> the `bddk` realm row EXISTS. If it still 404s, the problem is Keycloak-side, not data.")
        elif renamed:
            # A realm's name is its URL path segment; the admin console labels it
            # "Realm ID", directly above "Display name". Editing the wrong field renames
            # the realm and 404s the issuer while every client and user survives.
            print(f"=> the realm was RENAMED, not deleted: id {KNOWN_REALM_ID} is now named {renamed['name']!r}.")
            print("   Nothing was lost. Rename it back to `bddk` — do NOT build a fresh realm:")
            print("   .venv/bin/python deploy/keycloak/rebuild_realm.py   (handles the rename)")
        elif "master" in names and len(clients) > 0:
            print("=> `master` intact, no realm with the known id: a targeted deletion. Rebuild from scratch.")
        else:
            print("=> store looks freshly initialised: storage loss. Investigate the volume before rebuilding.")
    finally:
        await conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
