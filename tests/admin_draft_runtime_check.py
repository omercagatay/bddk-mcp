"""Runtime check called by the explicitly opted-in disposable LOGIN integration lane."""

import base64

import httpx

from bddk_mcp.admin.runtime import build_app_from_env
from bddk_mcp.store.doc_store import DocumentStore
from tests.conftest import SingleConnPool
from tests.test_admin_drafts import ReadOnlyStore, canonical_json, form_fields, key_files


async def check_public_admin_drafts(admin, public_dsn, tmp_path):
    # Seed as owner, then leave both PostgreSQL roles and canonical bytes untouched.
    doc = ReadOnlyStore().doc
    await DocumentStore(SingleConnPool(admin)).store_document(doc)
    before = await admin.fetchrow("SELECT * FROM public.documents WHERE document_id = 'doc'")
    epoch = await admin.fetchval("SELECT epoch FROM bddk_meta.corpus_state_epoch")
    key, private, public = key_files(tmp_path)
    draft_dir = tmp_path / "drafts"
    draft_dir.mkdir()
    env = {
        "BDDK_DATABASE_URL": public_dsn,
        "BDDK_ADMIN_DRAFT_DB": str(draft_dir / "draft.sqlite"),
        "BDDK_ADMIN_SIGNING_KEY": str(private),
        "BDDK_ADMIN_SIGNING_PUBLIC_KEY": str(public),
    }
    for revision in (0, 1):
        app, _, shutdown = await build_app_from_env(env)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://127.0.0.1") as client:
                fields = form_fields(await client.get("/documents/doc/edit"))
                assert fields["revision"] == str(revision)
                fields["title"] = "Runtime editorial draft"
                saved = await client.post("/documents/doc/edit", data=fields, headers={"origin": "http://127.0.0.1"})
                assert saved.status_code == 303
                fields = form_fields(await client.get("/documents/doc"))
                signed = await client.post("/documents/doc/sign", data=fields, headers={"origin": "http://127.0.0.1"})
                assert signed.status_code == 303
                artifact = (await client.get("/documents/doc/draft.json")).json()
                key.public_key().verify(
                    base64.b64decode(artifact["signature"]["signature_base64"]), canonical_json(artifact["payload"])
                )
        finally:
            await shutdown()
    assert await admin.fetchrow("SELECT * FROM public.documents WHERE document_id = 'doc'") == before
    assert await admin.fetchval("SELECT epoch FROM bddk_meta.corpus_state_epoch") == epoch
