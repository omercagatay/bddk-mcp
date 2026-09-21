"""Full legal publication exercise for the explicitly disposable LOGIN contract lane.

Only the embedding encoder is synthetic. Signature verification, schema-owner
import, exact membership, verifier/publisher LOGINs, active release and MCP SDK
calls use their real implementations. This is not a legal review of real law.
"""

from urllib.parse import urlsplit

import asyncpg
from mcp.shared.memory import create_connected_server_and_client_session

from bddk_mcp import cli
from bddk_mcp.core import config
from bddk_mcp.core.deps import Dependencies
from bddk_mcp.ingest.client import BddkApiClient
from bddk_mcp.server import create_mcp
from bddk_mcp.store import vector_store
from bddk_mcp.store.doc_store import DocumentStore
from tests.test_corpus_legal_evidence import write_materialized_legal_corpus


async def exercise_signed_legal_publication(admin, dsns, tmp_path, monkeypatch):
    assert all(urlsplit(dsn).hostname in {"localhost", "127.0.0.1", "::1"} for dsn in dsns.values())
    database_name = await admin.fetchval("SELECT current_database()")
    assert database_name.startswith("bddk_role_contract")
    monkeypatch.setenv("BDDK_ALLOW_INSECURE_DATABASE", "true")
    monkeypatch.setattr(config, "EXPECTED_DATABASE_NAME", database_name)
    monkeypatch.setattr(vector_store, "EMBEDDING_CHUNK_MODE", "character")

    async def deterministic_embedding(self, texts, *, prefix):
        assert prefix == "passage"
        return [[1.0] + [0.0] * (vector_store.EMBEDDING_DIMENSION - 1) for _ in texts]

    monkeypatch.setattr(vector_store.VectorStore, "_embed", deterministic_embedding)
    root = tmp_path / "signed-legal-corpus"
    root.mkdir()
    _manifest, trust = write_materialized_legal_corpus(root, bound=True)
    imported = await cli._bootstrap(
        dsns["ingestion"],
        root,
        False,
        require_quantified_freshness=True,
        require_verified_signature=True,
        trusted_signing_key=trust,
    )
    assert imported["documents"] == 1
    prepared = await cli._import_legal_evidence(
        dsns["schema"],
        root,
        trusted_signing_key=trust,
        accept_unmeasured_freshness=True,
    )
    assert prepared["legal_families"] == 1 and prepared["section_bindings"] == 1
    assert prepared["release_publication_required"] is True
    assert await admin.fetchval("SELECT count(*) FROM bddk_meta.active_corpus_release") == 0
    staged = await cli._verify_and_stage_corpus_release(
        dsns["release-verifier"],
        root,
        trusted_signing_key=trust,
        verifier_revision_sha256="a" * 64,
        verifier_image_digest="sha256:" + "b" * 64,
        valid_for_seconds=900,
        accept_unmeasured_freshness=True,
    )
    assert staged["chunk_artifact_match"] is True
    activated = await cli._activate_corpus_release(
        dsns["release-publisher"],
        request_id=staged["corpus_release_request"]["request_id"],
    )
    release_id = activated["active_corpus_release"]["release_id"]
    assert release_id == await admin.fetchval("SELECT release_id FROM bddk_meta.active_corpus_release")

    public = await asyncpg.create_pool(dsns["public"], min_size=1, max_size=3)
    client = BddkApiClient(public)
    try:
        assert not await public.fetchval(
            "SELECT has_table_privilege(current_user,'public.regulatory_legal_versions','SELECT')"
        )
        deps = Dependencies(pool=public, doc_store=DocumentStore(public), client=client, http=None)
        async with create_connected_server_and_client_session(
            create_mcp(deps, require_active_corpus_release=True)
        ) as session:
            args = {"document_id": "doc-1", "section_ref": "1", "as_of": "2024-06-30"}
            result = await session.call_tool("get_document_section", args)
            assert not result.isError
            assert result.structuredContent["answer_assessment"]["basis"] == "dated_version"
            outside = await session.call_tool("get_document_section", {**args, "as_of": "2025-06-30"})
            assert not outside.isError
            assert outside.structuredContent["answer_assessment"]["basis"] != "dated_version"
            await admin.execute("UPDATE public.regulatory_legal_versions SET validated_by='tampered-after-activation'")
            assert await admin.fetchval("SELECT count(*) FROM bddk_meta.active_corpus_release") == 0
            denied = await session.call_tool("get_document_section", args)
            assert denied.isError
            assert "CORPUS_RELEASE_UNAVAILABLE" in str(denied.content)
    finally:
        await client.close()
        await public.close()
