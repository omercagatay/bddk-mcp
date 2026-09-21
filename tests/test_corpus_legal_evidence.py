"""Signed legal evidence must be represented exactly, never admitted by row count alone."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from bddk_mcp.corpus_manifest import CorpusArtifact
from bddk_mcp.regulatory.legal_versions import LegalVersionBundle, canonical_bundle_sha256

FIXTURE = Path(__file__).parent / "fixtures/legal_versions/synthetic_one_family.json"


def write_materialized_legal_corpus(tmp_path, *, scope_reviewed_at="2026-07-16T00:00:00Z", bound=False):
    """Generate ephemeral, signed synthetic material; never use the project signing key."""
    import hashlib
    from datetime import datetime

    import yaml
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from bddk_mcp.corpus_manifest import canonical_manifest_payload, canonical_manifest_sha256
    from bddk_mcp.regulatory.legal_versions import (
        artifact_id_for,
        blob_id_for,
        event_id_for,
        evidence_id_for,
        status_assertion_id_for,
    )
    from tests.test_corpus_manifest import _trusted_key_path, _write_manifest

    manifest_path = _write_manifest(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["freshness"]["scope_reviewed_at"] = scope_reviewed_at
    bundle = portable_mapping()
    bindings = []
    if bound:
        from bddk_mcp.ingest.seed import _generate_seed_chunks
        from bddk_mcp.store.section_index import extract_document_sections
        from bddk_mcp.store.vector_store import VectorStore
        from tests.test_legal_versions import _trusted_citation_mapping

        text = "MADDE 1 - Sentetik ve yalnız test amaçlı tarihli hüküm."
        bundle = _trusted_citation_mapping(document_id="doc-1", section_content=text)
        section = extract_document_sections("doc-1", text)[0]
        version = bundle["versions"][0]
        bindings = [
            {
                "legal_version_id": version["legal_version_id"],
                "provision_id": version["provisions"][0]["provision_id"],
                "document_id": "doc-1",
                "document_sha256": version["legal_text_sha256"],
                "section_type": section.section_type,
                "section_ref": section.section_ref,
                "start_char": section.start_char,
                "end_char": section.end_char,
                "content_sha256": section.content_hash,
            }
        ]
        document = {
            "document_id": "doc-1",
            "title": "Synthetic signed legal evidence test",
            "category": "test",
            "decision_date": "",
            "decision_number": "",
            "source_url": "https://example.invalid/synthetic",
            "markdown_content": text,
            "content_hash": version["legal_text_sha256"],
            "downloaded_at": 1767225600.0,
            "extracted_at": 1767398400.0,
            "extraction_method": "synthetic-test",
            "total_pages": 1,
            "file_size": len(text.encode()),
        }
        cache = {
            k: document[k]
            for k in ("document_id", "title", "category", "decision_date", "decision_number", "source_url")
        }
        cache["content"] = text
        chunks, _ = _generate_seed_chunks(VectorStore(None), [document])
        for name, value in (("documents.json", [document]), ("decision_cache.json", [cache]), ("chunks.json", chunks)):
            raw = json.dumps(value).encode()
            (tmp_path / name).write_bytes(raw)
            item = next(a for a in manifest["artifacts"] if a["path"] == name)
            item.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), records=len(value))
        manifest["freshness"].update(
            source_observed_end=manifest["freshness"]["source_observed_start"],
            source_detection_slo_seconds=604800,
            publication_slo_seconds=1209600,
            max_manifest_age_seconds=31536000,
        )
    materials = {"review.txt": b"Synthetic test review, not actual legal approval."}
    review_hash = hashlib.sha256(materials["review.txt"]).hexdigest()
    blob_ids = {}
    for i, blob in enumerate(bundle["blobs"]):
        raw = f"Synthetic source {i}, not actual legislation.".encode()
        materials[f"source-{i}.txt"] = raw
        digest = hashlib.sha256(raw).hexdigest()
        new_id = blob_id_for(content_sha256=digest)
        blob_ids[blob["blob_id"]] = new_id
        blob.update(blob_id=new_id, content_sha256=digest)
    artifact_ids = {}
    for artifact in bundle["artifacts"]:
        artifact["blob_id"] = blob_ids[artifact["blob_id"]]
        new_id = artifact_id_for(
            blob_id=artifact["blob_id"],
            canonical_uri=artifact["canonical_uri"],
            retrieved_at=datetime.fromisoformat(artifact["retrieved_at"]),
        )
        artifact_ids[artifact["artifact_id"]] = new_id
        artifact["artifact_id"] = new_id
    for version in bundle["versions"]:
        version["source_artifact_ids"] = sorted(artifact_ids[a] for a in version["source_artifact_ids"])
        claims = [c for c in version["events"].values() if c] + version["status_assertions"] + version["provisions"]
        for item in [version, *claims]:
            if item["validation"]["review_record_sha256"]:
                item["validation"]["review_record_sha256"] = review_hash
        for item in claims:
            evidence = item["evidence"]
            evidence["artifact_id"] = artifact_ids[evidence["artifact_id"]]
            evidence["evidence_id"] = evidence_id_for(
                **{k: evidence[k] for k in ("artifact_id", "locator", "statement_sha256", "authority_level")}
            )
            if "event_id" in item:
                item["event_id"] = event_id_for(
                    legal_version_id=version["legal_version_id"],
                    event_type=item["event_type"],
                    event_date=datetime.fromisoformat(item["event_date"]).date(),
                    evidence_id=evidence["evidence_id"],
                    target_legal_version_id=item["target_legal_version_id"],
                )
            if "assertion_id" in item:
                item["assertion_id"] = status_assertion_id_for(
                    legal_version_id=version["legal_version_id"],
                    status=item["status"],
                    valid_from=datetime.fromisoformat(item["valid_from"]).date(),
                    valid_through=datetime.fromisoformat(item["valid_through"]).date(),
                    evidence_id=evidence["evidence_id"],
                )
        version["status_assertions"].sort(key=lambda x: x["assertion_id"])
    bundle["blobs"].sort(key=lambda x: x["blob_id"])
    bundle["artifacts"].sort(key=lambda x: x["artifact_id"])
    bundle["bundle_sha256"] = canonical_bundle_sha256(bundle)
    package = {"schema_version": 1, "bundles": [bundle], "bindings": bindings}
    materials["legal_evidence.json"] = json.dumps(package).encode()
    for name, raw in materials.items():
        (tmp_path / name).write_bytes(raw)
        manifest["artifacts"].append(
            {
                "role": "legal_evidence" if name == "legal_evidence.json" else "other",
                "path": name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
        )
    key = Ed25519PrivateKey.generate()
    public = key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    trust = _trusted_key_path(tmp_path)
    trust.write_bytes(public)
    manifest["integrity"].update(
        signature_status="verified",
        signature_algorithm="ed25519",
        signature_reference="corpus_scope.sig",
        signature_public_key_sha256=hashlib.sha256(public).hexdigest(),
    )
    manifest["integrity"]["manifest_sha256"] = canonical_manifest_sha256(manifest)
    (tmp_path / "corpus_scope.sig").write_bytes(key.sign(canonical_manifest_payload(manifest)))
    manifest_path.write_text(yaml.safe_dump(manifest))
    return manifest_path, trust


def fixture_bundle():
    return LegalVersionBundle.model_validate(json.loads(FIXTURE.read_text()))


def portable_mapping():
    """Synthetic data only; never copied into a published corpus."""
    mapping = json.loads(FIXTURE.read_text())
    mapping["fixture_only"] = False
    for artifact in mapping["artifacts"]:
        artifact["fixture_only"] = False
    mapping["bundle_sha256"] = canonical_bundle_sha256(mapping)
    return mapping


def test_manifest_can_bind_one_explicit_legal_evidence_artifact():
    artifact = CorpusArtifact(role="legal_evidence", path="legal_evidence.json", sha256="a" * 64, bytes=100, records=1)
    assert artifact.role == "legal_evidence"


def test_legal_projection_covers_all_columns_and_preserves_review_provenance():
    from bddk_mcp.regulatory.repository import _COLUMN_TYPES, project_bundle_rows

    bundle = fixture_bundle()
    rows = project_bundle_rows(bundle, imported_by="signed-corpus")
    assert set(rows) == set(_COLUMN_TYPES)
    for table, records in rows.items():
        for record in records:
            assert set(record) == set(_COLUMN_TYPES[table]), table
    assert rows["public.regulatory_relations"] == []
    assert len(rows["public.regulatory_legal_versions"]) == 2
    assert rows["public.regulatory_legal_versions"][0]["review_record_sha256"] == (
        bundle.versions[0].validation.review_record_sha256
    )
    receipt = rows["public.regulatory_family_imports"][0]
    assert receipt["bundle_sha256"] == bundle.bundle_sha256
    assert receipt["imported_by"] == "signed-corpus"
    assert receipt["predecessor_bundle_sha256"] is None
    assert json.loads(receipt["member_manifest"])["schema_version"] == 1


def test_production_legal_package_refuses_fixture_evidence():
    from bddk_mcp.regulatory.corpus_evidence import LegalEvidencePackage

    with pytest.raises(ValidationError, match="fixture"):
        LegalEvidencePackage(schema_version=1, bundles=[fixture_bundle()], bindings=[])


def test_package_checksum_and_unique_instrument_are_enforced():
    from bddk_mcp.regulatory.corpus_evidence import LegalEvidencePackage

    mapping = portable_mapping()
    package = LegalEvidencePackage(schema_version=1, bundles=[mapping], bindings=[])
    assert len(package.bundles) == 1
    with pytest.raises(ValidationError, match="duplicate"):
        LegalEvidencePackage(schema_version=1, bundles=[mapping, mapping], bindings=[])
    mapping["bundle_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="checksum"):
        LegalEvidencePackage(schema_version=1, bundles=[mapping], bindings=[])


def test_package_refuses_database_local_ids_even_in_checksummed_bundle():
    from bddk_mcp.regulatory.corpus_evidence import LegalEvidencePackage

    mapping = portable_mapping()
    mapping["versions"][0]["provisions"][0]["document_section_id"] = 42
    mapping["bundle_sha256"] = canonical_bundle_sha256(mapping)
    with pytest.raises(ValidationError, match="database-local"):
        LegalEvidencePackage(schema_version=1, bundles=[mapping], bindings=[])


def bound_package_mapping():
    mapping = portable_mapping()
    version = mapping["versions"][0]
    occurrence = version["provisions"][0]
    for artifact in mapping["artifacts"]:
        if artifact["artifact_id"] == occurrence["evidence"]["artifact_id"]:
            artifact["repository_document_id"] = "mapped-document"
    mapping["bundle_sha256"] = canonical_bundle_sha256(mapping)
    binding = {
        "legal_version_id": version["legal_version_id"],
        "provision_id": occurrence["provision_id"],
        "document_id": "mapped-document",
        "document_sha256": version["legal_text_sha256"],
        "section_type": "madde",
        "section_ref": "9",
        "start_char": 0,
        "end_char": 20,
        "content_sha256": occurrence["provision_text_sha256"],
    }
    return {"schema_version": 1, "bundles": [mapping], "bindings": [binding]}


def test_binding_must_cover_the_exact_document_version_and_provision():
    from bddk_mcp.regulatory.corpus_evidence import LegalEvidencePackage

    data = bound_package_mapping()
    assert len(LegalEvidencePackage.model_validate(data).bindings) == 1
    with pytest.raises(ValidationError, match="binding"):
        LegalEvidencePackage.model_validate({**data, "bindings": []})
    for field, value in (
        ("document_id", "other-document"),
        ("document_sha256", "0" * 64),
        ("content_sha256", "0" * 64),
    ):
        binding = {**data["bindings"][0], field: value}
        with pytest.raises(ValidationError, match="binding"):
            LegalEvidencePackage.model_validate({**data, "bindings": [binding]})
    with pytest.raises(ValidationError, match="duplicate"):
        LegalEvidencePackage.model_validate({**data, "bindings": data["bindings"] * 2})


def test_validated_package_and_bindings_cannot_be_mutated_after_admission():
    from bddk_mcp.regulatory.corpus_evidence import LegalEvidencePackage

    package = LegalEvidencePackage.model_validate(bound_package_mapping())
    with pytest.raises(ValidationError, match="frozen"):
        package.bindings = ()
    with pytest.raises(ValidationError, match="frozen"):
        package.bindings[0].document_id = "different-document"


def test_binding_refuses_boolean_offsets_reversed_spans_and_unknown_fields():
    from bddk_mcp.regulatory.corpus_evidence import LegalEvidencePackage

    data = bound_package_mapping()
    for changes in ({"start_char": True}, {"end_char": 0}, {"arbitrary_sql": "SELECT 1"}):
        with pytest.raises(ValidationError):
            LegalEvidencePackage.model_validate({**data, "bindings": [{**data["bindings"][0], **changes}]})


@pytest.mark.asyncio
async def test_portable_binding_resolves_local_id_and_derives_a_new_checksum_without_mutating_input():
    from bddk_mcp.regulatory.corpus_evidence import LegalEvidencePackage, resolve_legal_bundles

    package = LegalEvidencePackage.model_validate(bound_package_mapping())
    before = package.model_dump_json()

    class Connection:
        async def fetch(self, query, *args):
            assert "public.document_sections" in query and "public.documents" in query
            assert args[0] == "mapped-document"
            return [{"id": 137}]

    resolved = await resolve_legal_bundles(Connection(), package)
    assert package.model_dump_json() == before
    assert resolved[0].versions[0].provisions[0].document_section_id == 137
    assert resolved[0].bundle_sha256 == canonical_bundle_sha256(resolved[0])
    assert resolved[0].bundle_sha256 != package.bundles[0].bundle_sha256
    assert resolved[0].versions[0].validation == package.bundles[0].versions[0].validation


@pytest.mark.asyncio
@pytest.mark.parametrize("rows", [[], [{"id": 1}, {"id": 2}]])
async def test_binding_missing_or_ambiguous_abstains_before_import(rows):
    from bddk_mcp.regulatory.corpus_evidence import LegalEvidencePackage, resolve_legal_bundles

    class Connection:
        async def fetch(self, query, *args):
            return rows

    with pytest.raises(RuntimeError, match="exact section"):
        await resolve_legal_bundles(Connection(), LegalEvidencePackage.model_validate(bound_package_mapping()))


@pytest.mark.asyncio
async def test_exact_legal_membership_rejects_changed_review_missing_and_extra_rows():
    from bddk_mcp.regulatory.corpus_evidence import (
        LegalEvidencePackage,
        assert_legal_evidence_membership,
    )
    from bddk_mcp.regulatory.repository import project_bundle_rows

    package = LegalEvidencePackage(schema_version=1, bundles=[portable_mapping()], bindings=[])
    expected = project_bundle_rows(package.bundles[0], imported_by="signed-corpus")

    class Connection:
        async def fetch(self, query, *args):
            table = query.split("FROM ", 1)[1].split()[0]
            return expected[table]

    connection = Connection()
    await assert_legal_evidence_membership(connection, package)
    version = expected["public.regulatory_legal_versions"][0]
    original = version["review_record_sha256"]
    version["review_record_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="not exactly represented"):
        await assert_legal_evidence_membership(connection, package)
    version["review_record_sha256"] = original
    missing = expected["public.regulatory_evidence"].pop()
    with pytest.raises(RuntimeError, match="not exactly represented"):
        await assert_legal_evidence_membership(connection, package)
    expected["public.regulatory_evidence"].append(missing)
    expected["public.regulatory_relations"] = [{"relation_id": "unrepresented"}]
    with pytest.raises(RuntimeError, match="not exactly represented"):
        await assert_legal_evidence_membership(connection, package)


def test_every_source_blob_and_completed_review_must_be_in_the_signed_material_inventory():
    from bddk_mcp.regulatory.corpus_evidence import LegalEvidencePackage, assert_material_coverage
    from bddk_mcp.regulatory.repository import _review_records

    package = LegalEvidencePackage(schema_version=1, bundles=[portable_mapping()], bindings=[])
    bundle = package.bundles[0]
    hashes = {blob.content_sha256 for blob in bundle.blobs}
    hashes.update(review.review_record_sha256 for review in _review_records(bundle) if review.review_record_sha256)
    artifacts = [
        CorpusArtifact(role="other", path=f"legal/{i}.bin", sha256=h, bytes=30) for i, h in enumerate(sorted(hashes))
    ]
    assert_material_coverage(package, artifacts)
    for missing in range(len(artifacts)):
        with pytest.raises(RuntimeError, match="source or review"):
            assert_material_coverage(package, artifacts[:missing] + artifacts[missing + 1 :])
    with pytest.raises(RuntimeError, match="source or review"):
        assert_material_coverage(package, [a.model_copy(update={"role": "documents"}) for a in artifacts])


def test_optional_legal_loading_requires_a_verified_signature_and_rechecks_artifact_bytes(tmp_path):
    import hashlib
    from types import SimpleNamespace

    from bddk_mcp.regulatory.corpus_evidence import load_legal_evidence

    payload = b'{"schema_version":1,"bundles":[],"bindings":[]}'
    (tmp_path / "legal_evidence.json").write_bytes(payload)
    artifact = CorpusArtifact(
        role="legal_evidence",
        path="legal_evidence.json",
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
    )
    empty = SimpleNamespace(signature_sha256=None, manifest=SimpleNamespace(artifacts=[]))
    assert load_legal_evidence(tmp_path, empty) is None
    unsigned = SimpleNamespace(signature_sha256=None, manifest=SimpleNamespace(artifacts=[artifact]))
    with pytest.raises(RuntimeError, match="verified signature"):
        load_legal_evidence(tmp_path, unsigned)
    signed = SimpleNamespace(signature_sha256="a" * 64, manifest=SimpleNamespace(artifacts=[artifact]))
    (tmp_path / "legal_evidence.json").write_bytes(payload.replace(b"1", b"2"))
    with pytest.raises(RuntimeError, match="changed"):
        load_legal_evidence(tmp_path, signed)


def test_legal_loading_rejects_duplicate_json_keys_even_when_bytes_match_manifest(tmp_path):
    import hashlib
    from types import SimpleNamespace

    from bddk_mcp.regulatory.corpus_evidence import load_legal_evidence

    payload = b'{"schema_version":1,"schema_version":1,"bundles":[],"bindings":[]}'
    (tmp_path / "legal_evidence.json").write_bytes(payload)
    artifact = CorpusArtifact(
        role="legal_evidence",
        path="legal_evidence.json",
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
    )
    validation = SimpleNamespace(signature_sha256="a" * 64, manifest=SimpleNamespace(artifacts=[artifact]))
    with pytest.raises(RuntimeError, match="schema validation"):
        load_legal_evidence(tmp_path, validation)


@pytest.mark.asyncio
async def test_signed_legal_package_is_checked_inside_full_seed_membership():
    from bddk_mcp.ingest.seed import _assert_strict_seed_membership
    from bddk_mcp.regulatory.corpus_evidence import LegalEvidencePackage
    from bddk_mcp.regulatory.repository import project_bundle_rows

    package = LegalEvidencePackage(schema_version=1, bundles=[portable_mapping()], bindings=[])
    rows = project_bundle_rows(package.bundles[0], imported_by="signed-corpus")

    class Connection:
        async def fetch(self, query, *args):
            table = query.split("FROM ", 1)[1].split()[0]
            return rows.get(table, [])

        async def fetchval(self, query, *args):
            return sum(map(len, rows.values())) if "regulatory_instruments" in query else 0

    args = dict(
        expected_documents=[],
        expected_cache=[],
        expected_chunks=[],
        expected_embeddings=[],
        expected_sections={},
        retrieval_profile_sha256="7" * 64,
    )
    await _assert_strict_seed_membership(Connection(), **args, expected_legal_evidence=package)
    with pytest.raises(RuntimeError, match="not exactly represented"):
        await _assert_strict_seed_membership(Connection(), **args)
    rows["public.regulatory_source_blobs"][0]["content_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="not exactly represented"):
        await _assert_strict_seed_membership(Connection(), **args, expected_legal_evidence=package)


@pytest.mark.asyncio
async def test_postgres_signed_package_import_is_idempotent_and_exact(pg_pool):
    from bddk_mcp.regulatory.corpus_evidence import (
        LegalEvidencePackage,
        assert_legal_evidence_membership,
        import_legal_evidence,
    )

    package = LegalEvidencePackage(schema_version=1, bundles=[portable_mapping()], bindings=[])
    async with pg_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        try:
            await import_legal_evidence(connection, package)
            epoch = await connection.fetchval("SELECT epoch FROM bddk_meta.corpus_state_epoch")
            await import_legal_evidence(connection, package)
            assert await connection.fetchval("SELECT epoch FROM bddk_meta.corpus_state_epoch") == epoch
            await assert_legal_evidence_membership(connection, package)
            assert await connection.fetchval("SELECT count(*) FROM public.regulatory_family_imports") == 1
            changed = await connection.execute(
                "UPDATE public.regulatory_legal_versions SET validated_by='changed-reviewer' "
                "WHERE validation_state='validated'"
            )
            assert changed == "UPDATE 1"
            with pytest.raises(RuntimeError, match="not exactly represented"):
                await assert_legal_evidence_membership(connection, package)
        finally:
            await transaction.rollback()


@pytest.mark.asyncio
async def test_postgres_legal_import_rolls_back_all_prior_rows_on_late_failure(pg_pool):
    from bddk_mcp.regulatory.corpus_evidence import LegalEvidencePackage, import_legal_evidence

    package = LegalEvidencePackage(schema_version=1, bundles=[portable_mapping()], bindings=[])
    async with pg_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        try:
            await connection.execute("""
                CREATE FUNCTION pg_temp.reject_legal_receipt() RETURNS trigger LANGUAGE plpgsql AS $$
                BEGIN RAISE EXCEPTION 'injected late failure'; END $$;
                CREATE TRIGGER test_reject_legal_receipt BEFORE INSERT ON public.regulatory_family_imports
                FOR EACH ROW EXECUTE FUNCTION pg_temp.reject_legal_receipt();
            """)
            with pytest.raises(RuntimeError):
                await import_legal_evidence(connection, package)
            assert await connection.fetchval("SELECT count(*) FROM public.regulatory_instruments") == 0
            assert await connection.fetchval("SELECT count(*) FROM public.regulatory_legal_versions") == 0
        finally:
            await transaction.rollback()


def test_owner_import_cli_requires_explicit_trust_anchor_and_dispatches(monkeypatch, tmp_path, capsys):
    from bddk_mcp import cli

    calls = []

    async def run(dsn, seed_dir, **kwargs):
        calls.append((dsn, seed_dir, kwargs))
        return {"legal_families": 1, "section_bindings": 1}

    monkeypatch.setattr(cli, "_import_legal_evidence", run, raising=False)
    cli.main(
        [
            "import-legal-evidence",
            "--seed-dir",
            str(tmp_path),
            "--trusted-signing-key",
            str(tmp_path / "outside.pem"),
            "--accept-unmeasured-freshness",
        ]
    )
    assert calls == [
        (None, tmp_path, {"trusted_signing_key": tmp_path / "outside.pem", "accept_unmeasured_freshness": True})
    ]
    assert json.loads(capsys.readouterr().out)["legal_families"] == 1
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["import-legal-evidence"])
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["import-legal-evidence", "--trusted-signing-key", "trust.pem", "--allow-fixture"]
        )


@pytest.mark.asyncio
async def test_owner_import_refuses_wrong_identity_before_writes(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from bddk_mcp import cli

    validation = SimpleNamespace(manifest_sha256="a" * 64)
    package = SimpleNamespace(bundles=(), bindings=())
    pool = MagicMock()
    pool.close = AsyncMock()
    create_pool = AsyncMock(return_value=pool)
    owner_guard = AsyncMock(side_effect=RuntimeError("wrong actor"))
    importer = AsyncMock()
    monkeypatch.setattr("bddk_mcp.core.config.require_expected_database_name", lambda: "bddk_test")
    monkeypatch.setattr("bddk_mcp.db_transport.assert_database_transport", lambda value: value)
    monkeypatch.setattr("bddk_mcp.ingest.seed._manifest_seed_artifacts", lambda *a, **kw: (validation, {}))
    monkeypatch.setattr("bddk_mcp.regulatory.corpus_evidence.load_legal_evidence", lambda *a: package)
    monkeypatch.setattr("bddk_mcp.regulatory.corpus_evidence.import_legal_evidence", importer)
    monkeypatch.setattr("bddk_mcp.db_lifecycle.assert_schema_owner_identity", owner_guard)
    monkeypatch.setattr("asyncpg.create_pool", create_pool)
    with pytest.raises(RuntimeError, match="wrong actor"):
        await cli._import_legal_evidence(
            "postgresql://restricted", tmp_path, trusted_signing_key=tmp_path / "trust.pem"
        )
    owner_guard.assert_awaited_once_with(pool, "bddk_test")
    importer.assert_not_awaited()
    pool.close.assert_awaited_once()
    init = create_pool.await_args.kwargs["init"]
    assert init.func is owner_guard
    assert init.keywords == {"expected_database": "bddk_test"}


def test_real_signature_binds_legal_package_and_rechecks_source_material(tmp_path):
    from bddk_mcp.corpus_manifest import load_and_validate_corpus_manifest
    from bddk_mcp.regulatory.corpus_evidence import load_legal_evidence

    manifest, trust = write_materialized_legal_corpus(tmp_path)
    validation = load_and_validate_corpus_manifest(manifest, trusted_signing_key=trust, require_verified_signature=True)
    package = load_legal_evidence(tmp_path, validation)
    assert package is not None and len(package.bundles) == 1
    source = tmp_path / "source-0.txt"
    source.write_bytes(source.read_bytes().replace(b"Synthetic", b"Altered!!"))
    with pytest.raises(RuntimeError, match="changed"):
        load_legal_evidence(tmp_path, validation)


def test_signed_material_cannot_postdate_the_declared_scope_review(tmp_path):
    from bddk_mcp.corpus_manifest import load_and_validate_corpus_manifest
    from bddk_mcp.regulatory.corpus_evidence import load_legal_evidence

    manifest, trust = write_materialized_legal_corpus(tmp_path, scope_reviewed_at="2026-01-03T00:00:00Z")
    validation = load_and_validate_corpus_manifest(manifest, trusted_signing_key=trust, require_verified_signature=True)
    with pytest.raises(RuntimeError, match="scope review"):
        load_legal_evidence(tmp_path, validation)


def test_nested_families_keep_the_existing_bounded_bundle_limit(monkeypatch, tmp_path):
    from bddk_mcp.corpus_manifest import load_and_validate_corpus_manifest
    from bddk_mcp.regulatory import corpus_evidence

    manifest, trust = write_materialized_legal_corpus(tmp_path)
    validation = load_and_validate_corpus_manifest(manifest, trusted_signing_key=trust, require_verified_signature=True)
    monkeypatch.setattr(corpus_evidence, "_MAX_BUNDLE_BYTES", 100, raising=False)
    with pytest.raises(RuntimeError, match="family size limit"):
        corpus_evidence.load_legal_evidence(tmp_path, validation)


def test_seed_refuses_an_unmanifested_reserved_legal_artifact(tmp_path):
    from bddk_mcp.ingest.seed import _manifest_seed_artifacts
    from tests.test_corpus_manifest import _trusted_key_path, _write_manifest

    _write_manifest(tmp_path, signature_status="verified")
    (tmp_path / "legal_evidence.json").write_text("{}")
    with pytest.raises(RuntimeError, match="undeclared reserved"):
        _manifest_seed_artifacts(
            tmp_path,
            require_quantified_freshness=False,
            require_measured_freshness=False,
            require_verified_signature=True,
            trusted_signing_key=_trusted_key_path(tmp_path),
        )


@pytest.mark.asyncio
async def test_portable_import_reaches_dated_mcp_answer_without_exposing_legal_tables(pg_pool):
    from unittest.mock import MagicMock

    from mcp.shared.memory import create_connected_server_and_client_session

    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.regulatory.corpus_evidence import LegalEvidencePackage, import_legal_evidence
    from bddk_mcp.server import create_mcp
    from bddk_mcp.store.doc_store import DocumentStore
    from tests.test_legal_versions import _attach_repository_section, _PinnedPool, _trusted_citation_mapping

    document_id = "signed-package-mcp-test"
    text = "MADDE 1 - Sentetik ve yalnız test amaçlı tarihli hüküm."
    mapping = _trusted_citation_mapping(document_id=document_id, section_content=text)
    async with pg_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        try:
            await _attach_repository_section(connection, mapping, document_id=document_id, section_content=text)
            version = mapping["versions"][0]
            occurrence = version["provisions"][0]
            occurrence["document_section_id"] = None
            mapping["bundle_sha256"] = canonical_bundle_sha256(mapping)
            binding = {
                "legal_version_id": version["legal_version_id"],
                "provision_id": occurrence["provision_id"],
                "document_id": document_id,
                "document_sha256": version["legal_text_sha256"],
                "section_type": "article",
                "section_ref": "1",
                "start_char": 0,
                "end_char": len(text),
                "content_sha256": occurrence["provision_text_sha256"],
            }
            package = LegalEvidencePackage(schema_version=1, bundles=[mapping], bindings=[binding])
            await import_legal_evidence(connection, package)
            await connection.execute("CREATE ROLE bddk_dated_package_reader NOLOGIN")
            await connection.execute("GRANT USAGE ON SCHEMA public,bddk_meta TO bddk_dated_package_reader")
            await connection.execute("""
                GRANT SELECT ON public.documents,public.document_sections,public.regulatory_validated_section_citations
                TO bddk_dated_package_reader;
                GRANT EXECUTE ON FUNCTION bddk_meta.resolve_regulation_status(text,date) TO bddk_dated_package_reader;
                SET LOCAL ROLE bddk_dated_package_reader;
            """)
            assert not await connection.fetchval(
                "SELECT has_table_privilege(current_user,'public.regulatory_legal_versions','SELECT')"
            )
            pool = _PinnedPool(connection)
            deps = Dependencies(pool=pool, doc_store=DocumentStore(pool), client=MagicMock(), http=None)
            async with create_connected_server_and_client_session(create_mcp(deps)) as session:
                valid = await session.call_tool(
                    "get_document_section",
                    {
                        "document_id": document_id,
                        "section_ref": "1",
                        "as_of": "2024-06-30",
                        "quotation": text,
                    },
                )
                outside = await session.call_tool(
                    "get_document_section",
                    {
                        "document_id": document_id,
                        "section_ref": "1",
                        "as_of": "2025-06-30",
                        "quotation": text,
                    },
                )
            assert valid.isError is False and valid.structuredContent["status"] == "ok"
            assessment = valid.structuredContent["answer_assessment"]
            assert assessment["basis"] == "dated_version" and assessment["gaps"] == []
            assert assessment["scope_and_entailment"] == "not_assessed"
            assert assessment["quotation_status"] == "verified"
            assert {claim["role"] for claim in assessment["legal_evidence"]} >= {"publication", "effective", "status"}
            assert outside.structuredContent["status"] == "partial"
            assert outside.structuredContent["answer_assessment"]["basis"] != "dated_version"
        finally:
            await transaction.rollback()


def test_actual_source_candidate_does_not_invent_human_review_or_resolve_dates():
    from datetime import date

    from bddk_mcp.regulatory.corpus_evidence import LegalEvidencePackage
    from bddk_mcp.regulatory.legal_versions import ResolutionReason, resolve_as_of
    from bddk_mcp.regulatory.repository import _review_records

    path = (
        Path(__file__).parents[1] / "docs/evidence/legal-status/mevzuat-18799-2026-06-30/legal_evidence.candidate.json"
    )
    package = LegalEvidencePackage.model_validate_json(path.read_text())
    bundle = package.bundles[0]
    assert not bundle.fixture_only
    assert package.bindings[0].document_id == "mevzuat_18799"
    for review in _review_records(bundle):
        assert review.state.value == "in_review"
        assert review.validated_by is None and review.review_record_sha256 is None
    result = resolve_as_of(bundle, instrument_id=bundle.instrument.instrument_id, as_of=date(2026, 6, 30))
    assert not result.resolved
    assert result.reason is ResolutionReason.NO_VALIDATED_VERSION
