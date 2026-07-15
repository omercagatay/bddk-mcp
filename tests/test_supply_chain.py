"""Fail-closed contract tests for the isolated supply-chain evidence lane."""

from __future__ import annotations

import copy
import io
import json
import re
import subprocess
import tarfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

import scripts.install_supply_chain_tools as tool_installer
import scripts.supply_chain_evidence as evidence_tool
from scripts.install_supply_chain_tools import ToolInstallError, _extract_binary, install_tools, validate_manifest
from scripts.supply_chain_evidence import (
    EvidenceError,
    bind_container_evidence,
    canonicalize_cyclonedx,
    compare_distributions,
    create_evidence_manifest,
    create_provenance,
    enforce_policy,
    extract_wheel,
    load_external_material_manifest,
    normalize_sdist,
    validate_embedding_model_materials,
)

ROOT = Path(__file__).parents[1]
SUPPLY_CHAIN = ROOT / "supply-chain"
FIXTURES = Path(__file__).parent / "fixtures" / "supply_chain"
EVALUATION_TIME = datetime(2026, 7, 15, 12, tzinfo=UTC)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _clean_grype_report(*, built: str = "2026-07-15T08:00:00Z") -> dict:
    return {
        "matches": [],
        "descriptor": {
            "name": "grype",
            "version": "0.115.0",
            "db": {"status": {"built": built, "schemaVersion": "v6.0.2", "valid": True}},
            "configuration": {
                "only-fixed": False,
                "only-notfixed": False,
                "ignore": [],
                "exclude": [],
                "vex-documents": [],
                "vex-add": [],
            },
        },
        "ignoredMatches": [],
    }


def test_tool_manifest_uses_exact_official_release_checksums_and_builder_digest():
    manifest = _json(SUPPLY_CHAIN / "tools.json")
    tools = validate_manifest(manifest)
    assert set(tools) == {"buildx", "syft", "grype", "gitleaks"}
    assert all(re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) for record in tools.values())
    assert all("/releases/download/v" in record["url"] for record in tools.values())
    assert re.fullmatch(r"moby/buildkit:v\d+\.\d+\.\d+@sha256:[0-9a-f]{64}", manifest["builder_image"]["reference"])


def test_tool_manifest_rejects_mutable_or_unverified_downloads():
    manifest = _json(SUPPLY_CHAIN / "tools.json")
    mutable = copy.deepcopy(manifest)
    mutable["tools"]["syft"]["url"] = "https://example.test/syft-latest.tar.gz"
    with pytest.raises(ToolInstallError, match="official github.com"):
        validate_manifest(mutable)

    unverified = copy.deepcopy(manifest)
    unverified["tools"]["grype"]["sha256"] = "not-a-digest"
    with pytest.raises(ToolInstallError, match="SHA-256"):
        validate_manifest(unverified)

    mismatched_version = copy.deepcopy(manifest)
    mismatched_version["tools"]["syft"]["version"] = "1.latest"
    with pytest.raises(ToolInstallError, match="version is not exact"):
        validate_manifest(mismatched_version)

    wrong_repository = copy.deepcopy(manifest)
    wrong_repository["tools"]["syft"]["url"] = wrong_repository["tools"]["syft"]["url"].replace(
        "anchore/syft", "untrusted/syft"
    )
    with pytest.raises(ToolInstallError, match="official repository and version"):
        validate_manifest(wrong_repository)

    mutable_builder = copy.deepcopy(manifest)
    mutable_builder["builder_image"]["reference"] = "moby/buildkit:latest"
    with pytest.raises(ToolInstallError, match="BuildKit image"):
        validate_manifest(mutable_builder)


def test_tool_installer_rejects_a_cross_platform_manifest_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(tool_installer.platform, "system", lambda: "Linux")
    monkeypatch.setattr(tool_installer.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(
        tool_installer,
        "_download",
        lambda *_args, **_kwargs: pytest.fail("platform mismatch must fail before download"),
    )
    destination = tmp_path / "tools"

    with pytest.raises(ToolInstallError, match="does not match host 'linux_arm64'"):
        install_tools(SUPPLY_CHAIN / "tools.json", destination)

    assert not destination.exists()


def test_archive_extraction_requires_one_exact_bounded_regular_member(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    archive = tmp_path / "tool.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo("tool")
        info.size = 4
        bundle.addfile(info, io.BytesIO(b"safe"))
    target = tmp_path / "installed"
    _extract_binary(archive, "tool", target)
    assert target.read_bytes() == b"safe"

    monkeypatch.setattr(tool_installer, "MAX_TOOL_MEMBER_BYTES", 3)
    with pytest.raises(ToolInstallError, match="uncompressed size"):
        _extract_binary(archive, "tool", target)
    monkeypatch.setattr(tool_installer, "MAX_TOOL_MEMBER_BYTES", 128 * 1024 * 1024)

    malicious = tmp_path / "malicious.tar.gz"
    with tarfile.open(malicious, "w:gz") as bundle:
        info = tarfile.TarInfo("../tool")
        info.size = 4
        bundle.addfile(info, io.BytesIO(b"evil"))
    with pytest.raises(ToolInstallError, match="unsafe or duplicate"):
        _extract_binary(malicious, "tool", target)


def _write_sdist(path: Path, *, mtime: int, reverse: bool = False, link: bool = False) -> None:
    records = [("package", None), ("package/module.py", b"VALUE = 1\n")]
    if reverse:
        records.reverse()
    with tarfile.open(path, "w:gz") as bundle:
        for name, payload in records:
            info = tarfile.TarInfo(name)
            info.mtime = mtime
            info.uid = mtime
            info.gid = mtime
            if payload is None:
                info.type = tarfile.DIRTYPE
                info.mode = 0o775
                bundle.addfile(info)
            else:
                info.size = len(payload)
                info.mode = 0o664
                bundle.addfile(info, io.BytesIO(payload))
        if link:
            info = tarfile.TarInfo("package/link")
            info.type = tarfile.SYMTYPE
            info.linkname = "module.py"
            bundle.addfile(info)


def test_sdist_normalization_is_deterministic_and_rejects_links(tmp_path: Path):
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    _write_sdist(first, mtime=1, reverse=False)
    _write_sdist(second, mtime=2, reverse=True)
    first_digest = normalize_sdist(first, 1_784_116_352)
    second_digest = normalize_sdist(second, 1_784_116_352)
    assert first_digest == second_digest
    assert first.read_bytes() == second.read_bytes()

    unsafe = tmp_path / "unsafe.tar.gz"
    _write_sdist(unsafe, mtime=1, link=True)
    with pytest.raises(EvidenceError, match="link or unsupported"):
        normalize_sdist(unsafe, 1_784_116_352)


def test_wheel_extraction_enforces_path_and_uncompressed_size_bounds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    wheel = tmp_path / "fixture.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("package/__init__.py", b"safe")
    destination = tmp_path / "wheel"
    assert extract_wheel(wheel, destination) == 1
    assert (destination / "package" / "__init__.py").read_bytes() == b"safe"

    monkeypatch.setattr(evidence_tool, "MAX_WHEEL_FILE_BYTES", 3)
    with pytest.raises(EvidenceError, match="safe size"):
        extract_wheel(wheel, tmp_path / "oversized")

    malicious = tmp_path / "malicious.whl"
    with zipfile.ZipFile(malicious, "w") as archive:
        archive.writestr("../escape.py", b"unsafe")
    with pytest.raises(EvidenceError, match="unsafe or duplicate"):
        extract_wheel(malicious, tmp_path / "malicious")


def test_distribution_reproducibility_is_byte_for_byte_and_fail_closed(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for directory in (first, second):
        (directory / ".gitignore").write_text("*\n", encoding="utf-8")
        (directory / "bddk_mcp-1-py3-none-any.whl").write_bytes(b"wheel")
        (directory / "bddk_mcp-1.tar.gz").write_bytes(b"sdist")
    assert set(compare_distributions(first, second)) == {
        "bddk_mcp-1-py3-none-any.whl",
        "bddk_mcp-1.tar.gz",
    }

    (second / "bddk_mcp-1.tar.gz").write_bytes(b"changed")
    with pytest.raises(EvidenceError, match="different"):
        compare_distributions(first, second)


def test_provenance_is_deterministic_and_bound_to_subject_source_and_material(tmp_path: Path):
    artifact = tmp_path / "artifact.whl"
    material = tmp_path / "uv.lock"
    artifact.write_bytes(b"artifact")
    material.write_bytes(b"locked")
    kwargs = {
        "subject_name": "python/artifact.whl",
        "subject_digest": "c7c5c1d70c5dec44c2cb4c0bf2f24f585d879977fb3e2f590dcd21f8a92f9f53",
        "subject_kind": "python-wheel",
        "source_uri": "https://github.com/omercagatay/bddk-mcp",
        "source_commit": "a" * 40,
        "builder_id": "https://github.com/omercagatay/bddk-mcp/.github/workflows/supply-chain.yml",
        "build_type": "https://github.com/omercagatay/bddk-mcp/.github/workflows/supply-chain.yml#python-v1",
        "materials": [("uv.lock", material)],
    }
    first = create_provenance(**kwargs)
    second = create_provenance(**kwargs)
    assert first == second
    assert first["_type"] == "https://in-toto.io/Statement/v1"
    assert first["predicateType"] == "https://slsa.dev/provenance/v1"
    assert first["subject"][0]["digest"]["sha256"] == kwargs["subject_digest"]
    dependencies = first["predicate"]["buildDefinition"]["resolvedDependencies"]
    assert dependencies[0]["digest"] == {"gitCommit": "a" * 40}
    assert dependencies[1]["digest"]["sha256"] != kwargs["subject_digest"]


def test_embedding_model_manifest_is_an_exact_external_git_material():
    materials = load_external_material_manifest(SUPPLY_CHAIN / "model-assets.json")

    assert materials == [
        {
            "uri": (
                "git+https://huggingface.co/intfloat/multilingual-e5-base@d13f1b27baf31030b7fd040960d60d909913633f"
            ),
            "digest": {"gitCommit": "d13f1b27baf31030b7fd040960d60d909913633f"},
        }
    ]


def test_embedding_model_manifest_matches_runtime_and_container_recipes(tmp_path: Path):
    result = validate_embedding_model_materials(
        SUPPLY_CHAIN / "model-assets.json",
        ROOT / "bddk_mcp" / "core" / "config.py",
        [ROOT / "Dockerfile", ROOT / "Dockerfile.spaces"],
    )
    assert result["model_name"] == "intfloat/multilingual-e5-base"
    assert result["revision"] == "d13f1b27baf31030b7fd040960d60d909913633f"
    assert [record["path"] for record in result["container_recipes"]] == ["Dockerfile", "Dockerfile.spaces"]

    changed = tmp_path / "Dockerfile"
    changed.write_text(
        (ROOT / "Dockerfile").read_text(encoding="utf-8").replace("d13f1b27baf31030b7fd040960d60d909913633f", "a" * 40),
        encoding="utf-8",
    )
    with pytest.raises(EvidenceError, match="container recipe embedding model differs"):
        validate_embedding_model_materials(
            SUPPLY_CHAIN / "model-assets.json",
            ROOT / "bddk_mcp" / "core" / "config.py",
            [changed],
        )


def test_container_evidence_binds_buildx_local_image_and_syft_subject():
    manifest_digest = "b" * 64
    config_digest = "c" * 64
    local_reference = "bddk-mcp-standard:ci-" + "a" * 40
    buildx = {
        "containerimage.digest": "sha256:" + manifest_digest,
        "containerimage.config.digest": "sha256:" + config_digest,
        "containerimage.descriptor": {
            "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "digest": "sha256:" + manifest_digest,
            "size": 512,
            "platform": {"architecture": "amd64", "os": "linux"},
        },
        "image.name": "docker.io/library/" + local_reference,
    }
    raw_sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "metadata": {
            "component": {
                "type": "container",
                "name": local_reference,
                "version": "sha256:" + config_digest,
            }
        },
        "components": [{"type": "library", "name": "runtime"}],
    }
    inspect = [{"Id": "sha256:" + config_digest, "RepoTags": [local_reference]}]

    digest, sbom, identity = bind_container_evidence(
        buildx,
        raw_sbom,
        inspect,
        expected_local_reference=local_reference,
        subject_name="bddk-mcp-standard",
        source_date_epoch=1_784_116_352,
    )

    assert digest == manifest_digest
    assert identity["manifest_sha256"] == manifest_digest
    assert identity["config_sha256"] == config_digest
    assert identity["platform"] == "linux/amd64"
    assert sbom["metadata"]["component"]["hashes"] == [{"alg": "SHA-256", "content": manifest_digest}]

    wrong_inspect = [{"Id": "sha256:" + "d" * 64, "RepoTags": [local_reference]}]
    with pytest.raises(EvidenceError, match="local image config differs"):
        bind_container_evidence(
            buildx,
            raw_sbom,
            wrong_inspect,
            expected_local_reference=local_reference,
            subject_name="bddk-mcp-standard",
            source_date_epoch=1_784_116_352,
        )

    unbound_sbom = copy.deepcopy(raw_sbom)
    unbound_sbom["metadata"]["component"]["version"] = "latest"
    with pytest.raises(EvidenceError, match="does not identify"):
        bind_container_evidence(
            buildx,
            unbound_sbom,
            inspect,
            expected_local_reference=local_reference,
            subject_name="bddk-mcp-standard",
            source_date_epoch=1_784_116_352,
        )

    wrong_descriptor = copy.deepcopy(buildx)
    wrong_descriptor["containerimage.descriptor"]["digest"] = "sha256:" + "e" * 64
    with pytest.raises(EvidenceError, match="descriptor"):
        bind_container_evidence(
            wrong_descriptor,
            raw_sbom,
            inspect,
            expected_local_reference=local_reference,
            subject_name="bddk-mcp-standard",
            source_date_epoch=1_784_116_352,
        )


def test_cyclonedx_canonicalization_removes_timestamp_uuid_and_order_drift():
    digest = "b" * 64
    first = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": "urn:uuid:random-one",
        "metadata": {
            "timestamp": "2026-07-15T12:00:00Z",
            "component": {"type": "application", "name": "local-tag", "bom-ref": "old-root"},
        },
        "components": [
            {"type": "library", "name": "z", "bom-ref": "z"},
            {"type": "library", "name": "a", "bom-ref": "a"},
        ],
        "dependencies": [{"ref": "old-root", "dependsOn": ["z", "a"]}],
    }
    second = copy.deepcopy(first)
    second["serialNumber"] = "urn:uuid:random-two"
    second["metadata"]["timestamp"] = "2030-01-01T00:00:00Z"
    second["components"].reverse()
    second["dependencies"][0]["dependsOn"].reverse()

    normalized_first = canonicalize_cyclonedx(
        first,
        subject_name="bddk-mcp-standard",
        subject_digest=digest,
        subject_type="container",
        source_date_epoch=1_784_116_352,
    )
    normalized_second = canonicalize_cyclonedx(
        second,
        subject_name="bddk-mcp-standard",
        subject_digest=digest,
        subject_type="container",
        source_date_epoch=1_784_116_352,
    )
    assert normalized_first == normalized_second
    root = normalized_first["metadata"]["component"]
    assert root["hashes"] == [{"alg": "SHA-256", "content": digest}]
    assert normalized_first["dependencies"][0]["ref"] == f"urn:bddk-mcp:sha256:{digest}"


def test_high_vulnerability_fixture_and_secret_fixture_fail_closed():
    policy = _json(SUPPLY_CHAIN / "policy.json")
    vulnerable = _json(FIXTURES / "grype_high.json")
    result, violations = enforce_policy(
        policy,
        [("fixture.grype.json", vulnerable)],
        [],
        evaluation_time=EVALUATION_TIME,
    )
    assert not result["passed"]
    assert result["blocked_vulnerability_finding_count"] == 1
    assert "CVE-2099-0001" in violations[0]

    result, violations = enforce_policy(
        policy,
        [("clean.grype.json", _clean_grype_report())],
        _json(FIXTURES / "gitleaks_finding.json"),
        evaluation_time=EVALUATION_TIME,
    )
    assert not result["passed"]
    assert result["secret_finding_count"] == 1
    assert violations == ["unexcepted secret finding detected"]


def test_policy_accepts_clean_evidence_and_rejects_stale_or_malformed_database():
    policy = _json(SUPPLY_CHAIN / "policy.json")
    result, violations = enforce_policy(
        policy,
        [("clean.grype.json", _clean_grype_report())],
        [],
        evaluation_time=EVALUATION_TIME,
    )
    assert result["passed"]
    assert violations == []
    assert result["status"] == "repository_policy_passed_unsigned_evidence"
    assert result["external_approval_required"] is False
    assert result["release_promotion_eligible"] is False

    with pytest.raises(EvidenceError, match="stale"):
        enforce_policy(
            policy,
            [("stale.grype.json", _clean_grype_report(built="2026-07-01T00:00:00Z"))],
            [],
            evaluation_time=EVALUATION_TIME,
        )
    malformed = _clean_grype_report()
    del malformed["descriptor"]["db"]
    with pytest.raises(EvidenceError, match="database metadata"):
        enforce_policy(policy, [("bad.grype.json", malformed)], [], evaluation_time=EVALUATION_TIME)

    suppressed = _clean_grype_report()
    suppressed["descriptor"]["configuration"]["only-fixed"] = True
    with pytest.raises(EvidenceError, match="suppressive"):
        enforce_policy(policy, [("suppressed.grype.json", suppressed)], [], evaluation_time=EVALUATION_TIME)

    ignored_high = _clean_grype_report()
    ignored_high["ignoredMatches"] = _json(FIXTURES / "grype_high.json")["matches"]
    result, violations = enforce_policy(
        policy,
        [("ignored.grype.json", ignored_high)],
        [],
        evaluation_time=EVALUATION_TIME,
    )
    assert not result["passed"]
    assert result["suppressed_match_count"] == 1
    assert "suppressed High" in violations[0]


def test_exceptions_are_exact_owned_reasoned_and_time_bounded():
    policy = _json(SUPPLY_CHAIN / "policy.json")
    policy["vulnerabilities"]["exceptions"] = [
        {
            "id": "CVE-2099-0001",
            "package": "deliberately-vulnerable-fixture",
            "target": "fixture.grype.json",
            "reason": "Temporary fixture acceptance for a reviewed test only.",
            "owner": "security-reviewer",
            "approval_state": "pending_bank_release_review",
            "expires_on": "2026-07-16",
        }
    ]
    result, violations = enforce_policy(
        policy,
        [("fixture.grype.json", _json(FIXTURES / "grype_high.json"))],
        [],
        evaluation_time=EVALUATION_TIME,
    )
    assert result["passed"]
    assert violations == []
    assert result["applied_pending_vulnerability_exception_count"] == 1
    assert result["applied_pending_secret_exception_count"] == 0
    assert result["external_approval_required"] is True
    assert result["release_promotion_eligible"] is False
    assert result["status"] == "repository_policy_passed_external_approvals_pending"

    policy["vulnerabilities"]["exceptions"][0]["expires_on"] = "2026-07-14"
    with pytest.raises(EvidenceError, match="expired"):
        enforce_policy(
            policy,
            [("fixture.grype.json", _json(FIXTURES / "grype_high.json"))],
            [],
            evaluation_time=EVALUATION_TIME,
        )

    policy["vulnerabilities"]["exceptions"][0]["expires_on"] = "2026-07-16"
    policy["vulnerabilities"]["exceptions"][0]["approval_state"] = "bank_release_approved"
    with pytest.raises(EvidenceError, match="cannot claim bank release approval"):
        enforce_policy(
            policy,
            [("fixture.grype.json", _json(FIXTURES / "grype_high.json"))],
            [],
            evaluation_time=EVALUATION_TIME,
        )


def test_evidence_manifest_hashes_regular_files_and_rejects_symlinks(tmp_path: Path):
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "report.json").write_text("{}\n", encoding="utf-8")
    output = root / "manifest.json"
    manifest = create_evidence_manifest(root, output)
    assert manifest["files"][0]["path"] == "report.json"
    assert re.fullmatch(r"[0-9a-f]{64}", manifest["files"][0]["sha256"])

    (root / "link").symlink_to(root / "report.json")
    with pytest.raises(EvidenceError, match="non-regular"):
        create_evidence_manifest(root, output)


def test_workflow_is_isolated_immutable_pinned_and_does_not_claim_signing():
    workflow_path = ROOT / ".github" / "workflows" / "supply-chain.yml"
    workflow_text = workflow_path.read_text(encoding="utf-8")
    workflow = yaml.safe_load(workflow_text)
    assert isinstance(workflow, dict)
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["evidence"]["runs-on"] == "ubuntu-24.04"
    checkout = workflow["jobs"]["evidence"]["steps"][0]
    assert checkout["with"]["persist-credentials"] is False

    action_refs = re.findall(r"^\s*uses:\s*[^@\s]+@([^\s#]+)", workflow_text, flags=re.MULTILINE)
    assert action_refs
    assert all(re.fullmatch(r"[0-9a-f]{40}", reference) for reference in action_refs)
    assert "Dockerfile" in workflow_text and "Dockerfile.spaces" in workflow_text
    assert "--provenance=false" in workflow_text
    assert "--load" in workflow_text
    assert "BUILDX_METADATA_PROVENANCE" not in workflow_text
    assert "grype db update" in workflow_text
    assert "--config supply-chain/grype.yaml" in workflow_text
    assert "enforce-policy" in workflow_text
    assert "extract-wheel" in workflow_text
    assert "gitleaks git" in workflow_text and '--log-opts="--all"' in workflow_text
    assert "continue-on-error" not in workflow_text
    assert "|| true" not in workflow_text
    assert "only-fixed: false" in (SUPPLY_CHAIN / "grype.yaml").read_text(encoding="utf-8")
    assert "id-token: write" not in workflow_text
    assert "cosign" not in workflow_text.lower()

    tools = _json(SUPPLY_CHAIN / "tools.json")
    assert workflow["env"]["BUILDKIT_IMAGE"] == tools["builder_image"]["reference"]
    assert (
        _json(SUPPLY_CHAIN / "policy.json")["vulnerabilities"]["scanner_version"] == tools["tools"]["grype"]["version"]
    )
    grype_config = yaml.safe_load((SUPPLY_CHAIN / "grype.yaml").read_text(encoding="utf-8"))
    assert grype_config["ignore"] == []
    assert grype_config["only-fixed"] is False
    assert grype_config["only-notfixed"] is False

    for step in workflow["jobs"]["evidence"]["steps"]:
        if script := step.get("run"):
            syntax = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True, check=False)
            assert syntax.returncode == 0, f"{step.get('name')}: {syntax.stderr}"
