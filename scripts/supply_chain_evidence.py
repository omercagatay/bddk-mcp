#!/usr/bin/env python3
"""Create and enforce deterministic, reviewable supply-chain evidence."""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import io
import json
import os
import re
import stat
import sys
import tarfile
import tempfile
import uuid
import zipfile
from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
KNOWN_SEVERITIES = {"Unknown", "Negligible", "Low", "Medium", "High", "Critical"}
MAX_SDIST_MEMBERS = 100_000
MAX_SDIST_FILE_BYTES = 256 * 1024 * 1024
MAX_SDIST_TOTAL_BYTES = 512 * 1024 * 1024
MAX_WHEEL_MEMBERS = 100_000
MAX_WHEEL_FILE_BYTES = 256 * 1024 * 1024
MAX_WHEEL_TOTAL_BYTES = 512 * 1024 * 1024
VULNERABILITY_EXCEPTION_IDENTITY_FIELDS = (
    "target",
    "id",
    "namespace",
    "package",
    "version",
    "type",
    "match_type",
    "material_sha256",
)
EXCEPTION_GOVERNANCE_FIELDS = ("reason", "owner", "approval_state", "expires_on")


class EvidenceError(RuntimeError):
    """Raised when evidence is incomplete, malformed, stale, or violates policy."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read valid JSON evidence: {path}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise EvidenceError(f"subject is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, label: str) -> str:
    normalized = value.removeprefix("sha256:").lower()
    if not SHA256_RE.fullmatch(normalized):
        raise EvidenceError(f"{label} must be an exact SHA-256 digest")
    return normalized


def _split_mapping(value: str, label: str) -> tuple[str, str]:
    name, separator, target = value.partition("=")
    if not separator or not name or not target or name.startswith("/") or ".." in Path(name).parts:
        raise EvidenceError(f"{label} must use safe NAME=VALUE syntax")
    return name, target


def compare_distributions(first: Path, second: Path) -> dict[str, str]:
    def inventory(root: Path) -> dict[str, str]:
        if not root.is_dir():
            raise EvidenceError(f"distribution directory is missing: {root}")
        files: list[Path] = []
        for path in root.iterdir():
            if path.name == ".gitignore" and path.is_file() and not path.is_symlink():
                continue
            if not path.is_file() or path.is_symlink():
                raise EvidenceError("distribution directory contains an unsafe entry")
            files.append(path)
        expected = [path for path in files if path.suffix == ".whl" or path.name.endswith(".tar.gz")]
        if len(expected) != 2 or len(files) != 2:
            raise EvidenceError("each build must contain exactly one wheel and one source distribution")
        if sum(path.suffix == ".whl" for path in expected) != 1:
            raise EvidenceError("distribution build does not contain exactly one wheel")
        if sum(path.name.endswith(".tar.gz") for path in expected) != 1:
            raise EvidenceError("distribution build does not contain exactly one source distribution")
        return {path.name: _sha256(path) for path in expected}

    first_inventory = inventory(first)
    second_inventory = inventory(second)
    if first_inventory != second_inventory:
        raise EvidenceError("wheel/sdist rebuild produced different names or SHA-256 digests")
    return dict(sorted(first_inventory.items()))


def normalize_sdist(path: Path, source_date_epoch: int) -> str:
    """Rewrite a locally built sdist with stable tar and gzip metadata."""

    if source_date_epoch < 0 or not path.is_file() or path.is_symlink() or not path.name.endswith(".tar.gz"):
        raise EvidenceError("sdist normalization requires a regular .tar.gz file and non-negative epoch")
    records: list[tuple[tarfile.TarInfo, bytes | None]] = []
    names: set[str] = set()
    total_size = 0
    try:
        with tarfile.open(path, mode="r:gz") as source:
            members = source.getmembers()
            if not 1 <= len(members) <= MAX_SDIST_MEMBERS:
                raise EvidenceError("sdist member count is outside the safe limit")
            for member in members:
                parsed = PurePosixPath(member.name)
                if parsed.is_absolute() or ".." in parsed.parts or not member.name or member.name in names:
                    raise EvidenceError("sdist contains an unsafe or duplicate member name")
                names.add(member.name)
                if not (member.isfile() or member.isdir()):
                    raise EvidenceError("sdist contains a link or unsupported member type")
                payload: bytes | None = None
                if member.isfile():
                    if member.size < 0 or member.size > MAX_SDIST_FILE_BYTES:
                        raise EvidenceError("sdist member exceeds the safe size limit")
                    total_size += member.size
                    if total_size > MAX_SDIST_TOTAL_BYTES:
                        raise EvidenceError("sdist exceeds the safe expanded-size limit")
                    stream = source.extractfile(member)
                    if stream is None:
                        raise EvidenceError("sdist member cannot be read")
                    payload = stream.read(MAX_SDIST_FILE_BYTES + 1)
                    if len(payload) != member.size:
                        raise EvidenceError("sdist member size does not match its content")
                records.append((member, payload))
    except (OSError, tarfile.TarError) as exc:
        raise EvidenceError("sdist is not a valid gzip-compressed tar archive") from exc

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as raw:
            temporary_name = raw.name
            with gzip.GzipFile(
                filename="", mode="wb", compresslevel=9, fileobj=raw, mtime=source_date_epoch
            ) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as destination:
                    for original, payload in sorted(records, key=lambda item: item[0].name):
                        normalized = tarfile.TarInfo(original.name)
                        normalized.type = tarfile.DIRTYPE if original.isdir() else tarfile.REGTYPE
                        normalized.size = 0 if payload is None else len(payload)
                        normalized.mode = 0o755 if original.isdir() or original.mode & 0o111 else 0o644
                        normalized.mtime = source_date_epoch
                        normalized.uid = 0
                        normalized.gid = 0
                        normalized.uname = ""
                        normalized.gname = ""
                        destination.addfile(normalized, None if payload is None else io.BytesIO(payload))
        os.replace(temporary_name, path)
        temporary_name = None
    except OSError as exc:
        raise EvidenceError("cannot write normalized sdist") from exc
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return _sha256(path)


def extract_wheel(path: Path, destination: Path) -> int:
    """Safely expand a locally built wheel for package cataloging."""

    if not path.is_file() or path.is_symlink() or path.suffix != ".whl":
        raise EvidenceError("wheel extraction requires a regular .whl file")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise EvidenceError("wheel extraction destination must be absent or empty")
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    total_size = 0
    names: set[str] = set()
    extracted = 0
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if not 1 <= len(members) <= MAX_WHEEL_MEMBERS:
                raise EvidenceError("wheel member count is outside the safe limit")
            for member in members:
                parsed = PurePosixPath(member.filename)
                if (
                    parsed.is_absolute()
                    or ".." in parsed.parts
                    or not member.filename
                    or "\\" in member.filename
                    or member.filename in names
                ):
                    raise EvidenceError("wheel contains an unsafe or duplicate member name")
                names.add(member.filename)
                unix_mode = member.external_attr >> 16
                if stat.S_ISLNK(unix_mode) or member.flag_bits & 0x1:
                    raise EvidenceError("wheel contains a link or encrypted member")
                if member.file_size < 0 or member.file_size > MAX_WHEEL_FILE_BYTES:
                    raise EvidenceError("wheel member exceeds the safe size limit")
                total_size += member.file_size
                if total_size > MAX_WHEEL_TOTAL_BYTES:
                    raise EvidenceError("wheel exceeds the safe expanded-size limit")
                target = (destination / Path(*parsed.parts)).resolve()
                if destination_root != target and destination_root not in target.parents:
                    raise EvidenceError("wheel member resolves outside the extraction directory")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source:
                    payload = source.read(MAX_WHEEL_FILE_BYTES + 1)
                if len(payload) != member.file_size:
                    raise EvidenceError("wheel member size does not match its content")
                target.write_bytes(payload)
                target.chmod(0o644)
                extracted += 1
    except (OSError, zipfile.BadZipFile) as exc:
        raise EvidenceError("wheel is not a valid ZIP archive") from exc
    if extracted == 0:
        raise EvidenceError("wheel contains no files")
    return extracted


def _resolve_subject_digest(subject_file: Path | None, subject_sha256: str | None) -> str:
    if (subject_file is None) == (subject_sha256 is None):
        raise EvidenceError("provide exactly one of --subject-file or --subject-sha256")
    return _sha256(subject_file) if subject_file is not None else _require_sha256(str(subject_sha256), "subject")


def create_provenance(
    *,
    subject_name: str,
    subject_digest: str,
    subject_kind: str,
    source_uri: str,
    source_commit: str,
    builder_id: str,
    build_type: str,
    materials: Iterable[tuple[str, Path]],
    external_materials: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    digest = _require_sha256(subject_digest, "subject")
    if not subject_name or subject_name.startswith("/") or ".." in Path(subject_name).parts:
        raise EvidenceError("subject name must be stable and relative")
    if subject_kind not in {"python-wheel", "python-sdist", "container-image"}:
        raise EvidenceError("unsupported provenance subject kind")
    if (
        not source_uri.startswith("https://")
        or not builder_id.startswith("https://")
        or not build_type.startswith("https://")
    ):
        raise EvidenceError("source, builder, and build type identifiers must use HTTPS")
    if not GIT_COMMIT_RE.fullmatch(source_commit.lower()):
        raise EvidenceError("source commit must be a full Git commit digest")

    resolved: list[dict[str, Any]] = [
        {"uri": f"git+{source_uri}@{source_commit}", "digest": {"gitCommit": source_commit.lower()}}
    ]
    seen: set[str] = set()
    for name, path in sorted(materials, key=lambda item: item[0]):
        if name in seen or not name or name.startswith("/") or ".." in Path(name).parts:
            raise EvidenceError("material names must be unique, stable, and relative")
        seen.add(name)
        resolved.append({"uri": name, "digest": {"sha256": _sha256(path)}})
    external_uris: set[str] = set()
    for material in sorted(external_materials, key=lambda item: str(item.get("uri", ""))):
        uri = material.get("uri")
        material_digest = material.get("digest")
        if (
            not isinstance(uri, str)
            or not uri.startswith("git+https://")
            or not isinstance(material_digest, dict)
            or not GIT_COMMIT_RE.fullmatch(str(material_digest.get("gitCommit", "")))
            or uri in external_uris
        ):
            raise EvidenceError("external provenance material is invalid or duplicated")
        external_uris.add(uri)
        resolved.append({"uri": uri, "digest": {"gitCommit": material_digest["gitCommit"].lower()}})

    return {
        "_type": "https://in-toto.io/Statement/v1",
        "predicateType": "https://slsa.dev/provenance/v1",
        "subject": [{"name": subject_name, "digest": {"sha256": digest}}],
        "predicate": {
            "buildDefinition": {
                "buildType": build_type,
                "externalParameters": {"subjectKind": subject_kind},
                "internalParameters": {},
                "resolvedDependencies": resolved,
            },
            "runDetails": {"builder": {"id": builder_id}, "metadata": {}},
        },
    }


def load_external_material_manifest(path: Path) -> list[dict[str, Any]]:
    """Load exact immutable Git materials such as the embedded model snapshot."""

    raw = _read_json(path)
    if (
        not isinstance(raw, dict)
        or raw.get("schema_version") != 1
        or raw.get("license_review_status") != "pending_bank_review"
        or not isinstance(raw.get("materials"), list)
        or not raw["materials"]
    ):
        raise EvidenceError("external material manifest has an invalid schema")
    names: set[str] = set()
    uris: set[str] = set()
    result: list[dict[str, Any]] = []
    for material in raw["materials"]:
        if not isinstance(material, dict) or set(material) != {"name", "uri", "digest"}:
            raise EvidenceError("external material record has an invalid schema")
        name = material["name"]
        uri = material["uri"]
        digest = material["digest"]
        parsed = urlsplit(uri) if isinstance(uri, str) else None
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,127}", name)
            or name in names
            or parsed is None
            or parsed.scheme != "https"
            or parsed.netloc != "huggingface.co"
            or parsed.query
            or parsed.fragment
            or not isinstance(digest, dict)
            or set(digest) != {"gitCommit"}
            or not re.fullmatch(r"[0-9a-f]{40}", str(digest["gitCommit"]))
            or uri in uris
        ):
            raise EvidenceError("external material identity is invalid or duplicated")
        names.add(name)
        uris.add(uri)
        result.append(
            {
                "uri": f"git+{uri}@{digest['gitCommit']}",
                "digest": {"gitCommit": digest["gitCommit"]},
            }
        )
    return result


def validate_embedding_model_materials(
    manifest_path: Path,
    config_path: Path,
    dockerfiles: Iterable[Path],
) -> dict[str, Any]:
    """Bind the declared embedding model to runtime and image build defaults."""

    materials = load_external_material_manifest(manifest_path)
    if len(materials) != 1:
        raise EvidenceError("embedding material manifest must declare exactly one model")
    match = re.fullmatch(
        r"git\+https://huggingface\.co/([^@]+)@([0-9a-f]{40})",
        str(materials[0]["uri"]),
    )
    if match is None or materials[0]["digest"] != {"gitCommit": match.group(2)}:
        raise EvidenceError("embedding material must identify one exact Hugging Face Git commit")
    model_name, revision = match.groups()

    try:
        config_tree = ast.parse(config_path.read_text(encoding="utf-8"), filename=str(config_path))
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise EvidenceError("embedding runtime config cannot be parsed") from exc
    defaults: dict[str, str] = {}
    for node in config_tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name == "_DEFAULT_EMBEDDING_MODEL_REVISION" and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                defaults[name] = node.value.value
        elif name == "EMBEDDING_MODEL_NAME" and isinstance(node.value, ast.Call) and len(node.value.args) >= 2:
            default = node.value.args[1]
            if isinstance(default, ast.Constant) and isinstance(default.value, str):
                defaults[name] = default.value
    if defaults != {
        "EMBEDDING_MODEL_NAME": model_name,
        "_DEFAULT_EMBEDDING_MODEL_REVISION": revision,
    }:
        raise EvidenceError("runtime embedding model defaults differ from the external material manifest")

    dockerfile_records: list[dict[str, str]] = []
    model_pattern = re.compile(
        r"SentenceTransformer\(\s*(['\"])([^'\"]+)\1\s*,\s*revision\s*=\s*(['\"])([0-9a-f]{40})\3\s*\)"
    )
    paths = list(dockerfiles)
    if not paths:
        raise EvidenceError("at least one container recipe is required")
    for dockerfile in paths:
        try:
            content = dockerfile.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise EvidenceError("container recipe cannot be read") from exc
        matches = model_pattern.findall(content)
        if len(matches) != 1 or matches[0][1:] != (model_name, matches[0][2], revision):
            raise EvidenceError(
                f"container recipe embedding model differs from the external material: {dockerfile.name}"
            )
        dockerfile_records.append({"path": dockerfile.name, "sha256": _sha256(dockerfile)})

    return {
        "schema_version": 1,
        "model_name": model_name,
        "revision": revision,
        "manifest_sha256": _sha256(manifest_path),
        "runtime_config_sha256": _sha256(config_path),
        "container_recipes": sorted(dockerfile_records, key=lambda record: record["path"]),
    }


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        normalized = [_canonicalize(item) for item in value]
        return sorted(
            normalized, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
    return value


def _replace_string(value: Any, old: str, new: str) -> Any:
    if isinstance(value, dict):
        return {key: _replace_string(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_string(item, old, new) for item in value]
    return new if value == old else value


def canonicalize_cyclonedx(
    raw: Any,
    *,
    subject_name: str,
    subject_digest: str,
    subject_type: str,
    source_date_epoch: int,
) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("bomFormat") != "CycloneDX":
        raise EvidenceError("Syft output is not a CycloneDX JSON object")
    if not isinstance(raw.get("components"), list) or not raw["components"]:
        raise EvidenceError("CycloneDX SBOM has no component inventory")
    if subject_type not in {"file", "container"}:
        raise EvidenceError("CycloneDX subject type must be file or container")
    digest = _require_sha256(subject_digest, "SBOM subject")
    if source_date_epoch < 0:
        raise EvidenceError("SOURCE_DATE_EPOCH must not be negative")

    result = json.loads(json.dumps(raw))
    metadata = result.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise EvidenceError("CycloneDX metadata must be an object")
    timestamp = datetime.fromtimestamp(source_date_epoch, tz=UTC).isoformat().replace("+00:00", "Z")
    deterministic_ref = f"urn:bddk-mcp:sha256:{digest}"
    result["serialNumber"] = f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, deterministic_ref)}"
    metadata["timestamp"] = timestamp

    component = metadata.get("component")
    if not isinstance(component, dict):
        component = {}
        metadata["component"] = component
    old_ref = component.get("bom-ref")
    if isinstance(old_ref, str) and old_ref != deterministic_ref:
        result = _replace_string(result, old_ref, deterministic_ref)
        metadata = result["metadata"]
        component = metadata["component"]
    component["bom-ref"] = deterministic_ref
    component["type"] = subject_type
    component["name"] = subject_name
    component["hashes"] = [{"alg": "SHA-256", "content": digest}]
    return _canonicalize(result)


def _recursive_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _recursive_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _recursive_strings(item)


def bind_container_evidence(
    buildx_metadata: Any,
    raw_sbom: Any,
    image_inspect: Any,
    *,
    expected_local_reference: str,
    subject_name: str,
    source_date_epoch: int,
    expected_platform: str = "linux/amd64",
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Prove Buildx, the exact loaded image, and Syft refer to one build."""

    if not isinstance(buildx_metadata, dict):
        raise EvidenceError("Buildx metadata must be a JSON object")
    manifest_digest = _require_sha256(str(buildx_metadata.get("containerimage.digest", "")), "Buildx image")
    config_digest = _require_sha256(str(buildx_metadata.get("containerimage.config.digest", "")), "Buildx image config")
    descriptor = buildx_metadata.get("containerimage.descriptor")
    platform_os, separator, platform_architecture = expected_platform.partition("/")
    if (
        not separator
        or not platform_os
        or not platform_architecture
        or not isinstance(descriptor, dict)
        or descriptor.get("mediaType")
        not in {
            "application/vnd.docker.distribution.manifest.v2+json",
            "application/vnd.oci.image.manifest.v1+json",
        }
        or _require_sha256(str(descriptor.get("digest", "")), "Buildx image descriptor") != manifest_digest
        or not isinstance(descriptor.get("size"), int)
        or descriptor["size"] <= 0
        or descriptor.get("platform") != {"architecture": platform_architecture, "os": platform_os}
    ):
        raise EvidenceError("Buildx image descriptor is missing or differs from the requested result")

    if (
        not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._/-]{0,255}:[A-Za-z0-9][A-Za-z0-9._-]{0,127}", expected_local_reference
        )
        or not isinstance(image_inspect, list)
        or len(image_inspect) != 1
        or not isinstance(image_inspect[0], dict)
    ):
        raise EvidenceError("local image inspection evidence is invalid")
    image_name = buildx_metadata.get("image.name")
    if not isinstance(image_name, str) or not (
        image_name == expected_local_reference or image_name.endswith(f"/{expected_local_reference}")
    ):
        raise EvidenceError("Buildx image name differs from the expected local reference")
    inspected = image_inspect[0]
    if inspected.get("Id") != f"sha256:{config_digest}":
        raise EvidenceError("local image config differs from the Buildx result")
    repo_tags = inspected.get("RepoTags")
    if not isinstance(repo_tags, list) or expected_local_reference not in repo_tags:
        raise EvidenceError("local image tag is not bound to the inspected Buildx result")

    raw_strings = set(_recursive_strings(raw_sbom))
    if not any(config_digest in value for value in raw_strings):
        raise EvidenceError("Syft SBOM does not identify the inspected image config")
    sbom = canonicalize_cyclonedx(
        raw_sbom,
        subject_name=subject_name,
        subject_digest=manifest_digest,
        subject_type="container",
        source_date_epoch=source_date_epoch,
    )
    identity = {
        "schema_version": 1,
        "subject_name": subject_name,
        "manifest_sha256": manifest_digest,
        "config_sha256": config_digest,
        "local_reference": expected_local_reference,
        "platform": expected_platform,
        "buildx_metadata_sha256": hashlib.sha256(
            json.dumps(buildx_metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "raw_sbom_sha256": hashlib.sha256(
            json.dumps(raw_sbom, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    return manifest_digest, sbom, identity


def _parse_iso_datetime(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise EvidenceError(f"{label} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise EvidenceError(f"{label} must include a time zone")
    return parsed.astimezone(UTC)


def _parse_date(value: Any, label: str) -> date:
    if not isinstance(value, str):
        raise EvidenceError(f"{label} must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise EvidenceError(f"{label} must be an ISO date") from exc


def _validate_policy(raw: Any, evaluation_time: datetime) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(raw, dict) or raw.get("schema_version") != 2:
        raise EvidenceError("unsupported supply-chain policy schema")
    vulnerabilities = raw.get("vulnerabilities")
    secrets = raw.get("secrets")
    if not isinstance(vulnerabilities, dict) or not isinstance(secrets, dict):
        raise EvidenceError("policy must define vulnerability and secret rules")

    blocked = vulnerabilities.get("blocked_severities")
    if not isinstance(blocked, list) or not blocked or not all(item in KNOWN_SEVERITIES for item in blocked):
        raise EvidenceError("blocked vulnerability severities are invalid")
    if "Critical" not in blocked or "High" not in blocked:
        raise EvidenceError("repository policy must block both High and Critical vulnerabilities")
    max_age = vulnerabilities.get("maximum_database_age_hours")
    if not isinstance(max_age, int) or not 1 <= max_age <= 168:
        raise EvidenceError("vulnerability database age limit must be between 1 and 168 hours")
    scanner_version = vulnerabilities.get("scanner_version")
    if not isinstance(scanner_version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", scanner_version):
        raise EvidenceError("vulnerability scanner version must be exact")
    if secrets.get("block_all") is not True:
        raise EvidenceError("repository policy must block all unexcepted secrets")

    for kind, section, required in (
        (
            "vulnerability",
            vulnerabilities,
            set(VULNERABILITY_EXCEPTION_IDENTITY_FIELDS + EXCEPTION_GOVERNANCE_FIELDS),
        ),
        (
            "secret",
            secrets,
            {"fingerprint", *EXCEPTION_GOVERNANCE_FIELDS},
        ),
    ):
        exceptions = section.get("exceptions")
        if not isinstance(exceptions, list):
            raise EvidenceError(f"{kind} exceptions must be a list")
        identities: set[tuple[str, ...]] = set()
        for exception in exceptions:
            if not isinstance(exception, dict) or set(exception) != required:
                raise EvidenceError(f"{kind} exception has an invalid schema")
            if not all(isinstance(exception[key], str) and exception[key].strip() for key in required):
                raise EvidenceError(f"{kind} exception fields must be non-empty strings")
            if len(exception["reason"].strip()) < 12 or len(exception["owner"].strip()) < 3:
                raise EvidenceError(f"{kind} exception needs a substantive reason and owner")
            if exception["approval_state"] != "pending_bank_release_review":
                raise EvidenceError(f"{kind} exception cannot claim bank release approval in repository policy")
            if _parse_date(exception["expires_on"], f"{kind} exception expiry") < evaluation_time.date():
                raise EvidenceError(f"{kind} exception is expired")
            if kind == "vulnerability" and not SHA256_RE.fullmatch(exception["material_sha256"]):
                raise EvidenceError("vulnerability exception material_sha256 must be an exact lowercase SHA-256")
            identity_keys = VULNERABILITY_EXCEPTION_IDENTITY_FIELDS if kind == "vulnerability" else ("fingerprint",)
            identity = tuple(exception[key] for key in identity_keys)
            if identity in identities:
                raise EvidenceError(f"duplicate {kind} exception")
            identities.add(identity)
    return vulnerabilities, secrets


def _grype_db_time(report: dict[str, Any], expected_version: str) -> datetime:
    descriptor = report.get("descriptor")
    if (
        not isinstance(descriptor, dict)
        or descriptor.get("name") != "grype"
        or descriptor.get("version") != expected_version
    ):
        raise EvidenceError("vulnerability report lacks pinned Grype descriptor data")
    configuration = descriptor.get("configuration")
    if (
        not isinstance(configuration, dict)
        or configuration.get("only-fixed") is not False
        or configuration.get("only-notfixed") is not False
        or configuration.get("show-suppressed") is not True
        or configuration.get("exclude") != []
        or configuration.get("vex-documents") != []
        or configuration.get("vex-add") != []
    ):
        raise EvidenceError("Grype report used a suppressive or unreviewed scan configuration")
    database = descriptor.get("db")
    if not isinstance(database, dict):
        raise EvidenceError("vulnerability report lacks Grype database metadata")
    status = database.get("status")
    if (
        not isinstance(status, dict)
        or status.get("valid") is not True
        or not isinstance(status.get("schemaVersion"), str)
    ):
        raise EvidenceError("vulnerability report has an invalid Grype database status")
    return _parse_iso_datetime(status.get("built"), "Grype database build time")


def enforce_policy(
    policy: Any,
    grype_reports: Iterable[tuple[str, Any]],
    gitleaks_report: Any,
    *,
    evaluation_time: datetime,
    target_material_sha256: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    if evaluation_time.tzinfo is None:
        raise EvidenceError("policy evaluation time must include a time zone")
    evaluation_time = evaluation_time.astimezone(UTC)
    vulnerability_policy, secret_policy = _validate_policy(policy, evaluation_time)
    blocked = set(vulnerability_policy["blocked_severities"])
    material_digests: dict[str, str] = {}
    if target_material_sha256 is not None:
        if not isinstance(target_material_sha256, Mapping):
            raise EvidenceError("target material digests must be a mapping")
        for target, digest in target_material_sha256.items():
            if not isinstance(target, str) or not target:
                raise EvidenceError("target material names must be non-empty strings")
            material_digests[target] = _require_sha256(digest, f"target material {target}")

    vulnerability_exceptions = {
        tuple(item[key] for key in VULNERABILITY_EXCEPTION_IDENTITY_FIELDS)
        for item in vulnerability_policy["exceptions"]
    }
    for exception in vulnerability_policy["exceptions"]:
        target = exception["target"]
        actual_digest = material_digests.get(target)
        if actual_digest is None:
            raise EvidenceError(f"vulnerability exception target lacks material evidence: {target}")
        if actual_digest != exception["material_sha256"]:
            raise EvidenceError(f"vulnerability exception material differs from scanned target: {target}")
    secret_exceptions = {item["fingerprint"] for item in secret_policy["exceptions"]}

    violations: list[str] = []
    vulnerability_findings = 0
    unexcepted_vulnerability_findings = 0
    applied_pending_vulnerability_exceptions = 0
    applied_vulnerability_exception_identities: set[tuple[str, ...]] = set()
    suppressed_match_count = 0
    report_count = 0
    report_targets: set[str] = set()
    for target, raw in grype_reports:
        if not isinstance(target, str) or not target or target in report_targets:
            raise EvidenceError("Grype report target names must be unique non-empty strings")
        report_targets.add(target)
        report_count += 1
        if not isinstance(raw, dict) or not isinstance(raw.get("matches"), list):
            raise EvidenceError(f"invalid Grype report schema: {target}")
        ignored = raw.get("ignoredMatches", [])
        if not isinstance(ignored, list):
            raise EvidenceError(f"Grype ignored-match evidence is malformed: {target}")
        built = _grype_db_time(raw, vulnerability_policy["scanner_version"])
        if built > evaluation_time + timedelta(minutes=5):
            raise EvidenceError(f"Grype database build time is in the future: {target}")
        if evaluation_time - built > timedelta(hours=vulnerability_policy["maximum_database_age_hours"]):
            raise EvidenceError(f"Grype database is stale: {target}")
        reported_matches = [(item, False) for item in raw["matches"]]
        reported_matches.extend((item, True) for item in ignored)
        for match, suppressed in reported_matches:
            suppressed_match_count += int(suppressed)
            if not isinstance(match, dict):
                raise EvidenceError(f"invalid Grype match: {target}")
            vulnerability = match.get("vulnerability")
            artifact = match.get("artifact")
            if not isinstance(vulnerability, dict) or not isinstance(artifact, dict):
                raise EvidenceError(f"incomplete Grype match: {target}")
            vulnerability_id = vulnerability.get("id")
            namespace = vulnerability.get("namespace")
            severity = vulnerability.get("severity")
            package = artifact.get("name")
            version = artifact.get("version")
            package_type = artifact.get("type")
            match_details = match.get("matchDetails")
            if not all(
                isinstance(item, str) and item
                for item in (vulnerability_id, namespace, severity, package, version, package_type)
            ):
                raise EvidenceError(f"Grype match lacks identity fields: {target}")
            if not isinstance(match_details, list) or not match_details:
                raise EvidenceError(f"Grype match lacks exact match details: {target}")
            match_types = {
                detail.get("type")
                for detail in match_details
                if isinstance(detail, dict) and isinstance(detail.get("type"), str) and detail["type"]
            }
            if len(match_types) != 1 or len(match_types) != len(
                {detail.get("type") for detail in match_details if isinstance(detail, dict)}
            ):
                raise EvidenceError(f"Grype match must identify one exact match type: {target}")
            if any(not isinstance(detail, dict) or detail.get("type") not in match_types for detail in match_details):
                raise EvidenceError(f"Grype match must identify one exact match type: {target}")
            match_type = next(iter(match_types))
            if severity not in KNOWN_SEVERITIES:
                raise EvidenceError(f"Grype returned an unknown severity: {target}")
            material_sha256 = material_digests.get(target, "")
            exception_identity = (
                target,
                vulnerability_id,
                namespace,
                package,
                version,
                package_type,
                match_type,
                material_sha256,
            )
            if suppressed:
                violations.append(
                    f"{target}: suppressed {severity} vulnerability finding {vulnerability_id} in {package}"
                )
            if severity in blocked:
                vulnerability_findings += 1
                if exception_identity in vulnerability_exceptions:
                    applied_pending_vulnerability_exceptions += 1
                    applied_vulnerability_exception_identities.add(exception_identity)
                else:
                    unexcepted_vulnerability_findings += 1
                    violations.append(f"{target}: blocked {severity} vulnerability {vulnerability_id} in {package}")
    if report_count == 0:
        raise EvidenceError("at least one Grype report is required")
    unused_material_targets = set(material_digests) - report_targets
    if unused_material_targets:
        target = sorted(unused_material_targets)[0]
        raise EvidenceError(f"target material has no matching Grype report: {target}")
    unused_vulnerability_exceptions = vulnerability_exceptions - applied_vulnerability_exception_identities
    if unused_vulnerability_exceptions:
        target, vulnerability_id, _namespace, package, *_rest = sorted(unused_vulnerability_exceptions)[0]
        raise EvidenceError(
            f"unused vulnerability exception does not match scan evidence: {target}/{vulnerability_id}/{package}"
        )

    if not isinstance(gitleaks_report, list):
        raise EvidenceError("Gitleaks report must be a JSON array")
    secret_findings = 0
    unexcepted_secret_findings = 0
    applied_pending_secret_exceptions = 0
    reported_secret_fingerprints: set[str] = set()
    for finding in gitleaks_report:
        if not isinstance(finding, dict):
            raise EvidenceError("invalid Gitleaks finding")
        fingerprint = finding.get("Fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise EvidenceError("Gitleaks finding lacks a stable fingerprint")
        if fingerprint in reported_secret_fingerprints:
            raise EvidenceError("Gitleaks report contains a duplicate fingerprint")
        reported_secret_fingerprints.add(fingerprint)
        secret_findings += 1
        if fingerprint in secret_exceptions:
            applied_pending_secret_exceptions += 1
        else:
            unexcepted_secret_findings += 1
            violations.append("unexcepted secret finding detected")

    external_approval_required = bool(applied_pending_vulnerability_exceptions or applied_pending_secret_exceptions)
    evidence_integrity_passed = suppressed_match_count == 0 and unexcepted_secret_findings == 0
    passed = not violations
    result = {
        "schema_version": 1,
        "evaluation_time": evaluation_time.isoformat().replace("+00:00", "Z"),
        "grype_report_count": report_count,
        "blocked_vulnerability_finding_count": vulnerability_findings,
        "unexcepted_vulnerability_finding_count": unexcepted_vulnerability_findings,
        "suppressed_match_count": suppressed_match_count,
        "secret_finding_count": secret_findings,
        "unexcepted_secret_finding_count": unexcepted_secret_findings,
        "applied_pending_vulnerability_exception_count": applied_pending_vulnerability_exceptions,
        "applied_pending_secret_exception_count": applied_pending_secret_exceptions,
        "evidence_integrity_passed": evidence_integrity_passed,
        "external_approval_required": external_approval_required,
        "release_promotion_eligible": False,
        "violation_count": len(violations),
        "passed": passed,
        "status": (
            "repository_policy_failed"
            if not passed
            else (
                "repository_policy_passed_external_approvals_pending"
                if external_approval_required
                else "repository_policy_passed_unsigned_evidence"
            )
        ),
    }
    return result, violations


def create_evidence_manifest(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    output = output.resolve()
    if not root.is_dir() or root not in output.parents:
        raise EvidenceError("manifest output must be inside an existing evidence directory")
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path == output:
            continue
        if path.is_symlink():
            raise EvidenceError("evidence directory contains a non-regular file")
        if path.is_dir():
            continue
        if not path.is_file():
            raise EvidenceError("evidence directory contains a non-regular file")
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
                "size": path.stat().st_size,
            }
        )
    if not files:
        raise EvidenceError("evidence directory is empty")
    return {"schema_version": 1, "files": files}


def verify_evidence_manifest(root: Path, manifest_path: Path) -> dict[str, int]:
    if manifest_path.is_symlink():
        raise EvidenceError("evidence manifest must be a regular file")
    root = root.resolve()
    manifest_path = manifest_path.resolve()
    if not root.is_dir() or root not in manifest_path.parents or not manifest_path.is_file():
        raise EvidenceError("evidence manifest must be a regular file inside the evidence directory")

    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "files"}:
        raise EvidenceError("evidence manifest has an invalid schema")
    if manifest["schema_version"] != 1 or not isinstance(manifest["files"], list) or not manifest["files"]:
        raise EvidenceError("evidence manifest has an unsupported schema or empty inventory")

    expected: dict[str, tuple[str, int]] = {}
    for record in manifest["files"]:
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "size"}:
            raise EvidenceError("evidence manifest contains a malformed record")
        relative = record["path"]
        parsed = PurePosixPath(relative) if isinstance(relative, str) else PurePosixPath()
        if (
            not isinstance(relative, str)
            or not relative
            or parsed.is_absolute()
            or ".." in parsed.parts
            or parsed.as_posix() != relative
            or relative in expected
        ):
            raise EvidenceError("evidence manifest contains an unsafe or duplicate path")
        digest = record["sha256"]
        size = record["size"]
        if (
            not isinstance(digest, str)
            or not SHA256_RE.fullmatch(digest)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise EvidenceError("evidence manifest contains invalid digest or size metadata")
        expected[relative] = (digest, size)

    actual: dict[str, tuple[str, int]] = {}
    for path in root.rglob("*"):
        if path == manifest_path:
            continue
        if path.is_symlink():
            raise EvidenceError("evidence directory contains a non-regular file")
        if path.is_dir():
            continue
        if not path.is_file():
            raise EvidenceError("evidence directory contains a non-regular file")
        relative = path.relative_to(root).as_posix()
        actual[relative] = (_sha256(path), path.stat().st_size)
    if actual != expected:
        raise EvidenceError("downloaded evidence does not exactly match its manifest")
    return {"schema_version": 1, "verified_file_count": len(expected)}


def _command_compare(args: argparse.Namespace) -> None:
    _write_json(args.output, {"schema_version": 1, "artifacts": compare_distributions(args.first, args.second)})


def _command_normalize_sdist(args: argparse.Namespace) -> None:
    digest = normalize_sdist(args.input, args.source_date_epoch)
    _write_json(args.output, {"schema_version": 1, "artifact": args.input.name, "sha256": digest})


def _command_extract_wheel(args: argparse.Namespace) -> None:
    count = extract_wheel(args.input, args.destination)
    _write_json(args.output, {"schema_version": 1, "artifact": args.input.name, "file_count": count})


def _command_verify_model_materials(args: argparse.Namespace) -> None:
    _write_json(
        args.output,
        validate_embedding_model_materials(args.manifest, args.runtime_config, args.dockerfile),
    )


def _command_provenance(args: argparse.Namespace) -> None:
    materials = [(name, Path(path)) for name, path in (_split_mapping(value, "material") for value in args.material)]
    external_materials = (
        load_external_material_manifest(args.external_material_manifest)
        if args.external_material_manifest is not None
        else []
    )
    digest = _resolve_subject_digest(args.subject_file, args.subject_sha256)
    statement = create_provenance(
        subject_name=args.subject_name,
        subject_digest=digest,
        subject_kind=args.subject_kind,
        source_uri=args.source_uri,
        source_commit=args.source_commit,
        builder_id=args.builder_id,
        build_type=args.build_type,
        materials=materials,
        external_materials=external_materials,
    )
    _write_json(args.output, statement)


def _command_sbom(args: argparse.Namespace) -> None:
    digest = _resolve_subject_digest(args.subject_file, args.subject_sha256)
    sbom = canonicalize_cyclonedx(
        _read_json(args.input),
        subject_name=args.subject_name,
        subject_digest=digest,
        subject_type=args.subject_type,
        source_date_epoch=args.source_date_epoch,
    )
    _write_json(args.output, sbom)


def _command_container_evidence(args: argparse.Namespace) -> None:
    metadata = _read_json(args.buildx_metadata)
    materials = [(name, Path(path)) for name, path in (_split_mapping(value, "material") for value in args.material)]
    external_materials = (
        load_external_material_manifest(args.external_material_manifest)
        if args.external_material_manifest is not None
        else []
    )
    digest, sbom, identity = bind_container_evidence(
        metadata,
        _read_json(args.raw_sbom),
        _read_json(args.image_inspect),
        expected_local_reference=args.expected_local_reference,
        subject_name=args.subject_name,
        source_date_epoch=args.source_date_epoch,
    )
    provenance = create_provenance(
        subject_name=args.subject_name,
        subject_digest=digest,
        subject_kind="container-image",
        source_uri=args.source_uri,
        source_commit=args.source_commit,
        builder_id=args.builder_id,
        build_type=args.build_type,
        materials=materials,
        external_materials=external_materials,
    )
    _write_json(args.output_sbom, sbom)
    _write_json(args.output_provenance, provenance)
    _write_json(args.output_identity, identity)


def _target_material_digests(values: Iterable[str]) -> dict[str, str]:
    digests: dict[str, str] = {}
    for value in values:
        target, path = _split_mapping(value, "target material")
        if target in digests:
            raise EvidenceError(f"duplicate target material: {target}")
        digests[target] = _sha256(Path(path))
    return digests


def _evaluate_policy_args(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    evaluation_time = _parse_iso_datetime(args.evaluation_time, "policy evaluation time")
    reports = [(path.name, _read_json(path)) for path in args.grype_report]
    return enforce_policy(
        _read_json(args.policy),
        reports,
        _read_json(args.gitleaks_report),
        evaluation_time=evaluation_time,
        target_material_sha256=_target_material_digests(args.target_material),
    )


def _command_evaluate(args: argparse.Namespace) -> None:
    result, violations = _evaluate_policy_args(args)
    _write_json(args.output, result)
    if not result["evidence_integrity_passed"]:
        raise EvidenceError("; ".join(violations))


def _command_enforce(args: argparse.Namespace) -> None:
    result, violations = _evaluate_policy_args(args)
    _write_json(args.output, result)
    if violations:
        raise EvidenceError("; ".join(violations))


def _command_manifest(args: argparse.Namespace) -> None:
    _write_json(args.output, create_evidence_manifest(args.root, args.output))


def _command_verify_manifest(args: argparse.Namespace) -> None:
    verify_evidence_manifest(args.root, args.manifest)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare = subparsers.add_parser("compare-distributions", help="prove wheel/sdist rebuild reproducibility")
    compare.add_argument("--first", type=Path, required=True)
    compare.add_argument("--second", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.set_defaults(handler=_command_compare)

    normalize = subparsers.add_parser("normalize-sdist", help="rewrite one sdist with canonical tar/gzip metadata")
    normalize.add_argument("--input", type=Path, required=True)
    normalize.add_argument("--source-date-epoch", type=int, required=True)
    normalize.add_argument("--output", type=Path, required=True)
    normalize.set_defaults(handler=_command_normalize_sdist)

    extract = subparsers.add_parser("extract-wheel", help="safely expand a wheel for package-only SBOM cataloging")
    extract.add_argument("--input", type=Path, required=True)
    extract.add_argument("--destination", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.set_defaults(handler=_command_extract_wheel)

    model_materials = subparsers.add_parser(
        "verify-model-materials",
        help="bind the embedding model declaration to runtime defaults and container recipes",
    )
    model_materials.add_argument("--manifest", type=Path, required=True)
    model_materials.add_argument("--runtime-config", type=Path, required=True)
    model_materials.add_argument("--dockerfile", type=Path, action="append", required=True)
    model_materials.add_argument("--output", type=Path, required=True)
    model_materials.set_defaults(handler=_command_verify_model_materials)

    provenance = subparsers.add_parser("provenance", help="create a deterministic unsigned SLSA statement")
    provenance.add_argument("--subject-name", required=True)
    provenance.add_argument(
        "--subject-kind", choices=("python-wheel", "python-sdist", "container-image"), required=True
    )
    provenance_subject = provenance.add_mutually_exclusive_group(required=True)
    provenance_subject.add_argument("--subject-file", type=Path)
    provenance_subject.add_argument("--subject-sha256")
    provenance.add_argument("--source-uri", required=True)
    provenance.add_argument("--source-commit", required=True)
    provenance.add_argument("--builder-id", required=True)
    provenance.add_argument("--build-type", required=True)
    provenance.add_argument("--material", action="append", default=[])
    provenance.add_argument("--external-material-manifest", type=Path)
    provenance.add_argument("--output", type=Path, required=True)
    provenance.set_defaults(handler=_command_provenance)

    sbom = subparsers.add_parser("canonicalize-sbom", help="normalize and bind a Syft CycloneDX SBOM")
    sbom.add_argument("--input", type=Path, required=True)
    sbom.add_argument("--output", type=Path, required=True)
    sbom.add_argument("--subject-name", required=True)
    sbom.add_argument("--subject-type", choices=("file", "container"), required=True)
    sbom_subject = sbom.add_mutually_exclusive_group(required=True)
    sbom_subject.add_argument("--subject-file", type=Path)
    sbom_subject.add_argument("--subject-sha256")
    sbom.add_argument("--source-date-epoch", type=int, required=True)
    sbom.set_defaults(handler=_command_sbom)

    container = subparsers.add_parser(
        "container-evidence", help="bind Buildx result metadata and a Syft SBOM to one image digest"
    )
    container.add_argument("--buildx-metadata", type=Path, required=True)
    container.add_argument("--raw-sbom", type=Path, required=True)
    container.add_argument("--image-inspect", type=Path, required=True)
    container.add_argument("--expected-local-reference", required=True)
    container.add_argument("--output-sbom", type=Path, required=True)
    container.add_argument("--output-provenance", type=Path, required=True)
    container.add_argument("--output-identity", type=Path, required=True)
    container.add_argument("--subject-name", required=True)
    container.add_argument("--source-uri", required=True)
    container.add_argument("--source-commit", required=True)
    container.add_argument("--builder-id", required=True)
    container.add_argument("--build-type", required=True)
    container.add_argument("--source-date-epoch", type=int, required=True)
    container.add_argument("--material", action="append", default=[])
    container.add_argument("--external-material-manifest", type=Path)
    container.set_defaults(handler=_command_container_evidence)

    def add_policy_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--policy", type=Path, required=True)
        command.add_argument("--grype-report", type=Path, action="append", required=True)
        command.add_argument("--gitleaks-report", type=Path, required=True)
        command.add_argument("--target-material", action="append", default=[], metavar="TARGET=PATH")
        command.add_argument("--evaluation-time", required=True)
        command.add_argument("--output", type=Path, required=True)

    evaluate = subparsers.add_parser(
        "evaluate-policy",
        help="verify evidence integrity while reporting unexcepted vulnerabilities",
    )
    add_policy_arguments(evaluate)
    evaluate.set_defaults(handler=_command_evaluate)

    enforce = subparsers.add_parser("enforce-policy", help="fail on stale scans or unexcepted findings")
    add_policy_arguments(enforce)
    enforce.set_defaults(handler=_command_enforce)

    manifest = subparsers.add_parser("manifest", help="hash every evidence file into a canonical manifest")
    manifest.add_argument("--root", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.set_defaults(handler=_command_manifest)

    verify_manifest = subparsers.add_parser(
        "verify-manifest", help="verify the exact regular-file inventory in an evidence manifest"
    )
    verify_manifest.add_argument("--root", type=Path, required=True)
    verify_manifest.add_argument("--manifest", type=Path, required=True)
    verify_manifest.set_defaults(handler=_command_verify_manifest)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        args.handler(args)
    except EvidenceError as exc:
        print(f"supply-chain evidence failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
