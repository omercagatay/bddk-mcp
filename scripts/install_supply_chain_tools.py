#!/usr/bin/env python3
"""Install checksum-pinned supply-chain tools without executing remote scripts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

MAX_DOWNLOAD_BYTES = 160 * 1024 * 1024
MAX_TOOL_ARCHIVE_MEMBERS = 1_000
MAX_TOOL_MEMBER_BYTES = 128 * 1024 * 1024
MAX_TOOL_ARCHIVE_EXPANDED_BYTES = 512 * 1024 * 1024
ALLOWED_DOWNLOAD_HOSTS = {
    "github.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
}
EXPECTED_RELEASE_REPOSITORIES = {
    "buildx": "docker/buildx",
    "syft": "anchore/syft",
    "grype": "anchore/grype",
    "gitleaks": "gitleaks/gitleaks",
}


class ToolInstallError(RuntimeError):
    """Raised when a tool manifest or downloaded artifact is unsafe."""


def _host_platform() -> str:
    operating_system = platform.system().strip().lower()
    machine = platform.machine().strip().lower()
    normalized_machine = {
        "x86_64": "amd64",
        "amd64": "amd64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }.get(machine, machine)
    return f"{operating_system}_{normalized_machine}"


def _assert_manifest_matches_host(manifest: dict[str, Any]) -> None:
    declared = manifest.get("platform")
    actual = _host_platform()
    if declared != actual:
        raise ToolInstallError(f"tool manifest platform {declared!r} does not match host {actual!r}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ToolInstallError(f"cannot read tool manifest: {path}") from exc
    if not isinstance(value, dict):
        raise ToolInstallError("tool manifest must be a JSON object")
    return value


def _validate_https_release_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "github.com":
        raise ToolInstallError("tool URL must use an official github.com HTTPS release URL")
    parts = parsed.path.split("/")
    if len(parts) < 7 or parts[3:5] != ["releases", "download"]:
        raise ToolInstallError("tool URL must reference an immutable GitHub release asset")


def validate_manifest(manifest: dict[str, Any]) -> dict[str, dict[str, str]]:
    if manifest.get("schema_version") != 1 or manifest.get("platform") != "linux_amd64":
        raise ToolInstallError("unsupported tool manifest schema or platform")
    tools = manifest.get("tools")
    if not isinstance(tools, dict) or set(tools) != {"buildx", "syft", "grype", "gitleaks"}:
        raise ToolInstallError("tool manifest must pin buildx, syft, grype, and gitleaks")
    builder = manifest.get("builder_image")
    if not isinstance(builder, dict):
        raise ToolInstallError("tool manifest must pin the BuildKit builder image")
    reference = builder.get("reference")
    source = builder.get("source")
    builder_match = (
        re.fullmatch(r"moby/buildkit:v(\d+\.\d+\.\d+)@sha256:[0-9a-f]{64}", reference)
        if isinstance(reference, str)
        else None
    )
    if builder_match is None or source != (f"https://github.com/moby/buildkit/releases/tag/v{builder_match.group(1)}"):
        raise ToolInstallError("BuildKit image must use a digest and matching official release source")

    validated: dict[str, dict[str, str]] = {}
    for name, raw in tools.items():
        if not isinstance(raw, dict):
            raise ToolInstallError(f"invalid tool record: {name}")
        required = {"version", "url", "sha256", "format", "binary", "source"}
        if not required.issubset(raw):
            raise ToolInstallError(f"incomplete tool record: {name}")
        record = {key: str(raw[key]) for key in required}
        if re.fullmatch(r"\d+\.\d+\.\d+", record["version"]) is None:
            raise ToolInstallError(f"tool version is not exact: {name}")
        if len(record["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in record["sha256"]):
            raise ToolInstallError(f"invalid SHA-256 pin: {name}")
        if record["format"] not in {"binary", "tar.gz"}:
            raise ToolInstallError(f"unsupported archive format: {name}")
        if Path(record["binary"]).name != record["binary"]:
            raise ToolInstallError(f"unsafe binary name: {name}")
        _validate_https_release_url(record["url"])
        repository = EXPECTED_RELEASE_REPOSITORIES[name]
        tag = f"v{record['version']}"
        release_prefix = f"/{repository}/releases/download/{tag}/"
        if not urllib.parse.urlparse(record["url"]).path.startswith(release_prefix):
            raise ToolInstallError(f"tool URL does not match its official repository and version: {name}")
        expected_source = f"https://github.com/{repository}/releases/tag/{tag}"
        if record["source"] != expected_source:
            raise ToolInstallError(f"tool source does not match its official release: {name}")
        if record["format"] == "tar.gz":
            member = raw.get("member")
            if not isinstance(member, str) or Path(member).name != member:
                raise ToolInstallError(f"unsafe or missing archive member: {name}")
            record["member"] = member
        validated[str(name)] = record
    return validated


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "bddk-mcp-supply-chain/1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as output:  # noqa: S310
            final = urllib.parse.urlparse(response.geturl())
            if final.scheme != "https" or final.hostname not in ALLOWED_DOWNLOAD_HOSTS:
                raise ToolInstallError("release download redirected to an unapproved host")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > MAX_DOWNLOAD_BYTES:
                raise ToolInstallError("release artifact exceeds the download limit")
            written = 0
            while chunk := response.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_DOWNLOAD_BYTES:
                    raise ToolInstallError("release artifact exceeds the download limit")
                output.write(chunk)
    except (OSError, ValueError) as exc:
        raise ToolInstallError("failed to download a pinned release artifact") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_binary(archive: Path, member_name: str, destination: Path) -> None:
    try:
        with tarfile.open(archive, mode="r:gz") as bundle:
            members = bundle.getmembers()
            if not 1 <= len(members) <= MAX_TOOL_ARCHIVE_MEMBERS:
                raise ToolInstallError("release archive member count exceeds the safe limit")
            names: set[str] = set()
            expanded_size = 0
            for member in members:
                parsed = PurePosixPath(member.name)
                if parsed.is_absolute() or ".." in parsed.parts or not member.name or member.name in names:
                    raise ToolInstallError("release archive contains an unsafe or duplicate member name")
                names.add(member.name)
                if member.size < 0 or member.size > MAX_TOOL_MEMBER_BYTES:
                    raise ToolInstallError("release archive member exceeds the uncompressed size limit")
                expanded_size += member.size
                if expanded_size > MAX_TOOL_ARCHIVE_EXPANDED_BYTES:
                    raise ToolInstallError("release archive exceeds the expanded-size limit")
            matching = [member for member in members if member.name == member_name]
            if len(matching) != 1 or not matching[0].isfile():
                raise ToolInstallError("release archive does not contain exactly one expected binary")
            source = bundle.extractfile(matching[0])
            if source is None:
                raise ToolInstallError("cannot read expected binary from release archive")
            payload = source.read(MAX_TOOL_MEMBER_BYTES + 1)
            if len(payload) != matching[0].size:
                raise ToolInstallError("release binary size does not match its archive metadata")
            with destination.open("wb") as output:
                output.write(payload)
    except (OSError, tarfile.TarError) as exc:
        raise ToolInstallError("invalid release archive") from exc


def install_tools(manifest_path: Path, destination: Path) -> dict[str, str]:
    manifest = _load_json(manifest_path)
    tools = validate_manifest(manifest)
    _assert_manifest_matches_host(manifest)
    destination.mkdir(parents=True, exist_ok=True)
    installed: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="bddk-tools-") as temporary:
        temporary_path = Path(temporary)
        for name in sorted(tools):
            record = tools[name]
            download = temporary_path / f"{name}.download"
            _download(record["url"], download)
            if _sha256(download) != record["sha256"]:
                raise ToolInstallError(f"checksum mismatch for {name}")

            target = destination / record["binary"]
            staged = temporary_path / f"{name}.binary"
            if record["format"] == "binary":
                shutil.copyfile(download, staged)
            else:
                _extract_binary(download, record["member"], staged)
            os.chmod(staged, 0o755)
            os.replace(staged, target)
            installed[name] = record["version"]

    versions_path = destination / "versions.json"
    versions_path.write_text(json.dumps(installed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return installed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("supply-chain/tools.json"))
    parser.add_argument("--destination", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        installed = install_tools(args.manifest, args.destination)
    except ToolInstallError as exc:
        print(f"supply-chain tool installation failed: {exc}")
        return 2
    print("installed checksum-verified tools: " + ", ".join(f"{name}={version}" for name, version in installed.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
