"""Secret-free, offline acceptance checks for the OpenShift deployment starter.

The checks in this module validate repository-controlled inputs only.  They do
not contact an OpenShift cluster, an identity provider, PostgreSQL, or an image
registry and therefore cannot certify a bank deployment.  The generated
evidence deliberately hashes environment-specific identifiers and records all
live checks as pending external gates.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

_MAX_INPUT_BYTES = 1_048_576
_MAX_RENDERED_BYTES = 16_777_216
_KUSTOMIZE_VERSION = "v5.8.1"
_IMAGE_PLACEHOLDER = "REPLACE_IMAGE_REGISTRY/bddk-mcp@sha256:REPLACE_64_HEX_IMAGE_DIGEST"
_DIGEST_IMAGE = re.compile(r"^[^\s@]+@sha256:([a-f0-9]{64})$")
_DNS_NAME = re.compile(
    r"^(?=.{1,253}\.?$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.?$"
)
_DNS_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")
_HEX_40_64 = re.compile(r"^(?:[a-f0-9]{40}|[a-f0-9]{64})$")
_RELEASE_REQUEST_ID = re.compile(r"^corpus_release_request_sha256_[a-f0-9]{64}$")
_PLACEHOLDER = re.compile(r"REPLACE_[A-Z0-9_]+")
_REGULATORY_EGRESS_PURPOSES = frozenset({"regulatory_source", "enterprise_proxy"})
_SAFE_ERROR_FIELDS = frozenset(
    {
        "schema_version",
        "release",
        "version",
        "image",
        "previous_image",
        "kustomize_binary_sha256",
        "manifest_revision",
        "release_request_id",
        "platform",
        "namespace",
        "public_route_host",
        "operator_service_host",
        "operator_client_origin",
        "database_name",
        "jwt",
        "issuer",
        "jwks_url",
        "public_audience",
        "operator_audience",
        "public_required_scopes",
        "operator_required_scopes",
        "scope_claims",
        "algorithms",
        "access_token_types",
        "egress_policy_files",
        "required_egress",
        "id",
        "policy",
        "component",
        "purpose",
        "protocol",
        "port",
        "peer",
        "namespaceSelector",
        "podSelector",
        "ipBlock",
        "matchLabels",
        "cidr",
        "except",
        "rollback",
        "backup_evidence_id",
        "restore_drill_evidence_sha256",
        "runbook_revision",
        "database_strategy",
        "maximum_recovery_minutes",
    }
)


class OpenShiftAcceptanceError(RuntimeError):
    """A privacy-safe preflight input or repository error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _validate_dns_name(value: str, *, field: str) -> str:
    normalized = value.strip().lower().rstrip(".")
    if not _DNS_NAME.fullmatch(normalized):
        raise ValueError(f"{field} must be a valid DNS name")
    return normalized


def _validate_https(value: str, *, field: str, origin_only: bool = False) -> str:
    candidate = value.strip()
    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{field} must be a valid HTTPS URL") from exc
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError(f"{field} must be an absolute HTTPS URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(f"{field} must not contain credentials, query, or fragment")
    if port is not None and not 1 <= port <= 65535:
        raise ValueError(f"{field} contains an invalid port")
    _validate_dns_name(parsed.hostname, field=field)
    if origin_only and parsed.path not in ("", "/"):
        raise ValueError(f"{field} must be an origin without a path")
    return candidate.rstrip("/") if origin_only else candidate


def _validate_image(value: str) -> str:
    candidate = value.strip()
    match = _DIGEST_IMAGE.fullmatch(candidate)
    if not match or "REPLACE_" in candidate:
        raise ValueError("image must be an immutable sha256 digest reference")
    prefix = candidate.split("@", maxsplit=1)[0]
    if "://" in prefix or prefix.startswith(("/", ".")) or "/" not in prefix:
        raise ValueError("image must be a scheme-free registry/repository reference")
    final_component = prefix.rsplit("/", maxsplit=1)[-1]
    if ":" in final_component:
        raise ValueError("image must not combine a mutable tag with a digest")
    return candidate


class ReleaseInput(_StrictModel):
    version: str
    image: str
    previous_image: str
    manifest_revision: str
    release_request_id: str
    kustomize_binary_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator("version")
    @classmethod
    def _valid_version(cls, value: str) -> str:
        if not _VERSION.fullmatch(value):
            raise ValueError("version must be a semantic release version")
        return value

    @field_validator("image", "previous_image")
    @classmethod
    def _valid_image(cls, value: str) -> str:
        return _validate_image(value)

    @field_validator("manifest_revision")
    @classmethod
    def _valid_revision(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _HEX_40_64.fullmatch(normalized):
            raise ValueError("manifest_revision must be a full Git or content digest")
        return normalized

    @field_validator("release_request_id")
    @classmethod
    def _valid_release_request_id(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _RELEASE_REQUEST_ID.fullmatch(normalized):
            raise ValueError("release_request_id must identify one staged corpus release request")
        return normalized

    @model_validator(mode="after")
    def _images_differ(self) -> ReleaseInput:
        if self.image == self.previous_image:
            raise ValueError("current and rollback images must differ")
        return self


class PlatformInput(_StrictModel):
    namespace: str
    public_route_host: str
    operator_service_host: str
    operator_client_origin: str
    database_name: str

    @field_validator("namespace")
    @classmethod
    def _valid_namespace(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _DNS_LABEL.fullmatch(normalized):
            raise ValueError("namespace must be a DNS label")
        return normalized

    @field_validator("public_route_host", "operator_service_host")
    @classmethod
    def _valid_host(cls, value: str, info) -> str:
        return _validate_dns_name(value, field=info.field_name)

    @field_validator("operator_client_origin")
    @classmethod
    def _valid_origin(cls, value: str) -> str:
        return _validate_https(value, field="operator_client_origin", origin_only=True)

    @field_validator("database_name")
    @classmethod
    def _valid_database_name(cls, value: str) -> str:
        normalized = value.strip()
        if not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", normalized):
            raise ValueError("database_name must be an unquoted PostgreSQL identifier")
        return normalized

    @model_validator(mode="after")
    def _operator_host_matches_namespace(self) -> PlatformInput:
        expected = f"bddk-mcp-operator.{self.namespace}.svc"
        if self.operator_service_host != expected:
            raise ValueError("operator_service_host must be the exact in-namespace operator Service DNS name")
        return self


class JwtInput(_StrictModel):
    issuer: str
    jwks_url: str
    public_audience: str
    operator_audience: str
    public_required_scopes: tuple[str, ...] = ("bddk.read",)
    operator_required_scopes: tuple[str, ...] = ("bddk.operator",)
    scope_claims: tuple[Literal["scope", "scp"], ...] = ("scope", "scp")
    algorithms: tuple[Literal["RS256"], ...] = ("RS256",)
    access_token_types: tuple[Literal["at+jwt"], ...] = ("at+jwt",)

    @field_validator("issuer", "jwks_url")
    @classmethod
    def _valid_url(cls, value: str, info) -> str:
        return _validate_https(value, field=info.field_name)

    @field_validator("public_audience", "operator_audience")
    @classmethod
    def _valid_audience(cls, value: str, info) -> str:
        candidate = value.strip()
        if not _IDENTIFIER.fullmatch(candidate) or "REPLACE_" in candidate:
            raise ValueError(f"{info.field_name} must be a visible non-placeholder identifier")
        return candidate

    @model_validator(mode="after")
    def _exact_contract(self) -> JwtInput:
        if self.public_audience == self.operator_audience:
            raise ValueError("public and operator audiences must differ")
        if self.public_required_scopes != ("bddk.read",):
            raise ValueError("public_required_scopes must be exactly bddk.read")
        if self.operator_required_scopes != ("bddk.operator",):
            raise ValueError("operator_required_scopes must be exactly bddk.operator")
        if self.scope_claims != ("scope", "scp"):
            raise ValueError("scope_claims must describe the exact supported scope and scp claims")
        if self.algorithms != ("RS256",) or self.access_token_types != ("at+jwt",):
            raise ValueError("JWT algorithm and token type must retain the reviewed fail-closed profile")
        return self


class LabelSelector(_StrictModel):
    match_labels: dict[str, str] = Field(alias="matchLabels", min_length=1)

    @field_validator("match_labels")
    @classmethod
    def _safe_labels(cls, value: dict[str, str]) -> dict[str, str]:
        if any(not key.strip() or not item.strip() or "REPLACE_" in item for key, item in value.items()):
            raise ValueError("label selectors must contain resolved non-empty labels")
        return value


class IpBlock(_StrictModel):
    cidr: str
    except_: tuple[str, ...] = Field(default=(), alias="except")

    @field_validator("cidr")
    @classmethod
    def _bounded_cidr(cls, value: str) -> str:
        import ipaddress

        network = ipaddress.ip_network(value, strict=True)
        if network.prefixlen == 0:
            raise ValueError("a whole-Internet egress CIDR is forbidden")
        return str(network)

    @field_validator("except_")
    @classmethod
    def _valid_exceptions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        import ipaddress

        return tuple(str(ipaddress.ip_network(item, strict=True)) for item in value)

    @model_validator(mode="after")
    def _exceptions_are_subnets(self) -> IpBlock:
        import ipaddress

        network = ipaddress.ip_network(self.cidr)
        exceptions = [ipaddress.ip_network(item) for item in self.except_]
        if len(exceptions) != len(set(exceptions)) or any(
            item.version != network.version or not item.subnet_of(network) for item in exceptions
        ):
            raise ValueError("ipBlock exceptions must be unique subnets of cidr")
        return self


class EgressPeer(_StrictModel):
    namespace_selector: LabelSelector | None = Field(default=None, alias="namespaceSelector")
    pod_selector: LabelSelector | None = Field(default=None, alias="podSelector")
    ip_block: IpBlock | None = Field(default=None, alias="ipBlock")

    @model_validator(mode="after")
    def _not_open(self) -> EgressPeer:
        if self.ip_block is not None and (self.namespace_selector is not None or self.pod_selector is not None):
            raise ValueError("ipBlock cannot be combined with pod or namespace selectors")
        if self.ip_block is None and self.namespace_selector is None and self.pod_selector is None:
            raise ValueError("egress peer must constrain a destination")
        return self


class EgressRequirement(_StrictModel):
    id: str
    policy: str
    component: Literal["public", "operator", "lifecycle"]
    purpose: Literal["dns", "postgresql", "idp_jwks", "regulatory_source", "enterprise_proxy"]
    protocol: Literal["TCP", "UDP"]
    port: int = Field(ge=1, le=65535)
    peer: EgressPeer

    @field_validator("id", "policy")
    @classmethod
    def _safe_identifier(cls, value: str, info) -> str:
        if not _DNS_LABEL.fullmatch(value):
            raise ValueError(f"{info.field_name} must be a DNS label")
        return value


class RollbackInput(_StrictModel):
    backup_evidence_id: str
    restore_drill_evidence_sha256: str
    runbook_revision: str
    database_strategy: Literal["restore", "forward-fix"]
    maximum_recovery_minutes: int = Field(ge=1, le=10_080)

    @field_validator("backup_evidence_id")
    @classmethod
    def _safe_evidence_id(cls, value: str) -> str:
        candidate = value.strip()
        if not _IDENTIFIER.fullmatch(candidate) or any(
            token in candidate.lower() for token in ("secret", "token", "password")
        ):
            raise ValueError("backup_evidence_id must be a non-sensitive opaque identifier")
        return candidate

    @field_validator("restore_drill_evidence_sha256")
    @classmethod
    def _full_sha256(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not re.fullmatch(r"[a-f0-9]{64}", normalized):
            raise ValueError("restore_drill_evidence_sha256 must be a SHA-256 digest")
        return normalized

    @field_validator("runbook_revision")
    @classmethod
    def _runbook_revision(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _HEX_40_64.fullmatch(normalized):
            raise ValueError("runbook_revision must be a full Git or content digest")
        return normalized


class AcceptanceInput(_StrictModel):
    schema_version: Literal[1]
    release: ReleaseInput
    platform: PlatformInput
    jwt: JwtInput
    egress_policy_files: tuple[str, ...] = Field(min_length=1)
    required_egress: tuple[EgressRequirement, ...] = Field(min_length=1)
    rollback: RollbackInput

    @field_validator("egress_policy_files")
    @classmethod
    def _safe_relative_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for raw in value:
            path = Path(raw)
            if path.is_absolute() or ".." in path.parts or path.suffix not in {".yaml", ".yml"}:
                raise ValueError("egress policy paths must be relative YAML paths without traversal")
            normalized.append(path.as_posix())
        if len(set(normalized)) != len(normalized):
            raise ValueError("egress policy paths must be unique")
        return tuple(normalized)

    @model_validator(mode="after")
    def _complete_egress_matrix(self) -> AcceptanceInput:
        ids = [item.id for item in self.required_egress]
        if len(ids) != len(set(ids)):
            raise ValueError("required egress identifiers must be unique")
        permissions = [
            (
                item.policy,
                item.component,
                item.protocol,
                item.port,
                item.peer.model_dump_json(by_alias=True, exclude_none=True, exclude_defaults=True),
            )
            for item in self.required_egress
        ]
        if len(permissions) != len(set(permissions)):
            raise ValueError("required egress permissions must be unique")
        required_pairs = {
            ("public", "dns"),
            ("public", "postgresql"),
            ("public", "idp_jwks"),
            ("operator", "dns"),
            ("operator", "postgresql"),
            ("operator", "idp_jwks"),
            ("lifecycle", "dns"),
            ("lifecycle", "postgresql"),
        }
        observed = {(item.component, item.purpose) for item in self.required_egress}
        missing = required_pairs - observed
        missing_regulatory = {
            component
            for component in ("public", "operator")
            if not any(
                item.component == component
                and item.purpose in _REGULATORY_EGRESS_PURPOSES
                and item.protocol == "TCP"
                and item.port == 443
                for item in self.required_egress
            )
        }
        if missing or missing_regulatory:
            raise ValueError("required_egress does not cover every runtime and lifecycle dependency")
        if any(
            item.purpose in _REGULATORY_EGRESS_PURPOSES and (item.protocol != "TCP" or item.port != 443)
            for item in self.required_egress
        ):
            raise ValueError("regulatory source and enterprise proxy egress must be TCP port 443")
        if any(
            item.component == "lifecycle" and item.purpose not in {"dns", "postgresql"} for item in self.required_egress
        ):
            raise ValueError("lifecycle egress is limited to DNS and PostgreSQL")
        if not any(
            item.protocol == "UDP" and item.port == 53 and item.purpose == "dns" for item in self.required_egress
        ):
            raise ValueError("required_egress must include UDP DNS")
        return self


@dataclass(frozen=True, slots=True)
class CheckResult:
    id: str
    status: Literal["pass", "fail"]
    summary: str


@dataclass(frozen=True, slots=True)
class AcceptanceEvidence:
    status: Literal["preflight_passed_external_gates_pending", "preflight_failed"]
    generated_at: str
    input_sha256: str
    rendered_manifest_sha256: str
    release_version: str
    image_digest: str
    previous_image_digest: str
    renderer_sha256: str
    environment_fingerprint: str
    rollback_evidence: dict[str, str | int]
    checks: tuple[CheckResult, ...]
    external_gates: tuple[dict[str, str | bool], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "evidence_scope": "repository_offline_preflight_only",
            "bank_cluster_acceptance": False,
            "status": self.status,
            "generated_at": self.generated_at,
            "input_sha256": self.input_sha256,
            "rendered_manifest_sha256": self.rendered_manifest_sha256,
            "release_version": self.release_version,
            "image_digest": self.image_digest,
            "previous_image_digest": self.previous_image_digest,
            "renderer_sha256": self.renderer_sha256,
            "environment_fingerprint": self.environment_fingerprint,
            "rollback_evidence": self.rollback_evidence,
            "checks": [{"id": check.id, "status": check.status, "summary": check.summary} for check in self.checks],
            "external_gates": list(self.external_gates),
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n"


_EXTERNAL_GATES: tuple[dict[str, str | bool], ...] = tuple(
    {
        "id": identifier,
        "status": "not_run",
        "required_before_production": True,
        "reason": reason,
    }
    for identifier, reason in (
        ("cluster-admission", "requires target SCC, image policy, quota, and admission controllers"),
        ("route-service-ca", "requires live re-encrypt, operator-service, rotation, and CA-rollover tests"),
        ("idp-token-contract", "requires bank-issued access tokens and rejection cases"),
        ("postgresql-identities", "requires bank TLS, extensions, LOGIN membership, ACL, failover, and capacity tests"),
        ("network-enforcement", "requires the target CNI, DNS, firewall, proxy, and negative-connectivity tests"),
        (
            "lifecycle-jobs",
            "requires the approved corpus PVC, corpus-trust Secret, and ordered migrate, strict bootstrap, "
            "verify/stage, and activation Jobs in an isolated bank-like namespace",
        ),
        ("backup-restore-rollback", "requires a target backup, restore, upgrade, and rollback drill"),
        ("client-model-matrix", "requires release-specific MCP clients, models, citations, load, and timeout tests"),
    )
)


def _read_bounded(path: Path) -> bytes:
    try:
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
    except OSError as exc:
        raise OpenShiftAcceptanceError("input-unavailable", "an acceptance input file is unavailable") from exc
    if not resolved.is_file() or stat.st_size > _MAX_INPUT_BYTES:
        raise OpenShiftAcceptanceError("input-bounds", "an acceptance input file is not a bounded regular file")
    try:
        with resolved.open("rb") as handle:
            raw = handle.read(_MAX_INPUT_BYTES + 1)
    except OSError as exc:
        raise OpenShiftAcceptanceError("input-unavailable", "an acceptance input file is unavailable") from exc
    if len(raw) > _MAX_INPUT_BYTES:
        raise OpenShiftAcceptanceError("input-bounds", "an acceptance input file is not a bounded regular file")
    return raw


def load_acceptance_input(path: Path) -> tuple[AcceptanceInput, str]:
    """Load a bounded, strict, secret-free acceptance configuration."""
    raw = _read_bounded(path)
    try:
        payload = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise OpenShiftAcceptanceError("invalid-yaml", "acceptance input is not valid YAML") from exc
    if not isinstance(payload, dict):
        raise OpenShiftAcceptanceError("invalid-shape", "acceptance input must be a YAML mapping")
    if _PLACEHOLDER.search(json.dumps(payload, ensure_ascii=True, sort_keys=True)):
        raise OpenShiftAcceptanceError(
            "unresolved-config-placeholder", "acceptance input contains unresolved placeholders"
        )
    try:
        config = AcceptanceInput.model_validate(payload)
    except ValidationError as exc:
        locations = sorted({_safe_error_location(item["loc"]) for item in exc.errors(include_input=False)})
        suffix = ", ".join(locations[:8]) or "configuration"
        raise OpenShiftAcceptanceError("invalid-config", f"acceptance input is invalid at: {suffix}") from exc
    return config, hashlib.sha256(raw).hexdigest()


def _safe_error_location(location: tuple[str | int, ...]) -> str:
    pieces: list[str] = []
    for piece in location:
        if isinstance(piece, int):
            pieces.append("[]")
        elif piece in _SAFE_ERROR_FIELDS:
            pieces.append(piece)
        else:
            pieces.append("<extra>")
    return ".".join(pieces)


def _load_yaml_documents(path: Path) -> list[dict[str, Any]]:
    raw = _read_bounded(path)
    try:
        documents = [item for item in yaml.safe_load_all(raw) if item is not None]
    except yaml.YAMLError as exc:
        raise OpenShiftAcceptanceError("invalid-manifest", "a deployment manifest is not valid YAML") from exc
    if not documents or any(not isinstance(item, dict) for item in documents):
        raise OpenShiftAcceptanceError("invalid-manifest", "a deployment manifest has an invalid document shape")
    return documents


_RUNTIME_RESOURCES = (
    "serviceaccounts.yaml",
    "service-ca.yaml",
    "configmaps.yaml",
    "public-deployment.yaml",
    "operator-deployment.yaml",
    "services.yaml",
    "public-route.yaml",
    "networkpolicies.yaml",
)
_LIFECYCLE_RESOURCES = (
    "jobs/migrate.yaml",
    "jobs/bootstrap.yaml",
    "jobs/verify-stage-release.yaml",
    "jobs/activate-release.yaml",
)
_BANK_BOOTSTRAP_OVERLAY = "openshift-overlays/bank-bootstrap"
_BANK_BOOTSTRAP_PATCH = "bootstrap-job-patch.yaml"
_BANK_BOOTSTRAP_RESOURCES = (
    "../../openshift",
    "../../openshift/jobs",
)
_BANK_BOOTSTRAP_ARGS = [
    ".venv/bin/bddk-mcp",
    "bootstrap",
    "--seed-dir",
    "/var/run/bddk-mcp/corpus",
    "--reindex-existing",
    "--require-quantified-freshness",
    "--require-measured-freshness",
    "--require-verified-signature",
    "--trusted-signing-key",
    "/var/run/secrets/bddk-mcp/corpus-trust/ed25519-public-key.pem",
]
_VERIFY_STAGE_RELEASE_ARGS = [
    ".venv/bin/bddk-mcp",
    "verify-and-stage-corpus-release",
    "--seed-dir",
    "/var/run/bddk-mcp/corpus",
    "--trusted-signing-key",
    "/var/run/secrets/bddk-mcp/corpus-trust/ed25519-public-key.pem",
]
_ACTIVATE_RELEASE_ARGS = [
    ".venv/bin/bddk-mcp",
    "activate-corpus-release",
    "--request-id",
    "$(BDDK_RELEASE_REQUEST_ID)",
]
_BANK_CORPUS_MOUNT = {
    "name": "approved-corpus",
    "mountPath": "/var/run/bddk-mcp/corpus",
    "readOnly": True,
}
_BANK_TRUST_MOUNT = {
    "name": "corpus-signing-key",
    "mountPath": "/var/run/secrets/bddk-mcp/corpus-trust",
    "readOnly": True,
}
_BANK_CORPUS_VOLUME = {
    "name": "approved-corpus",
    "persistentVolumeClaim": {"claimName": "bddk-mcp-approved-corpus", "readOnly": True},
}
_BANK_TRUST_VOLUME = {
    "name": "corpus-signing-key",
    "secret": {
        "secretName": "bddk-mcp-corpus-trust",
        "defaultMode": 0o440,
        "items": [{"key": "ed25519-public-key.pem", "path": "ed25519-public-key.pem"}],
    },
}
_BASE_RENDERED_RESOURCES = frozenset(
    {
        *(
            ("v1", "ServiceAccount", f"bddk-mcp-{component}")
            for component in (
                "public",
                "operator",
                "lifecycle",
                "ingestion",
                "release-verifier",
                "release-publisher",
            )
        ),
        ("v1", "ConfigMap", "bddk-mcp-service-ca"),
        ("v1", "ConfigMap", "bddk-mcp-public-config"),
        ("v1", "ConfigMap", "bddk-mcp-operator-config"),
        ("apps/v1", "Deployment", "bddk-mcp-public"),
        ("apps/v1", "Deployment", "bddk-mcp-operator"),
        ("v1", "Service", "bddk-mcp-public"),
        ("v1", "Service", "bddk-mcp-operator"),
        ("route.openshift.io/v1", "Route", "bddk-mcp-public"),
        ("networking.k8s.io/v1", "NetworkPolicy", "bddk-mcp-default-deny-egress"),
        ("networking.k8s.io/v1", "NetworkPolicy", "bddk-mcp-default-deny-ingress"),
        ("networking.k8s.io/v1", "NetworkPolicy", "bddk-mcp-public-from-router"),
        ("networking.k8s.io/v1", "NetworkPolicy", "bddk-mcp-operator-from-approved-clients"),
    }
)
_COMMON_CONFIG_KEYS = {
    "MCP_TRANSPORT",
    "MCP_HOST",
    "PORT",
    "BDDK_TOOL_PROFILE",
    "BDDK_AUTO_SYNC",
    "BDDK_ALLOW_INSECURE_DATABASE",
    "BDDK_REQUIRE_ACTIVE_CORPUS_RELEASE",
    "BDDK_TELEMETRY_ENABLED",
    "BDDK_HTTP_ALLOWED_HOSTS",
    "BDDK_HTTP_ALLOWED_ORIGINS",
    "BDDK_JWT_ISSUER",
    "BDDK_JWT_RESOURCE",
    "BDDK_JWT_JWKS_URL",
    "BDDK_JWT_AUDIENCE",
    "BDDK_JWT_REQUIRED_SCOPES",
    "BDDK_JWT_ALGORITHMS",
    "BDDK_JWT_ACCESS_TOKEN_TYPES",
    "BDDK_HTTP_MAX_BODY_BYTES",
    "BDDK_HTTP_MAX_CONCURRENCY",
    "BDDK_HTTP_RATE_LIMIT_PER_MINUTE",
}
_PUBLIC_CONFIG_KEYS = _COMMON_CONFIG_KEYS
_OPERATOR_CONFIG_KEYS = _COMMON_CONFIG_KEYS | {
    "BDDK_OPERATOR_REMOTE_ENABLED",
    "BDDK_OPERATOR_JOB_DRAIN_TIMEOUT",
    "BDDK_OPERATOR_JOB_HISTORY",
}


def _validated_kustomization(openshift: Path) -> dict[str, Any]:
    raw = _read_bounded(openshift / "kustomization.yaml")
    try:
        kustomization = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise OpenShiftAcceptanceError("invalid-kustomization", "Kustomization is not valid YAML") from exc
    if not isinstance(kustomization, dict) or not isinstance(kustomization.get("resources"), list):
        raise OpenShiftAcceptanceError("invalid-kustomization", "Kustomization resource inventory is invalid")
    resources = kustomization["resources"]
    if len(resources) != len(set(resources)) or set(resources) != set(_RUNTIME_RESOURCES):
        raise OpenShiftAcceptanceError(
            "invalid-kustomization", "Kustomization must contain the exact reviewed runtime resource inventory"
        )
    return kustomization


def _validated_lifecycle_kustomization(openshift: Path) -> None:
    try:
        kustomization = yaml.safe_load(_read_bounded(openshift / "jobs" / "kustomization.yaml"))
    except yaml.YAMLError as exc:
        raise OpenShiftAcceptanceError("invalid-kustomization", "lifecycle Kustomization is not valid YAML") from exc
    if kustomization != {
        "apiVersion": "kustomize.config.k8s.io/v1beta1",
        "kind": "Kustomization",
        "resources": ["migrate.yaml", "bootstrap.yaml", "verify-stage-release.yaml", "activate-release.yaml"],
    }:
        raise OpenShiftAcceptanceError(
            "invalid-kustomization", "lifecycle Kustomization must contain the exact reviewed Job inventory"
        )


def _validated_bank_bootstrap_overlay(repository_root: Path) -> dict[str, Any]:
    overlay_root = repository_root / "deploy" / _BANK_BOOTSTRAP_OVERLAY
    try:
        overlay = yaml.safe_load(_read_bounded(overlay_root / "kustomization.yaml"))
        patch = yaml.safe_load(_read_bounded(overlay_root / _BANK_BOOTSTRAP_PATCH))
    except yaml.YAMLError as exc:
        raise OpenShiftAcceptanceError(
            "invalid-bank-bootstrap-overlay", "bank bootstrap overlay is not valid YAML"
        ) from exc

    expected_overlay = {
        "apiVersion": "kustomize.config.k8s.io/v1beta1",
        "kind": "Kustomization",
        "resources": list(_BANK_BOOTSTRAP_RESOURCES),
        "patches": [
            {
                "path": _BANK_BOOTSTRAP_PATCH,
                "target": {
                    "group": "batch",
                    "version": "v1",
                    "kind": "Job",
                    "name": "bddk-mcp-bootstrap-v5-0-1",
                },
            }
        ],
        "labels": [
            {
                "pairs": {"app.kubernetes.io/name": "bddk-mcp"},
                "includeSelectors": True,
                "includeTemplates": True,
            },
            {
                "pairs": {"app.kubernetes.io/version": "5.0.1"},
                "includeSelectors": False,
                "includeTemplates": True,
            },
        ],
    }
    expected_patch = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": "bddk-mcp-bootstrap-v5-0-1"},
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "bootstrap",
                            "args": _BANK_BOOTSTRAP_ARGS,
                            "volumeMounts": [_BANK_CORPUS_MOUNT, _BANK_TRUST_MOUNT],
                        }
                    ],
                    "volumes": [_BANK_CORPUS_VOLUME, _BANK_TRUST_VOLUME],
                }
            }
        },
    }
    if overlay != expected_overlay or patch != expected_patch:
        raise OpenShiftAcceptanceError(
            "invalid-bank-bootstrap-overlay",
            "bank bootstrap overlay differs from the exact reviewed promotion contract",
        )
    return overlay


def _bounded_file_sha256(path: Path, *, maximum_bytes: int) -> str:
    try:
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as exc:
        raise OpenShiftAcceptanceError("renderer-identity", "renderer binary identity is unavailable") from exc
    if not resolved.is_file() or not 1 <= metadata.st_size <= maximum_bytes:
        raise OpenShiftAcceptanceError("renderer-identity", "renderer binary is not a bounded regular file")
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise OpenShiftAcceptanceError("renderer-identity", "renderer binary identity is unavailable") from exc
    return digest.hexdigest()


def _kustomize_executable(expected_sha256: str) -> str:
    executable = shutil.which("kustomize")
    if executable is None:
        raise OpenShiftAcceptanceError(
            "kustomize-unavailable", f"checksum-verified Kustomize {_KUSTOMIZE_VERSION} is required for preflight"
        )
    try:
        result = subprocess.run(
            [executable, "version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OpenShiftAcceptanceError("kustomize-unavailable", "Kustomize version verification failed") from exc
    versions = re.findall(r"v\d+\.\d+\.\d+", result.stdout)
    if result.returncode != 0 or versions != [_KUSTOMIZE_VERSION]:
        raise OpenShiftAcceptanceError(
            "kustomize-version", f"preflight requires exactly Kustomize {_KUSTOMIZE_VERSION}"
        )
    if _bounded_file_sha256(Path(executable), maximum_bytes=256 * 1024 * 1024) != expected_sha256:
        raise OpenShiftAcceptanceError("kustomize-digest", "Kustomize binary differs from the reviewed SHA-256")
    return executable


def _replace_resource_placeholders(path: Path, substitutions: dict[str, str]) -> None:
    try:
        rendered = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise OpenShiftAcceptanceError("invalid-manifest", "a deployment manifest is unavailable") from exc
    for placeholder, value in substitutions.items():
        rendered = rendered.replace(placeholder, value)
    try:
        path.write_text(rendered, encoding="utf-8")
    except OSError as exc:
        raise OpenShiftAcceptanceError("render-failed", "temporary manifest preparation failed") from exc


def _render_repository_documents(
    repository_root: Path,
    substitutions: dict[str, str],
    namespace: str,
    egress: list[dict[str, Any]],
    kustomize_binary_sha256: str,
    release_version: str,
    expected_egress_policies: frozenset[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bytes]:
    source = repository_root / "deploy" / "openshift"
    _validated_kustomization(source)
    _validated_lifecycle_kustomization(source)
    overlay_source = repository_root / "deploy" / _BANK_BOOTSTRAP_OVERLAY
    overlay = _validated_bank_bootstrap_overlay(repository_root)
    executable = _kustomize_executable(kustomize_binary_sha256)
    try:
        with tempfile.TemporaryDirectory(prefix="bddk-mcp-kustomize-") as temporary:
            target_deploy = Path(temporary) / "deploy"
            target = target_deploy / "openshift"
            shutil.copytree(source, target)
            overlay_target = target_deploy / _BANK_BOOTSTRAP_OVERLAY
            shutil.copytree(overlay_source, overlay_target)
            for relative in (*_RUNTIME_RESOURCES, *_LIFECYCLE_RESOURCES):
                _replace_resource_placeholders(target / relative, substitutions)

            egress_name = "acceptance-egress.generated.yaml"
            (overlay_target / egress_name).write_text(yaml.safe_dump_all(egress, sort_keys=True), encoding="utf-8")
            overlay["resources"] = [*_BANK_BOOTSTRAP_RESOURCES, egress_name]
            overlay["namespace"] = namespace
            (overlay_target / "kustomization.yaml").write_text(
                yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8"
            )
            result = subprocess.run(
                [executable, "build", str(overlay_target)],
                check=False,
                capture_output=True,
                timeout=30,
            )
    except subprocess.TimeoutExpired as exc:
        raise OpenShiftAcceptanceError("render-timeout", "Kustomize render exceeded the bounded timeout") from exc
    except OSError as exc:
        raise OpenShiftAcceptanceError("render-failed", "Kustomize render could not be executed") from exc

    if result.returncode != 0:
        raise OpenShiftAcceptanceError("render-failed", "Kustomize rejected the deployment configuration")
    if not result.stdout or len(result.stdout) > _MAX_RENDERED_BYTES:
        raise OpenShiftAcceptanceError("render-bounds", "Kustomize output is empty or exceeds the size limit")
    try:
        rendered = [item for item in yaml.safe_load_all(result.stdout) if item is not None]
    except yaml.YAMLError as exc:
        raise OpenShiftAcceptanceError("render-invalid", "Kustomize output is not valid YAML") from exc
    if not rendered or any(not isinstance(item, dict) for item in rendered):
        raise OpenShiftAcceptanceError("render-invalid", "Kustomize output has an invalid document shape")
    if _PLACEHOLDER.search(result.stdout.decode("utf-8", errors="replace")):
        raise OpenShiftAcceptanceError("unresolved-placeholder", "rendered deployment contains unresolved placeholders")
    resource_keys: list[tuple[str, str, str]] = []
    for item in rendered:
        metadata = item.get("metadata")
        api_version = item.get("apiVersion")
        kind = item.get("kind")
        name = metadata.get("name") if isinstance(metadata, dict) else None
        if not all(isinstance(value, str) and value for value in (api_version, kind, name)):
            raise OpenShiftAcceptanceError("render-inventory", "rendered resource identity is incomplete")
        resource_keys.append((api_version, kind, name))
    if len(resource_keys) != len(set(resource_keys)):
        raise OpenShiftAcceptanceError("render-inventory", "rendered resource identities are not unique")
    job_suffix = release_version.replace(".", "-")
    expected_resources = {
        *_BASE_RENDERED_RESOURCES,
        ("batch/v1", "Job", f"bddk-mcp-migrate-v{job_suffix}"),
        ("batch/v1", "Job", f"bddk-mcp-bootstrap-v{job_suffix}"),
        ("batch/v1", "Job", f"bddk-mcp-verify-stage-release-v{job_suffix}"),
        ("batch/v1", "Job", f"bddk-mcp-activate-release-v{job_suffix}"),
        *(("networking.k8s.io/v1", "NetworkPolicy", policy_name) for policy_name in expected_egress_policies),
    }
    if set(resource_keys) != expected_resources:
        raise OpenShiftAcceptanceError("render-inventory", "rendered resource inventory differs from the review set")
    jobs = [item for item in rendered if item.get("kind") == "Job"]
    runtime = [item for item in rendered if item.get("kind") != "Job"]
    if len(jobs) != len(_LIFECYCLE_RESOURCES):
        raise OpenShiftAcceptanceError("render-inventory", "rendered lifecycle Job inventory is incomplete")
    job_rank = {
        "bddk-mcp-migrate": 0,
        "bddk-mcp-bootstrap": 1,
        "bddk-mcp-verify-stage-release": 2,
        "bddk-mcp-activate-release": 3,
    }
    try:
        jobs.sort(
            key=lambda item: next(
                rank for prefix, rank in job_rank.items() if item.get("metadata", {}).get("name", "").startswith(prefix)
            )
        )
    except StopIteration as exc:
        raise OpenShiftAcceptanceError("render-inventory", "rendered lifecycle Job names are invalid") from exc
    return runtime, jobs, result.stdout


def _load_egress_documents(config_path: Path, config: AcceptanceInput) -> list[dict[str, Any]]:
    root = config_path.resolve().parent
    documents: list[dict[str, Any]] = []
    for relative in config.egress_policy_files:
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise OpenShiftAcceptanceError("egress-path", "egress policy escaped its acceptance directory") from exc
        documents.extend(_load_yaml_documents(candidate))
    serialized = yaml.safe_dump_all(documents, sort_keys=True)
    if _PLACEHOLDER.search(serialized):
        raise OpenShiftAcceptanceError(
            "unresolved-egress-placeholder", "egress policies contain unresolved placeholders"
        )
    return documents


def _container(document: dict[str, Any]) -> dict[str, Any]:
    pod = document["spec"]["template"]["spec"]
    containers = pod.get("containers")
    _expect(isinstance(containers, list) and len(containers) == 1, "workloads require one reviewed container")
    _expect(not pod.get("initContainers"), "unreviewed init containers are forbidden")
    _expect(not pod.get("ephemeralContainers"), "unreviewed ephemeral containers are forbidden")
    return containers[0]


def _named(documents: list[dict[str, Any]], kind: str) -> dict[str, dict[str, Any]]:
    return {
        item["metadata"]["name"]: item
        for item in documents
        if item.get("kind") == kind and isinstance(item.get("metadata"), dict) and item["metadata"].get("name")
    }


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _check_release(
    runtime: list[dict[str, Any]], jobs: list[dict[str, Any]], config: AcceptanceInput, repository_root: Path
) -> None:
    workloads = list(_named(runtime, "Deployment").values()) + jobs
    images = {_container(item).get("image") for item in workloads}
    _expect(images == {config.release.image}, "workloads do not use one configured application image")
    _expect(
        all(_container(item).get("imagePullPolicy") == "IfNotPresent" for item in workloads),
        "immutable application workloads must retain the reviewed pull policy",
    )
    _validate_image(config.release.image)
    _validate_image(config.release.previous_image)
    kustomization = yaml.safe_load(_read_bounded(repository_root / "deploy" / "openshift" / "kustomization.yaml"))
    version_labels = [
        item.get("pairs", {}).get("app.kubernetes.io/version") for item in kustomization.get("labels", [])
    ]
    _expect(config.release.version in version_labels, "release version does not match the Kustomize release label")
    job_suffix = "v" + config.release.version.replace(".", "-")
    _expect(
        all(item["metadata"]["name"].endswith(job_suffix) for item in jobs), "lifecycle Job names do not match release"
    )


def _check_namespace(runtime: list[dict[str, Any]], jobs: list[dict[str, Any]], config: AcceptanceInput) -> None:
    documents = runtime + jobs
    _expect(documents, "rendered deployment is empty")
    _expect(
        all(item.get("metadata", {}).get("namespace") == config.platform.namespace for item in documents),
        "rendered resources are not bound to the configured namespace",
    )


def _check_route(runtime: list[dict[str, Any]], config: AcceptanceInput) -> None:
    routes = _named(runtime, "Route")
    _expect(set(routes) == {"bddk-mcp-public"}, "only the public component may have a Route")
    route = routes["bddk-mcp-public"]
    _expect(
        route.get("spec")
        == {
            "host": config.platform.public_route_host,
            "to": {"kind": "Service", "name": "bddk-mcp-public", "weight": 100},
            "port": {"targetPort": "https"},
            "tls": {"termination": "reencrypt", "insecureEdgeTerminationPolicy": "Redirect"},
            "wildcardPolicy": "None",
        },
        "public Route specification differs from the reviewed exposure boundary",
    )
    _expect(
        route["metadata"].get("annotations") == {"haproxy.router.openshift.io/timeout": "120s"},
        "public Route annotations differ from the reviewed set",
    )

    services = _named(runtime, "Service")
    _expect(set(services) == {"bddk-mcp-public", "bddk-mcp-operator"}, "service inventory mismatch")
    deployments = _named(runtime, "Deployment")
    for name, service in services.items():
        _expect(
            service["metadata"].get("annotations")
            == {"service.beta.openshift.io/serving-cert-secret-name": f"{name}-tls"},
            "service-serving certificate annotation mismatch",
        )
        component = name.removeprefix("bddk-mcp-")
        _expect(
            service.get("spec")
            == {
                "type": "ClusterIP",
                "selector": {
                    "app.kubernetes.io/name": "bddk-mcp",
                    "app.kubernetes.io/component": component,
                },
                "ports": [{"name": "https", "port": 443, "targetPort": "https", "protocol": "TCP"}],
            },
            "service specification differs from the reviewed internal-only boundary",
        )
        deployment = deployments[name]
        container = _container(deployment)
        env = {item["name"]: item.get("value") for item in container.get("env", []) if "value" in item}
        _expect(
            env.get("BDDK_TLS_CERT_FILE") == "/var/run/secrets/bddk-mcp/tls/tls.crt"
            and env.get("BDDK_TLS_KEY_FILE") == "/var/run/secrets/bddk-mcp/tls/tls.key",
            "application TLS file contract mismatch",
        )
        tls_mount = next(
            (item for item in container.get("volumeMounts", []) if item.get("name") == "service-tls"), None
        )
        tls_volume = next(
            (
                item
                for item in deployment["spec"]["template"]["spec"].get("volumes", [])
                if item.get("name") == "service-tls"
            ),
            None,
        )
        _expect(
            tls_mount == {"name": "service-tls", "mountPath": "/var/run/secrets/bddk-mcp/tls", "readOnly": True},
            "application TLS mount mismatch",
        )
        _expect(
            tls_volume == {"name": "service-tls", "secret": {"secretName": f"{name}-tls", "defaultMode": 0o440}},
            "application TLS Secret mismatch",
        )

    service_ca = _named(runtime, "ConfigMap")["bddk-mcp-service-ca"]
    _expect(
        service_ca["metadata"].get("annotations") == {"service.beta.openshift.io/inject-cabundle": "true"}
        and service_ca.get("data") == {},
        "OpenShift service CA injection contract mismatch",
    )


def _check_jwt(runtime: list[dict[str, Any]], config: AcceptanceInput) -> None:
    maps = {name: item["data"] for name, item in _named(runtime, "ConfigMap").items() if name.endswith("-config")}
    public = maps["bddk-mcp-public-config"]
    operator = maps["bddk-mcp-operator-config"]
    _expect(set(public) == _PUBLIC_CONFIG_KEYS, "public ConfigMap key inventory mismatch")
    _expect(set(operator) == _OPERATOR_CONFIG_KEYS, "operator ConfigMap key inventory mismatch")
    common = {
        "MCP_TRANSPORT": "streamable-http",
        "MCP_HOST": "0.0.0.0",
        "PORT": "8000",
        "BDDK_AUTO_SYNC": "false",
        "BDDK_ALLOW_INSECURE_DATABASE": "false",
        "BDDK_REQUIRE_ACTIVE_CORPUS_RELEASE": "true",
        "BDDK_TELEMETRY_ENABLED": "false",
        "BDDK_JWT_ISSUER": config.jwt.issuer,
        "BDDK_JWT_JWKS_URL": config.jwt.jwks_url,
        "BDDK_JWT_ALGORITHMS": "RS256",
        "BDDK_JWT_ACCESS_TOKEN_TYPES": "at+jwt",
        "BDDK_HTTP_MAX_BODY_BYTES": "1048576",
    }
    _expect(
        public
        == {
            **common,
            "BDDK_TOOL_PROFILE": "public",
            "BDDK_HTTP_ALLOWED_HOSTS": config.platform.public_route_host,
            "BDDK_HTTP_ALLOWED_ORIGINS": f"https://{config.platform.public_route_host}",
            "BDDK_JWT_RESOURCE": f"https://{config.platform.public_route_host}/mcp",
            "BDDK_JWT_AUDIENCE": config.jwt.public_audience,
            "BDDK_JWT_REQUIRED_SCOPES": "bddk.read",
            "BDDK_HTTP_MAX_CONCURRENCY": "32",
            "BDDK_HTTP_RATE_LIMIT_PER_MINUTE": "10000",
        },
        "public ConfigMap values differ from the reviewed runtime contract",
    )
    _expect(
        operator
        == {
            **common,
            "BDDK_TOOL_PROFILE": "operator",
            "BDDK_OPERATOR_REMOTE_ENABLED": "true",
            "BDDK_HTTP_ALLOWED_HOSTS": config.platform.operator_service_host,
            "BDDK_HTTP_ALLOWED_ORIGINS": config.platform.operator_client_origin,
            "BDDK_JWT_RESOURCE": f"https://{config.platform.operator_service_host}/mcp",
            "BDDK_JWT_AUDIENCE": config.jwt.operator_audience,
            "BDDK_JWT_REQUIRED_SCOPES": "bddk.operator",
            "BDDK_HTTP_MAX_CONCURRENCY": "8",
            "BDDK_HTTP_RATE_LIMIT_PER_MINUTE": "30",
            "BDDK_OPERATOR_JOB_DRAIN_TIMEOUT": "30",
            "BDDK_OPERATOR_JOB_HISTORY": "1000",
        },
        "operator ConfigMap values differ from the reviewed runtime contract",
    )


def _secret_ref(container: dict[str, Any], env_name: str) -> dict[str, str]:
    items = [item for item in container.get("env", []) if item.get("name") == env_name]
    _expect(len(items) == 1, f"{env_name} must have one exact Secret reference")
    return items[0]["valueFrom"]["secretKeyRef"]


def _check_identities(runtime: list[dict[str, Any]], jobs: list[dict[str, Any]], repository_root: Path) -> None:
    deployments = _named(runtime, "Deployment")
    public = deployments["bddk-mcp-public"]
    operator = deployments["bddk-mcp-operator"]
    _expect(
        _secret_ref(_container(public), "BDDK_DATABASE_URL")
        == {"name": "bddk-mcp-public-db", "key": "BDDK_DATABASE_URL"},
        "public DB Secret mismatch",
    )
    _expect(
        _secret_ref(_container(operator), "BDDK_OPERATOR_DATABASE_URL")
        == {"name": "bddk-mcp-operator-db", "key": "BDDK_OPERATOR_DATABASE_URL"},
        "operator DB Secret mismatch",
    )
    migrate, bootstrap, verifier, publisher = jobs
    _expect(
        _secret_ref(_container(migrate), "BDDK_SCHEMA_OWNER_DATABASE_URL")
        == {"name": "bddk-mcp-schema-owner-db", "key": "BDDK_SCHEMA_OWNER_DATABASE_URL"},
        "schema-owner DB Secret mismatch",
    )
    _expect(
        _secret_ref(_container(bootstrap), "BDDK_INGESTION_DATABASE_URL")
        == {"name": "bddk-mcp-ingestion-db", "key": "BDDK_INGESTION_DATABASE_URL"},
        "ingestion DB Secret mismatch",
    )
    _expect(
        _secret_ref(_container(verifier), "BDDK_RELEASE_VERIFIER_DATABASE_URL")
        == {
            "name": "bddk-mcp-release-verifier-db",
            "key": "BDDK_RELEASE_VERIFIER_DATABASE_URL",
        },
        "release-verifier DB Secret mismatch",
    )
    _expect(
        _secret_ref(_container(publisher), "BDDK_RELEASE_PUBLISHER_DATABASE_URL")
        == {
            "name": "bddk-mcp-release-publisher-db",
            "key": "BDDK_RELEASE_PUBLISHER_DATABASE_URL",
        },
        "release-publisher DB Secret mismatch",
    )
    all_secret_names = {
        "bddk-mcp-public-db",
        "bddk-mcp-operator-db",
        "bddk-mcp-schema-owner-db",
        "bddk-mcp-ingestion-db",
        "bddk-mcp-release-verifier-db",
        "bddk-mcp-release-publisher-db",
    }
    for workload in (public, operator, *jobs):
        container = _container(workload)
        serialized = yaml.safe_dump(workload)
        present = {name for name in all_secret_names if name in serialized}
        _expect(len(present) == 1, "a workload crosses a database identity boundary")
        _expect(
            not any("secretRef" in source for source in container.get("envFrom", [])),
            "whole-Secret imports are forbidden",
        )
        pod = workload["spec"]["template"]["spec"]
        ca_mount = next((item for item in container.get("volumeMounts", []) if item.get("name") == "postgres-ca"), None)
        ca_volume = next((item for item in pod.get("volumes", []) if item.get("name") == "postgres-ca"), None)
        _expect(
            ca_mount == {"name": "postgres-ca", "mountPath": "/var/run/configmaps/bddk-mcp/postgres", "readOnly": True},
            "PostgreSQL CA mount mismatch",
        )
        _expect(
            ca_volume
            == {
                "name": "postgres-ca",
                "configMap": {"name": "bddk-mcp-postgres-ca", "items": [{"key": "ca.crt", "path": "ca.crt"}]},
            },
            "PostgreSQL CA source mismatch",
        )

    secret_examples = _named(
        _load_yaml_documents(repository_root / "deploy" / "openshift" / "secrets.example.yaml"), "Secret"
    )
    expected_keys = {
        "bddk-mcp-public-db": "BDDK_DATABASE_URL",
        "bddk-mcp-operator-db": "BDDK_OPERATOR_DATABASE_URL",
        "bddk-mcp-schema-owner-db": "BDDK_SCHEMA_OWNER_DATABASE_URL",
        "bddk-mcp-ingestion-db": "BDDK_INGESTION_DATABASE_URL",
        "bddk-mcp-release-verifier-db": "BDDK_RELEASE_VERIFIER_DATABASE_URL",
        "bddk-mcp-release-publisher-db": "BDDK_RELEASE_PUBLISHER_DATABASE_URL",
        "bddk-mcp-telemetry-db": "BDDK_TELEMETRY_DATABASE_URL",
    }
    _expect(set(secret_examples) == set(expected_keys), "database Secret example inventory mismatch")
    values: list[str] = []
    for secret_name, key in expected_keys.items():
        string_data = secret_examples[secret_name].get("stringData")
        _expect(isinstance(string_data, dict) and set(string_data) == {key}, "database Secret key boundary mismatch")
        value = string_data[key]
        _expect("sslmode=verify-full" in value, "database Secret example must verify PostgreSQL TLS")
        _expect(
            "sslrootcert=%2Fvar%2Frun%2Fconfigmaps%2Fbddk-mcp%2Fpostgres%2Fca.crt" in value,
            "database Secret example must use the mounted PostgreSQL CA",
        )
        values.append(value)
    _expect(len(set(values)) == len(values), "database identities must not share example DSNs")
    _expect(
        "role%3Dbddk_schema_owner"
        in secret_examples["bddk-mcp-schema-owner-db"]["stringData"]["BDDK_SCHEMA_OWNER_DATABASE_URL"],
        "schema-owner identity must enter its exact NOLOGIN role",
    )


def _selector_labels(selector: dict[str, Any] | None) -> dict[str, str]:
    if not selector:
        return {}
    labels = selector.get("matchLabels", {})
    return labels if isinstance(labels, dict) else {}


def _peer_dump(peer: EgressPeer) -> dict[str, Any]:
    return peer.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)


def _check_network(runtime: list[dict[str, Any]], egress: list[dict[str, Any]], config: AcceptanceInput) -> None:
    base = _named(runtime, "NetworkPolicy")
    expected_policy_names = {
        "bddk-mcp-default-deny-egress",
        "bddk-mcp-default-deny-ingress",
        "bddk-mcp-public-from-router",
        "bddk-mcp-operator-from-approved-clients",
        *(item.policy for item in config.required_egress),
    }
    _expect(set(base) == expected_policy_names, "rendered NetworkPolicy inventory mismatch")
    deny_egress = base["bddk-mcp-default-deny-egress"]["spec"]
    _expect(
        deny_egress
        == {
            "podSelector": {"matchLabels": {"app.kubernetes.io/name": "bddk-mcp"}},
            "policyTypes": ["Egress"],
            "egress": [],
        },
        "default-deny egress policy mismatch",
    )
    deny_ingress = base["bddk-mcp-default-deny-ingress"]["spec"]
    _expect(
        deny_ingress.get("podSelector") == {"matchLabels": {"app.kubernetes.io/name": "bddk-mcp"}},
        "default-deny ingress selector mismatch",
    )
    _expect(
        deny_ingress.get("policyTypes") == ["Ingress"] and "ingress" not in deny_ingress,
        "default-deny ingress policy mismatch",
    )
    expected_ingress = {
        "bddk-mcp-public-from-router": {
            "podSelector": {
                "matchLabels": {
                    "app.kubernetes.io/name": "bddk-mcp",
                    "app.kubernetes.io/component": "public",
                }
            },
            "policyTypes": ["Ingress"],
            "ingress": [
                {
                    "from": [{"namespaceSelector": {"matchLabels": {"network.openshift.io/policy-group": "ingress"}}}],
                    "ports": [{"protocol": "TCP", "port": 8000}],
                }
            ],
        },
        "bddk-mcp-operator-from-approved-clients": {
            "podSelector": {
                "matchLabels": {
                    "app.kubernetes.io/name": "bddk-mcp",
                    "app.kubernetes.io/component": "operator",
                }
            },
            "policyTypes": ["Ingress"],
            "ingress": [
                {
                    "from": [{"namespaceSelector": {"matchLabels": {"bddk.bank/operator-client": "true"}}}],
                    "ports": [{"protocol": "TCP", "port": 8000}],
                }
            ],
        },
    }
    for policy_name, expected_spec in expected_ingress.items():
        _expect(base[policy_name].get("spec") == expected_spec, "ingress allow policy contract mismatch")

    policies = _named(egress, "NetworkPolicy")
    _expect(len(policies) == len(egress), "egress files may contain only named NetworkPolicy documents")
    expected_policies = {item.policy for item in config.required_egress}
    _expect(set(policies) == expected_policies, "egress overlay contains unreviewed or missing policies")

    observed_items: list[tuple[str, str, str, int, str]] = []
    for policy_name, policy in policies.items():
        spec = policy.get("spec", {})
        _expect(spec.get("policyTypes") == ["Egress"], "allow policy must be egress-only")
        labels = _selector_labels(spec.get("podSelector"))
        _expect(labels.get("app.kubernetes.io/name") == "bddk-mcp", "egress policy must select only this application")
        component = labels.get("app.kubernetes.io/component")
        _expect(component in {"public", "operator", "lifecycle"}, "egress policy must select an exact component")
        _expect(
            labels == {"app.kubernetes.io/name": "bddk-mcp", "app.kubernetes.io/component": component},
            "egress policy selector must match its declared component exactly",
        )
        for rule in spec.get("egress", []):
            _expect(set(rule) == {"to", "ports"}, "egress rules may contain only exact destinations and ports")
            peers = rule.get("to")
            ports = rule.get("ports")
            _expect(isinstance(peers, list) and peers, "egress rule must constrain destinations")
            _expect(isinstance(ports, list) and ports, "egress rule must constrain ports")
            for peer in peers:
                _expect(peer != {}, "open egress peers are forbidden")
                for port in ports:
                    _expect(set(port) == {"protocol", "port"}, "egress ports must be exact protocol/port pairs")
                    _expect(isinstance(port["port"], int), "named egress ports are not accepted by this harness")
                    observed_items.append(
                        (policy_name, component, port["protocol"], port["port"], json.dumps(peer, sort_keys=True))
                    )

    expected = {
        (item.policy, item.component, item.protocol, item.port, json.dumps(_peer_dump(item.peer), sort_keys=True))
        for item in config.required_egress
    }
    observed = set(observed_items)
    _expect(len(observed_items) == len(observed), "egress overlay contains duplicate effective permissions")
    _expect(observed == expected, "egress overlay differs from the declared least-privilege matrix")


def _check_workloads(runtime: list[dict[str, Any]], jobs: list[dict[str, Any]], config: AcceptanceInput) -> None:
    named_deployments = _named(runtime, "Deployment")
    _expect(set(named_deployments) == {"bddk-mcp-public", "bddk-mcp-operator"}, "deployment inventory mismatch")
    deployments = list(named_deployments.values())
    workloads = deployments + jobs
    for workload in workloads:
        pod = workload["spec"]["template"]["spec"]
        container = _container(workload)
        component = (
            workload["metadata"]["name"].removeprefix("bddk-mcp-").split("-v", maxsplit=1)[0]
            if workload.get("kind") == "Deployment"
            else workload["metadata"]["labels"].get("app.kubernetes.io/component")
        )
        expected_labels = {
            "app.kubernetes.io/name": "bddk-mcp",
            "app.kubernetes.io/component": component,
            "app.kubernetes.io/version": config.release.version,
        }
        _expect(component in {"public", "operator", "lifecycle"}, "workload component label mismatch")
        expected_metadata_labels = (
            expected_labels
            if workload.get("kind") == "Job"
            else {
                "app.kubernetes.io/name": "bddk-mcp",
                "app.kubernetes.io/version": config.release.version,
            }
        )
        _expect(
            workload["metadata"].get("labels") == expected_metadata_labels,
            "workload metadata labels mismatch",
        )
        _expect(
            workload["spec"]["template"]["metadata"].get("labels") == expected_labels,
            "workload pod-template labels mismatch",
        )
        if workload.get("kind") == "Deployment":
            _expect(
                workload["spec"].get("selector")
                == {
                    "matchLabels": {
                        "app.kubernetes.io/name": "bddk-mcp",
                        "app.kubernetes.io/component": component,
                    }
                },
                "Deployment selector mismatch",
            )
        _expect(
            not any(pod.get(field) for field in ("hostNetwork", "hostPID", "hostIPC")),
            "host namespace sharing is forbidden",
        )
        _expect(pod.get("automountServiceAccountToken") is False, "service account tokens must not auto-mount")
        _expect(
            pod.get("securityContext") == {"runAsNonRoot": True, "seccompProfile": {"type": "RuntimeDefault"}},
            "pod security context must match the reviewed restricted baseline",
        )
        security = container["securityContext"]
        _expect(
            security
            == {
                "allowPrivilegeEscalation": False,
                "readOnlyRootFilesystem": True,
                "capabilities": {"drop": ["ALL"]},
            },
            "container security context must match the reviewed restricted baseline",
        )
        _expect(
            container.get("resources", {}).get("requests") and container.get("resources", {}).get("limits"),
            "resources must be bounded",
        )
        _expect(
            any(item.get("mountPath") == "/tmp" for item in container.get("volumeMounts", [])),
            "workload requires bounded writable temporary storage",
        )
        runtime_workload = workload.get("kind") == "Deployment"
        corpus_admission_job = workload.get("kind") == "Job" and workload["metadata"]["name"].startswith(
            ("bddk-mcp-bootstrap", "bddk-mcp-verify-stage-release")
        )
        expected_volume_names = (
            {"postgres-ca", "runtime-tmp"}
            | ({"service-tls"} if runtime_workload else set())
            | ({"approved-corpus", "corpus-signing-key"} if corpus_admission_job else set())
        )
        volumes = pod.get("volumes", [])
        mounts = container.get("volumeMounts", [])
        _expect(
            {item.get("name") for item in volumes} == expected_volume_names
            and {item.get("name") for item in mounts} == expected_volume_names,
            "workload volume or mount inventory mismatch",
        )
        for volume in volumes:
            _expect(
                set(volume) <= {"name", "configMap", "secret", "emptyDir", "persistentVolumeClaim"},
                "unreviewed volume source is forbidden",
            )
        temporary = next(item for item in volumes if item.get("name") == "runtime-tmp")
        _expect(
            set(temporary) == {"name", "emptyDir"}
            and set(temporary["emptyDir"]) == {"sizeLimit"}
            and temporary["emptyDir"]["sizeLimit"],
            "temporary volume must have one explicit size limit",
        )
        temporary_mount = next(item for item in mounts if item.get("name") == "runtime-tmp")
        _expect(
            temporary_mount == {"name": "runtime-tmp", "mountPath": "/tmp"},
            "temporary mount contract mismatch",
        )
        secret_key_refs = [
            item["valueFrom"]["secretKeyRef"]
            for item in container.get("env", [])
            if isinstance(item.get("valueFrom"), dict) and "secretKeyRef" in item["valueFrom"]
        ]
        _expect(len(secret_key_refs) == 1, "workload requires one exact database Secret key reference")
        if runtime_workload:
            _expect(
                set(container)
                == {
                    "name",
                    "image",
                    "imagePullPolicy",
                    "args",
                    "envFrom",
                    "env",
                    "ports",
                    "livenessProbe",
                    "readinessProbe",
                    "securityContext",
                    "resources",
                    "volumeMounts",
                },
                "runtime container field inventory mismatch",
            )
            _expect(
                container.get("envFrom") == [{"configMapRef": {"name": f"bddk-mcp-{component}-config"}}],
                "runtime ConfigMap import mismatch",
            )
            expected_env_names = {
                "BDDK_DATABASE_URL" if component == "public" else "BDDK_OPERATOR_DATABASE_URL",
                "BDDK_TLS_CERT_FILE",
                "BDDK_TLS_KEY_FILE",
            }
            _expect(container.get("name") == "server", "runtime container name mismatch")
            _expect(
                "command" not in container
                and container.get("args") == [".venv/bin/bddk-mcp", "serve", "--profile", component],
                "runtime command contract mismatch",
            )
            _expect(
                container.get("ports") == [{"name": "https", "containerPort": 8000, "protocol": "TCP"}],
                "runtime container port contract mismatch",
            )
        else:
            _expect(
                set(container)
                == {
                    "name",
                    "image",
                    "imagePullPolicy",
                    "args",
                    "env",
                    "securityContext",
                    "resources",
                    "volumeMounts",
                },
                "lifecycle container field inventory mismatch",
            )
            _expect(not container.get("envFrom"), "lifecycle Jobs cannot import whole ConfigMaps or Secrets")
            job_name = workload["metadata"]["name"]
            if job_name.startswith("bddk-mcp-migrate"):
                expected_env_names = {"BDDK_EXPECTED_DATABASE_NAME", "BDDK_SCHEMA_OWNER_DATABASE_URL"}
            elif job_name.startswith("bddk-mcp-bootstrap"):
                expected_env_names = {"BDDK_INGESTION_DATABASE_URL"}
            elif job_name.startswith("bddk-mcp-verify-stage-release"):
                expected_env_names = {
                    "BDDK_RELEASE_VERIFIER_DATABASE_URL",
                    "BDDK_RELEASE_VERIFIER_REVISION_SHA256",
                    "BDDK_RELEASE_VERIFIER_IMAGE_DIGEST",
                    "BDDK_RELEASE_VERIFICATION_VALIDITY_SECONDS",
                }
            else:
                expected_env_names = {"BDDK_RELEASE_PUBLISHER_DATABASE_URL", "BDDK_RELEASE_REQUEST_ID"}
            _expect("command" not in container and not container.get("ports"), "lifecycle execution surface mismatch")
        _expect(
            {item.get("name") for item in container.get("env", [])} == expected_env_names,
            "workload environment inventory mismatch",
        )
    for deployment in deployments:
        container = _container(deployment)
        _expect(
            container["livenessProbe"]["httpGet"] == {"path": "/health/live", "port": "https", "scheme": "HTTPS"},
            "liveness probe mismatch",
        )
        _expect(
            container["readinessProbe"]["httpGet"] == {"path": "/health/ready", "port": "https", "scheme": "HTTPS"},
            "readiness probe mismatch",
        )
    _expect(
        [item["metadata"]["name"] for item in jobs]
        == [
            f"bddk-mcp-migrate-v{config.release.version.replace('.', '-')}",
            f"bddk-mcp-bootstrap-v{config.release.version.replace('.', '-')}",
            f"bddk-mcp-verify-stage-release-v{config.release.version.replace('.', '-')}",
            f"bddk-mcp-activate-release-v{config.release.version.replace('.', '-')}",
        ],
        "lifecycle Job order or naming mismatch",
    )
    _expect(
        named_deployments["bddk-mcp-public"]["spec"]["template"]["spec"].get("serviceAccountName") == "bddk-mcp-public"
        and named_deployments["bddk-mcp-operator"]["spec"]["template"]["spec"].get("serviceAccountName")
        == "bddk-mcp-operator",
        "runtime service-account separation mismatch",
    )
    _expect(
        named_deployments["bddk-mcp-operator"]["spec"].get("replicas") == 1
        and named_deployments["bddk-mcp-operator"]["spec"].get("strategy") == {"type": "Recreate"},
        "operator failover acceptance boundary mismatch",
    )
    _expect(
        _container(named_deployments["bddk-mcp-public"])["args"][-2:] == ["--profile", "public"]
        and _container(named_deployments["bddk-mcp-operator"])["args"][-2:] == ["--profile", "operator"],
        "runtime profile separation mismatch",
    )
    for index, job in enumerate(jobs):
        _expect(job["spec"].get("backoffLimit") == 1, "lifecycle Job backoff must be bounded")
        _expect(job["spec"].get("ttlSecondsAfterFinished") == 86400, "lifecycle Job evidence retention mismatch")
        pod = job["spec"]["template"]["spec"]
        expected_service_account = (
            "bddk-mcp-lifecycle",
            "bddk-mcp-ingestion",
            "bddk-mcp-release-verifier",
            "bddk-mcp-release-publisher",
        )[index]
        _expect(
            pod.get("serviceAccountName") == expected_service_account,
            "lifecycle service-account mismatch",
        )
        _expect(pod.get("restartPolicy") == "Never", "lifecycle Job restart policy mismatch")
    _expect(_container(jobs[0])["args"] == [".venv/bin/bddk-mcp", "migrate"], "migration Job command mismatch")
    _expect(
        _container(jobs[1])["args"] == _BANK_BOOTSTRAP_ARGS,
        "bootstrap Job command mismatch",
    )
    verifier_container = _container(jobs[2])
    publisher_container = _container(jobs[3])
    _expect(verifier_container["args"] == _VERIFY_STAGE_RELEASE_ARGS, "release verification Job command mismatch")
    _expect(publisher_container["args"] == _ACTIVATE_RELEASE_ARGS, "release activation Job command mismatch")
    verifier_env = {item["name"]: item.get("value") for item in verifier_container["env"] if "value" in item}
    expected_revision_sha256 = (
        config.release.manifest_revision
        if len(config.release.manifest_revision) == 64
        else hashlib.sha256(config.release.manifest_revision.encode("ascii")).hexdigest()
    )
    _expect(
        verifier_env
        == {
            "BDDK_RELEASE_VERIFIER_REVISION_SHA256": expected_revision_sha256,
            "BDDK_RELEASE_VERIFIER_IMAGE_DIGEST": config.release.image.rsplit("@", maxsplit=1)[1],
            "BDDK_RELEASE_VERIFICATION_VALIDITY_SECONDS": "900",
        },
        "release verification provenance or validity mismatch",
    )
    publisher_env = {item["name"]: item.get("value") for item in publisher_container["env"] if "value" in item}
    _expect(
        publisher_env == {"BDDK_RELEASE_REQUEST_ID": config.release.release_request_id},
        "release activation request mismatch",
    )
    expected_db = next(item for item in _container(jobs[0])["env"] if item["name"] == "BDDK_EXPECTED_DATABASE_NAME")
    _expect(expected_db["value"] == config.platform.database_name, "migration target-database guard mismatch")


def _check_bank_bootstrap(jobs: list[dict[str, Any]]) -> None:
    _expect(len(jobs) == 4, "strict bank promotion requires the reviewed four-stage lifecycle Jobs")
    bootstrap, verifier, publisher = jobs[1], jobs[2], jobs[3]
    _expect(
        bootstrap.get("metadata", {}).get("name", "").startswith("bddk-mcp-bootstrap-v"),
        "strict bank bootstrap Job identity mismatch",
    )
    container = _container(bootstrap)
    pod = bootstrap["spec"]["template"]["spec"]
    mounts = {item.get("name"): item for item in container.get("volumeMounts", [])}
    volumes = {item.get("name"): item for item in pod.get("volumes", [])}
    _expect(container.get("args") == _BANK_BOOTSTRAP_ARGS, "strict bank bootstrap arguments mismatch")
    _expect(
        mounts.get("approved-corpus") == _BANK_CORPUS_MOUNT and mounts.get("corpus-signing-key") == _BANK_TRUST_MOUNT,
        "strict bank bootstrap read-only mounts mismatch",
    )
    _expect(
        volumes.get("approved-corpus") == _BANK_CORPUS_VOLUME
        and volumes.get("corpus-signing-key") == _BANK_TRUST_VOLUME,
        "approved corpus and signing trust must use separate reviewed volume sources",
    )
    verifier_container = _container(verifier)
    verifier_pod = verifier["spec"]["template"]["spec"]
    verifier_mounts = {item.get("name"): item for item in verifier_container.get("volumeMounts", [])}
    verifier_volumes = {item.get("name"): item for item in verifier_pod.get("volumes", [])}
    _expect(verifier_container.get("args") == _VERIFY_STAGE_RELEASE_ARGS, "release verification arguments mismatch")
    _expect(
        verifier_pod.get("serviceAccountName") == "bddk-mcp-release-verifier",
        "release verification service-account mismatch",
    )
    _expect(
        verifier_mounts.get("approved-corpus") == _BANK_CORPUS_MOUNT
        and verifier_mounts.get("corpus-signing-key") == _BANK_TRUST_MOUNT
        and verifier_volumes.get("approved-corpus") == _BANK_CORPUS_VOLUME
        and verifier_volumes.get("corpus-signing-key") == _BANK_TRUST_VOLUME,
        "release verification trust mounts mismatch",
    )
    verifier_serialized = yaml.safe_dump(verifier, sort_keys=True)
    _expect(
        not any(
            forbidden in verifier_serialized
            for forbidden in (
                "bddk-mcp-release-publisher-db",
                "BDDK_RELEASE_PUBLISHER_DATABASE_URL",
                "BDDK_RELEASE_REQUEST_ID",
                "activate-corpus-release",
            )
        ),
        "release verifier crosses the activation boundary",
    )

    publisher_container = _container(publisher)
    publisher_pod = publisher["spec"]["template"]["spec"]
    publisher_mounts = {item.get("name"): item for item in publisher_container.get("volumeMounts", [])}
    publisher_volumes = {item.get("name"): item for item in publisher_pod.get("volumes", [])}
    _expect(publisher_container.get("args") == _ACTIVATE_RELEASE_ARGS, "release activation arguments mismatch")
    _expect(
        publisher_pod.get("serviceAccountName") == "bddk-mcp-release-publisher",
        "release activation service-account mismatch",
    )
    _expect(
        set(publisher_mounts) == {"postgres-ca", "runtime-tmp"}
        and set(publisher_volumes) == {"postgres-ca", "runtime-tmp"},
        "release activation Job must not receive corpus or trust volumes",
    )
    publisher_serialized = yaml.safe_dump(publisher, sort_keys=True)
    _expect(
        not any(
            forbidden in publisher_serialized
            for forbidden in (
                "approved-corpus",
                "corpus-signing-key",
                "bddk-mcp-approved-corpus",
                "bddk-mcp-corpus-trust",
                "/var/run/bddk-mcp/corpus",
                "/var/run/secrets/bddk-mcp/corpus-trust",
                "bddk-mcp-release-verifier-db",
                "BDDK_RELEASE_VERIFIER_DATABASE_URL",
                "verify-and-stage-corpus-release",
            )
        ),
        "release activation Job crosses the verifier or corpus boundary",
    )


def _check_telemetry(runtime: list[dict[str, Any]], repository_root: Path) -> None:
    deployments = _named(runtime, "Deployment")
    configs = _named(runtime, "ConfigMap")
    for name in ("bddk-mcp-public", "bddk-mcp-operator"):
        _expect(
            "BDDK_TELEMETRY_DATABASE_URL" not in yaml.safe_dump(deployments[name]), "baseline leaks telemetry identity"
        )
        _expect(
            configs[f"{name}-config"]["data"]["BDDK_TELEMETRY_ENABLED"] == "false",
            "baseline telemetry must be disabled",
        )
    overlay = yaml.safe_load(
        _read_bounded(repository_root / "deploy" / "openshift-overlays" / "telemetry" / "kustomization.yaml")
    )
    _expect(
        isinstance(overlay, dict)
        and set(overlay) == {"apiVersion", "kind", "resources", "patches"}
        and overlay.get("apiVersion") == "kustomize.config.k8s.io/v1beta1"
        and overlay.get("kind") == "Kustomization",
        "telemetry overlay Kustomization inventory mismatch",
    )
    _expect(overlay.get("resources") == ["../../openshift"], "telemetry overlay must build on the reviewed base")
    patches = overlay.get("patches")
    _expect(isinstance(patches, list) and len(patches) == 4, "telemetry overlay patch inventory mismatch")
    config_targets: set[str] = set()
    deployment_targets: set[str] = set()
    for patch in patches:
        target = patch.get("target", {})
        operations = yaml.safe_load(patch.get("patch", ""))
        _expect(isinstance(operations, list) and len(operations) == 1, "telemetry patch must have one exact operation")
        operation = operations[0]
        if target.get("kind") == "ConfigMap":
            config_targets.add(target.get("name"))
            _expect(
                operation == {"op": "replace", "path": "/data/BDDK_TELEMETRY_ENABLED", "value": "true"},
                "telemetry enablement patch mismatch",
            )
        else:
            _expect(
                target.get("group") == "apps" and target.get("version") == "v1" and target.get("kind") == "Deployment",
                "telemetry credential patch target mismatch",
            )
            name = target.get("name")
            deployment_targets.add(name)
            _expect(
                operation
                == {
                    "op": "add",
                    "path": "/spec/template/spec/containers/0/env/-",
                    "value": {
                        "name": "BDDK_TELEMETRY_DATABASE_URL",
                        "valueFrom": {
                            "secretKeyRef": {
                                "name": "bddk-mcp-telemetry-db",
                                "key": "BDDK_TELEMETRY_DATABASE_URL",
                            }
                        },
                    },
                },
                "telemetry credential patch mismatch",
            )
    _expect(
        config_targets == {"bddk-mcp-public-config", "bddk-mcp-operator-config"}
        and deployment_targets == {"bddk-mcp-public", "bddk-mcp-operator"},
        "telemetry overlay target isolation mismatch",
    )


def _check_rollback(config: AcceptanceInput) -> None:
    _validate_image(config.release.previous_image)
    _expect(config.rollback.restore_drill_evidence_sha256 != "0" * 64, "restore evidence digest must not be a sentinel")
    _expect(
        config.rollback.runbook_revision != "0" * len(config.rollback.runbook_revision),
        "runbook revision must not be a sentinel",
    )
    _expect(
        config.release.manifest_revision != "0" * len(config.release.manifest_revision),
        "manifest revision must not be a sentinel",
    )


def _run_check(identifier: str, function, *args) -> CheckResult:
    try:
        function(*args)
    except (AssertionError, AttributeError, IndexError, KeyError, StopIteration, TypeError, ValueError):
        return CheckResult(identifier, "fail", "repository-controlled contract failed; inspect the named check locally")
    return CheckResult(identifier, "pass", "repository-controlled contract satisfied")


def run_openshift_preflight(config_path: Path, repository_root: Path) -> AcceptanceEvidence:
    """Validate offline deployment contracts and return sanitized evidence."""
    config, input_sha256 = load_acceptance_input(config_path)
    verifier_revision_sha256 = (
        config.release.manifest_revision
        if len(config.release.manifest_revision) == 64
        else hashlib.sha256(config.release.manifest_revision.encode("ascii")).hexdigest()
    )
    substitutions = {
        _IMAGE_PLACEHOLDER: config.release.image,
        "REPLACE_RELEASE_VERIFIER_REVISION_SHA256": verifier_revision_sha256,
        "REPLACE_RELEASE_VERIFIER_IMAGE_DIGEST": config.release.image.rsplit("@", maxsplit=1)[1],
        "REPLACE_RELEASE_REQUEST_ID": config.release.release_request_id,
        "REPLACE_PUBLIC_ROUTE_HOST": config.platform.public_route_host,
        "REPLACE_OPERATOR_SERVICE_HOST": config.platform.operator_service_host,
        "REPLACE_OPERATOR_CLIENT_ORIGIN": config.platform.operator_client_origin.removeprefix("https://"),
        "REPLACE_BANK_IDP_ISSUER": config.jwt.issuer.removeprefix("https://"),
        "REPLACE_BANK_IDP_JWKS": config.jwt.jwks_url.removeprefix("https://"),
        "REPLACE_PUBLIC_AUDIENCE": config.jwt.public_audience,
        "REPLACE_OPERATOR_AUDIENCE": config.jwt.operator_audience,
        "REPLACE_DATABASE_NAME": config.platform.database_name,
    }
    egress = _load_egress_documents(config_path, config)
    runtime, jobs, manifest_bytes = _render_repository_documents(
        repository_root.resolve(),
        substitutions,
        config.platform.namespace,
        egress,
        config.release.kustomize_binary_sha256,
        config.release.version,
        frozenset(item.policy for item in config.required_egress),
    )
    environment_bytes = json.dumps(
        {
            "namespace": config.platform.namespace,
            "public": config.platform.public_route_host,
            "operator": config.platform.operator_service_host,
            "issuer": config.jwt.issuer,
            "database": config.platform.database_name,
        },
        sort_keys=True,
    ).encode("utf-8")
    checks = (
        _run_check("release-image", _check_release, runtime, jobs, config, repository_root),
        _run_check("namespace-render", _check_namespace, runtime, jobs, config),
        _run_check("route-tls", _check_route, runtime, config),
        _run_check("jwt-claim-contract", _check_jwt, runtime, config),
        _run_check("database-identity-ca", _check_identities, runtime, jobs, repository_root),
        _run_check("network-policy", _check_network, runtime, egress, config),
        _run_check("workloads-lifecycle", _check_workloads, runtime, jobs, config),
        _run_check("bank-bootstrap-trust", _check_bank_bootstrap, jobs),
        _run_check("telemetry-isolation", _check_telemetry, runtime, repository_root),
        _run_check("rollback-metadata", _check_rollback, config),
    )
    image_digest = config.release.image.rsplit("@sha256:", maxsplit=1)[1]
    previous_digest = config.release.previous_image.rsplit("@sha256:", maxsplit=1)[1]
    return AcceptanceEvidence(
        status=(
            "preflight_passed_external_gates_pending"
            if all(check.status == "pass" for check in checks)
            else "preflight_failed"
        ),
        generated_at=datetime.now(UTC).isoformat(),
        input_sha256=input_sha256,
        rendered_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        release_version=config.release.version,
        image_digest=image_digest,
        previous_image_digest=previous_digest,
        renderer_sha256=config.release.kustomize_binary_sha256,
        environment_fingerprint=hashlib.sha256(environment_bytes).hexdigest(),
        rollback_evidence={
            "database_strategy": config.rollback.database_strategy,
            "maximum_recovery_minutes": config.rollback.maximum_recovery_minutes,
            "manifest_revision": config.release.manifest_revision,
            "runbook_revision": config.rollback.runbook_revision,
            "restore_drill_evidence_sha256": config.rollback.restore_drill_evidence_sha256,
            "backup_evidence_id_sha256": hashlib.sha256(config.rollback.backup_evidence_id.encode("utf-8")).hexdigest(),
        },
        checks=checks,
        external_gates=_EXTERNAL_GATES,
    )


def sanitized_failure_evidence(error: OpenShiftAcceptanceError) -> str:
    """Return a stable failure envelope without echoing input values or paths."""
    payload = {
        "schema_version": 1,
        "evidence_scope": "repository_offline_preflight_only",
        "bank_cluster_acceptance": False,
        "status": "preflight_failed",
        "error": {"code": error.code, "message": str(error)},
        "external_gates": list(_EXTERNAL_GATES),
    }
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
