"""Offline OpenShift preflight and privacy-safe evidence tests."""

from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from bddk_mcp.openshift_acceptance import (
    OpenShiftAcceptanceError,
    load_acceptance_input,
    run_openshift_preflight,
    sanitized_failure_evidence,
)
from bddk_mcp.tools.registry import PUBLIC_TOOL_NAMES, TOOL_ANNOTATIONS
from scripts import openshift_acceptance as acceptance_cli

ROOT = Path(__file__).parents[1]
REQUIRES_KUSTOMIZE = pytest.mark.skipif(
    shutil.which("kustomize") is None,
    reason="the real pinned Kustomize renderer is unavailable; the required CI lane installs it",
)
KUSTOMIZE_BINARY_SHA256 = (
    hashlib.sha256(Path(shutil.which("kustomize") or __file__).resolve().read_bytes()).hexdigest()
    if shutil.which("kustomize") is not None
    else "a" * 64
)
CURRENT_IMAGE = "registry.bank.example/regulatory/bddk-mcp@sha256:" + "1" * 64
PREVIOUS_IMAGE = "registry.bank.example/regulatory/bddk-mcp@sha256:" + "2" * 64
DNS_PEER = {
    "namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "openshift-dns"}},
    "podSelector": {"matchLabels": {"dns.operator.openshift.io/daemonset-dns": "default"}},
}
POSTGRES_PEER = {"ipBlock": {"cidr": "192.0.2.40/32"}}
IDP_PEER = {"ipBlock": {"cidr": "192.0.2.41/32"}}
REGULATORY_PEER = {"ipBlock": {"cidr": "192.0.2.42/32"}}


def _requirement(
    identifier: str,
    policy: str,
    component: str,
    purpose: str,
    protocol: str,
    port: int,
    peer: dict,
) -> dict:
    return {
        "id": identifier,
        "policy": policy,
        "component": component,
        "purpose": purpose,
        "protocol": protocol,
        "port": port,
        "peer": deepcopy(peer),
    }


def _requirements() -> list[dict]:
    result: list[dict] = []
    for component in ("public", "operator", "lifecycle"):
        policy = f"bddk-mcp-{component}-required-egress"
        result.extend(
            [
                _requirement(f"{component}-dns-udp", policy, component, "dns", "UDP", 53, DNS_PEER),
                _requirement(f"{component}-dns-tcp", policy, component, "dns", "TCP", 53, DNS_PEER),
                _requirement(f"{component}-postgresql", policy, component, "postgresql", "TCP", 5432, POSTGRES_PEER),
            ]
        )
        if component != "lifecycle":
            result.append(_requirement(f"{component}-idp", policy, component, "idp_jwks", "TCP", 443, IDP_PEER))
            result.append(
                _requirement(
                    f"{component}-regulatory-source",
                    policy,
                    component,
                    "regulatory_source",
                    "TCP",
                    443,
                    REGULATORY_PEER,
                )
            )
    return result


def _egress_documents() -> list[dict]:
    documents: list[dict] = []
    for component in ("public", "operator", "lifecycle"):
        requirements = [item for item in _requirements() if item["component"] == component]
        rules: list[dict] = []
        for requirement in requirements:
            candidate = {
                "to": [requirement["peer"]],
                "ports": [{"protocol": requirement["protocol"], "port": requirement["port"]}],
            }
            if candidate not in rules:
                rules.append(candidate)
        documents.append(
            {
                "apiVersion": "networking.k8s.io/v1",
                "kind": "NetworkPolicy",
                "metadata": {"name": f"bddk-mcp-{component}-required-egress"},
                "spec": {
                    "podSelector": {
                        "matchLabels": {
                            "app.kubernetes.io/name": "bddk-mcp",
                            "app.kubernetes.io/component": component,
                        }
                    },
                    "policyTypes": ["Egress"],
                    "egress": rules,
                },
            }
        )
    return documents


def _config() -> dict:
    return {
        "schema_version": 1,
        "release": {
            "version": "5.0.1",
            "image": CURRENT_IMAGE,
            "previous_image": PREVIOUS_IMAGE,
            "manifest_revision": "3" * 40,
            "kustomize_binary_sha256": KUSTOMIZE_BINARY_SHA256,
        },
        "platform": {
            "namespace": "bddk-acceptance",
            "public_route_host": "mcp.acceptance.bank.example",
            "operator_service_host": "bddk-mcp-operator.bddk-acceptance.svc",
            "operator_client_origin": "https://audit.acceptance.bank.example",
            "database_name": "bddk_acceptance",
        },
        "jwt": {
            "issuer": "https://id.acceptance.bank.example/realms/bddk",
            "jwks_url": "https://id.acceptance.bank.example/realms/bddk/protocol/certs",
            "public_audience": "bddk-mcp-public",
            "operator_audience": "bddk-mcp-operator",
            "public_required_scopes": ["bddk.read"],
            "operator_required_scopes": ["bddk.operator"],
            "scope_claims": ["scope", "scp"],
            "algorithms": ["RS256"],
            "access_token_types": ["at+jwt"],
        },
        "egress_policy_files": ["egress.yaml"],
        "required_egress": _requirements(),
        "rollback": {
            "backup_evidence_id": "bank-backup-record-20260715",
            "restore_drill_evidence_sha256": "4" * 64,
            "runbook_revision": "5" * 40,
            "database_strategy": "restore",
            "maximum_recovery_minutes": 120,
        },
    }


def _write_inputs(tmp_path: Path, *, config: dict | None = None, egress: list[dict] | None = None) -> Path:
    config_path = tmp_path / "acceptance.yaml"
    config_path.write_text(yaml.safe_dump(config or _config(), sort_keys=False), encoding="utf-8")
    (tmp_path / "egress.yaml").write_text(yaml.safe_dump_all(egress or _egress_documents()), encoding="utf-8")
    return config_path


def _copy_repository_deployment(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    (root / "deploy").mkdir(parents=True)
    shutil.copytree(ROOT / "deploy" / "openshift", root / "deploy" / "openshift")
    shutil.copytree(ROOT / "deploy" / "openshift-overlays", root / "deploy" / "openshift-overlays")
    return root


def _checks(evidence) -> dict[str, str]:
    return {item.id: item.status for item in evidence.checks}


@REQUIRES_KUSTOMIZE
def test_valid_preflight_passes_offline_and_keeps_every_external_gate_pending(tmp_path: Path):
    config_path = _write_inputs(tmp_path)

    evidence = run_openshift_preflight(config_path, ROOT)

    assert evidence.status == "preflight_passed_external_gates_pending"
    assert set(_checks(evidence).values()) == {"pass"}
    assert _checks(evidence)["bank-bootstrap-trust"] == "pass"
    assert len(evidence.external_gates) == 8
    assert {gate["status"] for gate in evidence.external_gates} == {"not_run"}
    assert all(gate["required_before_production"] is True for gate in evidence.external_gates)
    assert len(evidence.rendered_manifest_sha256) == 64
    assert len(evidence.environment_fingerprint) == 64
    assert evidence.renderer_sha256 == KUSTOMIZE_BINARY_SHA256
    assert evidence.rollback_evidence == {
        "database_strategy": "restore",
        "maximum_recovery_minutes": 120,
        "manifest_revision": "3" * 40,
        "runbook_revision": "5" * 40,
        "restore_drill_evidence_sha256": "4" * 64,
        "backup_evidence_id_sha256": hashlib.sha256(b"bank-backup-record-20260715").hexdigest(),
    }

    report = evidence.to_json()
    assert '"evidence_scope": "repository_offline_preflight_only"' in report
    assert '"bank_cluster_acceptance": false' in report
    for sensitive_environment_value in (
        "mcp.acceptance.bank.example",
        "audit.acceptance.bank.example",
        "id.acceptance.bank.example",
        "bddk_acceptance",
        "bank-backup-record-20260715",
    ):
        assert sensitive_environment_value not in report
    assert "postgresql://" not in report
    assert "Bearer " not in report


def test_placeholder_image_is_rejected_without_echoing_the_value(tmp_path: Path):
    config = _config()
    placeholder = "REPLACE_IMAGE_REGISTRY/bddk-mcp@sha256:REPLACE_64_HEX_IMAGE_DIGEST"
    config["release"]["image"] = placeholder
    config_path = _write_inputs(tmp_path, config=config)

    with pytest.raises(OpenShiftAcceptanceError) as caught:
        load_acceptance_input(config_path)

    assert caught.value.code == "unresolved-config-placeholder"
    report = sanitized_failure_evidence(caught.value)
    assert placeholder not in report
    assert "unresolved placeholders" in report


@pytest.mark.parametrize(
    "image",
    [
        "registry.bank.example/regulatory/bddk-mcp:latest",
        "registry.bank.example/regulatory/bddk-mcp:5.0.1@sha256:" + "1" * 64,
        "https://registry.bank.example/regulatory/bddk-mcp@sha256:" + "1" * 64,
    ],
)
def test_mutable_or_non_oci_image_references_fail_input_validation(tmp_path: Path, image: str):
    config = _config()
    config["release"]["image"] = image
    config_path = _write_inputs(tmp_path, config=config)

    with pytest.raises(OpenShiftAcceptanceError) as caught:
        load_acceptance_input(config_path)

    assert caught.value.code == "invalid-config"
    assert image not in sanitized_failure_evidence(caught.value)


def test_whole_internet_egress_cannot_be_declared_as_required(tmp_path: Path):
    config = _config()
    postgresql = next(item for item in config["required_egress"] if item["id"] == "public-postgresql")
    postgresql["peer"]["ipBlock"]["cidr"] = "0.0.0.0/0"
    config_path = _write_inputs(tmp_path, config=config)

    with pytest.raises(OpenShiftAcceptanceError) as caught:
        load_acceptance_input(config_path)

    assert caught.value.code == "invalid-config"
    assert "0.0.0.0" not in sanitized_failure_evidence(caught.value)


def test_operator_service_dns_must_match_the_declared_namespace(tmp_path: Path):
    config = _config()
    config["platform"]["operator_service_host"] = "bddk-mcp-operator.another-namespace.svc"
    config_path = _write_inputs(tmp_path, config=config)

    with pytest.raises(OpenShiftAcceptanceError) as caught:
        load_acceptance_input(config_path)

    assert caught.value.code == "invalid-config"
    assert "another-namespace" not in sanitized_failure_evidence(caught.value)


@REQUIRES_KUSTOMIZE
def test_renderer_digest_mismatch_fails_before_rendering(tmp_path: Path):
    config = _config()
    config["release"]["kustomize_binary_sha256"] = "f" * 64
    config_path = _write_inputs(tmp_path, config=config)

    with pytest.raises(OpenShiftAcceptanceError) as caught:
        run_openshift_preflight(config_path, ROOT)

    assert caught.value.code == "kustomize-digest"


def test_checked_in_acceptance_templates_are_secret_free_fail_closed_shapes():
    with pytest.raises(OpenShiftAcceptanceError) as caught:
        load_acceptance_input(ROOT / "deploy" / "openshift" / "acceptance.example.yaml")

    assert caught.value.code == "unresolved-config-placeholder"
    policies = [
        item
        for item in yaml.safe_load_all(
            (ROOT / "deploy" / "openshift" / "acceptance-egress.example.yaml").read_text(encoding="utf-8")
        )
        if item
    ]
    assert {item["metadata"]["name"] for item in policies} == {
        "bddk-mcp-public-required-egress",
        "bddk-mcp-operator-required-egress",
        "bddk-mcp-lifecycle-required-egress",
    }
    serialized = yaml.safe_dump_all(policies)
    assert "password" not in serialized.lower()
    assert "token" not in serialized.lower()
    assert "postgresql://" not in serialized
    policies_by_component = {
        item["metadata"]["name"].removeprefix("bddk-mcp-").removesuffix("-required-egress"): item for item in policies
    }
    assert "REPLACE_REGULATORY_PROXY_CIDR" in yaml.safe_dump(policies_by_component["public"])
    assert "REPLACE_REGULATORY_PROXY_CIDR" in yaml.safe_dump(policies_by_component["operator"])
    assert "REPLACE_REGULATORY_PROXY_CIDR" not in yaml.safe_dump(policies_by_component["lifecycle"])

    acceptance = yaml.safe_load((ROOT / "deploy" / "openshift" / "acceptance.example.yaml").read_text())
    source_components = {
        item["component"]
        for item in acceptance["required_egress"]
        if item["purpose"] in {"regulatory_source", "enterprise_proxy"}
    }
    assert source_components == {"public", "operator"}


@pytest.mark.parametrize("component", ["public", "operator"])
def test_missing_required_egress_dependency_fails_strict_input_validation(tmp_path: Path, component: str):
    config = _config()
    config["required_egress"] = [
        item
        for item in config["required_egress"]
        if not (item["component"] == component and item["purpose"] == "regulatory_source")
    ]
    config_path = _write_inputs(tmp_path, config=config)

    with pytest.raises(OpenShiftAcceptanceError) as caught:
        load_acceptance_input(config_path)

    assert caught.value.code == "invalid-config"
    assert str(caught.value) == "acceptance input is invalid at: configuration"


def test_lifecycle_cannot_receive_regulatory_source_egress(tmp_path: Path):
    config = _config()
    config["required_egress"].append(
        _requirement(
            "lifecycle-regulatory-source",
            "bddk-mcp-lifecycle-required-egress",
            "lifecycle",
            "regulatory_source",
            "TCP",
            443,
            REGULATORY_PEER,
        )
    )
    config_path = _write_inputs(tmp_path, config=config)

    with pytest.raises(OpenShiftAcceptanceError) as caught:
        load_acceptance_input(config_path)

    assert caught.value.code == "invalid-config"


def test_live_outbound_public_tools_are_bound_to_runtime_regulatory_egress():
    expected_live_tools = {
        "search_bddk_institutions",
        "search_bddk_announcements",
        "get_bddk_bulletin",
        "get_bddk_bulletin_snapshot",
        "get_bddk_monthly",
        "analyze_bulletin_trends",
        "get_regulatory_digest",
        "compare_bulletin_metrics",
    }
    annotated_live_tools = {name for name in PUBLIC_TOOL_NAMES if TOOL_ANNOTATIONS[name].openWorldHint is True}
    assert annotated_live_tools == expected_live_tools

    source_permissions = {
        (item["component"], item["protocol"], item["port"])
        for item in _requirements()
        if item["purpose"] in {"regulatory_source", "enterprise_proxy"}
    }
    assert source_permissions == {("public", "TCP", 443), ("operator", "TCP", 443)}


@REQUIRES_KUSTOMIZE
def test_network_overlay_extra_rule_fails_closed(tmp_path: Path):
    documents = _egress_documents()
    documents[0]["spec"]["egress"].append(
        {"to": [{"ipBlock": {"cidr": "198.51.100.0/24"}}], "ports": [{"protocol": "TCP", "port": 8443}]}
    )
    config_path = _write_inputs(tmp_path, egress=documents)

    evidence = run_openshift_preflight(config_path, ROOT)

    assert evidence.status == "preflight_failed"
    assert _checks(evidence)["network-policy"] == "fail"
    assert "198.51.100.0" not in evidence.to_json()


@pytest.mark.parametrize(
    ("relative_path", "old", "new", "failed_check"),
    [
        ("deploy/openshift/public-route.yaml", "termination: reencrypt", "termination: edge", "route-tls"),
        (
            "deploy/openshift/public-route.yaml",
            "  wildcardPolicy: None\n",
            "  wildcardPolicy: None\n"
            "  alternateBackends:\n"
            "    - kind: Service\n"
            "      name: unreviewed-service\n"
            "      weight: 100\n",
            "route-tls",
        ),
        (
            "deploy/openshift/services.yaml",
            "  type: ClusterIP\n",
            "  type: LoadBalancer\n",
            "route-tls",
        ),
        (
            "deploy/openshift/configmaps.yaml",
            "BDDK_JWT_REQUIRED_SCOPES: bddk.read",
            "BDDK_JWT_REQUIRED_SCOPES: bddk.operator",
            "jwt-claim-contract",
        ),
        (
            "deploy/openshift/configmaps.yaml",
            'BDDK_ALLOW_INSECURE_DATABASE: "false"',
            'BDDK_ALLOW_INSECURE_DATABASE: "true"',
            "jwt-claim-contract",
        ),
        (
            "deploy/openshift/public-deployment.yaml",
            "readOnlyRootFilesystem: true",
            "readOnlyRootFilesystem: true\n            privileged: true",
            "workloads-lifecycle",
        ),
        (
            "deploy/openshift/public-deployment.yaml",
            'args: [".venv/bin/bddk-mcp", "serve", "--profile", "public"]',
            'args: ["/bin/sh", "-c", "unsafe", "--profile", "public"]',
            "workloads-lifecycle",
        ),
        (
            "deploy/openshift/public-deployment.yaml",
            "          imagePullPolicy: IfNotPresent\n",
            "          imagePullPolicy: IfNotPresent\n"
            "          lifecycle:\n"
            "            postStart:\n"
            "              exec:\n"
            '                command: ["/bin/sh", "-c", "unsafe"]\n',
            "workloads-lifecycle",
        ),
        (
            "deploy/openshift/jobs/migrate.yaml",
            "          imagePullPolicy: IfNotPresent\n",
            "          imagePullPolicy: IfNotPresent\n"
            "          lifecycle:\n"
            "            postStart:\n"
            "              exec:\n"
            '                command: ["/bin/sh", "-c", "unsafe"]\n',
            "workloads-lifecycle",
        ),
        (
            "deploy/openshift/public-deployment.yaml",
            "name: bddk-mcp-public-db",
            "name: bddk-mcp-operator-db",
            "database-identity-ca",
        ),
        (
            "deploy/openshift/operator-deployment.yaml",
            "path: /health/ready",
            "path: /ready",
            "workloads-lifecycle",
        ),
        (
            "deploy/openshift/operator-deployment.yaml",
            "name: BDDK_TLS_CERT_FILE",
            "name: BDDK_TELEMETRY_DATABASE_URL",
            "telemetry-isolation",
        ),
        (
            "deploy/openshift/networkpolicies.yaml",
            "network.openshift.io/policy-group: ingress",
            "network.openshift.io/policy-group: unreviewed",
            "network-policy",
        ),
        (
            "deploy/openshift/public-deployment.yaml",
            "app.kubernetes.io/component: public",
            "app.kubernetes.io/component: operator",
            "workloads-lifecycle",
        ),
        (
            "deploy/openshift/configmaps.yaml",
            'BDDK_TELEMETRY_ENABLED: "false"',
            'BDDK_TELEMETRY_ENABLED: "false"\n  BDDK_TELEMETRY_DATABASE_URL: forbidden-envfrom-credential',
            "jwt-claim-contract",
        ),
        (
            "deploy/openshift-overlays/telemetry/kustomization.yaml",
            "resources:\n  - ../../openshift\n",
            "resources:\n  - ../../openshift\nsecretGenerator:\n  - name: unreviewed-credentials\n    literals:\n      - UNREVIEWED=true\n",
            "telemetry-isolation",
        ),
    ],
)
@REQUIRES_KUSTOMIZE
def test_manifest_contract_tampering_is_attributed_to_a_named_check(
    tmp_path: Path, relative_path: str, old: str, new: str, failed_check: str
):
    config_path = _write_inputs(tmp_path)
    repository = _copy_repository_deployment(tmp_path)
    target = repository / relative_path
    source = target.read_text(encoding="utf-8")
    assert old in source
    target.write_text(source.replace(old, new, 1), encoding="utf-8")

    evidence = run_openshift_preflight(config_path, repository)

    assert evidence.status == "preflight_failed"
    assert _checks(evidence)[failed_check] == "fail"


@REQUIRES_KUSTOMIZE
def test_unreviewed_sidecar_fails_the_workload_inventory(tmp_path: Path):
    config_path = _write_inputs(tmp_path)
    repository = _copy_repository_deployment(tmp_path)
    target = repository / "deploy" / "openshift" / "public-deployment.yaml"
    source = target.read_text(encoding="utf-8")
    marker = "      containers:\n        - name: server"
    replacement = (
        "      containers:\n"
        "        - name: unreviewed-sidecar\n"
        "          image: REPLACE_IMAGE_REGISTRY/bddk-mcp@sha256:REPLACE_64_HEX_IMAGE_DIGEST\n"
        "          args: [sleep, '3600']\n"
        "        - name: server"
    )
    assert marker in source
    target.write_text(source.replace(marker, replacement, 1), encoding="utf-8")

    evidence = run_openshift_preflight(config_path, repository)

    assert evidence.status == "preflight_failed"
    assert _checks(evidence)["release-image"] == "fail"


@pytest.mark.parametrize(
    ("api_version", "kind", "name", "spec"),
    [
        ("rbac.authorization.k8s.io/v1", "RoleBinding", "unreviewed-binding", {"subjects": [], "roleRef": {}}),
        ("v1", "Pod", "unreviewed-pod", {"containers": [{"name": "unreviewed", "image": CURRENT_IMAGE}]}),
    ],
)
@REQUIRES_KUSTOMIZE
def test_unreviewed_rendered_resource_kind_fails_global_inventory(
    tmp_path: Path, api_version: str, kind: str, name: str, spec: dict
):
    config_path = _write_inputs(tmp_path)
    repository = _copy_repository_deployment(tmp_path)
    target = repository / "deploy" / "openshift" / "public-route.yaml"
    source = target.read_text(encoding="utf-8")
    injected = {"apiVersion": api_version, "kind": kind, "metadata": {"name": name}, "spec": spec}
    target.write_text(source + "---\n" + yaml.safe_dump(injected, sort_keys=False), encoding="utf-8")

    with pytest.raises(OpenShiftAcceptanceError) as caught:
        run_openshift_preflight(config_path, repository)

    assert caught.value.code == "render-inventory"


@REQUIRES_KUSTOMIZE
def test_sentinel_rollback_metadata_cannot_pass_preflight(tmp_path: Path):
    config = _config()
    config["rollback"]["restore_drill_evidence_sha256"] = "0" * 64
    config_path = _write_inputs(tmp_path, config=config)

    evidence = run_openshift_preflight(config_path, ROOT)

    assert evidence.status == "preflight_failed"
    assert _checks(evidence)["rollback-metadata"] == "fail"


def test_kustomization_resource_omission_fails_before_rendering(tmp_path: Path):
    config_path = _write_inputs(tmp_path)
    repository = _copy_repository_deployment(tmp_path)
    target = repository / "deploy" / "openshift" / "kustomization.yaml"
    source = target.read_text(encoding="utf-8")
    assert "  - public-route.yaml\n" in source
    target.write_text(source.replace("  - public-route.yaml\n", "", 1), encoding="utf-8")

    with pytest.raises(OpenShiftAcceptanceError) as caught:
        run_openshift_preflight(config_path, repository)

    assert caught.value.code == "invalid-kustomization"


def test_lifecycle_kustomization_resource_omission_fails_before_rendering(tmp_path: Path):
    config_path = _write_inputs(tmp_path)
    repository = _copy_repository_deployment(tmp_path)
    target = repository / "deploy" / "openshift" / "jobs" / "kustomization.yaml"
    source = target.read_text(encoding="utf-8")
    assert "  - bootstrap.yaml\n" in source
    target.write_text(source.replace("  - bootstrap.yaml\n", "", 1), encoding="utf-8")

    with pytest.raises(OpenShiftAcceptanceError) as caught:
        run_openshift_preflight(config_path, repository)

    assert caught.value.code == "invalid-kustomization"


@pytest.mark.parametrize(
    ("relative_path", "old", "new"),
    [
        (
            "bootstrap-job-patch.yaml",
            "--require-quantified-freshness",
            "--allow-unquantified-freshness",
        ),
        (
            "bootstrap-job-patch.yaml",
            "--require-measured-freshness",
            "--allow-unmeasured-freshness",
        ),
        (
            "bootstrap-job-patch.yaml",
            "--require-verified-signature",
            "--allow-unverified-signature",
        ),
        (
            "bootstrap-job-patch.yaml",
            "--trusted-signing-key",
            "--untrusted-signing-key",
        ),
        (
            "bootstrap-job-patch.yaml",
            "claimName: bddk-mcp-approved-corpus",
            "claimName: unreviewed-corpus",
        ),
        (
            "bootstrap-job-patch.yaml",
            "secretName: bddk-mcp-corpus-trust",
            "secretName: unreviewed-trust",
        ),
        (
            "kustomization.yaml",
            "patches:\n",
            "secretGenerator:\n  - name: generated-unreviewed-trust\n    literals: [key=value]\npatches:\n",
        ),
    ],
)
def test_bank_bootstrap_overlay_mutations_fail_before_rendering(tmp_path: Path, relative_path: str, old: str, new: str):
    config_path = _write_inputs(tmp_path)
    repository = _copy_repository_deployment(tmp_path)
    target = repository / "deploy" / "openshift-overlays" / "bank-bootstrap" / relative_path
    source = target.read_text(encoding="utf-8")
    assert old in source
    target.write_text(source.replace(old, new, 1), encoding="utf-8")

    with pytest.raises(OpenShiftAcceptanceError) as caught:
        run_openshift_preflight(config_path, repository)

    assert caught.value.code == "invalid-bank-bootstrap-overlay"


def test_preflight_requires_the_standalone_renderer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path = _write_inputs(tmp_path)
    monkeypatch.setattr("bddk_mcp.openshift_acceptance.shutil.which", lambda _name: None)

    with pytest.raises(OpenShiftAcceptanceError) as caught:
        run_openshift_preflight(config_path, ROOT)

    assert caught.value.code == "kustomize-unavailable"


def test_configuration_errors_do_not_expose_secret_shaped_extra_fields(tmp_path: Path):
    config = _config()
    config["jwt"]["super-secret-key-name"] = "super-secret-bank-value"
    config_path = _write_inputs(tmp_path, config=config)

    with pytest.raises(OpenShiftAcceptanceError) as caught:
        load_acceptance_input(config_path)

    report = json.loads(sanitized_failure_evidence(caught.value))
    assert report["status"] == "preflight_failed"
    assert "super-secret-key-name" not in json.dumps(report)
    assert "super-secret-bank-value" not in json.dumps(report)
    assert report["error"]["code"] == "invalid-config"


def test_cli_suppresses_unexpected_tracebacks_and_never_claims_bank_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    config_path = _write_inputs(tmp_path)

    def _unexpected(*_args, **_kwargs):
        raise RuntimeError("internal path /secret/location and token Bearer private-value")

    monkeypatch.setattr(acceptance_cli, "run_openshift_preflight", _unexpected)

    assert acceptance_cli.main(["--config", str(config_path), "--repository-root", str(ROOT)]) == 3
    report = json.loads(capsys.readouterr().out)
    assert report["evidence_scope"] == "repository_offline_preflight_only"
    assert report["bank_cluster_acceptance"] is False
    assert report["error"]["code"] == "unexpected-preflight-failure"
    assert "/secret/location" not in json.dumps(report)
    assert "private-value" not in json.dumps(report)


@REQUIRES_KUSTOMIZE
def test_cli_success_is_named_repository_preflight_with_external_gates_pending(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    config_path = _write_inputs(tmp_path)

    assert acceptance_cli.main(["--config", str(config_path), "--repository-root", str(ROOT)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "preflight_passed_external_gates_pending"
    assert report["evidence_scope"] == "repository_offline_preflight_only"
    assert report["bank_cluster_acceptance"] is False
    assert {gate["status"] for gate in report["external_gates"]} == {"not_run"}
