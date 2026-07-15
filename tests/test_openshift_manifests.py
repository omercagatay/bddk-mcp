"""Static acceptance checks for the OpenShift deployment starter."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
OPENSHIFT = ROOT / "deploy" / "openshift"
OPENSHIFT_TELEMETRY = ROOT / "deploy" / "openshift-overlays" / "telemetry"


def _documents(path: Path) -> list[dict]:
    return [document for document in yaml.safe_load_all(path.read_text(encoding="utf-8")) if document]


def _runtime_documents() -> list[dict]:
    names = (
        "serviceaccounts.yaml",
        "service-ca.yaml",
        "configmaps.yaml",
        "public-deployment.yaml",
        "operator-deployment.yaml",
        "services.yaml",
        "public-route.yaml",
        "networkpolicies.yaml",
    )
    return [document for name in names for document in _documents(OPENSHIFT / name)]


def _container(document: dict) -> dict:
    return document["spec"]["template"]["spec"]["containers"][0]


def _render_kustomization(directory: Path) -> list[dict]:
    required = os.getenv("BDDK_REQUIRE_KUSTOMIZE") == "1"
    if executable := shutil.which("kustomize"):
        command = [executable, "build", str(directory)]
    elif required:
        pytest.fail("BDDK_REQUIRE_KUSTOMIZE=1 but the standalone kustomize executable is unavailable")
    elif executable := shutil.which("kubectl"):
        command = [executable, "kustomize", str(directory)]
    elif executable := shutil.which("oc"):
        command = [executable, "kustomize", str(directory)]
    else:
        pytest.skip("kustomize, kubectl, and oc are unavailable")

    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return [document for document in yaml.safe_load_all(result.stdout) if document]


def test_required_kustomize_renderer_fails_closed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BDDK_REQUIRE_KUSTOMIZE", "1")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/kubectl" if name == "kubectl" else None)

    with pytest.raises(pytest.fail.Exception, match="BDDK_REQUIRE_KUSTOMIZE=1"):
        _render_kustomization(OPENSHIFT)


def _selector_map(documents: list[dict]) -> dict[tuple[str, str], dict]:
    selectors: dict[tuple[str, str], dict] = {}
    for document in documents:
        kind = document["kind"]
        name = document["metadata"]["name"]
        if kind == "Deployment":
            selectors[(kind, name)] = document["spec"]["selector"]
        elif kind == "Service":
            selectors[(kind, name)] = document["spec"]["selector"]
        elif kind == "NetworkPolicy":
            selectors[(kind, name)] = document["spec"]["podSelector"]
    return selectors


def test_kustomization_references_runtime_resources_but_not_secret_examples_or_jobs():
    kustomization = yaml.safe_load((OPENSHIFT / "kustomization.yaml").read_text(encoding="utf-8"))
    resources = set(kustomization["resources"])
    assert "public-deployment.yaml" in resources
    assert "operator-deployment.yaml" in resources
    assert "public-route.yaml" in resources
    assert "service-ca.yaml" in resources
    assert "secrets.example.yaml" not in resources
    assert not any(resource.startswith("jobs/") for resource in resources)
    assert "commonLabels" not in kustomization
    assert kustomization["labels"] == [
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
    ]


def test_release_version_is_absent_from_source_selectors():
    selectors = _selector_map(_runtime_documents())
    assert selectors
    for selector in selectors.values():
        assert "app.kubernetes.io/version" not in yaml.safe_dump(selector)


def test_rendered_release_change_preserves_every_selector(tmp_path: Path):
    current = _render_kustomization(OPENSHIFT)
    next_release_dir = tmp_path / "openshift"
    shutil.copytree(OPENSHIFT, next_release_dir)
    kustomization_path = next_release_dir / "kustomization.yaml"
    kustomization = kustomization_path.read_text(encoding="utf-8")
    assert "app.kubernetes.io/version: 5.0.1" in kustomization
    kustomization_path.write_text(
        kustomization.replace("app.kubernetes.io/version: 5.0.1", "app.kubernetes.io/version: 5.0.2"),
        encoding="utf-8",
    )
    next_release = _render_kustomization(next_release_dir)

    assert _selector_map(current) == _selector_map(next_release)
    for document in current:
        if document["kind"] == "Deployment":
            assert document["metadata"]["labels"]["app.kubernetes.io/version"] == "5.0.1"
            assert document["spec"]["template"]["metadata"]["labels"]["app.kubernetes.io/version"] == "5.0.1"
    for document in next_release:
        if document["kind"] == "Deployment":
            assert document["metadata"]["labels"]["app.kubernetes.io/version"] == "5.0.2"
            assert document["spec"]["template"]["metadata"]["labels"]["app.kubernetes.io/version"] == "5.0.2"


def test_public_is_the_only_routed_component():
    documents = _runtime_documents()
    routes = [document for document in documents if document["kind"] == "Route"]
    assert len(routes) == 1
    assert routes[0]["spec"]["to"]["name"] == "bddk-mcp-public"
    assert routes[0]["spec"]["port"]["targetPort"] == "https"
    assert routes[0]["spec"]["tls"]["termination"] == "reencrypt"
    assert routes[0]["spec"]["tls"]["insecureEdgeTerminationPolicy"] == "Redirect"


def test_services_generate_distinct_serving_certificates_and_expose_only_https():
    services = [document for document in _runtime_documents() if document["kind"] == "Service"]
    assert {service["metadata"]["name"] for service in services} == {
        "bddk-mcp-public",
        "bddk-mcp-operator",
    }
    for service in services:
        name = service["metadata"]["name"]
        assert service["metadata"]["annotations"]["service.beta.openshift.io/serving-cert-secret-name"] == f"{name}-tls"
        assert service["spec"]["ports"] == [{"name": "https", "port": 443, "targetPort": "https", "protocol": "TCP"}]

    service_ca = _documents(OPENSHIFT / "service-ca.yaml")[0]
    assert service_ca["kind"] == "ConfigMap"
    assert service_ca["metadata"]["annotations"]["service.beta.openshift.io/inject-cabundle"] == "true"
    assert service_ca["data"] == {}


def test_public_and_operator_use_distinct_profiles_identities_and_database_secrets():
    deployments = {
        document["metadata"]["name"]: document for document in _runtime_documents() if document["kind"] == "Deployment"
    }
    public = deployments["bddk-mcp-public"]
    operator = deployments["bddk-mcp-operator"]
    assert public["spec"]["template"]["spec"]["serviceAccountName"] == "bddk-mcp-public"
    assert operator["spec"]["template"]["spec"]["serviceAccountName"] == "bddk-mcp-operator"
    assert _container(public)["args"][-1] == "public"
    assert _container(operator)["args"][-1] == "operator"
    assert _container(public)["envFrom"] == [{"configMapRef": {"name": "bddk-mcp-public-config"}}]
    assert _container(operator)["envFrom"] == [{"configMapRef": {"name": "bddk-mcp-operator-config"}}]
    public_env = {item["name"]: item for item in _container(public)["env"]}
    operator_env = {item["name"]: item for item in _container(operator)["env"]}
    assert public_env["BDDK_DATABASE_URL"]["valueFrom"]["secretKeyRef"] == {
        "name": "bddk-mcp-public-db",
        "key": "BDDK_DATABASE_URL",
    }
    assert operator_env["BDDK_OPERATOR_DATABASE_URL"]["valueFrom"]["secretKeyRef"] == {
        "name": "bddk-mcp-operator-db",
        "key": "BDDK_OPERATOR_DATABASE_URL",
    }
    assert "BDDK_TELEMETRY_DATABASE_URL" not in public_env
    assert "BDDK_TELEMETRY_DATABASE_URL" not in operator_env
    assert "bddk-mcp-telemetry-db" not in yaml.safe_dump(public)
    assert "bddk-mcp-telemetry-db" not in yaml.safe_dump(operator)
    assert operator["spec"]["replicas"] == 1
    assert operator["spec"]["strategy"]["type"] == "Recreate"


def test_workloads_meet_restricted_security_and_probe_baseline():
    workloads = [document for document in _runtime_documents() if document["kind"] == "Deployment"] + [
        _documents(OPENSHIFT / "jobs" / name)[0] for name in ("migrate.yaml", "bootstrap.yaml")
    ]
    for workload in workloads:
        pod = workload["spec"]["template"]["spec"]
        container = pod["containers"][0]
        assert pod["automountServiceAccountToken"] is False
        assert pod["securityContext"]["runAsNonRoot"] is True
        assert pod["securityContext"]["seccompProfile"]["type"] == "RuntimeDefault"
        assert container["securityContext"]["allowPrivilegeEscalation"] is False
        assert container["securityContext"]["readOnlyRootFilesystem"] is True
        assert container["securityContext"]["capabilities"]["drop"] == ["ALL"]
        assert container["resources"]["requests"]
        assert container["resources"]["limits"]
        assert any(mount["mountPath"] == "/tmp" for mount in container["volumeMounts"])
        postgres_mount = next(mount for mount in container["volumeMounts"] if mount["name"] == "postgres-ca")
        assert postgres_mount == {
            "name": "postgres-ca",
            "mountPath": "/var/run/configmaps/bddk-mcp/postgres",
            "readOnly": True,
        }
        postgres_volume = next(volume for volume in pod["volumes"] if volume["name"] == "postgres-ca")
        assert postgres_volume["configMap"] == {
            "name": "bddk-mcp-postgres-ca",
            "items": [{"key": "ca.crt", "path": "ca.crt"}],
        }

    for workload in workloads[:2]:
        container = _container(workload)
        assert container["livenessProbe"]["httpGet"]["path"] == "/health/live"
        assert container["readinessProbe"]["httpGet"]["path"] == "/health/ready"
        assert container["livenessProbe"]["httpGet"]["scheme"] == "HTTPS"
        assert container["readinessProbe"]["httpGet"]["scheme"] == "HTTPS"
        assert container["ports"] == [{"name": "https", "containerPort": 8000, "protocol": "TCP"}]

        env = {item["name"]: item["value"] for item in container["env"] if "value" in item}
        assert env == {
            "BDDK_TLS_CERT_FILE": "/var/run/secrets/bddk-mcp/tls/tls.crt",
            "BDDK_TLS_KEY_FILE": "/var/run/secrets/bddk-mcp/tls/tls.key",
        }
        tls_mount = next(mount for mount in container["volumeMounts"] if mount["name"] == "service-tls")
        assert tls_mount["readOnly"] is True
        assert tls_mount["mountPath"] == "/var/run/secrets/bddk-mcp/tls"
        tls_volume = next(
            volume for volume in workload["spec"]["template"]["spec"]["volumes"] if volume["name"] == "service-tls"
        )
        assert tls_volume["secret"]["secretName"] == f"{workload['metadata']['name']}-tls"
        assert tls_volume["secret"]["defaultMode"] == 0o440


def test_lifecycle_jobs_use_distinct_database_secrets_and_service_account():
    migrate = _documents(OPENSHIFT / "jobs" / "migrate.yaml")[0]
    bootstrap = _documents(OPENSHIFT / "jobs" / "bootstrap.yaml")[0]
    assert migrate["spec"]["template"]["spec"]["serviceAccountName"] == "bddk-mcp-lifecycle"
    assert bootstrap["spec"]["template"]["spec"]["serviceAccountName"] == "bddk-mcp-lifecycle"
    assert "envFrom" not in _container(migrate)
    assert "envFrom" not in _container(bootstrap)
    assert _container(migrate)["env"] == [
        {
            "name": "BDDK_EXPECTED_DATABASE_NAME",
            "value": "REPLACE_DATABASE_NAME",
        },
        {
            "name": "BDDK_SCHEMA_OWNER_DATABASE_URL",
            "valueFrom": {
                "secretKeyRef": {
                    "name": "bddk-mcp-schema-owner-db",
                    "key": "BDDK_SCHEMA_OWNER_DATABASE_URL",
                }
            },
        },
    ]
    assert _container(bootstrap)["env"] == [
        {
            "name": "BDDK_INGESTION_DATABASE_URL",
            "valueFrom": {
                "secretKeyRef": {
                    "name": "bddk-mcp-ingestion-db",
                    "key": "BDDK_INGESTION_DATABASE_URL",
                }
            },
        }
    ]


def test_every_workload_uses_one_immutable_application_digest_placeholder():
    workloads = [document for document in _runtime_documents() if document["kind"] == "Deployment"] + [
        _documents(OPENSHIFT / "jobs" / name)[0] for name in ("migrate.yaml", "bootstrap.yaml")
    ]
    images = {_container(workload)["image"] for workload in workloads}
    assert images == {"REPLACE_IMAGE_REGISTRY/bddk-mcp@sha256:REPLACE_64_HEX_IMAGE_DIGEST"}
    assert all(_container(workload)["imagePullPolicy"] == "IfNotPresent" for workload in workloads)


def test_baseline_fails_closed_with_default_deny_egress():
    policies = [document for document in _runtime_documents() if document["kind"] == "NetworkPolicy"]
    deny = next(policy for policy in policies if policy["metadata"]["name"] == "bddk-mcp-default-deny-egress")
    assert deny["spec"] == {
        "podSelector": {"matchLabels": {"app.kubernetes.io/name": "bddk-mcp"}},
        "policyTypes": ["Egress"],
        "egress": [],
    }


def test_no_baseline_workload_imports_a_whole_secret():
    workloads = [document for document in _runtime_documents() if document["kind"] == "Deployment"] + [
        _documents(OPENSHIFT / "jobs" / name)[0] for name in ("migrate.yaml", "bootstrap.yaml")
    ]
    for workload in workloads:
        for source in _container(workload).get("envFrom", []):
            assert "secretRef" not in source


def test_telemetry_overlay_is_an_explicit_exact_key_opt_in():
    overlay = yaml.safe_load((OPENSHIFT_TELEMETRY / "kustomization.yaml").read_text(encoding="utf-8"))
    assert overlay["resources"] == ["../../openshift"]
    assert len(overlay["patches"]) == 4

    config_targets: set[str] = set()
    deployment_targets: set[str] = set()
    for patch in overlay["patches"]:
        operations = yaml.safe_load(patch["patch"])
        assert len(operations) == 1
        operation = operations[0]
        target = patch["target"]
        if target["kind"] == "ConfigMap":
            config_targets.add(target["name"])
            assert operation == {
                "op": "replace",
                "path": "/data/BDDK_TELEMETRY_ENABLED",
                "value": "true",
            }
        else:
            assert target["kind"] == "Deployment"
            deployment_targets.add(target["name"])
            assert operation["op"] == "add"
            assert operation["path"] == "/spec/template/spec/containers/0/env/-"
            assert operation["value"] == {
                "name": "BDDK_TELEMETRY_DATABASE_URL",
                "valueFrom": {
                    "secretKeyRef": {
                        "name": "bddk-mcp-telemetry-db",
                        "key": "BDDK_TELEMETRY_DATABASE_URL",
                    }
                },
            }

    assert config_targets == {"bddk-mcp-public-config", "bddk-mcp-operator-config"}
    assert deployment_targets == {"bddk-mcp-public", "bddk-mcp-operator"}


def test_rendered_baseline_excludes_telemetry_and_overlay_enables_it():
    baseline = _render_kustomization(OPENSHIFT)
    telemetry = _render_kustomization(OPENSHIFT_TELEMETRY)

    for documents, expected_enabled in ((baseline, "false"), (telemetry, "true")):
        config_maps = {
            document["metadata"]["name"]: document
            for document in documents
            if document["kind"] == "ConfigMap" and document["metadata"]["name"].endswith("-config")
        }
        deployments = [document for document in documents if document["kind"] == "Deployment"]
        assert config_maps
        assert deployments
        assert {config["data"]["BDDK_TELEMETRY_ENABLED"] for config in config_maps.values()} == {expected_enabled}
        for deployment in deployments:
            env = {item["name"]: item for item in _container(deployment)["env"]}
            if expected_enabled == "false":
                assert "BDDK_TELEMETRY_DATABASE_URL" not in env
                assert "bddk-mcp-telemetry-db" not in yaml.safe_dump(deployment)
            else:
                assert env["BDDK_TELEMETRY_DATABASE_URL"]["valueFrom"]["secretKeyRef"] == {
                    "name": "bddk-mcp-telemetry-db",
                    "key": "BDDK_TELEMETRY_DATABASE_URL",
                }


def test_secret_examples_assign_one_database_variable_per_identity():
    secrets = {
        document["metadata"]["name"]: document["stringData"]
        for document in _documents(OPENSHIFT / "secrets.example.yaml")
    }
    tls = "sslmode=verify-full&sslrootcert=%2Fvar%2Frun%2Fconfigmaps%2Fbddk-mcp%2Fpostgres%2Fca.crt"
    assert secrets == {
        "bddk-mcp-public-db": {"BDDK_DATABASE_URL": f"postgresql://REPLACE_PUBLIC_READER_DSN?{tls}"},
        "bddk-mcp-operator-db": {"BDDK_OPERATOR_DATABASE_URL": f"postgresql://REPLACE_OPERATOR_DSN?{tls}"},
        "bddk-mcp-schema-owner-db": {
            "BDDK_SCHEMA_OWNER_DATABASE_URL": (
                f"postgresql://REPLACE_SCHEMA_OWNER_DSN?options=-c%20role%3Dbddk_schema_owner&{tls}"
            )
        },
        "bddk-mcp-ingestion-db": {"BDDK_INGESTION_DATABASE_URL": f"postgresql://REPLACE_INGESTION_DSN?{tls}"},
        "bddk-mcp-telemetry-db": {"BDDK_TELEMETRY_DATABASE_URL": f"postgresql://REPLACE_TELEMETRY_WRITER_DSN?{tls}"},
    }


def test_remote_profiles_are_fail_closed_and_scoped():
    config_maps = {
        document["metadata"]["name"]: document["data"] for document in _documents(OPENSHIFT / "configmaps.yaml")
    }
    public = config_maps["bddk-mcp-public-config"]
    operator = config_maps["bddk-mcp-operator-config"]
    assert public["BDDK_JWT_REQUIRED_SCOPES"] == "bddk.read"
    assert operator["BDDK_JWT_REQUIRED_SCOPES"] == "bddk.operator"
    assert operator["BDDK_OPERATOR_REMOTE_ENABLED"] == "true"
    assert public["BDDK_TELEMETRY_ENABLED"] == "false"
    assert operator["BDDK_TELEMETRY_ENABLED"] == "false"
    for config in (public, operator):
        assert config["BDDK_ALLOW_INSECURE_DATABASE"] == "false"
        assert config["MCP_HOST"] == "0.0.0.0"
        assert config["BDDK_HTTP_ALLOWED_HOSTS"]
        assert config["BDDK_HTTP_ALLOWED_ORIGINS"].startswith("https://")
        assert config["BDDK_JWT_ISSUER"].startswith("https://")
        assert config["BDDK_JWT_JWKS_URL"].startswith("https://")
        assert config["BDDK_JWT_RESOURCE"].startswith("https://")
        assert config["BDDK_JWT_AUDIENCE"].startswith("REPLACE_")
        assert config["BDDK_JWT_ACCESS_TOKEN_TYPES"] == "at+jwt"


def test_docker_images_have_non_root_defaults_and_versioned_build_inputs():
    for name in ("Dockerfile", "Dockerfile.spaces"):
        dockerfile = (ROOT / name).read_text(encoding="utf-8")
        assert "ghcr.io/astral-sh/uv:0.11.14" in dockerfile
        assert "ghcr.io/astral-sh/uv:latest" not in dockerfile
        assert "revision='d13f1b27baf31030b7fd040960d60d909913633f'" in dockerfile
        assert "BDDK_EMBEDDING_MODEL_PATH=/app/embedding_model" in dockerfile
        assert "USER 10001:0" in dockerfile

    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert "!bddk_mcp/**" in dockerignore
    assert "!seed_data/**" in dockerignore
