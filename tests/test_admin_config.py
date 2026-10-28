from __future__ import annotations

import shlex
import tomllib
from pathlib import Path

import pytest

from bddk_mcp.admin.config import AdminConfig, AdminConfigError
from bddk_mcp.cli import build_parser


@pytest.fixture(autouse=True)
def _allow_insecure_database(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BDDK_ALLOW_INSECURE_DATABASE", "true")


def test_defaults_to_loopback() -> None:
    config = AdminConfig.from_env({"BDDK_DATABASE_URL": "postgresql://u:p@localhost:5432/bddk"})
    assert config.bind_host == "127.0.0.1"
    assert config.loopback_only is True
    assert config.port == 8100


def test_database_url_is_required() -> None:
    with pytest.raises(AdminConfigError, match="BDDK_DATABASE_URL"):
        AdminConfig.from_env({})


def test_non_loopback_bind_without_authentication_fails_closed() -> None:
    env = {
        "BDDK_DATABASE_URL": "postgresql://u:p@localhost:5432/bddk",
        "BDDK_ADMIN_HOST": "0.0.0.0",
    }
    with pytest.raises(AdminConfigError, match="authenticated or loopback-only"):
        AdminConfig.from_env(env)


def test_error_never_leaks_the_database_url() -> None:
    env = {"BDDK_DATABASE_URL": "postgresql://user:sup3rsecret@host:5432/bddk", "BDDK_ADMIN_HOST": "0.0.0.0"}
    with pytest.raises(AdminConfigError) as excinfo:
        AdminConfig.from_env(env)
    assert "sup3rsecret" not in str(excinfo.value)


def test_plaintext_dsn_is_rejected_without_insecure_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BDDK_ALLOW_INSECURE_DATABASE", raising=False)
    with pytest.raises(AdminConfigError, match="sslmode=verify-full"):
        AdminConfig.from_env({"BDDK_DATABASE_URL": "postgresql://u:p@localhost:5432/bddk"})


def _remote_env(**extra: str) -> dict[str, str]:
    env = {
        "BDDK_DATABASE_URL": "postgresql://u:p@localhost:5432/bddk",
        "BDDK_ADMIN_HOST": "0.0.0.0",
        "BDDK_ADMIN_PORT": "8443",
        "BDDK_ADMIN_REMOTE_ENABLED": "true",
        "BDDK_HTTP_ALLOWED_HOSTS": "admin.bank.example:8443",
        "BDDK_HTTP_ALLOWED_ORIGINS": "https://admin.bank.example",
        "BDDK_JWT_ISSUER": "https://id.bank.example/realms/bddk",
        "BDDK_JWT_RESOURCE": "https://mcp.bank.example/mcp",
        "BDDK_JWT_JWKS_URL": "https://id.bank.example/realms/bddk/jwks",
        "BDDK_JWT_AUDIENCE": "bddk-mcp",
        "BDDK_JWT_REQUIRED_SCOPES": "bddk.operator",
    }
    env.update(extra)
    return env


def test_remote_without_opt_in_fails_closed() -> None:
    with pytest.raises(AdminConfigError, match="BDDK_ADMIN_REMOTE_ENABLED"):
        AdminConfig.from_env(_remote_env(BDDK_ADMIN_REMOTE_ENABLED=""))


def test_remote_unauthenticated_opt_in_is_refused() -> None:
    with pytest.raises(AdminConfigError, match="cannot run unauthenticated"):
        AdminConfig.from_env(_remote_env(BDDK_HTTP_ALLOW_UNAUTHENTICATED="true"))


def test_remote_requires_operator_scope() -> None:
    with pytest.raises(AdminConfigError, match="bddk.operator"):
        AdminConfig.from_env(_remote_env(BDDK_JWT_REQUIRED_SCOPES="bddk.read"))


def test_remote_with_opt_in_and_jwt_is_accepted() -> None:
    config = AdminConfig.from_env(_remote_env())
    assert config.loopback_only is False
    assert config.http_security is not None
    assert "bddk.operator" in config.http_security.jwt_required_scopes


def test_port_falls_back_to_platform_port() -> None:
    config = AdminConfig.from_env({"BDDK_DATABASE_URL": "postgresql://u:p@localhost:5432/bddk", "PORT": "9001"})
    assert config.port == 9001


def test_railway_admin_starts_with_platform_port_and_no_bootstrap() -> None:
    root = Path(__file__).resolve().parents[1]
    admin = tomllib.loads((root / "deploy/railway/admin.toml").read_text())
    mcp = tomllib.loads((root / "railway.toml").read_text())
    assert admin["build"]["builder"] == "DOCKERFILE"
    assert admin["build"]["dockerfilePath"] == mcp["build"]["dockerfilePath"]
    assert admin["deploy"]["preDeployCommand"] == []
    assert admin["deploy"]["healthcheckPath"] == "/health/ready"
    assert "bootstrap" in mcp["deploy"]["preDeployCommand"]
    _, *bootstrap_argv = shlex.split(mcp["deploy"]["preDeployCommand"])
    bootstrap_args = build_parser().parse_args(bootstrap_argv)
    assert bootstrap_args.require_quantified_freshness and bootstrap_args.require_verified_signature
    assert "startCommand" not in mcp["deploy"]

    executable, *argv = shlex.split(admin["deploy"]["startCommand"])
    assert executable == "/app/.venv/bin/bddk-mcp"
    args = build_parser().parse_args(argv)
    assert args.command == "admin-ui"
    assert args.port is None
    env = _remote_env(BDDK_ADMIN_HOST=args.host, BDDK_ADMIN_PORT="", PORT="9001")
    config = AdminConfig.from_env(env)
    assert config.bind_host == "0.0.0.0"
    assert config.port == 9001
    assert config.loopback_only is False
    assert config.http_security is not None


def _ontology_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    proposals = tmp_path / "22599.proposals.json"
    proposals.write_text("{}", encoding="utf-8")
    env = {
        "BDDK_DATABASE_URL": "postgresql://u:p@localhost:5432/bddk",
        "BDDK_SEED_DIR": str(tmp_path / "corpus"),
        "BDDK_ONTOLOGY_PROPOSALS": str(proposals),
        "BDDK_ONTOLOGY_REVIEW_DB": str(tmp_path / "review" / "ontology-review.sqlite"),
    }
    env.update(extra)
    return env


def test_ontology_review_settings_are_resolved(tmp_path: Path) -> None:
    config = AdminConfig.from_env(_ontology_env(tmp_path, BDDK_ONTOLOGY_REVIEWER="owner@example"))
    assert config.ontology_proposals == tmp_path / "22599.proposals.json"
    assert config.ontology_review_db == tmp_path / "review" / "ontology-review.sqlite"
    assert config.ontology_reviewer == "owner@example"


def test_ontology_review_is_off_without_both_paths(tmp_path: Path) -> None:
    config = AdminConfig.from_env({"BDDK_DATABASE_URL": "postgresql://u:p@localhost:5432/bddk"})
    assert config.ontology_proposals is None and config.ontology_review_db is None
    assert config.ontology_reviewer == "project_owner"
    half = _ontology_env(tmp_path)
    del half["BDDK_ONTOLOGY_REVIEW_DB"]
    with pytest.raises(AdminConfigError, match="both"):
        AdminConfig.from_env(half)


def test_ontology_review_database_must_stay_outside_the_corpus(tmp_path: Path) -> None:
    env = _ontology_env(tmp_path, BDDK_ONTOLOGY_REVIEW_DB=str(tmp_path / "corpus" / "review.sqlite"))
    with pytest.raises(AdminConfigError, match="outside the corpus"):
        AdminConfig.from_env(env)


def test_ontology_reviewer_must_be_an_identifier(tmp_path: Path) -> None:
    with pytest.raises(AdminConfigError, match="BDDK_ONTOLOGY_REVIEWER"):
        AdminConfig.from_env(_ontology_env(tmp_path, BDDK_ONTOLOGY_REVIEWER="Ahmet Yılmaz"))
