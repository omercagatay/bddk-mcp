"""PostgreSQL transport policy tests."""

from __future__ import annotations

import pytest

from bddk_mcp.db_transport import (
    DatabaseTransportError,
    assert_database_transport,
    materialize_postgres_ca_from_env,
)


def test_verified_database_tls_requires_full_verification_and_absolute_ca(monkeypatch):
    monkeypatch.delenv("BDDK_ALLOW_INSECURE_DATABASE", raising=False)
    dsn = (
        "postgresql://runtime@postgres.bank.internal/bddk"
        "?sslmode=verify-full&sslrootcert=%2Fetc%2Fpki%2Fpostgres-ca.crt"
    )

    assert assert_database_transport(dsn) == dsn


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql://runtime@postgres.bank.internal/bddk",
        "postgresql://runtime@postgres.bank.internal/bddk?sslmode=prefer&sslrootcert=%2Fca.crt",
        "postgresql://runtime@postgres.bank.internal/bddk?sslmode=require&sslrootcert=%2Fca.crt",
        "postgresql://runtime@postgres.bank.internal/bddk?sslmode=verify-full",
        "postgresql://runtime@postgres.bank.internal/bddk?sslmode=verify-full&sslrootcert=relative.crt",
        "host=postgres.bank.internal dbname=bddk sslmode=verify-full sslrootcert=/ca.crt",
    ],
)
def test_unverified_database_transport_is_rejected_without_leaking_the_dsn(monkeypatch, dsn):
    monkeypatch.delenv("BDDK_ALLOW_INSECURE_DATABASE", raising=False)

    with pytest.raises(DatabaseTransportError) as exc_info:
        assert_database_transport(dsn)

    assert dsn not in str(exc_info.value)


def test_insecure_database_transport_requires_explicit_local_opt_in(monkeypatch):
    dsn = "postgresql://local-only@db/bddk"
    monkeypatch.setenv("BDDK_ALLOW_INSECURE_DATABASE", "true")

    assert assert_database_transport(dsn) == dsn


def test_postgres_ca_secret_is_written_only_when_set(tmp_path, monkeypatch):
    target = tmp_path / "ca.pem"
    monkeypatch.delenv("BDDK_POSTGRES_CA_CERT", raising=False)
    monkeypatch.setenv("BDDK_POSTGRES_CA_PATH", str(target))
    materialize_postgres_ca_from_env()
    assert not target.exists()

    monkeypatch.setenv("BDDK_POSTGRES_CA_CERT", "-----BEGIN CERTIFICATE-----\\nQUJD\\n-----END CERTIFICATE-----")
    materialize_postgres_ca_from_env()
    written = target.read_text()
    assert "BEGIN CERTIFICATE" in written and "\n" in written
    assert target.stat().st_mode & 0o777 == 0o600
