"""PostgreSQL transport policy tests."""

from __future__ import annotations

import pytest

from bddk_mcp.db_transport import DatabaseTransportError, assert_database_transport


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
