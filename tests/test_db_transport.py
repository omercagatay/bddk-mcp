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


@pytest.fixture
def ca_pem():
    from datetime import UTC, datetime, timedelta

    from cryptography import x509
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.x509.oid import NameOID

    key = Ed25519PrivateKey.generate()
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test-only database CA")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(UTC) - timedelta(days=1))
        .not_valid_after(datetime.now(UTC) + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, algorithm=None)
    )
    return cert.public_bytes(serialization.Encoding.PEM).decode()


def test_optional_environment_ca_is_materialized_without_weakening_tls(monkeypatch, tmp_path, ca_pem):
    import stat
    from urllib.parse import urlencode

    from bddk_mcp import db_transport

    path = tmp_path / "database-ca.pem"
    monkeypatch.setattr(db_transport, "_ENV_CA_PATH", path, raising=False)
    monkeypatch.delenv("BDDK_ALLOW_INSECURE_DATABASE", raising=False)
    monkeypatch.setenv("BDDK_DATABASE_CA_PEM", ca_pem)
    dsn = "postgresql://reader@db.invalid/bddk?" + urlencode({"sslmode": "verify-full", "sslrootcert": str(path)})
    assert assert_database_transport(dsn) == dsn
    assert path.read_text() == ca_pem
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(DatabaseTransportError):
        assert_database_transport(dsn.replace("verify-full", "require"))


@pytest.mark.parametrize("invalid_kind", ["malformed", "empty", "oversized", "non_ascii", "private_key"])
def test_invalid_ca_preserves_existing_file_and_does_not_leak_input(monkeypatch, tmp_path, ca_pem, invalid_kind):
    from urllib.parse import urlencode

    from bddk_mcp import db_transport

    path = tmp_path / "database-ca.pem"
    path.write_text(ca_pem)
    monkeypatch.setattr(db_transport, "_ENV_CA_PATH", path, raising=False)
    monkeypatch.delenv("BDDK_ALLOW_INSECURE_DATABASE", raising=False)
    invalid = {
        "malformed": "DO-NOT-LOG-THIS-INVALID-INPUT",
        "empty": "",
        "oversized": "x" * (64 * 1024 + 1),
        "non_ascii": "geçersiz sertifika",
        "private_key": ca_pem + "PRIVATE KEY material must never be accepted",
    }[invalid_kind]
    monkeypatch.setenv("BDDK_DATABASE_CA_PEM", invalid)
    dsn = "postgresql://reader@db.invalid/bddk?" + urlencode({"sslmode": "verify-full", "sslrootcert": str(path)})
    with pytest.raises(DatabaseTransportError) as error:
        assert_database_transport(dsn)
    assert "DO-NOT-LOG" not in str(error.value)
    assert path.read_text() == ca_pem


def test_ca_install_replaces_symlink_without_overwriting_its_target(monkeypatch, tmp_path, ca_pem):
    from urllib.parse import urlencode

    from bddk_mcp import db_transport

    target = tmp_path / "unrelated-file"
    target.write_text("preserve unrelated content")
    path = tmp_path / "database-ca.pem"
    path.symlink_to(target)
    monkeypatch.setattr(db_transport, "_ENV_CA_PATH", path, raising=False)
    monkeypatch.delenv("BDDK_ALLOW_INSECURE_DATABASE", raising=False)
    monkeypatch.setenv("BDDK_DATABASE_CA_PEM", ca_pem)
    dsn = "postgresql://reader@db.invalid/bddk?" + urlencode({"sslmode": "verify-full", "sslrootcert": str(path)})
    assert_database_transport(dsn)
    assert target.read_text() == "preserve unrelated content"
    assert not path.is_symlink()
    assert path.read_text() == ca_pem
