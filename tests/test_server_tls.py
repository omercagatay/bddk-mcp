"""Fail-closed TLS configuration tests for Streamable HTTP."""

from __future__ import annotations

import ssl
from unittest.mock import MagicMock, patch

import pytest

from bddk_mcp.http_security import HttpSecurityConfigError, load_http_security_config
from bddk_mcp.transport_tls import ServerTlsConfig, load_server_tls_config


def test_tls_is_optional_only_when_both_paths_are_absent():
    config = load_server_tls_config({})

    assert config.enabled is False
    assert config.uvicorn_options() == {}


@pytest.mark.parametrize(
    "env",
    (
        {"BDDK_TLS_CERT_FILE": "/tls/tls.crt"},
        {"BDDK_TLS_KEY_FILE": "/tls/tls.key"},
    ),
)
def test_tls_certificate_and_key_are_an_inseparable_pair(env):
    with pytest.raises(HttpSecurityConfigError, match="must be configured together"):
        load_server_tls_config(env)


def test_tls_material_is_parsed_and_passed_to_uvicorn():
    context = MagicMock()
    env = {
        "BDDK_TLS_CERT_FILE": "/tls/tls.crt",
        "BDDK_TLS_KEY_FILE": "/tls/tls.key",
    }
    with patch("bddk_mcp.transport_tls.ssl.SSLContext", return_value=context) as context_factory:
        tls = load_server_tls_config(env)

    context_factory.assert_called_once_with(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain.assert_called_once_with(certfile="/tls/tls.crt", keyfile="/tls/tls.key")

    from bddk_mcp.server import _uvicorn_options

    http = load_http_security_config({"MCP_HOST": "127.0.0.1", "PORT": "8443"})
    options = _uvicorn_options(http, tls)
    assert options["ssl_certfile"] == "/tls/tls.crt"
    assert options["ssl_keyfile"] == "/tls/tls.key"
    assert options["host"] == "127.0.0.1"
    assert options["port"] == 8443
    assert options["proxy_headers"] is False


@pytest.mark.parametrize("error", (OSError("missing"), ssl.SSLError("mismatch"), ValueError("bad")))
def test_invalid_tls_material_is_rejected_without_leaking_file_details(error):
    context = MagicMock()
    context.load_cert_chain.side_effect = error
    env = {
        "BDDK_TLS_CERT_FILE": "/private/bank/cert.pem",
        "BDDK_TLS_KEY_FILE": "/private/bank/key.pem",
    }
    with (
        patch("bddk_mcp.transport_tls.ssl.SSLContext", return_value=context),
        pytest.raises(HttpSecurityConfigError) as raised,
    ):
        load_server_tls_config(env)

    assert "/private/bank" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_disabled_tls_does_not_add_uvicorn_ssl_options():
    from bddk_mcp.server import _uvicorn_options

    http = load_http_security_config({})
    options = _uvicorn_options(http, ServerTlsConfig())

    assert "ssl_certfile" not in options
    assert "ssl_keyfile" not in options
