"""TLS material validation for the Streamable HTTP process."""

from __future__ import annotations

import os
import ssl
from collections.abc import Mapping
from dataclasses import dataclass

from bddk_mcp.http_security import HttpSecurityConfigError

_CERT_ENV = "BDDK_TLS_CERT_FILE"
_KEY_ENV = "BDDK_TLS_KEY_FILE"


@dataclass(frozen=True, slots=True)
class ServerTlsConfig:
    """Validated certificate and private-key paths for Uvicorn."""

    certfile: str | None = None
    keyfile: str | None = None

    @property
    def enabled(self) -> bool:
        """Return whether HTTPS is enabled for the process."""
        return self.certfile is not None

    def uvicorn_options(self) -> dict[str, str]:
        """Return only the TLS keyword arguments understood by Uvicorn."""
        if not self.enabled:
            return {}
        assert self.certfile is not None and self.keyfile is not None
        return {"ssl_certfile": self.certfile, "ssl_keyfile": self.keyfile}


def load_server_tls_config(env: Mapping[str, str] | None = None) -> ServerTlsConfig:
    """Load and validate an optional, inseparable TLS certificate/key pair.

    Certificate parsing happens before the listening socket is opened.  This
    keeps a partially configured or mismatched pair from silently falling back
    to plaintext HTTP.  Omitting both values deliberately leaves TLS disabled,
    which remains useful for loopback-only development and external proxies.
    """
    source = os.environ if env is None else env
    certfile = source.get(_CERT_ENV, "").strip()
    keyfile = source.get(_KEY_ENV, "").strip()

    if bool(certfile) != bool(keyfile):
        raise HttpSecurityConfigError(f"{_CERT_ENV} and {_KEY_ENV} must be configured together")
    if not certfile:
        return ServerTlsConfig()

    try:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=certfile, keyfile=keyfile)
    except (OSError, ValueError, ssl.SSLError):
        raise HttpSecurityConfigError(
            "Configured TLS certificate or private key is unreadable, invalid, encrypted, or mismatched"
        ) from None

    return ServerTlsConfig(certfile=certfile, keyfile=keyfile)
