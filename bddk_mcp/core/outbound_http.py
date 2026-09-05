"""Bounded outbound HTTP for approved regulatory upstreams.

This module protects explicit live BDDK and mevzuat requests. Application
checks are intentionally paired with a documented requirement for an
infrastructure egress allowlist because DNS validation and socket connection
cannot be atomic at the Python HTTP-client layer.
"""

import asyncio
import ipaddress
import logging
import random
import socket
from collections.abc import Awaitable, Callable, Collection
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from bddk_mcp.core.config import MAX_RETRIES

logger = logging.getLogger(__name__)

BDDK_HTTPS_HOSTS = frozenset({"bddk.org.tr", "www.bddk.org.tr"})
MEVZUAT_HTTPS_HOSTS = frozenset({"mevzuat.gov.tr", "www.mevzuat.gov.tr"})

_REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})
_MAX_UPSTREAM_REDIRECTS = 3


class OutboundHttpPolicyError(RuntimeError):
    """Privacy-safe rejection of an unsafe destination or response."""


def normalize_approved_https_url(
    candidate: str,
    *,
    base_url: str,
    allowed_hosts: Collection[str],
    boundary_name: str,
) -> str:
    """Canonicalize a URL inside an exact-host, default-port HTTPS boundary."""

    if not isinstance(candidate, str) or not candidate.strip():
        raise OutboundHttpPolicyError("Approved upstream URL is missing or invalid.")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in candidate) or "\\" in candidate:
        raise OutboundHttpPolicyError("Approved upstream URL is missing or invalid.")

    joined = urljoin(base_url, candidate.strip())
    try:
        parsed = urlsplit(joined)
        port = parsed.port
    except ValueError:
        raise OutboundHttpPolicyError("Approved upstream URL is missing or invalid.") from None

    host = (parsed.hostname or "").rstrip(".").lower()
    normalized_hosts = frozenset(value.rstrip(".").lower() for value in allowed_hosts)
    if (
        parsed.scheme.lower() != "https"
        or not host
        or host not in normalized_hosts
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
    ):
        raise OutboundHttpPolicyError(f"Upstream URL is outside the approved {boundary_name} HTTPS boundary.")

    netloc = host if port is None else f"{host}:{port}"
    return urlunsplit(("https", netloc, parsed.path or "/", parsed.query, ""))


async def assert_public_https_resolution(url: str) -> None:
    """Reject an approved hostname if any resolved address is non-public.

    This is defense in depth, not a substitute for an egress allowlist: DNS
    resolution and the later socket connection are not atomic in httpx.
    """

    host = urlsplit(url).hostname
    if host is None:
        raise OutboundHttpPolicyError("Approved upstream hostname could not be resolved.")
    try:
        answers = await asyncio.to_thread(socket.getaddrinfo, host, 443, type=socket.SOCK_STREAM)
        addresses = {item[4][0].split("%", 1)[0] for item in answers}
        parsed_addresses = {ipaddress.ip_address(address) for address in addresses}
    except (OSError, ValueError):
        raise OutboundHttpPolicyError("Approved upstream hostname could not be resolved.") from None
    if not parsed_addresses or any(not address.is_global for address in parsed_addresses):
        raise OutboundHttpPolicyError("Approved upstream hostname resolved outside the public network.")


async def _bounded_request_once(
    http: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    base_url: str,
    allowed_hosts: Collection[str],
    boundary_name: str,
    max_bytes: int,
    resolve: Callable[[str], Awaitable[None]],
    max_redirects: int,
    request_kwargs: dict,
) -> httpx.Response:
    """Execute one bounded request attempt, revalidating every redirect."""

    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    if max_redirects < 0:
        raise ValueError("max_redirects must not be negative")

    current_url = normalize_approved_https_url(
        url,
        base_url=base_url,
        allowed_hosts=allowed_hosts,
        boundary_name=boundary_name,
    )
    current_method = method.upper()
    current_kwargs = dict(request_kwargs)
    current_kwargs["follow_redirects"] = False

    for redirect_count in range(max_redirects + 1):
        await resolve(current_url)
        request = httpx.Request(current_method, current_url)
        async with http.stream(current_method, current_url, **current_kwargs) as streamed:
            status_code = streamed.status_code
            headers = dict(streamed.headers)
            # Buffered bodies are already decoded (or deliberately discarded).
            headers.pop("content-encoding", None)
            headers.pop("content-length", None)

            if status_code in _REDIRECT_STATUS_CODES:
                location = streamed.headers.get("location", "")
                if redirect_count >= max_redirects or not location:
                    raise OutboundHttpPolicyError("Approved upstream redirect policy was not satisfied.")
                current_url = normalize_approved_https_url(
                    location,
                    base_url=current_url,
                    allowed_hosts=allowed_hosts,
                    boundary_name=boundary_name,
                )
                if status_code == 303 or (status_code in {301, 302} and current_method == "POST"):
                    current_method = "GET"
                    for payload_key in ("content", "data", "files", "json"):
                        current_kwargs.pop(payload_key, None)
                continue

            # Error bodies are never needed for retry decisions and may be
            # attacker-controlled or unexpectedly large.
            if status_code >= 400 or current_method == "HEAD":
                return httpx.Response(status_code, headers=headers, content=b"", request=request)

            content_length = streamed.headers.get("content-length")
            if content_length:
                try:
                    declared_length = int(content_length)
                except ValueError:
                    raise OutboundHttpPolicyError("Approved upstream response length is invalid.") from None
                if declared_length < 0 or declared_length > max_bytes:
                    raise OutboundHttpPolicyError("Approved upstream response exceeds the download limit.")

            content = bytearray()
            async for chunk in streamed.aiter_bytes():
                if len(content) + len(chunk) > max_bytes:
                    raise OutboundHttpPolicyError("Approved upstream response exceeds the download limit.")
                content.extend(chunk)
            return httpx.Response(status_code, headers=headers, content=bytes(content), request=request)

    raise OutboundHttpPolicyError("Approved upstream redirect policy was not satisfied.")


async def bounded_request_with_retry(
    http: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    base_url: str,
    allowed_hosts: Collection[str],
    boundary_name: str,
    max_bytes: int,
    max_retries: int = MAX_RETRIES,
    max_redirects: int = _MAX_UPSTREAM_REDIRECTS,
    resolve: Callable[[str], Awaitable[None]] | None = None,
    **kwargs,
) -> httpx.Response:
    """Request an approved upstream with bounded streaming and safe retries.

    Policy failures are deterministic and are never retried. Retry logs expose
    only the approved boundary, HTTP method, attempt count, and exception type;
    URL paths, query strings, and exception messages are deliberately omitted.
    """

    if max_retries < 1:
        raise ValueError("max_retries must be positive")
    resolver = resolve or assert_public_https_resolution
    last_exc: httpx.HTTPStatusError | httpx.TransportError | None = None

    for attempt in range(max_retries):
        try:
            response = await _bounded_request_once(
                http,
                method,
                url,
                base_url=base_url,
                allowed_hosts=allowed_hosts,
                boundary_name=boundary_name,
                max_bytes=max_bytes,
                resolve=resolver,
                max_redirects=max_redirects,
                request_kwargs=kwargs,
            )
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code < 500 and exc.response.status_code != 429:
                raise
            last_exc = exc
        except httpx.TransportError as exc:
            last_exc = exc

        if attempt < max_retries - 1:
            logger.warning(
                "Retrying approved upstream HTTP request",
                extra={
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "http_method": method.upper(),
                    "upstream_boundary": boundary_name,
                    "error_type": type(last_exc).__name__,
                },
            )
            await asyncio.sleep(2**attempt + random.uniform(0, 1))

    assert last_exc is not None
    raise last_exc
