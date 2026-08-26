"""Adversarial tests for approved outbound HTTP boundaries."""

import logging
import socket
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from bddk_mcp.core.outbound_http import (
    BDDK_HTTPS_HOSTS,
    OutboundHttpPolicyError,
    assert_public_https_resolution,
    bounded_request_with_retry,
    normalize_approved_https_url,
)

_BASE_URL = "https://www.bddk.org.tr/"


async def _allow_test_resolution(_url: str) -> None:
    return None


def _normalize(url: str) -> str:
    return normalize_approved_https_url(
        url,
        base_url=_BASE_URL,
        allowed_hosts=BDDK_HTTPS_HOSTS,
        boundary_name="BDDK",
    )


async def _request(http: httpx.AsyncClient, url: str, *, max_bytes: int = 1024) -> httpx.Response:
    return await bounded_request_with_retry(
        http,
        "GET",
        url,
        base_url=_BASE_URL,
        allowed_hosts=BDDK_HTTPS_HOSTS,
        boundary_name="BDDK",
        max_bytes=max_bytes,
        max_retries=1,
        resolve=_allow_test_resolution,
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://www.bddk.org.tr/Mevzuat/Liste/50",
        "https://www.bddk.org.tr.evil.example/Mevzuat/Liste/50",
        "https://bddk.org.tr@evil.example/Mevzuat/Liste/50",
        "https://www.bddk.org.tr:444/Mevzuat/Liste/50",
        "https://127.0.0.1/Mevzuat/Liste/50",
        "https://www.bddk.org.tr\\@evil.example/Mevzuat/Liste/50",
    ],
)
def test_exact_https_boundary_rejects_lookalikes_and_unsafe_authorities(url):
    with pytest.raises(OutboundHttpPolicyError):
        _normalize(url)


def test_exact_https_boundary_canonicalizes_relative_url_and_removes_fragment():
    assert _normalize("/Mevzuat/Liste/50?lang=tr#local") == "https://www.bddk.org.tr/Mevzuat/Liste/50?lang=tr"


@pytest.mark.asyncio
async def test_public_resolution_rejects_private_answer(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))],
    )

    with pytest.raises(OutboundHttpPolicyError, match="outside the public network"):
        await assert_public_https_resolution("https://www.bddk.org.tr/")


@pytest.mark.asyncio
async def test_redirect_to_lookalike_is_rejected_before_second_request():
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(
            302,
            headers={"location": "https://www.bddk.org.tr.evil.example/private?token=do-not-log"},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(OutboundHttpPolicyError):
            await _request(http, "https://www.bddk.org.tr/start")

    assert requested == ["https://www.bddk.org.tr/start"]


@pytest.mark.asyncio
async def test_each_approved_redirect_is_revalidated():
    resolved: list[str] = []

    async def resolve(url: str) -> None:
        resolved.append(url)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/final?opaque=value"}, request=request)
        return httpx.Response(200, text="done", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        response = await bounded_request_with_retry(
            http,
            "GET",
            "https://www.bddk.org.tr/start",
            base_url=_BASE_URL,
            allowed_hosts=BDDK_HTTPS_HOSTS,
            boundary_name="BDDK",
            max_bytes=32,
            max_retries=1,
            resolve=resolve,
        )

    assert response.text == "done"
    assert resolved == ["https://www.bddk.org.tr/start", "https://www.bddk.org.tr/final?opaque=value"]


@pytest.mark.asyncio
async def test_declared_oversize_response_is_rejected():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-length": "4096"}, content=b"", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(OutboundHttpPolicyError, match="download limit"):
            await _request(http, "https://www.bddk.org.tr/catalog", max_bytes=16)


class _ChunkStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b"12345678"
        yield b"abcdefgh"


@pytest.mark.asyncio
async def test_chunked_oversize_response_is_rejected_without_content_length():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_ChunkStream(), request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(OutboundHttpPolicyError, match="download limit"):
            await _request(http, "https://www.bddk.org.tr/catalog", max_bytes=12)


@pytest.mark.asyncio
async def test_retry_log_omits_url_query_and_exception_message(caplog):
    attempts = 0
    secret = "opaque-query-secret"
    transport_detail = "private-proxy-password"

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError(transport_detail, request=request)
        return httpx.Response(200, text="ok", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with (
            patch("bddk_mcp.core.outbound_http.asyncio.sleep", new=AsyncMock()),
            caplog.at_level(logging.WARNING, logger="bddk_mcp.core.outbound_http"),
        ):
            response = await bounded_request_with_retry(
                http,
                "GET",
                f"https://www.bddk.org.tr/catalog?token={secret}",
                base_url=_BASE_URL,
                allowed_hosts=BDDK_HTTPS_HOSTS,
                boundary_name="BDDK",
                max_bytes=32,
                max_retries=2,
                resolve=_allow_test_resolution,
            )

    assert response.text == "ok"
    assert attempts == 2
    assert secret not in caplog.text
    assert transport_detail not in caplog.text
    assert "https://" not in caplog.text
    assert caplog.records[0].error_type == "ConnectError"
    assert caplog.records[0].upstream_boundary == "BDDK"


# -- Retry-decision matrix ----------------------------------------------------
# 4xx client errors are the caller's fault and must not be retried; 5xx and 429
# are transient and must be. This is the boundary's own policy
# (outbound_http.py: bounded_request_with_retry).


async def _request_with_retries(http: httpx.AsyncClient, *, max_retries: int) -> httpx.Response:
    return await bounded_request_with_retry(
        http,
        "GET",
        f"{_BASE_URL}probe",
        base_url=_BASE_URL,
        allowed_hosts=BDDK_HTTPS_HOSTS,
        boundary_name="BDDK",
        max_bytes=1024,
        max_retries=max_retries,
        resolve=_allow_test_resolution,
    )


def _counting_transport(*status_codes: int) -> tuple[httpx.MockTransport, list[int]]:
    """Return a transport replaying the given statuses, and its attempt log."""
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        index = min(len(attempts), len(status_codes) - 1)
        status = status_codes[index]
        attempts.append(status)
        return httpx.Response(status, text="", request=request)

    return httpx.MockTransport(handler), attempts


@pytest.mark.asyncio
async def test_client_error_is_not_retried():
    transport, attempts = _counting_transport(404)
    async with httpx.AsyncClient(transport=transport) as http:
        with pytest.raises(httpx.HTTPStatusError):
            await _request_with_retries(http, max_retries=3)
    assert len(attempts) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [500, 502, 429])
async def test_server_error_and_rate_limit_are_retried(status_code):
    transport, attempts = _counting_transport(status_code)
    async with httpx.AsyncClient(transport=transport) as http:
        with (
            patch("bddk_mcp.core.outbound_http.asyncio.sleep", new=AsyncMock()),
            pytest.raises(httpx.HTTPStatusError),
        ):
            await _request_with_retries(http, max_retries=3)
    assert len(attempts) == 3


@pytest.mark.asyncio
async def test_transient_server_error_then_success_returns_the_response():
    transport, attempts = _counting_transport(500, 200)
    async with httpx.AsyncClient(transport=transport) as http:
        with patch("bddk_mcp.core.outbound_http.asyncio.sleep", new=AsyncMock()):
            response = await _request_with_retries(http, max_retries=3)
    assert response.status_code == 200
    assert attempts == [500, 200]
