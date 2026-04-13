"""Tests for shared httpx client injection in BddkApiClient."""

import httpx
import pytest

from tests.conftest import MockPool


@pytest.mark.asyncio
async def test_client_accepts_external_http():
    from client import BddkApiClient

    external_http = httpx.AsyncClient(timeout=httpx.Timeout(5.0))
    client = BddkApiClient(pool=MockPool(), http=external_http)
    assert client._http is external_http
    assert client._owns_http is False
    await external_http.aclose()


@pytest.mark.asyncio
async def test_client_creates_own_http():
    from client import BddkApiClient

    client = BddkApiClient(pool=MockPool())
    assert client._http is not None
    assert client._owns_http is True
    await client.close()
