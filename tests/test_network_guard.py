"""The autouse network guard in conftest.py: outbound attempts fail, loopback works, ``network`` opts out."""

import asyncio
import socket

import pytest

REAL_GETADDRINFO = socket.getaddrinfo
TEST_NET = ("192.0.2.1", 80)  # RFC 5737 documentation address: never routed, so a broken guard fails fast


def test_outbound_connect_is_blocked(network_guard):
    with socket.socket() as sock:
        sock.settimeout(1)
        with pytest.raises(RuntimeError, match="Network access blocked in a unit test: connect to"):
            sock.connect(TEST_NET)
        with pytest.raises(RuntimeError, match="Network access blocked"):
            sock.connect_ex(TEST_NET)
    assert network_guard == [repr(TEST_NET)] * 2
    network_guard.clear()


def test_non_local_dns_lookup_is_blocked(network_guard):
    with pytest.raises(RuntimeError, match="DNS lookup of 'www.bddk.org.tr'"):
        socket.getaddrinfo("www.bddk.org.tr", 443)
    network_guard.clear()


async def test_event_loop_connect_to_an_ip_literal_is_blocked(network_guard):
    with pytest.raises(RuntimeError, match="Network access blocked"):
        await asyncio.wait_for(asyncio.open_connection(*TEST_NET), timeout=2)
    network_guard.clear()


def test_loopback_connect_and_lookup_work(network_guard):
    assert socket.getaddrinfo("localhost", 80)
    with socket.create_server(("127.0.0.1", 0)) as server, socket.create_connection(server.getsockname(), timeout=2):
        pass
    assert network_guard == []


async def test_event_loop_loopback_connection_works():
    server = await asyncio.start_server(lambda reader, writer: writer.close(), "127.0.0.1", 0)
    async with server:
        reader, writer = await asyncio.open_connection(*server.sockets[0].getsockname())
        writer.close()
        await writer.wait_closed()


@pytest.mark.network
def test_network_marked_test_is_exempt():
    assert socket.getaddrinfo is REAL_GETADDRINFO
    assert "connect" not in socket.socket.__dict__
