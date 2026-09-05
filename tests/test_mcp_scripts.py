"""The diagnostic scripts use the SDK with either MCP response encoding."""

from importlib import import_module
from types import SimpleNamespace

import httpx
import pytest
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp import FastMCP


@pytest.mark.parametrize("module_name", ["mcp_smoke", "mcp_fetch_full", "validate_all_pages"])
@pytest.mark.parametrize("json_response", [True, False])
async def test_scripts_fetch_pages_with_the_official_client(monkeypatch, capsys, module_name, json_response):
    module = import_module(f"scripts.{module_name}")
    server = FastMCP("script-check", json_response=json_response, stateless_http=True)
    calls = []

    @server.tool()
    def get_bddk_document(document_id: str, page_number: int) -> str:
        calls.append((document_id, page_number))
        return "Document page body"

    app = server.streamable_http_app()
    monkeypatch.setattr(module.sys, "argv", [module_name, "943"])
    if module_name == "validate_all_pages":
        monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="943|1\n"))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
            monkeypatch.setattr(
                module, "streamable_http_client", lambda url: streamable_http_client(url, http_client=client)
            )
            await module.main()
    assert calls == [("943", 1)]
    if module_name != "validate_all_pages":
        assert "Document page body" in capsys.readouterr().out
