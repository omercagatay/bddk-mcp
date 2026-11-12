"""The diagnostic script uses the SDK with either MCP response encoding."""

from importlib import import_module

import httpx
import pytest
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp import FastMCP


@pytest.mark.parametrize("module_name", ["mcp_fetch_full"])
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
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
            monkeypatch.setattr(
                module, "streamable_http_client", lambda url: streamable_http_client(url, http_client=client)
            )
            await module.main(["943"])
    assert calls == [("943", 1)]
    assert "Document page body" in capsys.readouterr().out


def test_help_is_an_option_and_the_url_defaults_to_the_local_endpoint(capsys):
    module = import_module("scripts.mcp_fetch_full")
    with pytest.raises(SystemExit) as excinfo:
        module._build_parser().parse_args(["--help"])
    assert excinfo.value.code == 0
    assert module.DEFAULT_URL in capsys.readouterr().out

    args = module._build_parser().parse_args(["943", "2", "3", "--url", "http://x/mcp"])
    assert (args.document_id, args.pages, args.url) == ("943", [2, 3], "http://x/mcp")
    assert module._build_parser().parse_args([]).url == module.DEFAULT_URL
