"""Official-client subprocess coverage for the installed stdio MCP command."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

import anyio
import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import LATEST_PROTOCOL_VERSION

from bddk_mcp import __version__
from bddk_mcp.corpus_manifest import CORPUS_SCOPE_WARNING
from bddk_mcp.tools.registry import PUBLIC_TOOL_NAMES
from bddk_mcp.tools.structured_outputs import SOURCE_DATA_BEGIN, SOURCE_DATA_END

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
E2E_SUPPORT = Path(__file__).resolve().parent / "e2e_support"


@pytest.mark.asyncio
async def test_installed_stdio_command_protocol_and_clean_shutdown(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """The installed command must expose, call, reject, recover, and exit cleanly."""
    executable = Path(sys.executable).with_name("bddk-mcp")
    assert executable.is_file(), f"installed bddk-mcp executable not found beside {sys.executable}"

    teardown_sentinel = tmp_path / "stdio-teardown-complete"
    child_env = {
        "BDDK_ADMIN_TOOLS": "false",
        "BDDK_AUTO_SYNC": "false",
        "BDDK_MCP_E2E_STUB": "1",
        "BDDK_MCP_E2E_TEARDOWN_SENTINEL": str(teardown_sentinel),
        "MCP_TRANSPORT": "stdio",
        "PYTHONPATH": os.pathsep.join((str(E2E_SUPPORT), str(REPOSITORY_ROOT))),
        "PYTHONUNBUFFERED": "1",
    }
    parameters = StdioServerParameters(
        command=str(executable),
        args=["serve", "--transport", "stdio"],
        cwd=REPOSITORY_ROOT,
        env=child_env,
    )

    caplog.set_level(logging.ERROR, logger="mcp.client.stdio")
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as child_stderr:
        with anyio.fail_after(20):
            async with stdio_client(parameters, errlog=child_stderr) as (read_stream, write_stream):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=5),
                ) as session:
                    initialized = await session.initialize()
                    assert initialized.protocolVersion == LATEST_PROTOCOL_VERSION
                    assert initialized.serverInfo.name == "BDDK"
                    assert initialized.serverInfo.version == __version__

                    listed = await session.list_tools()
                    assert {tool.name for tool in listed.tools} == set(PUBLIC_TOOL_NAMES)
                    assert len(listed.tools) == len(PUBLIC_TOOL_NAMES)

                    first = await session.call_tool(
                        "get_document_history",
                        {"document_id": "943"},
                    )
                    assert first.isError is False
                    first_text = first.content[0].text
                    assert SOURCE_DATA_BEGIN in first_text
                    assert "No version history found for document 943." in first_text
                    assert SOURCE_DATA_END in first_text
                    assert CORPUS_SCOPE_WARNING in first_text

                    rejected = await session.call_tool(
                        "get_document_history",
                        {"document_id": "943", "unexpected_argument": True},
                    )
                    assert rejected.isError is True
                    assert rejected.content[0].text.startswith("[ERROR:INVALID_INPUT] retryable=false")
                    assert "unexpected_argument" not in rejected.content[0].text
                    assert "pydantic.dev" not in rejected.content[0].text

                    follow_up = await session.call_tool(
                        "get_document_history",
                        {"document_id": "22599"},
                    )
                    assert follow_up.isError is False
                    follow_up_text = follow_up.content[0].text
                    assert SOURCE_DATA_BEGIN in follow_up_text
                    assert "No version history found for document 22599." in follow_up_text
                    assert SOURCE_DATA_END in follow_up_text
                    assert CORPUS_SCOPE_WARNING in follow_up_text
                    assert not teardown_sentinel.exists()

        child_stderr.seek(0)
        stderr_text = child_stderr.read()

    assert teardown_sentinel.read_text(encoding="utf-8") == "closed\n"
    assert "Traceback (most recent call last)" not in stderr_text
    # The official transport parses every stdout line as JSON-RPC.  A server
    # print or other protocol contamination triggers this client-side error.
    assert "Failed to parse JSONRPC message from server" not in caplog.text
