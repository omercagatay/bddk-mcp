from bddk_mcp.server import MCP_INSTRUCTIONS


def test_mcp_instructions_hide_tool_protocol_details():
    instructions = MCP_INSTRUCTIONS.lower()

    assert "tool schemas" in instructions
    assert "request/response" in instructions
    assert "tool traces" in instructions
    assert "internal reasoning" in instructions


def test_mcp_instructions_require_user_facing_answer_style():
    instructions = MCP_INSTRUCTIONS.lower()

    assert "user's language" in instructions
    assert "document_id" in instructions
    assert "section/page" in instructions
    assert "quality warning" in instructions
