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
    assert "actual title/name, not by document_id" in instructions
    assert "keep document_id only for tool calls and structured traceability" in instructions
    assert "document title/name and section/page reference" in instructions
    assert '"943 ilke 5"' not in instructions
    assert '"mevzuat_22599 madde 9"' not in instructions
    assert "section/page" in instructions
    assert "quality warning" in instructions


def test_mcp_instructions_forbid_presenting_repealed_or_draft_text_as_current_law():
    instructions = MCP_INSTRUCTIONS.lower()

    assert "mülga" in instructions and "taslak" in instructions
    assert "never present a document marked mülga" in instructions
    assert "absence of that mark does not establish that a text is in force" in instructions


def test_mcp_instructions_exempt_keyword_hits_from_the_relevance_rule():
    instructions = MCP_INSTRUCTIONS.lower()

    assert "below 50%" in instructions
    assert 'hits with match_strength "keyword" are one of three kinds' in instructions
    assert "(match_type fts_exact) a keyword match on a legal reference of the query that the chunk contains" in (
        instructions
    )
    assert (
        "(match_type fts) a keyword-only match in which the words of the query all occur in the chunk or the "
        "document title as separate words, which does not say that the phrase or a figure of the query occurs"
        in instructions
    )
    assert "every query word occurs" not in instructions
    assert "(match_type decision_number) a decision-number match" in instructions
    assert "no semantic score" in instructions
    assert "relevance 0.0 is not a low score" in instructions
    assert "verify the provision with get_document_section" in instructions
