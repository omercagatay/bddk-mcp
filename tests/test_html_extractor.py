"""html_to_markdown must not turn non-content nodes into text."""

from bddk_mcp.ingest.html_extractor import html_to_markdown


def test_comments_are_dropped():
    md = html_to_markdown("<p>MADDE 1</p><!--[if gte mso 9]><xml>zzz</xml><![endif]--><p>MADDE 2</p>")
    assert "zzz" not in md
    assert "mso" not in md
    assert "MADDE 1" in md
    assert "MADDE 2" in md


def test_doctype_is_dropped():
    assert html_to_markdown("<!DOCTYPE html><div>hello world</div>") == "hello world"


def test_cdata_processing_instruction_and_declaration_are_dropped():
    md = html_to_markdown("<div>kalan metin<![CDATA[gizli]]><?xml version='1.0'?><!ELEMENT x></div>")
    assert md == "kalan metin"


def test_comment_inside_paragraph_is_dropped():
    assert html_to_markdown("<p>bir <!-- gizli --> iki</p>") == "bir iki"
