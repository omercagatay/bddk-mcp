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


def test_nested_table_renders_each_inner_cell_once_without_ghost_row():
    md = html_to_markdown(
        "<table><tr><td>Outer<table><tr><td>INNER1</td><td>INNER2</td></tr></table></td><td>X</td></tr></table>"
    )
    assert md.count("INNER1") == 1
    assert md.count("INNER2") == 1
    assert md.splitlines() == ["| Outer INNER1 ; INNER2 | X |", "|---|---|"]


def test_nested_table_inside_tbody_and_div_is_flattened_once():
    md = html_to_markdown(
        "<table><tbody><tr><th>A</th><th>B</th></tr>"
        "<tr><td><div>Kalem<table><tbody><tr><td>i1</td></tr><tr><td>i2</td></tr></tbody></table></div></td>"
        "<td>2</td></tr></tbody></table>"
    )
    assert md.count("i1") == 1
    assert md.count("i2") == 1
    assert len(md.splitlines()) == 3


def test_omitted_closing_tr_keeps_every_row_once():
    # html.parser nests the next <tr> inside the previous one when </tr> is omitted.
    md = html_to_markdown(
        "<html><body><table><tr><td>a1</td><td>a2</td><tr><td>b1</td><td>b2</td><tr><td>c1</td><td>c2</td>"
        "</table></body></html>"
    )
    for cell in ("a1", "a2", "b1", "b2", "c1", "c2"):
        assert md.count(cell) == 1
    assert md.splitlines() == ["| a1 | a2 |", "|---|---|", "| b1 | b2 |", "| c1 | c2 |"]


def test_omitted_closing_td_keeps_every_cell_once():
    md = html_to_markdown("<table><tr><td>a1<td>a2</tr><tr><td>b1<td>b2</tr></table>")
    for cell in ("a1", "a2", "b1", "b2"):
        assert md.count(cell) == 1
    assert md.splitlines() == ["| a1 | a2 |", "|---|---|", "| b1 | b2 |"]


def test_omitted_closing_td_and_tr_keep_every_cell_once():
    md = html_to_markdown("<table><tr><th>H1<th>H2<tr><td>x1<td>x2<tr><td>y1<td>y2</table>")
    for cell in ("H1", "H2", "x1", "x2", "y1", "y2"):
        assert md.count(cell) == 1
    assert md.splitlines() == ["| H1 | H2 |", "|---|---|", "| x1 | x2 |", "| y1 | y2 |"]


def test_colspan_zero_keeps_cell_text():
    md = html_to_markdown(
        "<table><tr><td>A</td><td>B</td></tr><tr><td colspan='0'>LOST TEXT</td><td>c</td></tr></table>"
    )
    assert "| LOST TEXT | c |" in md


def test_non_numeric_and_negative_spans_count_as_one():
    md = html_to_markdown(
        "<table><tr><td>A</td><td>B</td></tr>"
        "<tr><td colspan='abc'>x</td><td rowspan='-3'>y</td></tr>"
        "<tr><td colspan='2;'>z</td><td>w</td></tr></table>"
    )
    assert md.splitlines()[2:] == ["| x | y |", "| z | w |"]


def test_huge_colspan_is_clamped_to_one_hundred_columns():
    md = html_to_markdown("<table>" + "<tr><td colspan='300000'>a</td></tr>" * 20 + "</table>")
    header = md.splitlines()[0]
    assert header.count("|") == 101
    assert len(md) < 10_000


def test_pre_keeps_its_line_breaks():
    md = html_to_markdown("<body><p>MADDE 1</p><pre>### EK-3\n\nEK:3\nA = B + C</pre></body>")
    assert md.splitlines() == ["MADDE 1", "", "### EK-3", "", "EK:3", "A = B + C"]
