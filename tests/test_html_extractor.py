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


def test_padded_table_grid_is_bounded_and_falls_back_to_flattened_text(caplog):
    import time

    wide = "<table><tr>" + '<td colspan="100">x</td>' * 400 + "</tr>" + "<tr></tr>" * 800 + "</table>"
    plain = "<table><tr>" + "<td>x</td>" * 2000 + "</tr>" + "<tr></tr>" * 4000 + "</table>"
    for html in (wide, plain):
        started = time.monotonic()
        with caplog.at_level("WARNING", logger="bddk_mcp.ingest.html_extractor"):
            out = html_to_markdown(html)
        assert time.monotonic() - started < 10
        assert len(out) < 1_000_000
        assert "x" in out
        assert "|" not in out
    assert "exceeds the 200000-cell grid bound" in caplog.text


def test_rowspans_count_toward_the_table_grid_bound(monkeypatch, caplog):
    from bddk_mcp.ingest import html_extractor

    # One row of three 100x100 cells: the row itself is 300 cells wide, so only the
    # 29,700 pending rowspan entries can cross the lowered bound.
    monkeypatch.setattr(html_extractor, "_MAX_TABLE_CELLS", 10_000)
    spanning = "<table><tr>" + '<td rowspan="100" colspan="100">y</td>' * 3 + "</tr></table>"
    with caplog.at_level("WARNING", logger="bddk_mcp.ingest.html_extractor"):
        out = html_to_markdown(spanning)
    assert "y" in out
    assert "|" not in out
    assert "exceeds the 10000-cell grid bound" in caplog.text

    # The same row without rowspans stays a grid under the same bound.
    flat_row = "<table><tr>" + '<td colspan="100">y</td>' * 3 + "</tr></table>"
    assert "|" in html_to_markdown(flat_row)


def test_many_tables_below_the_table_bound_share_a_document_bound(monkeypatch, caplog):
    from bddk_mcp.ingest import html_extractor

    monkeypatch.setattr(html_extractor, "_MAX_TABLE_CELLS", 1_000)
    monkeypatch.setattr(html_extractor, "_MAX_DOCUMENT_CELLS", 5_000)
    # Each table is 30x30 = 900 padded cells, just below the per-table bound.
    table = "<table><tr>" + '<td colspan="30">z</td>' + "</tr>" + "<tr></tr>" * 29 + "</table>"
    with caplog.at_level("WARNING", logger="bddk_mcp.ingest.html_extractor"):
        out = html_to_markdown("<body>" + table * 20 + "</body>")
    grids = [block for block in out.split("\n\n") if block.startswith("|")]
    assert len(grids) == 5  # 5 x 900 = 4,500 cells; a sixth would pass 5,000
    assert out.count("z") == 20
    assert "grid bound of the document" in caplog.text


def test_document_grid_bound_limits_output_with_real_constants():
    import time

    # 200 tables of 400x400 padded cells, each below the per-table bound.
    table = "<table><tr>" + '<td colspan="100">w</td>' * 4 + "</tr>" + "<tr></tr>" * 399 + "</table>"
    started = time.monotonic()
    out = html_to_markdown("<body>" + table * 200 + "</body>")
    assert time.monotonic() - started < 10
    assert out.count("w") >= 200
    assert len(out) < 10_000_000


def test_ordinary_colspan_table_still_renders_as_a_grid():
    out = html_to_markdown(
        '<table><tr><th colspan="2">Başlık</th><th>C</th></tr><tr><td>a</td><td>b</td><td>c</td></tr>'
        "<tr><td>d</td><td>e</td><td>f</td></tr></table>"
    )
    assert "| Başlık |  | C |" in out
    assert "| a | b | c |" in out
    assert "| d | e | f |" in out
