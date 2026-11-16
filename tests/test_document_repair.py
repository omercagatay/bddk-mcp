"""Tests for scripts/document_repair: guards, text-stage transforms, merge rules, policy, GPU path, font fallback.

No GPU, no network, no poppler and no Cambria Math font: the guard cases are cut from the real 1336 / 1341 runs of
2026-11-15 (tests/fixtures/document_repair/guard_cases.json), PDFs are generated here, and the pdftotext call is
replaced by pdfplumber's own text of the same page.
"""

from __future__ import annotations

import io
import json
import subprocess
import tarfile
from collections import Counter
from pathlib import Path

import pytest

from scripts.document_repair import assemble as assemble_mod
from scripts.document_repair import glyphmap, gpu, guards, pipeline, textlayer
from scripts.document_repair.__main__ import main as cli_main

FIXTURE = Path(__file__).parent / "fixtures" / "document_repair" / "guard_cases.json"
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))
COUNTERS = ("glyphs", "sup", "sub", "prose_words", "all_words")


def _ref(case):
    ref = dict(case["ref"])
    for k in COUNTERS:
        ref[k] = Counter(ref[k])
    return ref


def _case(tag):
    return next(c for c in CASES if tag in c["chosen_for"])


# ---------------------------------------------------------------- guards on labelled regions


# No recognised formula failed G6 on 2026-11-15: its failing case is the mutation test below.
REAL_CASES = [(g, w) for g in ("G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7") for w in ("pass", "fail")]


@pytest.mark.parametrize(("guard", "want"), [c for c in REAL_CASES if c != ("G6", "fail")])
def test_guard_on_real_region(guard, want):
    case = _case(f"{guard}_{want}")
    res = guards.check(_ref(case), case["meta"], case["md"])
    assert bool(res[guard][0]) is (want == "pass"), res[guard]
    assert [bool(res[guard][0]), res[guard][1]] == case["expected"][guard]


def test_every_fixture_case_reproduces_all_recorded_guards():
    for case in CASES:
        res = guards.check(_ref(case), case["meta"], case["md"])
        assert {g: [bool(res[g][0]), res[g][1]] for g in case["expected"]} == case["expected"], case["id"]


def test_g3_catches_the_dotted_capital_i_read_as_plain_i():
    case = next(c for c in CASES if c["id"] == "1341_p011_0")
    assert case["label"][0] == "W"
    res = guards.check(_ref(case), case["meta"], case["md"])
    assert res["G3"][0] is False and "İ" in res["G3"][1]


def test_g6_rejects_a_rule_the_page_does_not_have():
    """Mutation of a passing formula (labelled): one extra \\overline adds a rule but no glyph."""
    case = _case("G6_pass")
    assert case["id"] == "1336_p008_2" and "D_b" in case["md"]
    mutated = case["md"].replace("D_b", "\\overline{D}_b", 1)
    res = guards.check(_ref(case), case["meta"], mutated)
    assert res["G6"] == (False, "pdf rules 0 vs latex frac/sqrt/overline 1")
    assert res["G3"][0] is True


def test_g0_rejects_truncation_and_missing_meta():
    case = _case("G0_pass")
    ref = _ref(case)
    assert guards.g0(ref, {"eos": False, "truncated": True}, case["md"]) == (False, "truncated")
    assert guards.g0(ref, pipeline.NO_META, case["md"])[0] is False
    assert guards.g0(ref, case["meta"], "  ") == (False, "empty output")


# ---------------------------------------------------------------- policy


def _row(kind, failing=(), **extra):
    row = {"id": "p001_0", "kind": kind, "page": 1}
    for g in pipeline.GUARDS:
        row[g] = [g not in failing, ""]
    for k in pipeline.AGREE:
        row["G8_" + k] = [("G8_" + k) not in failing, ""]
    row.update(extra)
    return row


def test_recommended_policy_accepts_formulas_on_g0_to_g3_g5_g6_only():
    assert pipeline.accept(_row("formula"), "recommended") == (True, [])
    assert pipeline.accept(_row("formula", ["G4", "G7", "G8_1b"]), "recommended") == (True, [])
    for g in ("G0", "G1", "G2", "G3", "G5", "G6"):
        assert pipeline.accept(_row("formula", [g]), "recommended") == (False, [g])


def test_recommended_policy_never_accepts_a_table():
    assert pipeline.accept(_row("table"), "recommended") == (False, ["tables: review"])


def test_missing_output_is_rejected_under_every_policy():
    for policy in pipeline.POLICIES:
        assert pipeline.accept({"missing_output": True, "kind": "formula"}, policy) == (
            False,
            ["no recognition output"],
        )


def test_all_policy_needs_every_agreement():
    assert pipeline.accept(_row("table"), "all") == (True, [])
    assert pipeline.accept(_row("formula", ["G4"]), "all") == (False, ["G4"])
    row = _row("formula")
    row["G8_ctx"] = [None, "no output"]
    assert pipeline.accept(row, "all") == (False, ["G8_ctx"])


def test_unknown_policy_is_refused(tmp_path):
    with pytest.raises(ValueError):
        pipeline.run(tmp_path / "x.pdf", tmp_path / "out", policy="lenient")


# ---------------------------------------------------------------- text-stage transforms


def test_normalise_follows_the_seed_convention():
    raw = "   Karar Sayısı : 11234     Karar Tarihi : 06.11.2025  \n\n\n\n  Madde 1 – Metin   \nİDD\n\n"
    assert textlayer.normalise(raw) == "Karar Sayısı : 11234\n\nKarar Tarihi : 06.11.2025\n\nMadde 1 – Metin\nİDD\n"


def test_decode_poppler_restores_split_surrogates_and_folds_math_letters():
    raw = "\U0001d43e".encode("utf-16-be")
    hi, lo = raw[:2], raw[2:]
    cesu = (chr(int.from_bytes(hi, "big")) + chr(int.from_bytes(lo, "big"))).encode("utf-8", "surrogatepass")
    assert textlayer.decode_poppler(b"x = " + cesu + b"\r\n") == "x = K\n"


def test_join_pages_keeps_a_sentence_across_a_page_break():
    assert textlayer.join_pages(["Birinci sayfa bu", "cümle devam eder."]) == "Birinci sayfa bu\ncümle devam eder."
    assert textlayer.join_pages(["Bitti.", "Yeni paragraf."]) == "Bitti.\n\nYeni paragraf."


def _char(t, x0, base, size, pos=None):
    return {"t": t, "x0": x0, "x1": x0 + 0.5 * size, "base": base, "size": size, "pos": pos}


def test_wrap_scripts_marks_raised_and_lowered_runs_but_not_letters_inside_a_word():
    line = "Tutar 20231 K1 aBc"
    chars = {
        6: _char("2", 0, 0, 10),
        10: _char("1", 0, 0, 6, "sup"),
        13: _char("1", 0, 0, 6, "sub"),
        16: _char("B", 0, 0, 6, "sup"),
    }
    amap = [chars.get(i) for i in range(len(line))]
    out, runs = assemble_mod.wrap_scripts(line, amap, 0)
    assert out == "Tutar 2023$^{1}$ K$_{1}$ aBc"
    assert runs == 2


def test_classify_scripts_marks_a_footnote_marker_before_its_text_and_a_glued_one():
    marker = _char("1", 50, 390, 7)
    text = [_char(ch, 55 + 5 * i, 392, 10) for i, ch in enumerate("Dipnot")]
    figure = [_char(ch, 50 + 6 * i, 100, 12) for i, ch in enumerate("2023")]
    glued = _char("2", 75, 95, 7)
    chars = [marker, *text, *figure, glued]
    guards.classify_scripts(chars, 12)
    assert marker["pos"] == "sup" and glued["pos"] == "sup"
    assert all(c["pos"] is None for c in text + figure)


def _band(chars, text):
    return {"chars": chars, "text": text, "x0": min(c["x0"] for c in chars)}


def test_footnotes_start_at_a_raised_number_and_continue_on_the_next_page():
    def body(text, x0=50):
        return _band([_char(ch, x0 + 6 * i, 0, 12) for i, ch in enumerate(text)], text)

    fn = _band(
        [_char("1", 50, 0, 7, "sup"), *[_char(ch, 56 + 5 * i, 0, 10) for i, ch in enumerate("Dipnot")]], "1Dipnot"
    )
    page1 = [body("Gövde metni."), fn, body("sürüyor")]
    page2 = [body("devamı burada."), body("Yeni gövde.")]
    assert assemble_mod.footnotes([({}, page1), ({}, page2)]) == [(1, 0), (None, 1)]


# ---------------------------------------------------------------- synthetic PDFs (no poppler)


def _pdf(pages, path):
    """A minimal PDF: pages = [[(font, size, x, y, text), ...], ...] with F1 Helvetica and F2 Symbol."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None]
    kids = []
    fonts = "<< /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> /F2 << /Type /Font /Subtype /Type1 /BaseFont /Symbol >> >>"
    for runs in pages:
        stream = "".join(
            f"BT /{f} {s} Tf {x} {y} Td ({t.replace('(', chr(92) + '(').replace(')', chr(92) + ')')}) Tj ET\n"
            for f, s, x, y, t in runs
        )
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}endstream")
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 400] /Resources << /Font {fonts} >> "
            f"/Contents {len(objs)} 0 R >>"
        )
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = io.BytesIO(), []
    out.write(b"%PDF-1.4\n")
    for i, o in enumerate(objs, 1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n{o}\nendobj\n".encode("latin-1"))
    xref = out.tell()
    out.write(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(out.getvalue())
    return path


@pytest.fixture
def no_poppler(monkeypatch):
    """pdftotext replaced by pdfplumber's text of each page; any subprocess call (pdftotext, ssh) fails the test."""
    import pdfplumber

    def pages(pdf):
        with pdfplumber.open(pdf) as doc:
            return [p.extract_text() or "" for p in doc.pages]

    def no_subprocess(*args, **kwargs):
        raise AssertionError(f"unexpected subprocess call: {args!r}")

    monkeypatch.setattr(textlayer, "pdftotext_pages", pages)
    monkeypatch.setattr(subprocess, "run", no_subprocess)
    monkeypatch.setattr(subprocess, "Popen", no_subprocess)


@pytest.fixture
def pdfminer_restored(monkeypatch):
    """glyphmap.install patches pdfminer's font classes; undo it after the test."""
    from pdfminer.pdffont import PDFCIDFont, PDFSimpleFont

    for cls in (PDFSimpleFont, PDFCIDFont):
        monkeypatch.setattr(cls, "to_unichr", cls.to_unichr)
    monkeypatch.setattr(glyphmap, "_ORIG", None)
    monkeypatch.setattr(glyphmap, "_MAPS", {})
    yield


BODY = [("F1", 12, 40, 360, "Birinci madde metni burada yer alir."), ("F1", 12, 40, 340, "Formul asagidadir:")]
FORMULA_A = ("F2", 12, 60, 300, "a + b = c")
FORMULA_B = ("F2", 12, 60, 270, "d - e = f")
AFTER = [("F1", 12, 40, 230, "Formulden sonraki metin."), ("F1", 12, 40, 210, "Son satir.")]


def _two_formula_pdf(tmp_path):
    return _pdf([[*BODY, FORMULA_A, FORMULA_B, *AFTER]], tmp_path / "9001.pdf")


def test_no_gpu_path_lists_every_region_for_review_and_sends_nothing(
    tmp_path, no_poppler, pdfminer_restored, monkeypatch, capsys
):
    monkeypatch.delenv("BDDK_REPAIR_GPU_HOST", raising=False)
    monkeypatch.setattr(gpu, "run", lambda *a, **k: pytest.fail("GPU step called without a host"))
    pdf = _two_formula_pdf(tmp_path)
    out = tmp_path / "out"
    assert cli_main([str(pdf), str(out), "--math-font", str(tmp_path / "absent.ttc")]) == 0
    regions = json.loads((out / "regions.json").read_text(encoding="utf-8"))
    counts = json.loads((out / "counts.json").read_text(encoding="utf-8"))
    assert counts["regions"] == len(regions) == 2 and counts["by_kind"] == {"formula": 2}
    assert all(r["accepted"] is False and r["failed_guards"] == ["no recognition output"] for r in regions)
    assert counts["accepted"] == 0 and counts["gpu_seconds"] is None and counts["policy"] == "recommended"
    md = (out / "repaired.md").read_text(encoding="utf-8")
    assert "Birinci madde metni" in md and "Son satir." in md
    assert sorted(p.name for p in (out / "masked").glob("*_a.png")) == ["9001_p001_0_a.png", "9001_p001_1_a.png"]
    text = capsys.readouterr().out
    assert "Cambria Math font not found" in text and "every one is listed for review" in text


def test_no_gpu_run_ignores_recognitions_left_in_a_reused_directory(tmp_path, no_poppler, pdfminer_restored):
    pdf = _two_formula_pdf(tmp_path)
    out = tmp_path / "out"
    (out / "rec").mkdir(parents=True)
    for rid in ("p001_0", "p001_1"):
        (out / "rec" / f"9001_{rid}_a.md").write_text("$$\\alpha + \\beta = \\chi$$", encoding="utf-8")
    (out / "rec" / "meta.jsonl").write_text(
        "".join(json.dumps({"name": f"9001_{r}_a", "eos": True, "truncated": False}) + "\n" for r in ("p001_0",)),
        encoding="utf-8",
    )
    counts = pipeline.run(pdf, out, math_font=tmp_path / "absent.ttc")
    regions = json.loads((out / "regions.json").read_text(encoding="utf-8"))
    assert counts["accepted"] == 0
    assert all(r["failed_guards"] == ["no recognition output"] for r in regions)


def test_footnotes_follow_the_body_of_the_last_page(tmp_path, no_poppler, pdfminer_restored):
    glyphmap.configure(tmp_path / "absent.ttc")
    page1 = [*BODY, ("F1", 7, 40, 64, "1"), ("F1", 10, 45, 60, "Dipnot metni burada.")]
    page2 = [("F1", 12, 40, 360, "Ikinci sayfanin metni."), ("F1", 12, 40, 340, "Govde burada biter.")]
    pdf = _pdf([page1, page2], tmp_path / "9003.pdf")
    _, pages, regs = pipeline.detect(pdf)
    assert regs == []
    md, rep = assemble_mod.assemble(pdf, pages, regs, {})
    assert rep["pages"][0]["footnote_start_band"] == 2
    assert md.index("Formul asagidadir:") < md.index("Ikinci sayfanin metni.") < md.index("Dipnot metni")
    assert "$^{1}$" in md.split("Govde burada biter.")[1]
    assert md.endswith("Dipnot metni burada.\n")


def _run_with_accepted(tmp_path, accepted_ids):
    pdf = _two_formula_pdf(tmp_path)
    _, pages, regs = pipeline.detect(pdf)
    accepted = {r["id"]: f"$$X_{{{r['id'][-1]}}}$$" for r in regs if r["id"] in accepted_ids}
    return regs, *assemble_mod.assemble(pdf, pages, regs, accepted)


def test_merge_replaces_stacked_regions_only_together(tmp_path, no_poppler, pdfminer_restored):
    glyphmap.configure(tmp_path / "absent.ttc")
    regs, md, rep = _run_with_accepted(tmp_path, {"p001_0", "p001_1"})
    assert [r["id"] for r in regs] == ["p001_0", "p001_1"]
    assert all(rep["regions"][r["id"]]["clean"] for r in regs)
    assert "$$X_{0}$$" in md and "$$X_{1}$$" in md
    assert "Formulden sonraki metin." in md and "Formul asagidadir:" in md


def test_merge_keeps_a_region_whose_neighbour_is_not_replaced(tmp_path, no_poppler, pdfminer_restored):
    glyphmap.configure(tmp_path / "absent.ttc")
    regs, md, rep = _run_with_accepted(tmp_path, {"p001_0"})
    assert rep["regions"]["p001_0"]["clean"] is False
    assert rep["regions"]["p001_0"]["blocked_by_unmerged_neighbour"] == "p001_1"
    assert "$$X_" not in md


def test_merge_requires_separable_lines(tmp_path, no_poppler, pdfminer_restored, monkeypatch):
    """pdftotext -layout can print a formula row between words of the lines around it; then the region's glyphs
    sit inside body lines (not as a prefix or suffix), its lines are not separable and nothing is replaced."""
    glyphmap.configure(tmp_path / "absent.ttc")
    pdf = _pdf([[*BODY, FORMULA_A, *AFTER]], tmp_path / "9002.pdf")
    plain = textlayer.pdftotext_pages(pdf)[0].splitlines()
    k = next(i for i, ln in enumerate(plain) if "asagidadir" in ln)
    assert plain[k] == "Formul asagidadir:" and plain[k + 2] == "Formulden sonraki metin."
    mixed = f"asagidadir: {plain[k + 1]} Formulden"
    plain[k : k + 3] = ["Formul", mixed, "sonraki metin."]
    monkeypatch.setattr(textlayer, "pdftotext_pages", lambda _pdf: [chr(10).join(plain)])
    _, pages, regs = pipeline.detect(pdf)
    assert [r["kind"] for r in regs] == ["formula"]
    md, rep = assemble_mod.assemble(pdf, pages, regs, {regs[0]["id"]: "$$Y$$"})
    assert rep["regions"][regs[0]["id"]]["leak_out"] > 0
    assert rep["regions"][regs[0]["id"]]["clean"] is False
    assert "$$Y$$" not in md and mixed in md


# ---------------------------------------------------------------- Cambria Math font absent


def test_absent_font_recovers_nothing_and_leaves_codes_unmapped(tmp_path, pdfminer_restored, monkeypatch):
    assert glyphmap.configure(tmp_path / "absent.ttc") is None
    assert glyphmap._system() == ({}, {})

    class Font:
        basefont = "ABCDEF+CambriaMath"
        unicode_map = None

    from pdfminer.pdffont import PDFSimpleFont

    class Pdf:
        pages = []

    glyphmap.install(Pdf())
    glyphmap._MAPS["ABCDEF+CambriaMath"] = {}
    assert PDFSimpleFont.to_unichr(Font(), 35) == "(cid:35)"
    assert glyphmap.family_key("ABCDEF+CambriaMath") == "cambriamath"
    assert glyphmap.family_key("ABCDEF+Calibri") is None


# ---------------------------------------------------------------- GPU step (ssh), faked


@pytest.mark.parametrize("bad", ["-oProxyCommand=sh", "host;rm", "a b", "", "../x"])
def test_gpu_refuses_unsafe_host_alias(bad, tmp_path):
    with pytest.raises(ValueError):
        gpu.run(bad, "1336", tmp_path, {}, tmp_path, "bddk-repair", "python3")


@pytest.mark.parametrize("field", ["doc", "workdir", "python"])
def test_gpu_refuses_unsafe_remote_words(field, tmp_path):
    args = {"doc": "1336", "workdir": "bddk-repair", "python": "python3"}
    args[field] = "x$(id)" if field != "workdir" else "../etc"
    with pytest.raises(ValueError):
        gpu.run("gpuhost", args["doc"], tmp_path, {}, tmp_path, args["workdir"], args["python"])


def _tar(members):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data, kind in members:
            info = tarfile.TarInfo(name)
            if kind == "sym":
                info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
                tar.addfile(info)
            else:
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_gpu_unpacks_only_the_expected_output_files(tmp_path):
    data = _tar(
        [
            ("./1336_p007_0_a.md", b"$$x$$", "file"),
            ("./meta.jsonl", b"{}\n", "file"),
            ("../../evil.md", b"x", "file"),
            ("./1336_p007_0_a.png", b"x", "file"),
            ("./9999_p007_0_a.md", b"x", "file"),
            ("./1336_p008_0_b.md", b"", "sym"),
        ]
    )
    assert gpu._unpack_outputs(data, "1336", tmp_path) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["1336_p007_0_a.md", "meta.jsonl"]


def test_gpu_sends_only_this_runs_images_and_quotes_every_remote_word(tmp_path, monkeypatch):
    masked = tmp_path / "masked"
    masked.mkdir()
    for name in ("1336_p007_0_a", "1336_p007_0_b", "1336_p007_0_c", "1336_p007_0_d", "1999_p001_0_a"):
        (masked / f"{name}.png").write_bytes(b"png")
    lists = {"a": ["1336_p007_0_a"], "b": ["1336_p007_0_b"], "c": ["1336_p007_0_c"], "d": ["1336_p007_0_d"]}
    calls = []

    class Proc:
        def __init__(self, rc=0, out=b""):
            self.returncode, self.stdout = rc, out

    def fake(cmd, **kw):
        calls.append((cmd, kw))
        assert cmd[:6] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "--"] and cmd[6] == "gpuhost"
        if cmd[7].startswith("tar cf"):
            return Proc(out=_tar([("./1336_p007_0_a.md", b"$$x$$", "file")]))
        return Proc()

    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr(gpu.time, "sleep", lambda s: None)
    gpu.run("gpuhost", "1336", masked, lists, tmp_path / "rec", "bddk-repair", ".venv/bin/python")
    uploads = [kw["input"] for cmd, kw in calls if "input" in kw]
    assert len(uploads) == 1
    with tarfile.open(fileobj=io.BytesIO(uploads[0])) as tar:
        names = sorted(tar.getnames())
        l4b = tar.extractfile("l4b.txt").read().decode()
    assert names == sorted([f"{n}.png" for v in lists.values() for n in v] + ["l1b.txt", "l4b.txt"])
    assert l4b == "1336_p007_0_a\n1336_p007_0_c\n1336_p007_0_d\n"
    remote = [cmd[7] for cmd, _ in calls]
    assert remote[0] == "rm -rf bddk-repair/docs/1336 && mkdir -p bddk-repair/docs/1336/img bddk-repair/docs/1336/out"
    assert ".venv/bin/python -I bddk-repair/recognize.py 4b" in remote[2]
    assert (tmp_path / "rec" / "1336_p007_0_a.md").read_text() == "$$x$$"


def test_gpu_ssh_failure_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: type("P", (), {"returncode": 255, "stdout": b""})())
    with pytest.raises(RuntimeError, match="exit 255"):
        gpu.run("gpuhost", "1336", tmp_path, {}, tmp_path, "bddk-repair", "python3")
