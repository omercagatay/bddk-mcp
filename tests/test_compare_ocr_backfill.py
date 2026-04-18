"""Unit tests for the compare_ocr_backfill metrics."""

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_ocr_backfill.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("compare_ocr_backfill", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compare_ocr_backfill"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestTextMetrics:
    def test_form_drops_counts_empty_form_blocks(self):
        m = _load_module()
        text = "foo <form></form> bar <form>\n  \n</form> baz <form>not empty</form>"
        assert m.count_form_drops(text) == 2

    def test_form_drops_zero_when_absent(self):
        m = _load_module()
        assert m.count_form_drops("no forms here") == 0

    def test_latex_markers_counts_all_patterns(self):
        m = _load_module()
        text = r"$x_{i}$ and $y^{2}$ with \frac{a}{b}, \sum, \Delta, \cdot, \sqrt{2}, \min, \max"
        assert m.count_latex_markers(text) == 9

    def test_md_image_refs_counts_markdown_and_html(self):
        m = _load_module()
        text = "![alt](x.png) and ![](y.png) and <img src='z.png'/> and <img   src='w.png' >"
        assert m.count_md_image_refs(text) == 4

    def test_md_image_refs_ignores_empty_img_tag(self):
        m = _load_module()
        text = "<img> and <img  >"
        assert m.count_md_image_refs(text) == 0


class TestRegressionFlag:
    def test_flag_true_when_form_drops_worsen(self):
        m = _load_module()
        before = {"form_drops": 1, "latex_markers": 5}
        after = {"form_drops": 2, "latex_markers": 5}
        assert m.regression_flag(before, after) is True

    def test_flag_true_when_latex_markers_regress(self):
        m = _load_module()
        before = {"form_drops": 0, "latex_markers": 10}
        after = {"form_drops": 0, "latex_markers": 8}
        assert m.regression_flag(before, after) is True

    def test_flag_false_when_improved(self):
        m = _load_module()
        before = {"form_drops": 3, "latex_markers": 5}
        after = {"form_drops": 0, "latex_markers": 7}
        assert m.regression_flag(before, after) is False

    def test_flag_false_when_equal(self):
        m = _load_module()
        before = {"form_drops": 0, "latex_markers": 5}
        after = {"form_drops": 0, "latex_markers": 5}
        assert m.regression_flag(before, after) is False


class TestSilentDropCandidate:
    def test_candidate_when_pdf_has_more_images_than_md(self):
        m = _load_module()
        row = {"pdf_image_count": 3, "md_image_refs": 1, "latex_markers": 0}
        assert m.is_silent_drop_candidate(row) is True

    def test_not_candidate_when_latex_offsets_images(self):
        m = _load_module()
        row = {"pdf_image_count": 3, "md_image_refs": 0, "latex_markers": 5}
        assert m.is_silent_drop_candidate(row) is False

    def test_not_candidate_when_no_pdf_images(self):
        m = _load_module()
        row = {"pdf_image_count": 0, "md_image_refs": 0, "latex_markers": 0}
        assert m.is_silent_drop_candidate(row) is False
