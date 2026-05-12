"""Markdown sanitization and quality assessment helpers.

The storage sanitizer is conservative: it removes extraction noise that has no
legal meaning while keeping Markdown structure intact. The context sanitizer is
stricter because its output is sent to LLMs and MCP clients.
"""

from __future__ import annotations

import re
import textwrap

from pydantic import BaseModel, Field

_KNOWN_FAIL_DOCUMENT_IDS = {
    "1043",
    "1045",
    "1305",
    "1313",
    "1314",
    "1334",
    "903",
    "905",
    "907",
    "mevzuat_16290",
    "mevzuat_21192",
}

_EMBEDDED_ARTIFACT_MARKER = "[removed embedded image/formula artifact]"
_DATA_URI_RE = re.compile(r"data:image/[a-z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+", re.IGNORECASE)
_CID_RE = re.compile(r"\bcid:[^\s\])>\"']+", re.IGNORECASE)
_IMG_TAG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_DANGEROUS_TAG_RE = re.compile(r"</?(?:div|table|tr|td|img|span|font)\b[^>]*>", re.IGNORECASE)
_RAW_HTML_TAG_RE = re.compile(r"<(?:div|table|tr|td|img|span|font)\b[^>]*>", re.IGNORECASE)
_HTML_ATTR_RE = re.compile(r"\b(?:style|class|width|height|src|align|valign)=[\"'][^\"']*[\"']", re.IGNORECASE)
_HTML_ENTITY_RE = re.compile(r"&(?:nbsp|ccedil|Ccedil|ouml|Ouml|uuml|Uuml|#[0-9]+|#x[0-9a-f]+);", re.IGNORECASE)
_EMPTY_BOLD_LINE_RE = re.compile(r"(?m)^[ \t]*\*\*[ \t]*\*\*[ \t]*$")
_ISOLATED_BOLD_LINE_RE = re.compile(r"(?m)^[ \t]*\*\*[ \t]*$")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_UNDERSCORE_LEADER_RE = re.compile(r"(?m)^[ \t]*_{10,}[ \t]*$")
_DASH_LEADER_RE = re.compile(r"(?m)^[ \t]*-{10,}[ \t]*$")
_DOT_LEADER_RE = re.compile(r"(?m)(?<!\.)\.{10,}(?!\.)")
_INVISIBLE_SPACE_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_BDDK_CIRCULAR_AUTHOR_RE = re.compile(r"\b([A-ZÇĞİÖŞÜ]{2,})BaşkanSayı\s*:\s*")
_BDDK_CIRCULAR_SUBJECT_RE = re.compile(r"(\d{4})Konu\s*:\s*")
_BDDK_CIRCULAR_NUMBER_RE = re.compile(r"\b(GENELGE)(\d{4}/\d+)\b")
_CAMELCASE_TRANSITION_RE = re.compile(r"[a-zçğıöşü][A-ZÇĞİÖŞÜ]")
_KNOWN_MIXED_CASE_TERMS = {
    "HashCalc",
}
_MIXED_CASE_UNIT_RE = re.compile(r"^\d+(?:kW|MW|GW|kWh|MWh|GWh)$")
_FORMULA_REF_RE = re.compile(
    r"aşağıdaki form[üu]l(?:[üu]|ler)?"
    r"|yer alan form[üu]l(?:[üu]|ler)?"
    r"|form[üu]l(?:[üu]|ler)?\s+(?:ile|uyarınca|vasıtasıyla|aracılığıyla|kullanılarak)",
    re.IGNORECASE,
)
_LATEX_OR_IMAGE_RE = re.compile(r"\$\$|!\[[^\]]*]\([^)]*\)|<img\b|data:image/", re.IGNORECASE)
_INLINE_FORMULA_RE = re.compile(
    r"(?im)"
    r"(?:(?:^|[.;:\n]\s*)"
    r"(?:\([A-ZÇĞİÖŞÜa-zçğıöşü0-9]+\)\s*)?"
    r"[A-ZÇĞİÖŞÜa-zçğıöşüΑ-Ωα-ω0-9]"
    r"[A-ZÇĞİÖŞÜa-zçğıöşüΑ-Ωα-ω0-9\s*/().,*+\-%×÷]*"
    r"\s*=\s*"
    r"(?=[^\n]{3,240})"
    r"(?=[^\n]*(?:\d|[+*/×÷^∑√]|maksimum|minimum|min|max|α|β|γ|ρ))"
    r"[^\n]{3,240}"
    r"|\b[A-ZÇĞİÖŞÜΑ-Ω]{2,}[A-ZÇĞİÖŞÜΑ-Ω0-9]*\s*/\s*\([^)\n]{3,120}[+\-*/×÷][^)\n]{1,120}\))",
)


class QualityAssessment(BaseModel):
    """Document-level Markdown quality label and signal counts."""

    document_id: str = ""
    label: str = "clean"
    flags: list[str] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    warning: str = ""


def sanitize_markdown_for_storage(text: str) -> str:
    """Normalize storage-safe extraction artifacts while preserving legal text."""
    if not text:
        return text

    out = text.replace("\u00a0", " ")
    out = _INVISIBLE_SPACE_RE.sub("", out)
    out = out.replace("\f", "\n")
    out = _EMPTY_BOLD_LINE_RE.sub("", out)
    out = _ISOLATED_BOLD_LINE_RE.sub("", out)
    out = _UNDERSCORE_LEADER_RE.sub("", out)
    out = _DASH_LEADER_RE.sub("", out)
    out = _DOT_LEADER_RE.sub(" ... ", out)
    out = _CID_RE.sub("", out)
    out = _repair_pdf_spacing_loss(out)
    out = _BLANK_LINES_RE.sub("\n\n", out)
    return out.strip() + ("\n" if out.endswith("\n") else "")


def sanitize_markdown_for_context(text: str, max_line_length: int = 1000) -> str:
    """Sanitize Markdown before it is sent to an LLM or MCP client."""
    if not text:
        return text

    out = sanitize_markdown_for_storage(text)
    out = _IMG_TAG_RE.sub(_EMBEDDED_ARTIFACT_MARKER, out)
    out = _DATA_URI_RE.sub(_EMBEDDED_ARTIFACT_MARKER, out)
    out = _CID_RE.sub(_EMBEDDED_ARTIFACT_MARKER, out)
    out = _DANGEROUS_TAG_RE.sub("", out)
    out = _CONTROL_CHAR_RE.sub("", out)
    out = _BLANK_LINES_RE.sub("\n\n", out)
    return _wrap_long_lines(out, max_line_length=max_line_length)


def assess_markdown_quality(text: str, document_id: str = "") -> QualityAssessment:
    """Return deterministic quality flags and a clean/warning/fail label."""
    counts = _count_signals(text)
    flags = [name for name, count in counts.items() if count > 0]

    fail = (
        counts["raw_html_tag"] > 0
        or counts["data_uri_image"] > 0
        or counts["cid_marker"] >= 20
        or counts["replacement_char"] > 0
        or (counts["very_long_lines_gt_3000"] > 0 and _has_raw_markup_or_blob(text))
        or document_id in _KNOWN_FAIL_DOCUMENT_IDS
    )

    warning = (
        counts["control_char"] > 0
        or counts["formula_ref_without_latex_or_image"] > 0
        or counts["long_underscore_run"] > 0
        or counts["camelcase_concat"] > 0
        or counts["repeated_para_blocks_gt2"] > 0
    )

    label = "fail" if fail else "warning" if warning else "clean"
    return QualityAssessment(
        document_id=document_id,
        label=label,
        flags=flags,
        counts=counts,
        warning=_quality_warning(label, flags),
    )


def _count_signals(text: str) -> dict[str, int]:
    cid_count = len(_CID_RE.findall(text))
    data_uris = _DATA_URI_RE.findall(text)
    long_lines = [len(line) for line in text.splitlines()]
    words = re.findall(r"\S+", text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    duplicate_paragraphs = len(paragraphs) - len(set(paragraphs))

    return {
        "control_char": len(_CONTROL_CHAR_RE.findall(text)),
        "data_uri_image": len(data_uris),
        "wmf_data_uri": sum(1 for uri in data_uris if uri.lower().startswith("data:image/x-wmf")),
        "cid_marker": cid_count,
        "image_placeholder": text.lower().count("[image") + text.lower().count("[resim"),
        "html_attr": len(_HTML_ATTR_RE.findall(text)),
        "html_entity": len(_HTML_ENTITY_RE.findall(text)),
        "empty_bold": len(_EMPTY_BOLD_LINE_RE.findall(text)),
        "isolated_bold_line": len(_ISOLATED_BOLD_LINE_RE.findall(text)),
        "raw_html_tag": len(_RAW_HTML_TAG_RE.findall(text)),
        "malformed_table_rows": _count_malformed_table_rows(text),
        "excessive_pipe_density": _count_excessive_pipe_density(text),
        "long_lines_gt_1000": sum(1 for n in long_lines if n > 1000),
        "very_long_lines_gt_3000": sum(1 for n in long_lines if n > 3000),
        "long_words_ge35": sum(1 for word in words if len(word) >= 35),
        "extreme_words_ge55": sum(1 for word in words if len(word) >= 55),
        "legal_heading_concat": len(re.findall(r"(?:MADDE|Madde|İlke|ILKE)\s*\d+[A-ZÇĞİÖŞÜ]", text)),
        "repeated_para_blocks_gt2": _count_repeated_para_blocks(text),
        "duplicate_paragraphs": max(0, duplicate_paragraphs),
        "replacement_char": text.count("\ufffd"),
        "long_underscore_run": len(re.findall(r"_{10,}", text)),
        "camelcase_concat": _count_camelcase_concat(text),
        "formula_ref_without_latex_or_image": int(_has_formula_ref_without_extractable_formula(text)),
    }


def _wrap_long_lines(text: str, max_line_length: int) -> str:
    if max_line_length <= 0:
        return text
    wrapped: list[str] = []
    for line in text.splitlines():
        if len(line) <= max_line_length:
            wrapped.append(line)
            continue
        wrapped.extend(
            textwrap.wrap(
                line,
                width=max_line_length,
                break_long_words=True,
                break_on_hyphens=False,
                replace_whitespace=False,
                drop_whitespace=False,
            )
        )
    return "\n".join(wrapped)


def _has_raw_markup_or_blob(text: str) -> bool:
    lower = text.lower()
    return "data:image/" in lower or "base64" in lower or bool(_RAW_HTML_TAG_RE.search(text))


def _count_malformed_table_rows(text: str) -> int:
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if "|" in stripped and not (stripped.startswith("|") and stripped.endswith("|")):
            count += 1
    return count


def _count_excessive_pipe_density(text: str) -> int:
    return sum(1 for line in text.splitlines() if len(line) > 80 and line.count("|") >= 12)


def _repair_pdf_spacing_loss(text: str) -> str:
    out = _BDDK_CIRCULAR_AUTHOR_RE.sub(r"\1\nBaşkan\nSayı: ", text)
    out = _BDDK_CIRCULAR_SUBJECT_RE.sub(r"\1\nKonu: ", out)
    out = _BDDK_CIRCULAR_NUMBER_RE.sub(r"\1 \2", out)
    replacements = {
        "HakkındaYönetmeliğ": "Hakkında Yönetmeliğ",
        "ilişkinYönetmelik": "ilişkin Yönetmelik",
        "ConsistencyAssessment": "Consistency Assessment",
        "StandartYaklaşım": "Standart Yaklaşım",
    }
    for needle, replacement in replacements.items():
        out = out.replace(needle, replacement)
    return out


def _count_camelcase_concat(text: str) -> int:
    return sum(1 for match in _CAMELCASE_TRANSITION_RE.finditer(text) if not _is_camelcase_false_positive(text, match))


def _is_camelcase_false_positive(text: str, match: re.Match[str]) -> bool:
    start, end = _token_bounds(text, match.start(), match.end())
    token = text[start:end]
    if token in _KNOWN_MIXED_CASE_TERMS:
        return True
    if _MIXED_CASE_UNIT_RE.fullmatch(token):
        return True
    if _is_inside_url_context(text, start):
        return True
    if len(token) <= 3:
        return True

    upper_index = match.start() + 1
    upper_run = 0
    while upper_index + upper_run < len(text) and text[upper_index + upper_run].isupper():
        upper_run += 1
    return upper_run >= 2


def _token_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    while start > 0 and _is_token_char(text[start - 1]):
        start -= 1
    while end < len(text) and _is_token_char(text[end]):
        end += 1
    return start, end


def _is_token_char(char: str) -> bool:
    return char.isalnum()


def _is_inside_url_context(text: str, start: int) -> bool:
    prefix = text[max(0, start - 80) : start].lower()
    return "http://" in prefix or "https://" in prefix or "www." in prefix


def _count_repeated_para_blocks(text: str) -> int:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) >= 160]
    counts: dict[str, int] = {}
    repeated = 0
    for para in paragraphs:
        counts[para] = counts.get(para, 0) + 1
        if counts[para] == 3:
            repeated += 1
    return repeated


def _has_formula_ref_without_extractable_formula(text: str) -> bool:
    if not _FORMULA_REF_RE.search(text):
        return False
    if _LATEX_OR_IMAGE_RE.search(text):
        return False
    return not any(
        _INLINE_FORMULA_RE.search(text[max(0, match.start() - 500) : match.end() + 1500])
        for match in _FORMULA_REF_RE.finditer(text)
    )


def _quality_warning(label: str, flags: list[str]) -> str:
    if label == "fail":
        return (
            "This document contains severe extraction artifacts; do not rely on it for audit-grade "
            "or calculation-level answers without source review."
        )
    if label == "warning":
        if "formula_ref_without_latex_or_image" in flags:
            return "This document may reference formulas that were not extracted; verify formula-level answers against source."
        return "This document contains extraction-quality warnings; verify critical details against source."
    return ""
