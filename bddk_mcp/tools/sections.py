"""Section-level document retrieval and search tools."""

from __future__ import annotations

import bisect
import hashlib
import logging
import math
import re
import unicodedata
from typing import TYPE_CHECKING

from bddk_mcp.core.config import PAGE_SIZE
from bddk_mcp.quality.markdown_quality import (
    FORMULA_EXTRACTION_WARNING,
    QualityAssessment,
    assess_markdown_quality,
    is_formula_aware,
    mentions_formula,
    unsafe_verbatim_reason,
)
from bddk_mcp.quotations import ExactQuotationCheck, check_exact_quotation
from bddk_mcp.store.doc_store import fts_prefix_stem_query, normalize_search_text
from bddk_mcp.store.legal_ref import document_id_candidates, parse_legal_refs, turkish_casefold
from bddk_mcp.store.section_index import (
    GOVDE_HEADING,
    split_section_truncation_notice,
    truncation_notice_original_length,
)
from bddk_mcp.tools.contract_types import (
    DocumentId,
    HeadingFilter,
    OptionalDocumentId,
    OptionalQuotation,
    OptionalSectionStartChar,
    SectionQuery,
    SectionRef,
    SectionResultLimit,
    SectionType,
)
from bddk_mcp.tools.errors import NOT_FOUND, tool_error
from bddk_mcp.tools.structured_outputs import (
    LEGAL_APPLICABILITY_WARNING,
    STRUCTURELESS_GOVDE_LABEL,
    UNTRUSTED_SOURCE_WARNING,
    CrossWindowMatch,
    DocumentSectionResponse,
    DocumentSectionToolResult,
    EvidenceReference,
    QualityMetadata,
    SectionAnswerAssessment,
    SectionItem,
    SectionSearchResponse,
    SectionSearchToolResult,
    WindowReference,
    decision_identifier_text,
    document_status_notice,
    source_date_text,
    structured_tool_result,
    utc_date,
)
from bddk_mcp.tools.tool_logging import logged_tool

if TYPE_CHECKING:
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.store.doc_store import StoredDocumentSection

logger = logging.getLogger(__name__)


_MAX_EXACT_SECTION_CHARS = 30_000
_MAX_SEARCH_EXCERPT_CHARS = 2_000
_MAX_DISAMBIGUATION_RESULTS = 10
_SECTION_TRUNCATION_WARNING = (
    "One or more section bodies were returned as bounded excerpts. Use an exact document/section reference "
    "or paginated full-document retrieval before relying on omitted text."
)
_EXACT_REFERENCE_MISSING_WARNING = (
    "The requested exact provision was not found. Returned lexical matches are other sections, "
    "not evidence for that requested provision. Check the reference or retrieve the full document."
)
_LEGAL_APPLICABILITY_WARNING = LEGAL_APPLICABILITY_WARNING
_NESTED_ROW_HINT_START = (
    "fikra and bent rows are search-oriented and are not unique addresses (many rows can share one reference); "
)
_NESTED_ROW_HINT = (
    _NESTED_ROW_HINT_START + 'where a stored parent article contains a row it is shown as "Parent:"; retrieve the '
    "parent madde (section_type=madde, or gecici_madde/ek_madde) for the full provision."
)
_PARENT_ARTICLE_TYPES = ("madde", "gecici_madde", "ek_madde")


def _nested_row_hint(parent_types: set[str]) -> str:
    """The fikra/bent hint, naming the kind of parent the listed rows actually have."""
    if parent_types and parent_types <= set(_PARENT_ARTICLE_TYPES):
        return _NESTED_ROW_HINT
    if parent_types:
        kinds = " or ".join(sorted(parent_types))
        types = ", ".join(f"section_type={kind}" for kind in sorted(parent_types))
        return (
            _NESTED_ROW_HINT_START + 'where a stored parent provision contains a row it is shown as "Parent:"; '
            f"retrieve that parent {kinds} ({types}) for the full provision."
        )
    return (
        _NESTED_ROW_HINT_START + 'where a stored parent provision contains a row it is shown as "Parent:"; '
        "retrieve the parent provision (madde, gecici_madde/ek_madde, paragraf or ilke) that contains the row "
        "for the full provision."
    )


_CAPPED_QUOTATION_NOTE = (
    "This reference is a capped stored window of the provision: the quotation was compared only with that "
    "window, and text beyond it was NOT compared. The rest of the provision is in full in get_bddk_document; "
    "parts of it are also stored as later rows of the same document (nested fikra/bent/paragraf rows and/or "
    "govde windows), which do not necessarily cover every character."
)
_Parents = dict[tuple[str, str, int], tuple[str, str]]
# Value: (type, ref, true end code point, row runs past that end) of the capped named section a govde row continues.
# The fifth element is (start code point, stored window length) of that capped section, when it was found.
_Continuation = tuple[str, str, int, bool, tuple[int, int] | None]
_Continuations = dict[tuple[str, str, int], _Continuation]
_STRUCTURELESS_LABEL = STRUCTURELESS_GOVDE_LABEL
_STRUCTURELESS_WARNING = (
    "One or more hits are full-text windows of a document that has no article structure (for example a Kurul "
    "Kararı). Cite such a document by its title and decision date/number, not as an article (madde) or as a "
    "govde window."
)
_DECISION_TEXT_LABEL = "leading decision text of a Kurul Kararı whose only named sections are annexes"
_DECISION_TEXT_WARNING = (
    "One or more hits are the leading decision text of a Kurul Kararı whose only named sections are its annexes. "
    "Cite it by the decision's title and decision date/number, not as an article (madde) or as a govde window."
)
_BODY_TEXT_LABEL = (
    "body text of a document whose only named sections are annexes or paragraphs (outside those named sections)"
)
_BODY_TEXT_WARNING = (
    "One or more hits are body text of a document whose only named sections are annexes or paragraphs, outside "
    "those named sections. Cite it by the document title, not as a named annex or paragraph."
)
_GOVDE_WARNING = (
    "One or more hits are govde remainder (unparsed body/footnote text), not madde/ilke/paragraf identities. "
    "Do not cite them as a legal provision."
)

# Turkish question words and question particles. A question such as
# "Denetim komitesi en az kaç üyeden oluşur?" is searched without them: the provision that answers
# it never contains "kaç". Reviewed list; matched on the diacritic-free search form of the word.
_QUERY_QUESTION_WORDS = frozenset(
    {
        "acaba",
        "hangi",
        "hangisi",
        "hangileri",
        "kaç",
        "kaçtır",
        "kim",
        "kimdir",
        "kimler",
        "kimlerdir",
        "kimlerden",
        "ne",
        "nedir",
        "neler",
        "nelerdir",
        "neden",
        "niçin",
        "nasıl",
        "nasıldır",
        "mi",
        "mı",
        "mu",
        "mü",
        "midir",
        "mıdır",
        "mudur",
        "müdür",
        "misin",
        "mısın",
        "musun",
        "müsün",
    }
)
# The predicate and frame words of a question ("... kimlerden oluşur?", "ne sıklıkla yapılır?"): left out
# only when the query is a question (it has a question word or a "?"), because in a statement they are
# content ("komite en az iki üyeden oluşur").
_QUESTION_FRAME_WORDS = frozenset(
    {
        "gerekir",
        "gerekmektedir",
        "olmalı",
        "olmalıdır",
        "oluşur",
        "sıklıkla",
        "yapılır",
        "yapılmalı",
        "yapılmalıdır",
    }
)
# "yüzde 15" and "%15" are the same figure; the source may write either, so "yüzde" is never required.
_QUERY_PERCENT_WORDS = frozenset({"yüzde", "yuzde"})

_LOOSE_SECTION_SEARCH_STOPWORDS = (
    _QUERY_QUESTION_WORDS
    | _QUESTION_FRAME_WORDS
    | {
        "acaba",
        "altına",
        "aşağıdaki",
        "aşağıdakilerden",
        "bir",
        "biri",
        "değildir",
        "göre",
        "hangi",
        "hangisi",
        "hakkında",
        "için",
        "ilişkin",
        "olan",
        "olarak",
        "ve",
        "veya",
        "yanlıştır",
        "yer",
    }
)
# The same lists on the diacritic-free search form, so "kac", "olusur" and "icin" match like "kaç",
# "oluşur" and "için" (the FTS search itself ignores diacritics).
_NORMALIZED_QUESTION_WORDS = frozenset(normalize_search_text(word) for word in _QUERY_QUESTION_WORDS)
_NORMALIZED_QUESTION_FRAME_WORDS = frozenset(normalize_search_text(word) for word in _QUESTION_FRAME_WORDS)
_NORMALIZED_PERCENT_WORDS = frozenset(normalize_search_text(word) for word in _QUERY_PERCENT_WORDS)
_NORMALIZED_LOOSE_STOPWORDS = frozenset(normalize_search_text(word) for word in _LOOSE_SECTION_SEARCH_STOPWORDS)
# Standalone FTS on these floods IPC / generic-ratio articles and crowds out
# the provision that actually contains the distinctive tokens.
_LOOSE_FTS_SKIP = {
    "asgari",
    "azami",
    "aşım",
    "ceza",
    "cezası",
    "idari",
    "oran",
    "oranı",
    "para",
    "yaptırım",
    "yüzde",
}


_ARTICLE_TYPES = frozenset({"madde", "gecici_madde", "ek_madde"})
_LETTERED_STORED_REF_RE = re.compile(r"^(\d+)/([^\W\d_])$")


def display_section_ref(section_type: str | None, section_ref: str | None) -> str:
    """A stored section reference as the source prints it; the stored form stays the lookup key.

    An inserted article keeps its slash with an upper-case letter ("66/a" -> "66/A"); a lettered paragraph
    of a KGK standard is printed without one ("21/a" -> "21A"). Every other reference is shown as stored.
    """
    ref = section_ref or ""
    match = _LETTERED_STORED_REF_RE.match(ref)
    if match is None:
        return ref
    if section_type in _ARTICLE_TYPES:
        return f"{match.group(1)}/{match.group(2).upper()}"
    if section_type == "paragraf":
        return f"{match.group(1)}{match.group(2).upper()}"
    return ref


def _ref_text(section_type: str | None, section_ref: str | None) -> str:
    """'<type> <reference>' for the text of an answer, the reference in its printed form."""
    return f"{section_type} {display_section_ref(section_type, section_ref)}"


def _section_quality(section: StoredDocumentSection) -> QualityAssessment:
    local = assess_markdown_quality(section.content, document_id=section.doc_id)
    document = section.document_quality
    severity = {"clean": 0, "warning": 1, "unknown": 2, "fail": 3}
    quality = local
    if document is not None:
        worst = max((document, local), key=lambda item: severity[item.label])
        quality = worst.model_copy(update={"flags": list(dict.fromkeys([*document.flags, *local.flags]))})
    if (
        section.document_extraction_method
        and not is_formula_aware(section.document_extraction_method)
        and mentions_formula(section.content)
    ):
        quality = quality.model_copy(
            update={
                "label": "warning" if quality.label == "clean" else quality.label,
                "flags": [*quality.flags, "formula_unaware_extraction"],
                "warning": " ".join(filter(None, (quality.warning, FORMULA_EXTRACTION_WARNING))),
            }
        )
    return quality


def _quality_lines(quality: QualityAssessment, *, prefix: str) -> list[str]:
    """Return consistent user-visible quality metadata for a non-clean section."""
    if quality.label not in {"warning", "fail", "unknown"}:
        return []
    flags = ", ".join(quality.flags) if quality.flags else "none"
    lines = [f"{prefix}Quality: {quality.label}", f"{prefix}Quality flags: {flags}"]
    if quality.warning:
        lines.append(f"{prefix}⚠ Quality warning: {quality.warning}")
    return lines


def _quality_metadata(quality: QualityAssessment) -> QualityMetadata:
    return QualityMetadata(
        label=quality.label,  # type: ignore[arg-type]
        flags=list(quality.flags),
        warning=quality.warning or None,
    )


_EXCERPT_SNAP_CHARS = 40


def _window_start(match_offset: int, length: int, max_chars: int) -> int:
    """Start of a ``max_chars`` window that shows ``match_offset`` a quarter of the way in."""
    return max(0, min(match_offset - max_chars // 4, length - max_chars))


def _best_term_window_start(text: str, terms: list[str], max_chars: int) -> int:
    """Start of the window that contains the most distinct query terms; ties go to the earliest start.

    Candidate windows place one occurrence of a term a quarter of the way in, as the phrase window does.
    """
    occurrences = {
        term: [match.start() for match in re.finditer(re.escape(term), text, re.IGNORECASE)] for term in terms
    }
    occurrences = {term: offsets for term, offsets in occurrences.items() if offsets}
    if not occurrences:
        return 0
    candidates = sorted(
        {_window_start(offset, len(text), max_chars) for offsets in occurrences.values() for offset in offsets}
    )
    best_start, best_count = candidates[0], -1
    for start in candidates:
        count = sum(
            1
            for term, offsets in occurrences.items()
            if (index := bisect.bisect_left(offsets, start)) < len(offsets)
            and offsets[index] + len(term) <= start + max_chars
        )
        if count > best_count:
            best_start, best_count = start, count
    return best_start


def _snap_to_word_start(text: str, start: int) -> int:
    """Move a window start that falls inside a word back to that word's start, when it is close by."""
    if start <= 0 or text[start - 1].isspace() or text[start].isspace():
        return start
    for index in range(start - 1, max(-1, start - 1 - _EXCERPT_SNAP_CHARS), -1):
        if text[index].isspace():
            return index + 1
    return start


def _leading_space(section: StoredDocumentSection, raw_content: str) -> int:
    """Whitespace before the stored text in the section's document range (offsets count it, the content does not)."""
    source_range = section.normalized_source_range
    return len(source_range) - len(source_range.lstrip()) if source_range.strip() == raw_content else 0


def _section_excerpt(
    section: StoredDocumentSection,
    *,
    max_chars: int,
    query: str | None = None,
) -> tuple[str, bool, int, int]:
    """Return a bounded excerpt and absolute offsets into normalized content."""

    # Stored truncation notices are storage metadata, never legal characters.
    raw_content, truncation_notice = split_section_truncation_notice(section.content or "")
    if len(raw_content) <= max_chars:
        local_start = 0
        local_end = len(raw_content)
    else:
        local_start = 0
        if query:
            # Search the original string so Turkish İ / Unicode case changes
            # cannot shift normalized code-point offsets. Prefer the complete
            # phrase (including PDF line wraps) over the window of its words.
            phrase = re.search(r"\s+".join(re.escape(word) for word in query.split()), raw_content, re.IGNORECASE)
            if phrase:
                local_start = _window_start(phrase.start(), len(raw_content), max_chars)
            else:
                local_start = _best_term_window_start(raw_content, _loose_search_terms(query), max_chars)
            local_start = _snap_to_word_start(raw_content, local_start)
        local_end = min(len(raw_content), local_start + max_chars)

    # Verbatim contract: the excerpt is an exact stored substring, never a
    # sanitized/summarized rendering of it.
    excerpt = raw_content[local_start:local_end]
    unsafe_reason = unsafe_verbatim_reason(excerpt)
    if unsafe_reason is not None:
        tool_error(
            "VERBATIM_UNAVAILABLE",
            f"Section {section.doc_id}/{_ref_text(section.section_type, section.section_ref)} cannot be returned verbatim: "
            f"its stored text contains an unsafe extraction artifact ({unsafe_reason}). This server does not "
            "rewrite or summarize section text; repair and re-publish the source instead.",
            retryable=False,
        )
    leading_space = _leading_space(section, raw_content)
    return (
        excerpt,
        local_start > 0 or local_end < len(raw_content) or truncation_notice is not None,
        section.start_char + leading_space + local_start,
        min(section.end_char, section.start_char + leading_space + local_end),
    )


def _search_excerpt_lines(
    section: StoredDocumentSection, excerpt_start: int, excerpt_end: int, *, truncated: bool
) -> list[str]:
    """Where a search excerpt lies in its section, printed before the excerpt when it is not the whole section."""
    if not truncated:
        return []
    raw_content = split_section_truncation_notice(section.content or "")[0]
    if excerpt_start > section.start_char + _leading_space(section, raw_content):
        return [
            f"  Excerpt from inside the section: returned normalized excerpt range [{excerpt_start}, {excerpt_end}) "
            f"of [{section.start_char}, {section.end_char}); the text before code point {excerpt_start} is omitted"
        ]
    return [f"  Returned normalized excerpt range: [{excerpt_start}, {excerpt_end}) (section body truncated)"]


def _capped_span(section: StoredDocumentSection) -> tuple[bool, int | None]:
    """Return whether the stored section is a capped window, and the uncapped span end if recorded."""
    _legal_text, notice = split_section_truncation_notice(section.content or "")
    if notice is None:
        return False, None
    original_length = truncation_notice_original_length(notice)
    return True, None if original_length is None else section.start_char + original_length


def _document_page(offset: int) -> int:
    """The get_bddk_document page (1-based, PAGE_SIZE code points each) that holds a code point offset."""
    return offset // PAGE_SIZE + 1


def _page_range_text(start: int, end: int) -> str:
    """'pages A-B' (or 'page A') of get_bddk_document that hold the code points [start, end)."""
    first, last = _document_page(start), _document_page(max(start, end - 1))
    return f"page {first}" if first == last else f"pages {first}-{last}"


def _capped_span_line(section: StoredDocumentSection) -> str | None:
    capped, original_end = _capped_span(section)
    if not capped:
        return None
    # The stored window is the span [start_char, end_char); its stored text is stripped, so it can be a few
    # whitespace characters shorter than the window when the window ends at a break.
    stored_length = section.end_char - section.start_char
    if original_end is None:
        return (
            f"- Stored window capped at {stored_length:,} characters of a longer provision; the provision "
            f"continues past code point {section.end_char} (get_bddk_document from page "
            f"{_document_page(section.end_char)}, {PAGE_SIZE:,} characters per page). {_CAPPED_REMAINDER_TEXT}"
        )
    return (
        f"- Stored window capped at {stored_length:,} of {original_end - section.start_char:,} characters; "
        f"the provision continues to code point {original_end} (uncapped span [{section.start_char}, "
        f"{original_end})). The whole provision is on get_bddk_document "
        f"{_page_range_text(section.start_char, original_end)} ({PAGE_SIZE:,} characters per page); the part "
        f"past this stored window is on {_page_range_text(section.end_char, original_end)}. {_CAPPED_REMAINDER_TEXT}"
    )


_CAPPED_REMAINDER_TEXT = (
    "The rest of the provision is in full in get_bddk_document; parts of it are also stored as later rows of this "
    "document (nested fikra/bent/paragraf rows and/or govde windows), which do not necessarily cover every character."
)


def _continuation_of(section: StoredDocumentSection, continuations: _Continuations) -> _Continuation | None:
    """Return (type, ref) of the capped named section a govde row continues, if any."""
    if section.section_type != "govde":
        return None
    return continuations.get((section.doc_id, section.source_content_hash, section.start_char))


def _continuation_label(parent: _Continuation) -> str:
    if parent[4] is None:
        return f"continuation of {_ref_text(parent[0], parent[1])} beyond its stored window"
    start, stored_length = parent[4]
    return (
        f"continuation of {_ref_text(parent[0], parent[1])} beyond its {stored_length:,}-character stored window "
        f"(the provision is {parent[2] - start:,} characters long)"
    )


def _continuation_overrun(parent: _Continuation) -> str | None:
    """State the out-of-provision tail when the govde window has text past the capped provision's true end."""
    if not parent[3]:
        return None
    return f"the characters from code point {parent[2]} on are outside {_ref_text(parent[0], parent[1])}"


def _govde_kind(section: StoredDocumentSection, continuations: _Continuations) -> str | None:
    """Classify a govde row as exactly one kind; None for a named section."""
    if section.section_type != "govde":
        return None
    if _continuation_of(section, continuations):
        return "continuation"
    if section.document_structureless:
        return "full_text_window"
    if section.document_decision_window:
        return "decision_text"
    if section.document_annex_or_paragraph_only:
        return "body_text"
    return "remainder"


def _govde_label(section: StoredDocumentSection, continuations: _Continuations) -> str | None:
    """Label a govde row by its kind; None for a named section and for genuine remainder."""
    kind = _govde_kind(section, continuations)
    if kind == "continuation":
        parent = _continuation_of(section, continuations)
        assert parent is not None
        label = _continuation_label(parent)
        if overrun := _continuation_overrun(parent):
            label = f"{label} up to code point {parent[2]}; {overrun}"
        return label
    return {
        "full_text_window": _STRUCTURELESS_LABEL,
        "decision_text": _DECISION_TEXT_LABEL,
        "body_text": _BODY_TEXT_LABEL,
    }.get(kind or "")


def _heading_suffix(section: StoredDocumentSection, continuations: _Continuations) -> str:
    """The heading part of a title line.

    The parser stores one generic heading on every govde row ("no structural heading; body/footnote
    remainder").  For a row that is a full-text window or the continuation of a capped provision that
    wording contradicts the kind label printed below it, so the title line leaves it out.
    """
    if not section.heading or (section.heading == GOVDE_HEADING and _govde_label(section, continuations)):
        return ""
    return f" — {section.heading}"


def _continuation_warning(section: StoredDocumentSection, parent: _Continuation) -> str:
    belongs = _ref_text(parent[0], parent[1])
    if overrun := _continuation_overrun(parent):
        belongs = f"{belongs} up to code point {parent[2]}; {overrun}"
    return (
        f"{section.doc_id} govde {section.section_ref} is the {_continuation_label(parent)}: this text belongs to "
        f"{belongs}. Cite that provision and verify the full text with get_bddk_document."
    )


async def _find_continuations(deps: Dependencies, sections: list[StoredDocumentSection]) -> _Continuations:
    """Capped named sections continued by the govde rows, with each capped section's stored window length."""
    if not any(section.section_type == "govde" for section in sections):
        return {}
    found = await deps.doc_store.find_capped_continuations(sections)
    windows: dict[tuple[str, str, str, str, int], tuple[int, int] | None] = {}
    for (doc_id, content_hash, _start), (section_type, section_ref, end, _overrun) in found.items():
        key = (doc_id, content_hash, section_type, section_ref, end)
        if key in windows:
            continue
        windows[key] = None
        for row in await deps.doc_store.get_document_section(
            doc_id, section_type=section_type, section_ref=section_ref
        ):
            legal_text, notice = split_section_truncation_notice(row.content or "")
            if (
                row.source_content_hash == content_hash
                and (length := truncation_notice_original_length(notice)) is not None
                and row.start_char + length == end
            ):
                windows[key] = (row.start_char, row.end_char - row.start_char)
                break
    return {
        key: (section_type, section_ref, end, overrun, windows.get((key[0], key[1], section_type, section_ref, end)))
        for key, (section_type, section_ref, end, overrun) in found.items()
    }


async def _find_parents(deps: Dependencies, sections: list[StoredDocumentSection]) -> _Parents:
    if not any(section.section_type in {"fikra", "bent"} for section in sections):
        return {}
    return await deps.doc_store.find_parent_articles(sections)


def _parent_of(section: StoredDocumentSection, parents: _Parents) -> tuple[str, str] | None:
    return parents.get((section.doc_id, section.source_content_hash, section.start_char))


def _parent_line(section: StoredDocumentSection, parents: _Parents, *, prefix: str) -> list[str]:
    parent = _parent_of(section, parents)
    return [f"{prefix}Parent: {_ref_text(parent[0], parent[1])}"] if parent else []


def _with_parent(item: SectionItem, section: StoredDocumentSection, parents: _Parents) -> SectionItem:
    parent = _parent_of(section, parents)
    if parent is None:
        return item
    return item.model_copy(update={"parent_section_type": parent[0], "parent_section_ref": parent[1]})


def _section_item(
    section: StoredDocumentSection,
    *,
    max_chars: int = _MAX_SEARCH_EXCERPT_CHARS,
    query: str | None = None,
    continuations: _Continuations | None = None,
) -> SectionItem:
    content, truncated, excerpt_start, excerpt_end = _section_excerpt(
        section,
        max_chars=max_chars,
        query=query,
    )
    capped, original_end = _capped_span(section)
    parent = _continuation_of(section, continuations or {})
    return SectionItem(
        document_id=section.doc_id,
        section_type=section.section_type,
        section_ref=section.section_ref,
        display_section_ref=display_section_ref(section.section_type, section.section_ref),
        # Same rule as the text title line: a generic govde heading that contradicts the kind label is left out.
        heading=_heading_suffix(section, continuations or {}).removeprefix(" — "),
        start_char=section.start_char,
        end_char=section.end_char,
        page_start=section.page_start,
        page_end=section.page_end,
        content=content,
        content_truncated=truncated,
        excerpt_start_char=excerpt_start,
        excerpt_end_char=excerpt_end,
        content_hash=section.content_hash,
        rank=section.rank,
        quality=_quality_metadata(_section_quality(section)),
        stored_span_capped=capped,
        original_end_char=original_end,
        continuation_of_section_type=parent[0] if parent else None,
        continuation_of_section_ref=parent[1] if parent else None,
        continuation_of_section_end_char=parent[2] if parent else None,
        document_structureless=section.document_structureless,
        govde_kind=_govde_kind(section, continuations or {}),  # type: ignore[arg-type]
    )


def _section_evidence(section: StoredDocumentSection) -> EvidenceReference:
    return EvidenceReference(
        document_id=section.doc_id,
        title=section.document_title or None,
        source_url=section.document_source_url or None,
        decision_date=section.document_decision_date or None,
        decision_number=section.document_decision_number or None,
        source_downloaded_on=utc_date(section.document_downloaded_at),
        category=section.document_category or None,
        extraction_method=section.document_extraction_method or None,
        retrieval_source="section_index",
        page_start=section.page_start,
        page_end=section.page_end,
        section_type=section.section_type,
        section_ref=section.section_ref,
        start_char=section.start_char,
        end_char=section.end_char,
        content_hash=section.content_hash,
        quality=_quality_metadata(_section_quality(section)),
    )


async def _window_neighbours(
    deps: Dependencies, section: StoredDocumentSection, continuations: _Continuations
) -> list[StoredDocumentSection]:
    """The rows before and after a govde row that is one of several consecutive windows of one text.

    That is a full-text window of a structureless document, a govde row continuing a capped provision
    (its neighbours are the capped provision itself or the other govde rows continuing it), or a govde row
    of any other kind (decision text, body text, remainder) that touches another govde row of the same
    stored text: a long uncovered span is split into touching windows. Empty for a named section and for
    a window that has no neighbour.
    """
    kind = _govde_kind(section, continuations)
    if kind is None:
        return []
    rows = sorted(
        (
            row
            for row in await deps.doc_store.get_document_section(section.doc_id, section_type="govde")
            if row.source_content_hash == section.source_content_hash
        ),
        key=lambda row: row.start_char,
    )
    position = next((index for index, row in enumerate(rows) if row.start_char == section.start_char), None)
    if position is None:
        return []
    before = rows[position - 1] if position > 0 else None
    after = rows[position + 1] if position + 1 < len(rows) else None
    if kind == "full_text_window":
        return [row for row in (before, after) if row is not None]
    if kind != "continuation":
        return [
            row
            for row in (before, after)
            if row is not None and (row.end_char == section.start_char or row.start_char == section.end_char)
        ]
    parent = _continuation_of(section, continuations)
    assert parent is not None
    linked = await _find_continuations(deps, [row for row in (before, after) if row is not None])

    def continues_same(row: StoredDocumentSection | None) -> bool:
        other = None if row is None else _continuation_of(row, linked)
        return other is not None and other[:3] == parent[:3]

    neighbours = []
    if continues_same(before):
        neighbours.append(before)
    elif parent[4] is not None:
        neighbours.extend(
            row
            for row in await deps.doc_store.get_document_section(
                section.doc_id, section_type=parent[0], section_ref=parent[1]
            )
            if row.source_content_hash == section.source_content_hash and row.start_char == parent[4][0]
        )
    if continues_same(after):
        neighbours.append(after)
    return [row for row in neighbours if row is not None]


def _window_reference(section: StoredDocumentSection) -> WindowReference:
    return WindowReference(
        section_type=section.section_type,
        section_ref=section.section_ref,
        start_char=section.start_char,
        end_char=section.end_char,
    )


def _cross_window_match(
    section: StoredDocumentSection, neighbours: list[StoredDocumentSection], quotation: str
) -> CrossWindowMatch | None:
    """The one exact occurrence of the quotation in this window's document text joined with a touching neighbour."""
    for neighbour in neighbours:
        first, second = sorted((neighbour, section), key=lambda row: row.start_char)
        if first.end_char != second.start_char or not first.normalized_source_range:
            continue
        joined = first.normalized_source_range + second.normalized_source_range
        found = joined.find(quotation)
        if found >= 0 and joined.find(quotation, found + 1) < 0:
            return CrossWindowMatch(
                start_char=first.start_char + found,
                end_char=first.start_char + found + len(quotation),
                windows=[_window_reference(first), _window_reference(second)],
            )
    return None


def _assess_quotation(
    sections: list[StoredDocumentSection],
    quotation: str,
    *,
    neighbours: list[StoredDocumentSection] | None = None,
) -> SectionAnswerAssessment:
    """Check a proposed quotation character-for-character against one resolved section."""

    if len(sections) != 1:
        return SectionAnswerAssessment(
            basis="insufficient",
            quotation_status="unavailable",
            gaps=["exact_section_not_resolved"],
        )
    section = sections[0]
    # The stored hash binds the stored content, truncation notice included. Verify that
    # binding first, then compare only the legal text: the notice is storage metadata,
    # never legal text. It is trailing, so offsets into the legal text stay valid.
    if hashlib.sha256(section.content.encode("utf-8")).hexdigest() != section.content_hash:
        check = ExactQuotationCheck(status="unavailable", reason="reference_hash_mismatch")
    else:
        legal_text, notice = split_section_truncation_notice(section.content)
        check = check_exact_quotation(
            legal_text,
            quotation,
            expected_reference_sha256=hashlib.sha256(legal_text.encode("utf-8")).hexdigest(),
        )
        if notice is not None:
            # A capped section stores only a window of the provision. A quotation absent
            # from that window may lie in the part that was not compared, so the check is
            # inconclusive, and a match of the whole window is still only an excerpt.
            if check.status == "mismatch":
                check = check.model_copy(update={"status": "unavailable", "reason": "reference_truncated"})
            elif check.extent == "whole_reference":
                check = check.model_copy(update={"extent": "excerpt"})
        elif neighbours:
            # One of several consecutive windows of one text: a quotation absent from this window may
            # cross into a neighbouring one, so the check is inconclusive; a whole window is an excerpt.
            if check.status == "mismatch":
                check = check.model_copy(update={"status": "unavailable", "reason": "reference_window"})
            elif check.extent == "whole_reference":
                check = check.model_copy(update={"extent": "excerpt"})
    cross_window = (
        _cross_window_match(section, neighbours, quotation)
        if neighbours and check.reason == "reference_window"
        else None
    )
    if check.status == "exact_reference_match":
        status = "exact_reference_match"
    elif check.status == "mismatch":
        status = "not_found"
    else:
        status = "unavailable"
    return SectionAnswerAssessment(
        basis="local_text",
        quotation_status=status,  # type: ignore[arg-type]
        quotation_check=check,
        gaps=[] if status == "exact_reference_match" else [f"quotation_{check.reason}"],
        neighbouring_windows=[_window_reference(row) for row in neighbours] if neighbours else None,
        cross_window_match=cross_window,
    )


def _span_text(section: StoredDocumentSection, start: int, end: int) -> str:
    """'[s, e) of this provision (document code points [S, E))' for offsets into the section's legal text."""
    offset = section.start_char + _leading_space(section, split_section_truncation_notice(section.content)[0])
    return f"[{start}, {end}) of this provision (document code points [{offset + start}, {offset + end}))"


_NEAR_MATCH_DIFFERENCE_TEXT = {
    "whitespace": "whitespace and line breaks",
    "apostrophes_or_quotation_marks": "typographic vs ASCII apostrophes or quotation marks",
    "asterisk_or_underscore_markers": "asterisk or underscore markers (** or __: Markdown emphasis or footnote/blank markers)",
    "soft_hyphens": "soft hyphens",
}


def _quotation_diagnostic_lines(
    assessment: SectionAnswerAssessment, section: StoredDocumentSection | None
) -> list[str]:
    """Diagnostics printed next to the character-exact status; they never change it."""
    check = assessment.quotation_check
    if check is None or section is None:
        return []
    lines: list[str] = []
    if check.occurrences:
        spans = ", ".join(_span_text(section, span.start_char, span.end_char) for span in check.occurrences)
        more = (
            f" (the first {len(check.occurrences)} are listed)"
            if check.occurrence_count > len(check.occurrences)
            else ""
        )
        lines.append(
            f"The quotation occurs exactly {check.occurrence_count} times in the stored provision{more}, at "
            f"{spans}; quote enough surrounding characters to make it unique."
        )
    if check.near_match is not None:
        near = check.near_match
        differences = ", ".join(_NEAR_MATCH_DIFFERENCE_TEXT[difference] for difference in near.differences)
        lines.append(
            f"Near match (the status above is unchanged): the quotation differs from the stored text only by "
            f"{differences}, at {_span_text(section, near.start_char, near.end_char)}. Quote those stored "
            f"characters exactly, as given between the markers below:"
        )
        lines.extend(["<<<", near.stored_text, ">>>"])
    elif check.near_match_places == "several":
        lines.append(
            "Near match: after ignoring only whitespace, apostrophe/quotation-mark, asterisk/underscore-marker "
            "(Markdown emphasis or footnote/blank markers) and "
            "soft-hyphen differences, the quotation matches the stored text at several places, so no single "
            "stored span is named; the status above is unchanged."
        )
    if assessment.neighbouring_windows and check.reason == "reference_window":
        rows = ", ".join(
            f"{_ref_text(row.section_type, row.section_ref)} (start_char={row.start_char})"
            for row in assessment.neighbouring_windows
        )
        lines.append(
            "This row is one of several consecutive windows of one text, and the quotation was compared with "
            "this window only: it was not found here, but it may cross into the neighbouring window. Check it "
            f"against {rows}."
        )
        if (match := assessment.cross_window_match) is not None:
            windows = " and ".join(
                f"{_ref_text(row.section_type, row.section_ref)} [{row.start_char}, {row.end_char})"
                for row in match.windows
            )
            lines.append(
                f"The quotation occurs exactly once in the stored text of {windows} joined, at document code points "
                f"[{match.start_char}, {match.end_char}); it crosses the boundary between those windows."
            )
    return lines


def _section_warnings(
    sections: list[StoredDocumentSection],
    *,
    content_truncated: bool = False,
    continuations: _Continuations | None = None,
) -> list[str]:
    quality_warnings = list(
        dict.fromkeys(quality.warning for section in sections if (quality := _section_quality(section)).warning)
    )
    warnings = (
        [UNTRUSTED_SOURCE_WARNING, _LEGAL_APPLICABILITY_WARNING, *quality_warnings] if sections else quality_warnings
    )
    warnings.extend(
        dict.fromkeys(notice for section in sections if (notice := document_status_notice(section.document_category)))
    )
    if content_truncated:
        warnings.append(_SECTION_TRUNCATION_WARNING)
    kinds: set[str] = set()
    for section in sections:
        kind = _govde_kind(section, continuations or {})
        if kind == "continuation":
            parent = _continuation_of(section, continuations or {})
            assert parent is not None
            warnings.append(_continuation_warning(section, parent))
        elif kind:
            kinds.add(kind)
    # Only genuine remainder keeps the warning that it is not a legal provision identity.
    for kind, warning in (
        ("full_text_window", _STRUCTURELESS_WARNING),
        ("decision_text", _DECISION_TEXT_WARNING),
        ("body_text", _BODY_TEXT_WARNING),
        ("remainder", _GOVDE_WARNING),
    ):
        if kind in kinds:
            warnings.append(warning)
    return warnings


def _provenance_lines(section: StoredDocumentSection, *, prefix: str) -> list[str]:
    """Decision identifiers and the source download date of the section's document."""
    lines = []
    if decision := decision_identifier_text(section.document_decision_date, section.document_decision_number):
        lines.append(f"{prefix}Decision date / number: {decision}")
    lines.append(f"{prefix}Text obtained from the source on: {source_date_text(section.document_downloaded_at)}")
    return lines


def _format_section(
    section: StoredDocumentSection,
    *,
    include_content: bool = True,
    quality: QualityAssessment | None = None,
    continuations: _Continuations | None = None,
    parents: _Parents | None = None,
) -> str:
    heading = _heading_suffix(section, continuations or {})
    lines = [
        f"### {section.doc_id} — {_ref_text(section.section_type, section.section_ref)}{heading}",
        f"- Document ID: {section.doc_id}",
        f"- Section: {_ref_text(section.section_type, section.section_ref)}",
        (f"- Normalized Markdown code-point range: [{section.start_char}, {section.end_char}); not source PDF pages"),
    ]
    lines.extend(_parent_line(section, parents or {}, prefix="- "))
    if capped_line := _capped_span_line(section):
        lines.append(capped_line)
    if label := _govde_label(section, continuations or {}):
        lines.append(f"- Section kind: {label}")
    if section.document_title:
        lines.append(f"- Document title: {section.document_title}")
    if status_notice := document_status_notice(section.document_category):
        lines.append(f"- Document status: {status_notice}")
    lines.extend(_provenance_lines(section, prefix="- "))
    if section.document_source_url:
        lines.append(f"- Catalog source (not a validated citation): {section.document_source_url}")
    if section.document_extraction_method:
        lines.append(f"- Extraction method: {section.document_extraction_method}")
    if section.page_start is not None:
        page_end = section.page_end if section.page_end is not None else section.page_start
        lines.append(f"- Normalized page window: {section.page_start}-{page_end} (not verified source PDF pages)")
    quality = quality or _section_quality(section)
    lines.extend(_quality_lines(quality, prefix="- "))
    if include_content:
        excerpt, truncated, excerpt_start, excerpt_end = _section_excerpt(
            section,
            max_chars=_MAX_EXACT_SECTION_CHARS,
        )
        if truncated:
            lines.append(
                f"- Returned normalized excerpt range: [{excerpt_start}, {excerpt_end}) (section body truncated)"
            )
        lines.extend(["", excerpt])
    return "\n".join(lines)


def _section_preview(section: StoredDocumentSection, *, length: int = 220) -> str:
    text = " ".join(section.content.split())
    if len(text) <= length:
        return text
    return text[:length].rsplit(" ", 1)[0]


def _section_key(section: StoredDocumentSection) -> tuple[str, str, str, str]:
    return (section.doc_id, section.section_type, section.section_ref, section.content_hash)


def _normalize_optional(value: str | int | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value.lower() if value else None


def _loose_search_text(text: str) -> str:
    # Search-only acronym equivalence; never rewrite returned source evidence.
    return re.sub(r"\blgd\b", "thk", turkish_casefold(text))


def _loose_search_terms(query: str, *, limit: int = 16) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for raw_term in re.findall(r"\w+", query, flags=re.UNICODE):
        term = _loose_search_text(raw_term)
        if len(term) < 3 or normalize_search_text(term) in _NORMALIZED_LOOSE_STOPWORDS or term in seen:
            continue
        terms.append(term)
        seen.add(term)
        if len(terms) >= limit:
            break
    return terms


def _is_control(char: str) -> bool:
    return ord(char) < 0x20 or ord(char) == 0x7F


def _controls_as_spaces(text: str) -> str:
    """Treat C0 control characters (and DEL) in a query as word separators."""
    return "".join(" " if _is_control(char) else char for char in text)


def _visible_controls(text: str) -> str:
    """Echo text with each C0 control character (and DEL) as a visible escape such as \\x00, never raw."""
    return "".join(f"\\x{ord(char):02x}" if _is_control(char) else char for char in text)


def _strip_format_chars(text: str) -> str:
    """Remove Unicode format characters (soft hyphen, zero-width space/joiner, BOM) from a search query."""
    return "".join(char for char in text if unicodedata.category(char) != "Cf")


def _required_search_text(query: str) -> tuple[str, list[str]]:
    """Return the query text the exact-word and stem passes require, and the words left out.

    Question words are left out, and in a question also its frame words ("oluşur", "yapılır"); so is
    "yüzde" (a source may write "%" instead).
    When nothing would remain, every word is kept: a word is never dropped when it is the only one.
    """
    words = re.findall(r"\w+", query, flags=re.UNICODE)
    question = "?" in query or any(normalize_search_text(word) in _NORMALIZED_QUESTION_WORDS for word in words)
    not_required = (
        _NORMALIZED_QUESTION_WORDS
        | _NORMALIZED_PERCENT_WORDS
        | (_NORMALIZED_QUESTION_FRAME_WORDS if question else frozenset())
    )
    kept: list[str] = []
    dropped: list[str] = []
    for word in words:
        (dropped if normalize_search_text(word) in not_required else kept).append(word)
    if not kept or not dropped:
        return query, []
    return " ".join(kept), dropped


def _loose_scored_text(section: StoredDocumentSection) -> str:
    """Heading and body that the loose ranking scores; the generic govde placeholder heading is not text."""
    heading = "" if section.heading == GOVDE_HEADING else section.heading
    return f"{heading} {section.content}"


def _loose_section_score(section: StoredDocumentSection, terms: list[str], weights: dict[str, float]) -> float:
    text = _loose_search_text(_loose_scored_text(section))
    score = sum(weights[term] for term in terms if term in text)
    # Penalize large cross-topic spans without rewarding tiny fragments:
    # short clauses (up to 100 words) compete on coverage, not brevity.
    score /= 1 + math.log(max(1, len(text.split()) / 100))
    if section.section_type == "gecici_madde" and "geçici" not in terms and "gecici" not in terms:
        score -= 10_000
    return score


def _document_id_phrase(document_id: str) -> str:
    """Name an ID inferred from the query: "the number 943", "the document ID kgk_tms_12"."""
    return f"the number {document_id}" if document_id.isdigit() else f"the document ID {document_id}"


async def _any_document_stored(deps: Dependencies, document_id: str) -> bool:
    for candidate in document_id_candidates(document_id):
        if await deps.doc_store.has_document(candidate):
            return True
    return False


async def _require_stored_document(deps: Dependencies, document_id: str) -> None:
    """Raise NOT_FOUND for an explicit document_id that is not a stored document (as get_bddk_document does)."""
    if not await _any_document_stored(deps, document_id):
        tool_error(
            NOT_FOUND,
            f"Document {document_id} is not available in the local store. This MCP server is airlocked "
            "and does not fetch from live BDDK / mevzuat.gov.tr sources at runtime.",
            retryable=False,
        )


async def _search_sections_strict_with_stem_top_up(
    deps: Dependencies,
    query: str,
    *,
    candidate_doc_ids: list[str | None],
    section_type: str | None,
    limit: int,
) -> tuple[list[StoredDocumentSection], set[tuple[str, str, str, str]]]:
    """Strict FTS pass, topped up with the prefix-stem pass when it returns fewer than ``limit`` hits.

    The top-up uses the same document scope (the candidate that produced the strict hits, or
    each candidate in turn when there were none) and section filter. Strict hits come first;
    top-up hits are de-duplicated against them and carry no rank. Returns the hits and the
    keys of the top-up hits.
    """
    hits: list[StoredDocumentSection] = []
    used_doc_id: str | None = None
    for candidate_doc_id in candidate_doc_ids:
        hits = await deps.doc_store.search_document_sections(
            query,
            document_id=candidate_doc_id,
            section_type=section_type,
            limit=limit,
        )
        if hits:
            used_doc_id = candidate_doc_id
            break
    if len(hits) >= limit or fts_prefix_stem_query(query) is None:
        return hits, set()
    seen = {_section_key(section) for section in hits}
    stem_keys: set[tuple[str, str, str, str]] = set()
    scope = [used_doc_id] if hits else candidate_doc_ids
    top_up: list[StoredDocumentSection] = []
    for candidate_doc_id in scope:
        stem_hits = await deps.doc_store.search_document_sections(
            query,
            document_id=candidate_doc_id,
            section_type=section_type,
            limit=limit + len(hits),
            prefix_stem=True,
        )
        top_up = [section for section in stem_hits if _section_key(section) not in seen]
        if top_up:
            break
    top_up = [section.model_copy(update={"rank": None}) for section in top_up[: limit - len(hits)]]
    stem_keys = {_section_key(section) for section in top_up}
    return hits + top_up, stem_keys


async def _search_sections_loose(
    deps: Dependencies,
    query: str,
    *,
    document_id: str | None,
    section_type: str | None,
    limit: int,
) -> list[StoredDocumentSection]:
    terms = _loose_search_terms(query)
    if not terms:
        return []
    fts_terms = [term for term in terms if (len(term) >= 5 or term == "thk") and term not in _LOOSE_FTS_SKIP]
    if not fts_terms:
        fts_terms = terms
    if "thk" in fts_terms:
        fts_terms = [*fts_terms, "lgd"]

    merged: dict[tuple[str, str, str, str], StoredDocumentSection] = {}
    # Retrieve a wider pool before cross-term ranking, rather than letting
    # single-word top-k lists discard the multi-concept match permanently.
    candidate_limit = min(100, max(40, limit * 4))
    for term in fts_terms:
        term_hits = await deps.doc_store.search_document_sections(
            term,
            document_id=document_id,
            section_type=section_type,
            limit=candidate_limit,
            include_source_range=False,
        )
        for section in term_hits:
            # Per-term ranks come from different FTS queries and are not
            # comparable; surfacing them would mislead rank-gating clients.
            merged[_section_key(section)] = section.model_copy(update={"rank": None})

    texts = [_loose_search_text(_loose_scored_text(s)) for s in merged.values()]
    # Rarity already captures specificity. Squared character length favored
    # long title words over short substantive terms such as test, veri and THK.
    weights = {
        term: 1 + math.log((1 + len(texts)) / (1 + sum(term in text for text in texts)))
        if term not in _LOOSE_FTS_SKIP
        else 1.0
        for term in terms
    }
    ranked = sorted(
        merged.values(),
        key=lambda section: (-_loose_section_score(section, terms, weights), section.start_char, _section_key(section)),
    )
    top = ranked[:limit]
    # Only the hits that leave this function need their verbatim source range.
    return await deps.doc_store.hydrate_source_ranges(top) if top else []


def register(mcp, deps: Dependencies) -> None:
    """Register section tools on the given MCP instance."""

    @mcp.tool()
    @logged_tool(logger)
    async def get_document_section(
        document_id: DocumentId,
        section_type: SectionType = None,
        section_ref: SectionRef = None,
        heading: HeadingFilter = None,
        quotation: OptionalQuotation = None,
        start_char: OptionalSectionStartChar = None,
    ) -> DocumentSectionToolResult:
        """
        Retrieve exact structural sections from a stored BDDK document.

        Use when the user asks for a specific article/principle/paragraph such as
        `943 İlke 5` or `mevzuat_22599 Madde 9`.

        Args:
            document_id: Stored document ID, e.g. `943` or `mevzuat_22599`
            section_type: Optional exact section type, e.g. madde, gecici_madde, ek_madde, ilke, paragraf, ek
            section_ref: Optional exact section reference, e.g. 9 or 66/A (inserted article)
            heading: Optional heading substring filter
            quotation: Optional proposed quotation; verify it in the exact provision,
                preserving every character, including whitespace. An exact stored-reference
                match does not establish original-source fidelity, a paraphrase or a duty.
            start_char: Optional exact start offset that selects one of several sections
                sharing type, reference and heading; copy it from the disambiguation list.
                It is valid only for the corpus file being served and is not a citation.

        A document_id that is not a stored document is a NOT_FOUND error (a bare
        numeric ID is also tried as `mevzuat_<id>` and `bddk_<id>`).

        fikra and bent rows are search-oriented, not unique addresses: many rows can
        share one reference. Where a stored parent provision (a madde, gecici_madde or
        ek_madde, or in a guide or standard a paragraf or ilke) contains a row it is
        shown as "Parent:"; fetch that parent for the full provision.

        Supplying quotation returns answer_assessment with the mechanical match result.
        The quotation is compared with the section's legal text after its stored hash
        is confirmed; a storage truncation notice on a capped section is not legal text
        and is excluded, so reference_sha256 is the hash of exactly the text compared.
        A capped section holds only a stored window of its provision (at most 20,000
        characters; the answer states its length and the provision's true length, and
        the get_bddk_document pages that hold the rest), so text beyond that window is
        not compared: a quotation absent from the window is
        reported as unavailable (gap quotation_reference_truncated), never not_found,
        and a match is never reported as the whole reference. The same holds for a govde
        row that is one of several consecutive windows of one text (a document without
        article structure, the continuation of a long provision, or any govde row that
        touches another govde row of the same text): a quotation absent
        from it is unavailable (gap quotation_reference_window), the neighbouring rows
        are named with their start_char, and an exact occurrence in this window joined
        with a neighbour is reported with its document offsets.
        The status is always character-exact. Next to a not_found status the answer may
        name the one stored place the quotation matches once only whitespace/line
        breaks, typographic vs ASCII apostrophes and quotation marks, asterisk or
        underscore markers ** / __ (Markdown emphasis or footnote/blank markers)
        and soft hyphens are ignored, with those stored characters to quote instead; an
        ambiguous quotation lists where it occurs.
        Scope and semantic entailment still require review; never infer them from a
        quotation match.
        """
        sections = []
        for candidate in document_id_candidates(document_id):
            sections = await deps.doc_store.get_document_section(
                candidate,
                section_type=_normalize_optional(section_type),
                section_ref=_normalize_optional(section_ref),
                heading=heading,
                limit=_MAX_DISAMBIGUATION_RESULTS + 1,
                start_char=start_char,
            )
            if sections:
                break
        section_item = quality = None
        continuations = await _find_continuations(deps, sections[:_MAX_DISAMBIGUATION_RESULTS])
        parents = await _find_parents(deps, sections[:_MAX_DISAMBIGUATION_RESULTS])
        if len(sections) == 1:
            quality = _section_quality(sections[0])
            section_item = _with_parent(
                _section_item(sections[0], max_chars=_MAX_EXACT_SECTION_CHARS, continuations=continuations),
                sections[0],
                parents,
            )
        assessment = None
        assessment_text = ""
        if quotation is not None:
            neighbours = await _window_neighbours(deps, sections[0], continuations) if len(sections) == 1 else []
            assessment = _assess_quotation(sections, quotation, neighbours=neighbours)
            assessment_text = (
                f"\n\nAnswer evidence checks: {assessment.basis}\n"
                f"Quotation in stored reference (no normalization): {assessment.quotation_status}\n"
                "Original-source fidelity is NOT established by this character comparison.\n"
                f"Evidence gaps: {', '.join(assessment.gaps) or 'none in the requested mechanical checks'}\n"
                "Scope and semantic entailment are NOT assessed. A matching quotation does not prove "
                "a broader claim, obligation, frequency, threshold, or applicability to a particular bank."
            )
            if len(sections) == 1 and split_section_truncation_notice(sections[0].content)[1] is not None:
                assessment_text += f"\n{_CAPPED_QUOTATION_NOTE}"
            if diagnostics := _quotation_diagnostic_lines(assessment, sections[0] if len(sections) == 1 else None):
                assessment_text += "\n" + "\n".join(diagnostics)
        assessment_incomplete = bool(assessment and assessment.gaps)
        if not sections:
            # A document without the requested section is "no results"; an unknown document is NOT_FOUND.
            await _require_stored_document(deps, document_id)
            query = " ".join(
                str(part) for part in (document_id, section_type or "", section_ref or "", heading or "") if part
            )
            output = (
                f"No section found for document {document_id} with the requested filters.\n"
                f"Try search_document_sections with query: {_visible_controls(query or document_id)}"
            )
            return structured_tool_result(
                DocumentSectionResponse(
                    status="no_results",
                    text=output + assessment_text,
                    answer_assessment=assessment,
                    requested_document_id=document_id,
                    section_type=_normalize_optional(section_type),
                    section_ref=_normalize_optional(section_ref),
                    heading=heading,
                )
            )

        if len(sections) == 1:
            assert section_item is not None and quality is not None
            section = sections[0]
            return structured_tool_result(
                DocumentSectionResponse(
                    status="partial" if section_item.content_truncated or assessment_incomplete else "ok",
                    text=_format_section(section, quality=quality, continuations=continuations, parents=parents)
                    + assessment_text,
                    answer_assessment=assessment,
                    evidence=[_section_evidence(section)],
                    warnings=_section_warnings(
                        [section], content_truncated=section_item.content_truncated, continuations=continuations
                    ),
                    requested_document_id=document_id,
                    section_type=_normalize_optional(section_type),
                    section_ref=_normalize_optional(section_ref),
                    heading=heading,
                    results=[section_item],
                )
            )

        result_sections = sections[:_MAX_DISAMBIGUATION_RESULTS]
        more_results = len(sections) > len(result_sections)
        lines = [
            (
                f"More than {_MAX_DISAMBIGUATION_RESULTS} sections matched. "
                if more_results
                else f"Multiple sections matched ({len(result_sections)}). "
            )
            + "Narrow by section_type, section_ref, or heading, or repeat the call with start_char=<value> "
            "from a line below to select exactly one section (start_char is valid only for this corpus file "
            "and is not a citation):\n"
        ]
        # Every listed section comes from the one resolved document.
        if result_sections[0].document_title:
            lines.append(f"- Document title: {result_sections[0].document_title}")
        lines.extend(_provenance_lines(result_sections[0], prefix="- "))
        for status_notice in dict.fromkeys(
            notice for section in result_sections if (notice := document_status_notice(section.document_category))
        ):
            lines.append(f"- Document status: {status_notice}")
        if any(section.section_type in {"fikra", "bent"} for section in result_sections):
            lines.append(_nested_row_hint({parent[0] for parent in parents.values()}))
        for section in result_sections:
            heading_text = _heading_suffix(section, continuations)
            lines.append(
                f"- {section.doc_id} {_ref_text(section.section_type, section.section_ref)}{heading_text} "
                f"(start_char={section.start_char}, hash={section.content_hash[:12]})"
            )
            lines.extend(_parent_line(section, parents, prefix="  "))
            if label := _govde_label(section, continuations):
                lines.append(f"  Section kind: {label}")
            lines.extend(_quality_lines(_section_quality(section), prefix="  "))
            lines.append(f"  {_section_preview(section)}")
        if more_results:
            lines.append("... additional matches omitted; add a section reference or heading filter.")
        result_items = [
            _with_parent(_section_item(section, continuations=continuations), section, parents)
            for section in result_sections
        ]
        result_content_truncated = more_results or any(item.content_truncated for item in result_items)
        return structured_tool_result(
            DocumentSectionResponse(
                status="partial" if result_content_truncated or assessment_incomplete else "ok",
                text="\n".join(lines) + assessment_text,
                answer_assessment=assessment,
                evidence=[_section_evidence(section) for section in result_sections],
                warnings=_section_warnings(
                    result_sections,
                    content_truncated=result_content_truncated,
                    continuations=continuations,
                ),
                requested_document_id=document_id,
                section_type=_normalize_optional(section_type),
                section_ref=_normalize_optional(section_ref),
                heading=heading,
                results=result_items,
            )
        )

    @mcp.tool()
    @logged_tool(logger)
    async def search_document_sections(
        query: SectionQuery,
        document_id: OptionalDocumentId = None,
        section_type: SectionType = None,
        limit: SectionResultLimit = 10,
    ) -> SectionSearchToolResult:
        """
        Search section-level content in stored BDDK documents.

        Parses exact legal references from the query and uses them as filters when
        explicit arguments are not provided.

        Args:
            query: Turkish legal query, e.g. `943 İlke 5 model validasyonu`
            document_id: Optional document ID filter; an ID that is not a stored
                document is a NOT_FOUND error
            section_type: Optional section type filter
            limit: Maximum number of section results

        Without `section_type`, every named section and every `govde` row is
        searched; nested fıkra/bent rows are omitted unless `section_type` requests
        them. Named sections and govde rows are ranked by the same FTS rank. Each
        govde hit is labelled as exactly one kind: (1) a full-text window of a
        document with no article structure (for example a Kurul Kararı), cited by
        the document's title and decision date/number, not as an article; (2) the
        continuation of a named provision beyond its stored window (at most 20,000
        characters), which belongs to that provision and is labelled with it; (3) the
        leading decision text of a Kurul Kararı whose only named sections are
        annexes, cited as that decision; (4) body text of a document whose only
        named sections are annexes or paragraphs; (5) remainder text between
        provisions (preamble, footnotes, unparsed leftovers), which is not a legal
        provision identity.

        Each FTS hit includes a `Match rank` (SQLite FTS5 BM25 over the section
        heading and body, heading weighted 2x, sign-flipped so that higher is
        better): ranks are comparable within one query's results and can be
        used to gate low-confidence retrieval.

        Turkish question words (kaç, nedir, nasıl, mi ...), in a question also its
        frame words (oluşur, yapılır ...), and "yüzde" (sources also write "%") are
        not required words; the answer lists them under "not searched". Invisible
        format characters are ignored.

        When the exact-word pass returns fewer than `limit` hits, it is topped up
        with an inflection-tolerant pass (word roots and prefixes, e.g. "süresi"
        also finds "süresinin", "denetlenmesi" also finds "denetimi"). Those hits
        follow the exact-word hits, carry no rank and are marked `stem_match`. The
        loose token fallback runs when both passes find nothing, and also when
        every hit is an inflection-tolerant one; its hits then follow them.

        A bare number or a `kgk_`/`rg_` document ID in the query (typically at its
        start, e.g. `943 İlke 5`) is read as a document ID only when such a
        document is stored; only the first such candidate in the query is
        considered (a later number is never promoted). It scopes the search as a preference, not a filter: it limits
        the search to that document only when the exact-reference, exact-word or
        inflection-tolerant pass finds something there; the text then says so and
        `inferred_document_filter_applied` is true. Otherwise the search falls back
        to every document and the text adds a note saying that the ID was not
        applied (`inferred_document_filter_dropped` is true).
        """
        # Invisible format characters (soft hyphen, zero-width space) split words: never search them.
        # Control characters (NUL ...) separate words. The query is echoed as given, except that a
        # control character is shown as an escape, never raw.
        search_query = _strip_format_chars(_controls_as_spaces(query))
        echoed_query = _visible_controls(query)
        refs = parse_legal_refs(search_query)
        # An explicit document_id is a hard filter. A number inferred from the
        # query text is only a preference: it is ignored when no such document
        # is stored, and dropped when the exact-reference, strict and
        # inflection-tolerant passes find nothing in it; it never scopes the
        # loose fallback.
        # Only the first candidate is considered: an unstored number is query content (an amount, a
        # day count, a year), so a later number is never promoted over it and nothing is stripped.
        candidate_query_doc_id = None if document_id else (refs.document_ids[0] if refs.document_ids else None)
        query_doc_id = candidate_query_doc_id
        filter_drop_reason: str | None = None
        if query_doc_id and not await _any_document_stored(deps, query_doc_id):
            filter_drop_reason = "no stored document has that ID"
            query_doc_id = None
        inferred_doc_id = document_id or query_doc_id
        explicit_section_type = _normalize_optional(section_type)
        # The exact reference is the first parsed one; with an explicit section_type only a reference of
        # that type qualifies, so "İlke 5" with section_type=paragraf is not read as paragraf 5.
        parsed_refs = [ref for ref in refs.sections if not explicit_section_type or ref[0] == explicit_section_type]
        mismatched_ref = None if parsed_refs or not refs.sections else refs.sections[0]
        parsed_section_type, inferred_section_ref = parsed_refs[0] if parsed_refs else (None, None)
        inferred_section_type = explicit_section_type or parsed_section_type
        exact_ref_detected = bool(inferred_doc_id and inferred_section_type and inferred_section_ref)
        # The filter actually applied: an explicit section_type always; a type parsed from the query
        # text only while a document is identified.
        applied_section_type = inferred_section_type if exact_ref_detected else explicit_section_type
        candidate_doc_ids: list[str | None] = (
            list(document_id_candidates(inferred_doc_id)) if inferred_doc_id else [None]
        )
        # The document number names the scope, it is not a content word: no section contains its own
        # document ID, so the scoped passes search the query without it. The unscoped retry keeps the
        # raw query, where the number may be content ("kredi 90 gün gecikme").
        scoped_query = search_query
        if query_doc_id:
            start, end = refs.document_id_spans[0]
            scoped_query = f"{search_query[:start].rstrip()} {search_query[end:].lstrip()}".strip()
        # Question words and "yüzde" are not required by the exact-word and stem passes.
        required_scoped_query, not_searched = _required_search_text(scoped_query)

        exact_hits = []
        if exact_ref_detected:
            for candidate_doc_id in candidate_doc_ids:
                exact_hits = await deps.doc_store.get_document_section(
                    candidate_doc_id,
                    section_type=_normalize_optional(inferred_section_type),
                    section_ref=_normalize_optional(inferred_section_ref),
                    limit=limit,
                )
                if exact_hits:
                    break
        hits, stem_keys = await _search_sections_strict_with_stem_top_up(
            deps,
            required_scoped_query,
            candidate_doc_ids=candidate_doc_ids,
            section_type=applied_section_type,
            limit=limit,
        )
        loose_fallback_used = False
        # Set when the inferred document is stored but the exact provision named in the query is not in it.
        missing_in_dropped_doc: str | None = None
        if not hits and not exact_hits and query_doc_id:
            # Nothing exact or strict inside the inferred document: a loose match there must not
            # outrank a strict match elsewhere, so repeat strict, then loose, with no document scope.
            filter_drop_reason = f"no section in document {query_doc_id} matched the query"
            if exact_ref_detected:
                missing_in_dropped_doc = query_doc_id
                filter_drop_reason = (
                    f"document {query_doc_id} is stored, but {_ref_text(parsed_section_type, inferred_section_ref)} "
                    "was not "
                    "found in it and no other section of it matched the query"
                )
            inferred_doc_id = None
            # With the inferred document dropped no document is identified, so neither the section
            # reference nor its type is applied to the unscoped search.
            exact_ref_detected = False
            applied_section_type = explicit_section_type
            hits, stem_keys = await _search_sections_strict_with_stem_top_up(
                deps,
                _required_search_text(search_query)[0],
                candidate_doc_ids=[None],
                section_type=applied_section_type,
                limit=limit,
            )
            candidate_doc_ids = [None]
        if not hits and not exact_hits:
            # The in-document loose loop runs only for an explicit document_id argument.
            for candidate_doc_id in candidate_doc_ids:
                hits = await _search_sections_loose(
                    deps,
                    search_query,
                    document_id=candidate_doc_id,
                    section_type=applied_section_type,
                    limit=limit,
                )
                if hits:
                    break
            loose_fallback_used = bool(hits)
        elif not exact_hits and len(hits) < limit and all(_section_key(hit) in stem_keys for hit in hits):
            # Only inflection-tolerant hits: a weak prefix match must not block the provision the query
            # names, so the loose pass also runs, in the same scope, and its hits follow the stem hits.
            seen = {_section_key(hit) for hit in hits}
            loose_hits = await _search_sections_loose(
                deps,
                scoped_query if inferred_doc_id else search_query,
                document_id=hits[0].doc_id if inferred_doc_id else None,
                section_type=applied_section_type,
                limit=limit,
            )
            extra = [hit for hit in loose_hits if _section_key(hit) not in seen][: limit - len(hits)]
            hits = hits + extra
            loose_fallback_used = bool(extra)
        # A section reference without an identified document cannot be looked up, so it is not applied.
        # Computed after the drop decision so that a dropped inferred document reports it as not applied.
        unapplied_ref_note: str | None = None
        if missing_in_dropped_doc:
            unapplied_ref_note = (
                f"Note: {_ref_text(parsed_section_type, inferred_section_ref)} was not found in document "
                f"{missing_in_dropped_doc}, which {_document_id_phrase(missing_in_dropped_doc)} in the query "
                "identified; the "
                "reference was not applied to the search over every document. Check the reference, or retrieve "
                "that document with get_bddk_document."
            )
        elif inferred_section_ref and not exact_ref_detected:
            unapplied_ref_note = (
                f"Note: {_ref_text(parsed_section_type, inferred_section_ref)} in the query was not applied as a filter "
                "because no document was identified; find the document id with search_bddk_regulations, then call "
                f"get_document_section(document_id, section_type='{parsed_section_type}', "
                f"section_ref='{inferred_section_ref}')."
            )
        elif mismatched_ref:
            unapplied_ref_note = (
                f"Note: {_ref_text(mismatched_ref[0], mismatched_ref[1])} in the query was not used as an exact reference "
                f"because section_type={explicit_section_type} was passed."
            )
        reported_section_ref = None if unapplied_ref_note else _normalize_optional(inferred_section_ref)
        filter_dropped = filter_drop_reason is not None
        inferred_filter_applied = bool(query_doc_id and not filter_dropped)
        filter_note = (
            f"Note: {_document_id_phrase(candidate_query_doc_id)} in the query was not applied as a document filter "
            f"({filter_drop_reason}).\n"
            if filter_dropped
            else (
                f"Note: the search was limited to document {query_doc_id} because of "
                f"{_document_id_phrase(query_doc_id)} in the query; pass document_id or remove the "
                f"{'number' if query_doc_id.isdigit() else 'document ID'} to search every document.\n"
                if inferred_filter_applied
                else ""
            )
        )
        not_searched_note = (
            f"Note: not searched as required words: {', '.join(not_searched)} (Turkish question words and the "
            "frame words of a question are left out; 'yüzde' is optional because sources also write '%').\n"
            if not_searched
            else ""
        )
        if exact_hits:
            # Exact-ref lookups carry no FTS rank; inherit it from the FTS
            # duplicate being deduplicated away so the top hit is not the
            # only unscored result.
            fts_ranks = {_section_key(s): s.rank for s in hits if s.rank is not None}
            exact_hits = [
                s.model_copy(update={"rank": fts_ranks.get(_section_key(s))}) if s.rank is None else s
                for s in exact_hits
            ]
            seen = {_section_key(section) for section in exact_hits}
            hits = (exact_hits + [section for section in hits if _section_key(section) not in seen])[:limit]
            # The explicitly requested provision is an exact-reference hit even when the prefix top-up
            # also found it: never label it as an inflection-tolerant match.
            stem_keys -= seen
        if not hits and document_id:
            # An explicit document_id that names no stored document is NOT_FOUND, not "no sections".
            await _require_stored_document(deps, document_id)
        if not hits:
            passed_filters = [
                name for name, value in (("document_id", document_id), ("section_type", section_type)) if value
            ]
            remove_hint = f"remove the {'/'.join(passed_filters)} filter, " if passed_filters else ""
            output = (
                f"NO RESULTS: No document sections found matching '{echoed_query}'.\n"
                + filter_note
                + not_searched_note
                + (f"{unapplied_ref_note}\n" if unapplied_ref_note else "")
                + f"Try a broader query, {remove_hint}or retrieve the full document."
            )
            return structured_tool_result(
                SectionSearchResponse(
                    status="no_results",
                    text=output,
                    query=query,
                    document_id=inferred_doc_id,
                    section_type=applied_section_type,
                    section_ref=reported_section_ref,
                    parsed_section_ref=_normalize_optional(inferred_section_ref),
                    exact_reference_detected=exact_ref_detected,
                    loose_fallback_used=False,
                    warnings=[unapplied_ref_note] if unapplied_ref_note else [],
                    inferred_document_filter_dropped=filter_dropped,
                    query_words_not_searched=not_searched,
                )
            )

        continuations = await _find_continuations(deps, hits)
        parents = await _find_parents(deps, hits)
        lines = [f"Found {len(hits)} section result(s) for '{echoed_query}':\n"]
        if unapplied_ref_note:
            lines.insert(0, f"{unapplied_ref_note}\n")
        if not_searched_note:
            lines.insert(0, not_searched_note)
        if filter_note:
            lines.insert(0, filter_note)
        has_exact_word_hits = any(_section_key(hit) not in stem_keys for hit in hits)
        for hit in hits:
            heading = _heading_suffix(hit, continuations)
            lines.append(f"**{hit.doc_id} — {_ref_text(hit.section_type, hit.section_ref)}{heading}**")
            lines.append(f"  Document ID: {hit.doc_id}")
            if hit.document_title:
                lines.append(f"  Document title: {hit.document_title}")
            if status_notice := document_status_notice(hit.document_category):
                lines.append(f"  Document status: {status_notice}")
            lines.extend(_provenance_lines(hit, prefix="  "))
            if hit.document_source_url:
                lines.append(f"  Catalog source (not a validated citation): {hit.document_source_url}")
            lines.append(f"  Section: {_ref_text(hit.section_type, hit.section_ref)}")
            lines.extend(_parent_line(hit, parents, prefix="  "))
            if label := _govde_label(hit, continuations):
                lines.append(f"  Note: {label}")
            elif hit.section_type == "govde":
                lines.append("  Note: govde remainder — not a legal provision identity")
            lines.append(
                f"  Normalized Markdown code-point range: [{hit.start_char}, {hit.end_char}); not source PDF pages"
            )
            if hit.rank is not None:
                # "relative" guards against conflation with the percent-scale
                # relevance gate in the server instructions (store search).
                lines.append(f"  Match rank: {hit.rank:.4f} (relative, FTS)")
            if _section_key(hit) in stem_keys:
                lines.append(
                    "  Match: inflection-tolerant match (word prefixes of the query terms, not the exact words); "
                    + ("no rank, listed after the exact-word hits" if has_exact_word_hits else "no rank")
                )
            lines.extend(_quality_lines(_section_quality(hit), prefix="  "))
            excerpt, truncated, excerpt_start, excerpt_end = _section_excerpt(
                hit, max_chars=_MAX_SEARCH_EXCERPT_CHARS, query=search_query
            )
            if excerpt:
                lines.extend(_search_excerpt_lines(hit, excerpt_start, excerpt_end, truncated=truncated))
                lines.append(excerpt)
                if truncated:
                    lines.append("  [excerpt truncated — call get_document_section for the full section]")
            lines.append("")
        result_items = [
            _with_parent(_section_item(hit, query=search_query, continuations=continuations), hit, parents).model_copy(
                update={"stem_match": _section_key(hit) in stem_keys}
            )
            for hit in hits
        ]
        result_content_truncated = any(item.content_truncated for item in result_items)
        return structured_tool_result(
            SectionSearchResponse(
                status="partial" if result_content_truncated or (exact_ref_detected and not exact_hits) else "ok",
                text="\n".join(lines),
                evidence=[_section_evidence(hit) for hit in hits],
                warnings=[
                    *_section_warnings(hits, content_truncated=result_content_truncated, continuations=continuations),
                    *(
                        [_EXACT_REFERENCE_MISSING_WARNING]
                        if (exact_ref_detected and not exact_hits) or missing_in_dropped_doc
                        else []
                    ),
                    *([unapplied_ref_note] if unapplied_ref_note else []),
                    *(
                        [
                            "Loose token fallback was used: results may match only part of the query. "
                            "Search each substantive issue separately and verify exact provisions; "
                            "this ranking is not a legal-confidence score."
                        ]
                        if loose_fallback_used
                        else []
                    ),
                ],
                query=query,
                document_id=inferred_doc_id,
                section_type=applied_section_type,
                section_ref=reported_section_ref,
                parsed_section_ref=_normalize_optional(inferred_section_ref),
                exact_reference_detected=exact_ref_detected,
                loose_fallback_used=loose_fallback_used,
                inferred_document_filter_dropped=filter_dropped,
                inferred_document_filter_applied=inferred_filter_applied,
                query_words_not_searched=not_searched,
                results=result_items,
            )
        )
