"""Vector store over the SQLite corpus: chunking, embeddings, hybrid search.

Architecture:
  - Table ``chunks``: section-aware token chunks with float16 embeddings and
    an FTS5 index over normalized text (``chunks_fts``)
  - Embedding model: multilingual-e5-base (offline-first via BDDK_EMBEDDING_MODEL_PATH)
  - Dense retrieval: exact cosine similarity with numpy over all chunk vectors
    (``ponytail``: brute force is exact and takes milliseconds up to a few
    hundred thousand chunks; add an ANN index such as hnswlib beyond that)
  - Hybrid search: dense + FTS5 BM25 fused with Reciprocal Rank Fusion
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import sqlite3
import threading
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from bddk_mcp.core.config import (
    EMBEDDING_CHUNK_TARGET_TOKENS,
    EMBEDDING_CHUNK_TOKEN_OVERLAP,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_PATH,
    EMBEDDING_MODEL_REVISION,
    SEMANTIC_RELEVANCE_THRESHOLD,
)
from bddk_mcp.quality.markdown_quality import assess_markdown_quality
from bddk_mcp.store.corpus_db import CorpusDB
from bddk_mcp.store.doc_store import fts_match_query, normalize_search_text
from bddk_mcp.store.legal_ref import (
    DECISION_NUMBER_SUFFIX,
    LegalRefs,
    canonical_section_ref,
    parse_legal_refs,
    turkish_casefold,
)
from bddk_mcp.store.search_glossary import GlossaryEntry, query_abbreviation
from bddk_mcp.store.section_index import DocumentSection, extract_document_sections

logger = logging.getLogger(__name__)


class VectorIndexConsistencyError(RuntimeError):
    """The vector index cannot be published for a missing or different source."""


class RetrievalProfileError(RuntimeError):
    """The corpus was embedded with a different model or chunking than this runtime."""


class SemanticSearchReadinessError(RuntimeError):
    """The serving runtime cannot execute the advertised semantic-search path."""


class SemanticSearchUnavailableError(RuntimeError):
    """The embedding runtime failed deterministically and requires operator action."""


_PASSAGE_PREFIX = "passage"
_QUERY_PREFIX = "query"
_PREFIX_SEPARATOR = ": "
_EMBEDDING_ENCODE_BATCH_SIZE = 32
_RRF_K = 60
_HYBRID_VECTOR_CANDIDATE_LIMIT = 50
_HYBRID_VECTOR_FETCH_LIMIT = 100
_HYBRID_FTS_CANDIDATE_LIMIT = 50
_SEARCH_SNIPPET_CHARS = 800
_SCORE_DECIMAL_PLACES = 4
_RRF_SCORE_DECIMAL_PLACES = 6
_FTS_GATE_PENALTY = 0.65
# The former score-gap cut: hits within this distance of the top ordering score were the only ones
# returned. Nothing is cut any more, but those hits keep their former place ahead of the others.
_GAP_ORDER_THRESHOLD = 0.051
_HIGH_CONFIDENCE_THRESHOLD = 0.70
_MEDIUM_CONFIDENCE_THRESHOLD = 0.50
_FTS_TOKEN_RE = re.compile(r"[0-9A-Za-zÇĞİÖŞÜçğıöşü]{3,}")
_FTS_RELAXED_STOPWORDS = {
    "acaba",
    "ancak",
    "bir",
    "buna",
    "göre",
    "gore",
    "hangi",
    "için",
    "icin",
    "ile",
    "kaç",
    "kac",
    "kadar",
    "nasıl",
    "nasil",
    "nedir",
    "nelerdir",
    "olan",
    "olarak",
    "olabilir",
    "olmalıdır",
    "olmalidir",
    "veya",
}
_FTS_RELAXED_MIN_TERMS = 3
_FTS_RELAXED_MAX_TERMS = 12
_LEXICAL_RELEVANCE_BOOST = 0.045
_PHRASE_RELEVANCE_BOOST = 0.03
_PHRASE_RELEVANCE_MAX = 0.06
_TURKISH_STEM_SUFFIXES = ("leri", "lari", "inin", "unun", "nin", "nun", "ler", "lar", "si", "su", "i", "u")
_EMBEDDING_DTYPE = np.float16
# Query normalisation (VectorStore.search): the keyword side folds case and diacritics, the embedding does
# not, so the embedded query is normalised. Letters only Turkish has (with the circumflex vowels â, î, û);
# a query with none of them may have been typed without them.
_TURKISH_LETTERS = frozenset("çğıöşüâîûÇĞİÖŞÜÂÎÛ")
_QUERY_WORD_RE = re.compile(r"[^\W\d_]+")
_VOCABULARY_WORD_RE = re.compile(r"[A-Za-zÇĞİÖŞÜÂÎÛçğıöşüâîû]+")
# An all-upper-case query is lower-cased only when it has at least this many words of four or more letters:
# "TFRS 9", "MADDE 5", "BDDK" or "SYR" are codes and references, not shouted sentences.
_SHOUTED_MIN_LONG_WORDS = 2
# Function words: a query made only of these has nothing to search for and is treated like an empty one.
# Reviewed list, compared on the folded form.
_QUERY_FUNCTION_WORDS = frozenset(
    normalize_search_text(word)
    for word in (
        *_FTS_RELAXED_STOPWORDS,
        *"ama bu bunlar da de daha diye en fakat gibi hem her ise ki mi mı mu mü ne o ve ya yani şu çok".split(),
    )
)
# "8948", "8948 sayılı karar", "8948 sayılı Kurul Kararı": a decision number on its own (folded query; the
# decision-number pattern of legal_ref).
_DECISION_NUMBER_QUERY_RE = re.compile(rf"^\s*(\d{{3,6}})(?:{DECISION_NUMBER_SUFFIX})?\s*$")
# A word typed without Turkish letters is replaced by the corpus form only when that form occurs at least
# this often in the chunk texts and the typed form less than this share as often.
_RESTORE_MIN_COUNT = 20
_RESTORE_MAX_TYPED_SHARE = 0.05
# ... and only when no second Turkish spelling of the same folded form occurs at least this share as often.
_RESTORE_AMBIGUOUS_SHARE = 0.20


@dataclass(frozen=True)
class RelevanceCalibration:
    """Maps the internal ordering score of one embedding model to the displayed relevance.

    ``displayed = clamp((ordering_score - floor) / (ceiling - floor), 0, 1)``.
    ``semantic_threshold`` is the default minimum raw cosine for a dense hit.
    """

    floor: float
    ceiling: float
    semantic_threshold: float


# Any model without a measured entry: identity mapping and the old 0.50 raw
# threshold. Nothing has been measured for such a model.
_IDENTITY_RELEVANCE_CALIBRATION = RelevanceCalibration(floor=0.0, ceiling=1.0, semantic_threshold=0.50)

# Measured 2026-11-12 on the 319-document corpus (13,721 chunks) with the pinned
# revision of intfloat/multilingual-e5-base, 101 queries (scripts/calibrate_relevance.py):
# 40 Turkish in-domain questions with expected documents, 6 English questions on
# corpus topics, 10 near-domain questions on instruments outside the corpus, 40
# off-topic questions and 5 gibberish strings. Raw top-1 cosines: in-domain
# 0.852-0.914, off-topic 0.739-0.852; every expected document scored >= 0.822.
# A calibration on a small query set, not a retrieval evaluation.
RELEVANCE_CALIBRATIONS: dict[str, RelevanceCalibration] = {
    "intfloat/multilingual-e5-base": RelevanceCalibration(floor=0.82, ceiling=0.96, semantic_threshold=0.80),
}


def relevance_calibration(embedding_model: str) -> RelevanceCalibration:
    """The calibration recorded for ``embedding_model``, or the identity mapping."""
    return RELEVANCE_CALIBRATIONS.get(embedding_model, _IDENTITY_RELEVANCE_CALIBRATION)


def semantic_threshold(embedding_model: str) -> float:
    """Minimum raw cosine for a dense hit: BDDK_SEMANTIC_THRESHOLD, else the model's default."""
    if SEMANTIC_RELEVANCE_THRESHOLD is not None:
        return SEMANTIC_RELEVANCE_THRESHOLD
    return relevance_calibration(embedding_model).semantic_threshold


def displayed_relevance(ordering_score: float, calibration: RelevanceCalibration) -> float:
    """Calibrated relevance in [0, 1]; monotonic in the ordering score, so ranking is unchanged."""
    span = calibration.ceiling - calibration.floor
    return min(1.0, max(0.0, (ordering_score - calibration.floor) / span))


# Loaded embedding models shared by every VectorStore in this process, keyed by
# (loader class, model reference, load kwargs).
_LOADED_EMBEDDING_MODELS: dict[tuple[object, str, tuple[tuple[str, object], ...]], object] = {}
_LOADED_EMBEDDING_MODELS_LOCK = threading.Lock()
_LOADED_EMBEDDING_MODELS_LIMIT = 4


# -- Chunking -----------------------------------------------------------------


@dataclass(frozen=True)
class DocumentChunk:
    """A text chunk plus optional legal section metadata."""

    chunk_text: str
    start_char: int
    end_char: int
    section_type: str = ""
    section_ref: str = ""
    section_start_char: int | None = None
    section_end_char: int | None = None
    section_content_hash: str = ""


@dataclass(frozen=True)
class _ChunkSpan:
    start_char: int
    end_char: int
    section: DocumentSection | None = None


@dataclass(frozen=True)
class _TextUnit:
    start_char: int
    end_char: int
    text: str
    token_count: int


_WORD_UNIT_RE = re.compile(r"\S+\s*", re.MULTILINE)


def _chunk_document(
    doc_id: str,
    text: str,
    tokenizer=None,
    target_tokens: int = EMBEDDING_CHUNK_TARGET_TOKENS,
    token_overlap: int = EMBEDDING_CHUNK_TOKEN_OVERLAP,
) -> list[DocumentChunk]:
    """Split text into chunks and attach best-effort legal section metadata."""
    if not text:
        return []
    sections = extract_document_sections(doc_id, text)
    if tokenizer is None:
        raise RetrievalProfileError("Token chunking is configured but the embedding tokenizer is unavailable.")
    return _chunk_document_by_tokens(doc_id, text, sections, tokenizer, target_tokens, token_overlap)


def _chunk_document_by_tokens(
    doc_id: str,
    text: str,
    sections: list[DocumentSection],
    tokenizer,
    target_tokens: int,
    token_overlap: int,
) -> list[DocumentChunk]:
    target_tokens = max(1, target_tokens)
    token_overlap = max(0, min(token_overlap, target_tokens - 1))
    chunks: list[DocumentChunk] = []
    for span in _chunk_spans(text, sections):
        for start_char, end_char in _token_budget_ranges(
            text=text,
            span=span,
            tokenizer=tokenizer,
            target_tokens=target_tokens,
            token_overlap=token_overlap,
        ):
            chunk_text = text[start_char:end_char]
            if not chunk_text.strip():
                continue
            section = span.section or _section_for_chunk(start_char, end_char, sections)
            chunks.append(
                DocumentChunk(
                    chunk_text=chunk_text,
                    start_char=start_char,
                    end_char=end_char,
                    section_type=section.section_type if section else "",
                    section_ref=section.section_ref if section else "",
                    section_start_char=section.start_char if section else None,
                    section_end_char=section.end_char if section else None,
                    section_content_hash=section.content_hash if section else "",
                )
            )
    return chunks


# A stretch of text that belongs to no section and holds only caption lines (a
# "İKİNCİ BÖLÜM" part caption, a Rehber sub-heading "Kredi Riskinde Önemli
# Artışın Değerlendirilmesi", a document title) is no provision: embedded on its
# own it matches a topic query almost perfectly and has nothing behind it.  It is
# merged into the chunk of the section that follows it.  A caption line is at
# most _CAPTION_MAX_CHARS long, opens with a capital letter or a number (or, set
# directly under another caption line, continues it in lower case), carries no
# sentence or clause punctuation at its end, is no table row and carries no
# bracketed editorial note such as "69–70 [Silinmiştir]".  An upper-case
# line must name a part, carry an outline number or run to several words, unless
# the stretch holds a part caption or opens the document; so a signature block
# ("İMZA" / "ADI-SOYADI" / "UNVAN") stays its own chunk.
_CAPTION_MAX_CHARS = 120
_CAPTION_MIN_UPPER_WORDS = 3
_CAPTION_MARKUP_RE = re.compile(r"^[#*\s]+|[*\s]+$")
_CAPTION_END_RE = re.compile(r"[.!?…;:,][”’\"']*$")
_CAPTION_PART_RE = re.compile(r"BÖLÜM|KISIM|\bBölüm\b|\bKısım\b")
_CAPTION_OUTLINE_RE = re.compile(r"^(?:[IVXLC]+|\d+(?:\.\d+)*)\.\s")
_CAPTION_NOTE_RE = re.compile(r"\[[^\]]*\]")


def _is_caption_line(line: str, *, continues: bool, short_upper_ok: bool) -> bool:
    line = _CAPTION_MARKUP_RE.sub("", line)
    if (
        not line
        or len(line) > _CAPTION_MAX_CHARS
        or not (line[:1].isupper() or line[:1].isdigit() or (continues and line[:1].islower()))
        or not any(ch.isalpha() for ch in line)
        or line.startswith("|")
        or _CAPTION_NOTE_RE.search(line)
        or _CAPTION_END_RE.search(line)
    ):
        return False
    if line.upper() == line:
        return bool(
            short_upper_ok
            or _CAPTION_PART_RE.search(line)
            or _CAPTION_OUTLINE_RE.match(line)
            or len(line.split()) >= _CAPTION_MIN_UPPER_WORDS
        )
    return True


def _is_caption_stretch(stretch: str, *, opens_document: bool = False) -> bool:
    lines = [line.strip() for line in stretch.strip().splitlines()]
    if not lines:
        return False
    short_upper_ok = opens_document or any(_CAPTION_PART_RE.search(line) for line in lines)
    for index, line in enumerate(lines):
        if not line:
            continue
        continues = index > 0 and bool(lines[index - 1])
        if not _is_caption_line(line, continues=continues, short_upper_ok=short_upper_ok):
            return False
    return True


def _chunk_spans(text: str, sections: list[DocumentSection]) -> list[_ChunkSpan]:
    if not sections:
        return [_ChunkSpan(start_char=0, end_char=len(text))] if text.strip() else []

    spans: list[_ChunkSpan] = []
    sorted_sections = sorted(sections, key=lambda section: (section.start_char, section.end_char))
    cursor = 0
    for index, section in enumerate(sorted_sections):
        caption_start: int | None = None
        if cursor < section.start_char and text[cursor : section.start_char].strip():
            uncovered = not any(
                max(cursor, other.start_char) < min(section.start_char, other.end_char) for other in sorted_sections
            )
            if uncovered and _is_caption_stretch(text[cursor : section.start_char], opens_document=cursor == 0):
                caption_start = cursor
            else:
                spans.append(_ChunkSpan(start_char=cursor, end_char=section.start_char))

        later_starts = [
            later.start_char
            for later in sorted_sections[index + 1 :]
            if section.start_char < later.start_char < section.end_char
        ]
        end_char = min(section.end_char, later_starts[0]) if later_starts else section.end_char
        if section.start_char < end_char and text[section.start_char : end_char].strip():
            # A caption stretch opens the following section's span: the chunk stays
            # one contiguous slice of the document and is bound to that section.
            span_start = section.start_char if caption_start is None else caption_start
            spans.append(_ChunkSpan(start_char=span_start, end_char=end_char, section=section))
        elif caption_start is not None:
            spans.append(_ChunkSpan(start_char=caption_start, end_char=section.start_char))
        cursor = max(cursor, end_char)

    if cursor < len(text) and text[cursor:].strip():
        spans.append(_ChunkSpan(start_char=cursor, end_char=len(text)))
    return spans


def _token_budget_ranges(
    text: str,
    span: _ChunkSpan,
    tokenizer,
    target_tokens: int,
    token_overlap: int,
) -> list[tuple[int, int]]:
    units = _text_units(text[span.start_char : span.end_char], span.start_char, tokenizer, target_tokens)
    ranges: list[tuple[int, int]] = []
    current: list[_TextUnit] = []
    current_tokens = 0

    for unit in units:
        if current and current_tokens + unit.token_count > target_tokens:
            ranges.append((current[0].start_char, current[-1].end_char))
            current = _overlap_units(current, token_overlap)
            current_tokens = sum(item.token_count for item in current)
            if current and current_tokens + unit.token_count > target_tokens:
                current = []
                current_tokens = 0

        current.append(unit)
        current_tokens += unit.token_count

    if current:
        ranges.append((current[0].start_char, current[-1].end_char))
    return ranges


def _text_units(text: str, absolute_start: int, tokenizer, target_tokens: int) -> list[_TextUnit]:
    units: list[_TextUnit] = []
    for match in _WORD_UNIT_RE.finditer(text):
        unit_text = match.group(0)
        start_char = absolute_start + match.start()
        token_count = max(1, _count_tokens(unit_text, tokenizer))
        if token_count <= target_tokens:
            units.append(
                _TextUnit(
                    start_char=start_char,
                    end_char=absolute_start + match.end(),
                    text=unit_text,
                    token_count=token_count,
                )
            )
        else:
            units.extend(_split_oversized_unit(unit_text, start_char, tokenizer, target_tokens))
    return units


def _split_oversized_unit(text: str, absolute_start: int, tokenizer, target_tokens: int) -> list[_TextUnit]:
    units: list[_TextUnit] = []
    cursor = 0
    while cursor < len(text):
        lo = 1
        hi = len(text) - cursor
        best = 1
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = text[cursor : cursor + mid]
            if _count_tokens(candidate, tokenizer) <= target_tokens or mid == 1:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        chunk_text = text[cursor : cursor + best]
        units.append(
            _TextUnit(
                start_char=absolute_start + cursor,
                end_char=absolute_start + cursor + best,
                text=chunk_text,
                token_count=max(1, _count_tokens(chunk_text, tokenizer)),
            )
        )
        cursor += best
    return units


def _overlap_units(units: list[_TextUnit], token_overlap: int) -> list[_TextUnit]:
    if token_overlap <= 0:
        return []
    kept: list[_TextUnit] = []
    total = 0
    for unit in reversed(units):
        if total + unit.token_count > token_overlap:
            break
        kept.append(unit)
        total += unit.token_count
    return list(reversed(kept))


def _count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _model_reference(embedding_model: str = EMBEDDING_MODEL_NAME) -> tuple[str, dict]:
    """Return the embedding model reference and the offline-first load kwargs."""
    model_ref = EMBEDDING_MODEL_PATH if EMBEDDING_MODEL_PATH else embedding_model
    kwargs: dict = {"local_files_only": bool(EMBEDDING_MODEL_PATH), "trust_remote_code": False}
    if not EMBEDDING_MODEL_PATH and EMBEDDING_MODEL_REVISION:
        kwargs["revision"] = EMBEDDING_MODEL_REVISION
    return model_ref, kwargs


def _torch_device() -> str:
    """Pick the device once instead of loading the weights twice on CPU-only hosts."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - torch missing or a broken CUDA probe
        return "cpu"


def _load_torch_model(loader, model_ref: str, **model_kwargs):
    """Load a sentence-transformers model on the selected device, falling back to CPU."""
    device = _torch_device()
    try:
        return loader(model_ref, device=device, **model_kwargs), device
    except (RuntimeError, ValueError, AssertionError):
        if device == "cpu":
            raise
        return loader(model_ref, device="cpu", **model_kwargs), "cpu"


def _load_embedding_tokenizer(embedding_model: str = EMBEDDING_MODEL_NAME):
    model_ref, kwargs = _model_reference(embedding_model)
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_ref, use_fast=True, **kwargs)
    except Exception as exc:
        logger.error(
            "Embedding tokenizer load failed (%s source)",
            "local" if EMBEDDING_MODEL_PATH else "remote",
            extra={"error_type": type(exc).__name__},
        )
        raise RetrievalProfileError("The embedding tokenizer could not be loaded.") from None


def _section_for_chunk(start_char: int, end_char: int, sections: list[DocumentSection]) -> DocumentSection | None:
    for section in sections:
        if section.start_char <= start_char < section.end_char:
            return section
    overlapping = [
        section for section in sections if max(start_char, section.start_char) < min(end_char, section.end_char)
    ]
    if not overlapping:
        return None
    return max(
        overlapping,
        key=lambda section: min(end_char, section.end_char) - max(start_char, section.start_char),
    )


# -- Query helpers ------------------------------------------------------------


def _turkish_lower(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


def _turkish_upper(text: str) -> str:
    return text.replace("i", "İ").replace("ı", "I").upper()


@dataclass(frozen=True)
class QueryVocabulary:
    """Word counts of the chunk texts (lower-cased with Turkish rules); per folded form, its most frequent
    surface form where that form differs from the folded one and no second such form is comparably common;
    and every folded form that occurs."""

    counts: dict[str, int] = field(default_factory=dict)
    surface: dict[str, str] = field(default_factory=dict)
    folded: frozenset[str] = frozenset()
    upper_only: frozenset[str] = frozenset()
    """Lower-cased words that the chunk texts spell only in upper case (codes such as BDDK, TFRS, SYR)."""


def _build_query_vocabulary(texts) -> QueryVocabulary:
    spelled = Counter(_VOCABULARY_WORD_RE.findall("\n".join(texts)))
    counts: Counter[str] = Counter()
    in_capitals: set[str] = set()
    otherwise: set[str] = set()
    for token, count in spelled.items():
        word = _turkish_lower(token)
        counts[word] += count
        (in_capitals if len(token) >= 2 and token.isupper() else otherwise).add(word)
    spellings: dict[str, list[tuple[int, str]]] = {}
    folded_forms: set[str] = set()
    for word, count in counts.items():
        folded = normalize_search_text(word)
        folded_forms.add(folded)
        if folded != word:
            spellings.setdefault(folded, []).append((count, word))
    surface: dict[str, str] = {}
    for folded, forms in spellings.items():
        forms.sort(reverse=True)
        # "kismi" can be "kısmi" or "kısmı", "ayni" "aynı" or "aynî": two comparably common Turkish
        # spellings leave the typed word as it is.
        if len(forms) > 1 and forms[1][0] >= _RESTORE_AMBIGUOUS_SHARE * forms[0][0]:
            continue
        surface[folded] = forms[0][1]
    return QueryVocabulary(
        counts=dict(counts),
        surface=surface,
        folded=frozenset(folded_forms),
        upper_only=frozenset(in_capitals - otherwise),
    )


def _looks_turkish(text: str, vocabulary: QueryVocabulary) -> bool:
    """At least half of the words of three or more letters occur in the corpus vocabulary in some form."""
    words = [word for word in _QUERY_WORD_RE.findall(text) if len(word) >= 3]
    known = sum(1 for word in words if normalize_search_text(word) in vocabulary.folded)
    return 2 * known >= len(words)


def _restored_word(word: str, vocabulary: QueryVocabulary) -> str:
    if len(word) >= 2 and word.isupper():
        return word  # an abbreviation inside a mixed-case query
    lowered = word.lower()
    candidate = vocabulary.surface.get(normalize_search_text(lowered))
    if candidate is None or candidate == lowered:
        return word
    count = vocabulary.counts.get(candidate, 0)
    if count < _RESTORE_MIN_COUNT or vocabulary.counts.get(lowered, 0) >= _RESTORE_MAX_TYPED_SHARE * count:
        return word
    if word[:1].isupper():
        return _turkish_upper(candidate[:1]) + candidate[1:]
    return candidate


def normalize_query(query: str, vocabulary: QueryVocabulary | None) -> str:
    """The query as it is embedded: NFC; an all-upper-case query with at least two words of four or more
    letters lower-cased (Turkish rules when it has a Turkish letter), except its glossary abbreviations and
    the words the corpus spells only in upper case; in a query without any Turkish letter whose words are mostly in the corpus vocabulary,
    each word typed without its Turkish letters restored from that vocabulary when one spelling clearly
    dominates. A query that has a Turkish letter is otherwise left as typed, and so is an all-caps word
    inside a mixed-case query.
    """
    text = unicodedata.normalize("NFC", query)
    words = [word for word in _QUERY_WORD_RE.findall(text) if len(word) >= 2]
    has_turkish_letter = any(char in _TURKISH_LETTERS for char in text)
    long_words = sum(1 for word in words if len(word) >= 4)
    if words and all(word.isupper() for word in words) and long_words >= _SHOUTED_MIN_LONG_WORDS:
        lower = _turkish_lower if has_turkish_letter else str.lower

        def lowered(match: re.Match[str]) -> str:
            word = match.group(0)
            if _query_abbreviation_at(match) is not None or (
                vocabulary is not None and _turkish_lower(word) in vocabulary.upper_only
            ):
                return word  # an abbreviation in a shouted sentence
            return lower(word)

        text = _QUERY_WORD_RE.sub(lowered, text)
    if has_turkish_letter or vocabulary is None or not _looks_turkish(text, vocabulary):
        return text
    return _QUERY_WORD_RE.sub(lambda match: _restored_word(match.group(0), vocabulary), text)


def _query_abbreviation_at(match: re.Match[str]) -> GlossaryEntry | None:
    """The glossary entry of a word match of the query, by the rule of ``query_abbreviation`` (in capitals,
    a whole word, no abbreviating full stop: never Turkish "bkz.")."""
    return query_abbreviation(match.group(0), match.string[match.end() :], match.string[: match.start()])


def _glossary_entries(query: str) -> list[GlossaryEntry]:
    """The glossary entries whose abbreviation is a word of the query as ``query_abbreviation`` reads it, in
    query order, each once; an entry whose expansion the query already contains is left out."""
    folded_query = " ".join(re.findall(r"\w+", normalize_search_text(query)))
    entries: dict[str, GlossaryEntry] = {}
    for match in _QUERY_WORD_RE.finditer(query):
        if (entry := _query_abbreviation_at(match)) is not None:
            if f" {normalize_search_text(entry.expansion)} " in f" {folded_query} ":
                continue
            entries.setdefault(entry.abbreviation, entry)
    return list(entries.values())


def _expanded_query(query: str, entries: list[GlossaryEntry]) -> str:
    """The query with each glossary abbreviation replaced by its expansion (for the strict keyword pass)."""
    by_abbreviation = {entry.abbreviation: entry.expansion for entry in entries}

    def expanded(match: re.Match[str]) -> str:
        entry = _query_abbreviation_at(match)
        return by_abbreviation.get(entry.abbreviation, match.group(0)) if entry is not None else match.group(0)

    return _QUERY_WORD_RE.sub(expanded, query)


def _has_content_word(query: str) -> bool:
    """A word that is neither a function word nor a single letter."""
    return any(
        token not in _QUERY_FUNCTION_WORDS and not (len(token) == 1 and token.isalpha())
        for token in re.findall(r"\w+", normalize_search_text(query))
    )


class SearchHits(list):
    """Search hits plus how the query was read.

    ``searched_as`` is the normalised query when it differs from the typed one; ``expansions`` lists the
    (abbreviation, expansion) pairs of the glossary that were applied.
    """

    searched_as: str | None = None
    expansions: list[tuple[str, str]] = []
    no_content_words: bool = False


def _date_pattern(date: str) -> re.Pattern[str]:
    """A query date as it can stand in a chunk: "31.12.2019" also as "31/12/2019" or "31-12-2019"."""
    day_month_year = re.fullmatch(r"(\d{2})\.(\d{2})\.(\d{4})", date)
    if day_month_year is None:
        return re.compile(rf"(?<![\d/]){re.escape(date)}(?![\d/])")
    day, month, year = (part.lstrip("0") or "0" for part in day_month_year.groups())
    return re.compile(rf"(?<![\d./-])0?{day}[./-]0?{month}[./-]{day_month_year.group(3)}(?!\d)")


def _hit_contains_reference(hit: dict, refs: LegalRefs) -> bool:
    """Whether the chunk itself holds a legal reference of the query.

    A section reference counts when the chunk text names it with its kind word ("MADDE 9", "9 uncu
    madde") or the chunk is bound to that section; a decision number when the chunk has "<n> sayılı" or
    the document carries that number; a date when the chunk has that date. Separate tokens that happen to
    match ("madde" and a "9" inside "20/9/2017") do not count.
    """
    text = str(hit.get("chunk_text") or hit.get("snippet") or "")
    if refs.sections:
        bound = (str(hit.get("section_type") or ""), canonical_section_ref(str(hit.get("section_ref") or "")))
        named = set(parse_legal_refs(text).sections) | {bound}
        if any(ref in named for ref in refs.sections):
            return True
    if refs.decision_numbers:
        folded = turkish_casefold(text)
        for number in refs.decision_numbers:
            if str(hit.get("decision_number") or "") == number or re.search(rf"(?<!\d){number}\s+sayılı", folded):
                return True
    return any(_date_pattern(date).search(text) for date in refs.dates)


# A section reference as the corpus prints it: the kind word, then the number ("md. 9" -> "madde 9").
_SECTION_KIND_WORDS = {
    "madde": "madde",
    "gecici_madde": "geçici madde",
    "ek_madde": "ek madde",
    "ilke": "ilke",
    "paragraf": "paragraf",
    "ek": "ek",
}


def _canonical_reference_query(query: str) -> str:
    """The query with every section reference that ``parse_legal_refs`` recognises written as the corpus
    prints it ("md. 9", "m. 9", "9 uncu maddesi" -> "madde 9"; "EK-1" -> "ek 1"), for the keyword passes;
    a query without one is returned unchanged."""
    parts: list[str] = []
    end = 0
    for start, stop, section_type, ref in parse_legal_refs(query).section_spans:
        if start < end:
            continue  # overlaps a reference already rewritten
        parts.append(query[end:start])
        parts.append(f"{_SECTION_KIND_WORDS[section_type]} {ref}")
        end = stop
    if not parts:
        return query
    parts.append(query[end:])
    return "".join(parts)


# A percentage ("%12", "% 0,90", "yüzde 12") and a lettered reference ("66/A") in the search-normalised
# query. The chunk index splits "%12" into the token 12 and "66/A" into 66 and a: the strict pass searches
# the parts of each as one phrase, and a percentage is checked on the chunk itself (_has_percent_figures).
_PERCENT_FIGURE_RE = re.compile(r"(?:%\s*|\byuzde\s+)(\d+(?:[.,]\d+)*)")
_LETTERED_REFERENCE_RE = re.compile(r"\b(\d+)\s*/\s*([^\W\d_])(?!\w)")
_WORD_TOKEN_RE = re.compile(r"\w+")


def _strict_fts_query(query: str) -> str | None:
    """The strict all-words MATCH expression (``fts_match_query``), except that the number of a percentage
    is required without "%" or "yüzde" (the chunk may write either; the index keeps neither) and a lettered
    reference ("66/A") is one phrase ("66 a"); a query with neither gives ``fts_match_query`` exactly."""
    text = normalize_search_text(query)
    spans = [(match.start(), match.end(), match.group(1)) for match in _PERCENT_FIGURE_RE.finditer(text)]
    spans += [(match.start(), match.end(), match.group(0)) for match in _LETTERED_REFERENCE_RE.finditer(text)]
    if not spans:
        return fts_match_query(query)
    terms: list[str] = []
    end = 0
    for start, stop, phrase in sorted(spans):
        if start < end:
            continue
        terms.extend(f'"{token}"' for token in _WORD_TOKEN_RE.findall(text[end:start]))
        terms.append('"' + " ".join(_WORD_TOKEN_RE.findall(phrase)) + '"')
        end = stop
    terms.extend(f'"{token}"' for token in _WORD_TOKEN_RE.findall(text[end:]))
    return " ".join(terms) or None


def _percent_figures(query: str) -> list[str]:
    """The numbers of the percentages of the query ("%12", "yüzde 0,90" -> "12", "0,90")."""
    return list(dict.fromkeys(match.group(1) for match in _PERCENT_FIGURE_RE.finditer(normalize_search_text(query))))


def _has_percent_figures(hit: dict, figures: list[str]) -> bool:
    """Whether the chunk (or its document title) holds every percentage of the query, written with "%" or
    "yüzde" right before the number ("%12", "% 12", "yüzde 12"; a decimal comma or point either way)."""
    text = normalize_search_text(f"{hit.get('title') or ''} {hit.get('chunk_text') or hit.get('snippet') or ''}")
    for figure in figures:
        number = "[.,]".join(re.escape(part) for part in re.split(r"[.,]", figure))
        if not re.search(rf"(?:%\s*|\byuzde\s+){number}(?![.,]?\d)", text):
            return False
    return True


def _relaxed_fts_query(query: str) -> str | None:
    """Loose fallback: OR of prefix-matched content words when the strict AND query is empty."""
    terms: list[str] = []
    seen: set[str] = set()
    for match in _FTS_TOKEN_RE.finditer(query):
        term = normalize_search_text(match.group(0))
        if match.group(0).lower() in _FTS_RELAXED_STOPWORDS or term in seen:
            continue
        seen.add(term)
        terms.append(term)
        if len(terms) >= _FTS_RELAXED_MAX_TERMS:
            break

    if len(terms) < _FTS_RELAXED_MIN_TERMS:
        return None
    return " OR ".join(f'"{term}"*' for term in terms)


def _normalize_search_text(text: str) -> str:
    return normalize_search_text(text)


def _stem_search_term(term: str) -> str:
    for suffix in _TURKISH_STEM_SUFFIXES:
        if len(term) > len(suffix) + 3 and term.endswith(suffix):
            return term[: -len(suffix)]
    return term


def _phrase_match_count(query: str, text: str) -> int:
    normalized_query = _normalize_search_text(query)
    normalized_text_terms = [
        _stem_search_term(match.group(0)) for match in re.finditer(r"[0-9a-z]{3,}", _normalize_search_text(text))
    ]
    normalized_text = " ".join(normalized_text_terms)
    stopwords = {_normalize_search_text(term) for term in _FTS_RELAXED_STOPWORDS}
    terms = [
        _stem_search_term(match.group(0))
        for match in re.finditer(r"[0-9a-z]{3,}", normalized_query)
        if match.group(0) not in stopwords
    ]

    phrases: set[str] = set()
    for size in (3, 2):
        for index in range(0, len(terms) - size + 1):
            phrases.add(" ".join(terms[index : index + size]))

    return sum(1 for phrase in phrases if phrase in normalized_text)


def _split_by_former_gap(semantic_hits: list[dict], gate_factor: float) -> tuple[list[dict], list[dict]]:
    """Split ordered semantic hits into those the former score-gap cut kept and those it dropped.

    Used only to order the hits; nothing is dropped.
    """
    if not semantic_hits:
        return [], []

    def gap_score(hit: dict) -> float:
        return round(hit["ordering_score"] * gate_factor, _SCORE_DECIMAL_PLACES)

    top = gap_score(semantic_hits[0])
    kept = [h for h in semantic_hits if top - gap_score(h) <= _GAP_ORDER_THRESHOLD]
    admitted = [h for h in semantic_hits if top - gap_score(h) > _GAP_ORDER_THRESHOLD]
    return kept, admitted


def _shown_as_strong_or_moderate(hit: dict) -> bool:
    return hit["relevance"] >= _MEDIUM_CONFIDENCE_THRESHOLD


def _keyword_only_hits(fts_hits: list[dict], shown_doc_ids: set[str]) -> list[dict]:
    """Strict all-words FTS matches of documents without a semantic hit, in BM25 order.

    They follow the strong and moderate semantic hits, precede the weak ones, and carry no semantic
    score; hits of the relaxed OR fallback
    (only some query words) are never kept this way.
    """
    hits: list[dict] = []
    for fts_hit in fts_hits:
        if fts_hit.get("fts_relaxed") or fts_hit["doc_id"] in shown_doc_ids:
            continue
        hit = dict(fts_hit)
        hit.update(
            match_type="fts",
            relevance=0.0,
            semantic_relevance=0.0,
            ordering_score=0.0,
            relevance_clamped=False,
            confidence="low",
        )
        hits.append(hit)
    return sorted(hits, key=lambda h: h.get("fts_rank", 0.0), reverse=True)


def _quality_metadata(text: str, doc_id: str) -> dict:
    quality = assess_markdown_quality(text or "", document_id=doc_id)
    return {"quality_label": quality.label, "quality_flags": quality.flags}


async def _document_quality_by_ids(db: CorpusDB, doc_ids: list[str]) -> dict[str, dict]:
    """Assess quality from full stored documents, not the matching chunk, on a worker thread."""
    unique_ids = list(dict.fromkeys(doc_ids))
    if not unique_ids:
        return {}
    placeholders = ",".join("?" for _ in unique_ids)
    rows = await db.run(
        lambda conn: conn.execute(
            f"SELECT document_id, markdown_content FROM documents WHERE document_id IN ({placeholders})",
            unique_ids,
        ).fetchall()
    )
    content_by_id = {row["document_id"]: row["markdown_content"] or "" for row in rows}
    return await asyncio.to_thread(
        lambda: {doc_id: _quality_metadata(content_by_id.get(doc_id, ""), doc_id) for doc_id in unique_ids}
    )


def _document_metadata_from_row(row: sqlite3.Row) -> dict:
    return {
        "decision_date": row["decision_date"] or "",
        "decision_number": row["decision_number"] or "",
        "source_url": row["source_url"] or "",
    }


def _section_metadata_from_row(row: sqlite3.Row) -> dict:
    return {
        "section_type": row["section_type"] or "",
        "section_ref": row["section_ref"] or "",
        "section_start_char": row["section_start_char"],
        "section_end_char": row["section_end_char"],
        "section_content_hash": row["section_content_hash"] or "",
    }


def _embedding_blob(vector) -> bytes:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    if array.shape[0] != EMBEDDING_DIMENSION or not np.all(np.isfinite(array)) or not np.any(array):
        raise VectorIndexConsistencyError("An embedding vector is malformed.")
    return array.astype(_EMBEDDING_DTYPE).tobytes()


def chunk_digest(texts: list[str]) -> str:
    """Identity of a chunk list: exported embeddings are reused only for exactly these texts."""
    digest = hashlib.sha256()
    for text in texts:
        digest.update(f"{len(text)}:".encode())
        digest.update(text.encode())
    return digest.hexdigest()


@dataclass
class _VectorIndex:
    """Every current chunk vector in memory, aligned with its row identity."""

    token: tuple[int, int]
    rowids: np.ndarray
    doc_ids: list[str]
    categories: np.ndarray
    matrix: np.ndarray
    vocabulary: QueryVocabulary | None = None

    @classmethod
    def load(cls, conn: sqlite3.Connection, token: tuple[int, int]) -> _VectorIndex:
        rows = conn.execute(
            "SELECT c.rowid AS rowid, c.doc_id, d.category, c.embedding FROM chunks AS c "
            "JOIN documents AS d ON d.document_id = c.doc_id AND d.content_hash = c.content_hash "
            "ORDER BY c.rowid"
        ).fetchall()
        width = EMBEDDING_DIMENSION * np.dtype(_EMBEDDING_DTYPE).itemsize
        if any(len(row["embedding"]) != width for row in rows):
            raise VectorIndexConsistencyError("A stored embedding has the wrong size for this runtime.")
        if rows:
            matrix = np.frombuffer(b"".join(row["embedding"] for row in rows), dtype=_EMBEDDING_DTYPE)
            matrix = matrix.reshape(len(rows), EMBEDDING_DIMENSION).astype(np.float32)
        else:
            matrix = np.zeros((0, EMBEDDING_DIMENSION), dtype=np.float32)
        return cls(
            token=token,
            rowids=np.array([row["rowid"] for row in rows], dtype=np.int64),
            doc_ids=[row["doc_id"] for row in rows],
            categories=np.array([row["category"] or "" for row in rows], dtype=object),
            matrix=matrix,
            vocabulary=_build_query_vocabulary(
                row["chunk_text"] or ""
                for row in conn.execute(
                    "SELECT c.chunk_text FROM chunks AS c "
                    "JOIN documents AS d ON d.document_id = c.doc_id AND d.content_hash = c.content_hash"
                )
            ),
        )


class VectorStore:
    """Chunk index with exact cosine search, FTS5 keyword search and RRF fusion.

    Usage::

        store = VectorStore(db)
        await store.add_document(doc_id="1291", title="...", content="...")
        results = await store.search("sermaye yeterliliği hesaplama", limit=10)
    """

    def __init__(
        self,
        db: CorpusDB,
        embedding_model: str = EMBEDDING_MODEL_NAME,
        *,
        embed_fn=None,
        tokenizer=None,
    ) -> None:
        self._db = db
        self._embedding_model = embedding_model
        # Test seams: ``embed_fn`` (texts -> unit vectors) replaces the model and
        # ``tokenizer`` (``encode(text, add_special_tokens=False)``) the chunk tokenizer.
        self._embed_fn = embed_fn
        self._chunk_tokenizer_fn = tokenizer
        self._embedding_load_lock = threading.Lock()
        self._semantic_search_ready = False
        self._semantic_search_failure_type: str | None = None
        self._index: _VectorIndex | None = None
        self._index_lock = threading.Lock()

    @property
    def semantic_search_ready(self) -> bool:
        """Whether the complete read-only semantic-search path passed its startup probe."""
        return self._semantic_search_ready and self._semantic_search_failure_type is None

    def _raise_if_semantic_search_unavailable(self) -> None:
        if self._semantic_search_failure_type is not None:
            raise SemanticSearchUnavailableError("Semantic search is unavailable in this runtime.")

    async def close(self) -> None:
        logger.info("VectorStore closed")

    # -- Model loading -----------------------------------------------------------

    def _ensure_embeddings(self) -> None:
        """Lazy-load the embedding model on first search/add."""
        if self._embed_fn is not None:
            return
        with self._embedding_load_lock:
            if self._embed_fn is not None:
                return
            from sentence_transformers import SentenceTransformer

            model_ref, model_kwargs = _model_reference(self._embedding_model)
            model_kwargs["backend"] = "torch"
            model_source = "local" if EMBEDDING_MODEL_PATH else "remote"
            cache_key = (SentenceTransformer, model_ref, tuple(sorted(model_kwargs.items())))
            with _LOADED_EMBEDDING_MODELS_LOCK:
                loaded_model = _LOADED_EMBEDDING_MODELS.get(cache_key)
                if loaded_model is None:
                    logger.info("Loading embedding model (%s source)", model_source)
                    loaded_model, device = _load_torch_model(SentenceTransformer, model_ref, **model_kwargs)
                    logger.info("Loaded embedding model (%s execution, %s source)", device, model_source)
                    if len(_LOADED_EMBEDDING_MODELS) >= _LOADED_EMBEDDING_MODELS_LIMIT:
                        _LOADED_EMBEDDING_MODELS.clear()
                    _LOADED_EMBEDDING_MODELS[cache_key] = loaded_model
            # sentence-transformers renamed get_sentence_embedding_dimension to
            # get_embedding_dimension; older releases have only the old name.
            dimension_of = getattr(loaded_model, "get_embedding_dimension", None)
            if dimension_of is None:
                dimension_of = loaded_model.get_sentence_embedding_dimension
            dimension = dimension_of()
            if dimension != EMBEDDING_DIMENSION:
                raise RuntimeError(f"Embedding model dimension must be {EMBEDDING_DIMENSION} for the corpus.")

            def encode(texts: list[str]):
                return loaded_model.encode(
                    texts,
                    batch_size=_EMBEDDING_ENCODE_BATCH_SIZE,
                    show_progress_bar=False,
                    output_value="sentence_embedding",
                    precision="float32",
                    convert_to_numpy=True,
                    convert_to_tensor=False,
                    normalize_embeddings=True,
                )

            self._embed_fn = encode

    def _chunk_tokenizer(self):
        if self._chunk_tokenizer_fn is None:
            self._chunk_tokenizer_fn = _load_embedding_tokenizer(self._embedding_model)
        return self._chunk_tokenizer_fn

    def load_chunk_tokenizer(self) -> None:
        """Load the chunk tokenizer now, so a build fails before it writes when the tokenizer is unavailable."""
        self._chunk_tokenizer()

    async def _embed(self, texts: list[str], prefix: str = _PASSAGE_PREFIX) -> np.ndarray:
        """Embed texts (unit vectors, float32) on a worker thread."""
        if prefix not in {_PASSAGE_PREFIX, _QUERY_PREFIX}:
            raise RetrievalProfileError("Unknown embedding prefix.")
        self._raise_if_semantic_search_unavailable()
        prefixed = [f"{prefix}{_PREFIX_SEPARATOR}{text}" for text in texts]
        try:
            await asyncio.to_thread(self._ensure_embeddings)
            embeddings = await asyncio.to_thread(self._embed_fn, prefixed)
        except Exception as error:
            self._semantic_search_ready = False
            self._semantic_search_failure_type = type(error).__name__
            logger.exception("Semantic embedding path failed (error_type=%s)", self._semantic_search_failure_type)
            raise SemanticSearchUnavailableError("Semantic search is unavailable in this runtime.") from None
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim != 2 or array.shape != (len(texts), EMBEDDING_DIMENSION):
            raise VectorIndexConsistencyError("The embedding output has the wrong shape.")
        return array

    # -- Add documents -----------------------------------------------------------

    async def add_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        category: str = "",
        decision_date: str = "",
        decision_number: str = "",
        source_url: str = "",
        *,
        cached_embeddings=None,
    ) -> int:
        """Embed and publish a document's chunks. Returns the number of chunks.

        ``cached_embeddings(digest, count)`` may return the vectors exported earlier
        for exactly these chunk texts (see ``chunk_digest``); a seed whose chunks and
        model are unchanged then builds without loading the embedding model.
        """
        if not content.strip():
            return 0
        tokenizer = await asyncio.to_thread(self._chunk_tokenizer)
        chunks = _chunk_document(doc_id, content, tokenizer=tokenizer)
        if not chunks:
            return 0
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        texts = [chunk.chunk_text for chunk in chunks]
        embeddings = cached_embeddings(chunk_digest(texts), len(texts)) if cached_embeddings else None
        if embeddings is None or embeddings.shape != (len(texts), EMBEDDING_DIMENSION):
            embeddings = await self._embed(texts)
        rows = [
            (
                doc_id,
                index,
                title,
                chunk.chunk_text,
                chunk.start_char,
                chunk.end_char,
                chunk.section_type,
                chunk.section_ref,
                chunk.section_start_char,
                chunk.section_end_char,
                chunk.section_content_hash,
                content_hash,
                _embedding_blob(vector),
                normalize_search_text(title),
                normalize_search_text(chunk.chunk_text),
            )
            for index, (chunk, vector) in enumerate(zip(chunks, embeddings, strict=True))
        ]

        def _publish(conn: sqlite3.Connection) -> int:
            stored = conn.execute("SELECT content_hash FROM documents WHERE document_id = ?", (doc_id,)).fetchone()
            if stored is None or stored["content_hash"] != content_hash:
                raise VectorIndexConsistencyError(
                    "Vector chunks were not published because the stored document hash does not match."
                )
            meta = {row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM meta")}
            expected = {"embedding_model": self._embedding_model, "embedding_dimension": str(EMBEDDING_DIMENSION)}
            for key, value in expected.items():
                if key in meta and meta[key] != value:
                    raise RetrievalProfileError(
                        "The corpus file was built with a different embedding model; "
                        "delete it and run `bddk-mcp build` again."
                    )
                conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)", (key, value))
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.executemany(
                "INSERT INTO chunks (doc_id, chunk_index, title, chunk_text, start_char, end_char, section_type, "
                "section_ref, section_start_char, section_end_char, section_content_hash, content_hash, embedding, "
                "title_norm, chunk_text_norm) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            return len(rows)

        written = await self._db.transaction(_publish)
        logger.debug("Added document to vector index (chunks=%d)", written)
        return written

    # -- Index --------------------------------------------------------------------

    def _current_index(self) -> _VectorIndex:
        """Return the in-memory vector index, reloading it after any corpus write."""
        token = self._db.change_token()
        with self._index_lock:
            if self._index is None or self._index.token != token:
                self._index = self._db.run_sync(lambda conn: _VectorIndex.load(conn, token))
                logger.info("Vector index loaded (chunks=%d)", len(self._index.doc_ids))
            return self._index

    # -- Search: public API ----------------------------------------------------

    async def search(
        self,
        query: str,
        limit: int = 10,
        category: str | None = None,
    ) -> list[dict]:
        """Search documents with hybrid (dense + keyword) retrieval.

        The query is normalised once (``normalize_query``) for the embedding and the phrase matching;
        the returned ``SearchHits`` say what was searched when that differs from the typed query. A query
        word that is an abbreviation of the reviewed glossary (``search_glossary``), typed in capitals and
        not abbreviated by a full stop (never Turkish "bkz."), gets its expansion
        appended to the embedded query, and the strict keyword pass also runs with the expansion in its
        place.
        Document quality is assessed once, for the returned hits only.
        """
        self._raise_if_semantic_search_unavailable()
        if not _has_content_word(query):
            empty = SearchHits()
            empty.no_content_words = True
            return empty
        index = await asyncio.to_thread(self._current_index)
        searched = normalize_query(query, index.vocabulary)
        entries = _glossary_entries(searched)
        if entries:
            expansions = " ".join(dict.fromkeys(entry.expansion for entry in entries))
            found = await self._hybrid_search(
                searched,
                limit,
                category,
                embed_query=f"{searched} {expansions}",
                fts_alternative_query=_expanded_query(searched, entries),
            )
        else:
            found = await self._hybrid_search(searched, limit, category)
        if number := _DECISION_NUMBER_QUERY_RE.match(normalize_search_text(searched)):
            decisions = await self._decision_number_hits(number.group(1), category)
            decision_ids = {hit["doc_id"] for hit in decisions}
            found = (decisions + [hit for hit in found if hit["doc_id"] not in decision_ids])[:limit]
        hits = SearchHits(found)
        hits.searched_as = searched if searched != query else None
        hits.expansions = [(entry.abbreviation, entry.expansion) for entry in entries]
        quality_by_id = await _document_quality_by_ids(self._db, [hit["doc_id"] for hit in hits])
        for hit in hits:
            hit.update(quality_by_id[hit["doc_id"]])
        return hits

    async def _decision_number_hits(self, number: str, category: str | None) -> list[dict]:
        """The stored documents whose decision number is ``number``, each with its first current chunk."""
        where = "d.decision_number = ?" + (" AND d.category = ?" if category else "")
        params = [number, category] if category else [number]
        rows = await self._db.run(
            lambda conn: conn.execute(
                "SELECT c.doc_id, d.title, d.category, d.decision_date, d.decision_number, d.source_url, "
                "c.chunk_text, c.section_type, c.section_ref, c.section_start_char, c.section_end_char, "
                "c.section_content_hash FROM chunks AS c "
                "JOIN documents AS d ON d.document_id = c.doc_id AND d.content_hash = c.content_hash "
                f"WHERE {where} AND c.chunk_index = ("
                "SELECT MIN(x.chunk_index) FROM chunks AS x WHERE x.doc_id = c.doc_id AND x.content_hash = c.content_hash"
                ") ORDER BY c.doc_id",
                params,
            ).fetchall()
        )
        return [
            {
                "doc_id": row["doc_id"],
                "title": row["title"] or "",
                "category": row["category"] or "",
                **_document_metadata_from_row(row),
                "snippet": (row["chunk_text"] or "")[:_SEARCH_SNIPPET_CHARS],
                "fts_rank": 0.0,
                "relevance": 0.0,
                "semantic_relevance": 0.0,
                "ordering_score": 0.0,
                "relevance_clamped": False,
                "match_type": "decision_number",
                "decision_number_match": True,
                "confidence": "low",
                **_section_metadata_from_row(row),
            }
            for row in rows
        ]

    async def assert_semantic_search_ready(self) -> None:
        """Fail closed unless model loading, encoding and corpus retrieval all work.

        An empty result is healthy: this probe verifies the executable retrieval
        path, not the relevance of a particular corpus snapshot.
        """
        if self._semantic_search_ready:
            return
        try:
            await self.search("bankacılık düzenlemeleri", limit=1)
        except SemanticSearchUnavailableError:
            raise
        except Exception as error:
            logger.exception("Semantic search readiness probe failed (error_type=%s)", type(error).__name__)
            raise SemanticSearchReadinessError("Semantic search readiness probe failed.") from None
        self._semantic_search_ready = True

    # -- Dense retrieval --------------------------------------------------------

    async def _vector_search(
        self,
        query: str,
        limit: int = 10,
        category: str | None = None,
        fetch_limit: int | None = None,
    ) -> list[dict]:
        """Exact cosine similarity over every current chunk vector."""
        query_embedding = (await self._embed([query], prefix=_QUERY_PREFIX))[0]
        if fetch_limit is None:
            fetch_limit = min(limit * 5, 100)
        index = await asyncio.to_thread(self._current_index)
        if not index.doc_ids:
            return []
        scores = index.matrix @ query_embedding
        if category:
            scores = np.where(index.categories == category, scores, -np.inf)
        count = min(fetch_limit, scores.shape[0])
        candidate_positions = np.argpartition(-scores, count - 1)[:count]
        candidate_positions = candidate_positions[np.argsort(-scores[candidate_positions], kind="stable")]
        candidate_positions = [int(position) for position in candidate_positions if np.isfinite(scores[position])]
        if not candidate_positions:
            return []
        rowids = [int(index.rowids[position]) for position in candidate_positions]
        placeholders = ",".join("?" for _ in rowids)
        rows = await self._db.run(
            lambda conn: conn.execute(
                f"SELECT c.rowid AS rowid, c.doc_id, d.title, d.category, d.decision_date, d.decision_number, "
                f"d.source_url, c.chunk_text, "
                f"c.section_type, c.section_ref, c.section_start_char, c.section_end_char, c.section_content_hash "
                f"FROM chunks AS c JOIN documents AS d ON d.document_id = c.doc_id "
                f"WHERE c.rowid IN ({placeholders})",
                rowids,
            ).fetchall()
        )
        row_by_id = {row["rowid"]: row for row in rows}

        # Deduplicate by doc_id, keep best score
        seen: dict[str, dict] = {}
        for position, rowid in zip(candidate_positions, rowids, strict=True):
            row = row_by_id.get(rowid)
            if row is None:
                continue
            did = row["doc_id"]
            distance = float(1.0 - scores[position])
            if did not in seen or distance < seen[did]["distance"]:
                seen[did] = {
                    "doc_id": did,
                    "title": row["title"] or "",
                    "category": row["category"] or "",
                    **_document_metadata_from_row(row),
                    "snippet": (row["chunk_text"] or "")[:_SEARCH_SNIPPET_CHARS],
                    "distance": distance,
                    "relevance": round(1 - distance, _SCORE_DECIMAL_PLACES),
                    "semantic_relevance": round(1 - distance, _SCORE_DECIMAL_PLACES),
                    "fts_rank": 0.0,
                    "match_type": "vector",
                    **_section_metadata_from_row(row),
                }

        hits = sorted(seen.values(), key=lambda x: x["distance"])
        return hits[:limit]

    # -- Keyword retrieval -----------------------------------------------------

    async def _fts_search(
        self,
        query: str,
        limit: int = 50,
        category: str | None = None,
        alternative_query: str | None = None,
        document_limit: int | None = None,
    ) -> list[dict]:
        """FTS5 search on chunk text (plus title) with BM25 scoring.

        Hits from the relaxed OR fallback carry ``fts_relaxed=True``: they
        matched only some query words, so they never count as exact matches.
        ``alternative_query`` (the query with abbreviations expanded) runs as a
        second strict pass; a document keeps its best-ranked chunk of the two.
        With ``document_limit``, the strict matches are also fetched per document
        (best chunk of each) until that many documents are found; documents that
        only this pass found carry ``keyword_tail=True`` and serve only the
        keyword-only list, so the fused ranking sees the same candidates as before.
        Every pass searches the section references of the query as the corpus prints them
        (``_canonical_reference_query``).
        """
        query = _canonical_reference_query(query)
        if alternative_query:
            alternative_query = _canonical_reference_query(alternative_query)
        rows = await self._fts_rows(_strict_fts_query(query), limit=limit, category=category)
        if alternative_query:
            rows = [
                *rows,
                *await self._fts_rows(_strict_fts_query(alternative_query), limit=limit, category=category),
            ]
        relaxed = False
        if not rows and (relaxed_query := _relaxed_fts_query(query)):
            rows = await self._fts_rows(relaxed_query, limit=limit, category=category)
            relaxed = True

        seen: dict[str, dict] = {}
        for row in rows:
            did = row["doc_id"]
            rank = float(row["fts_rank"])
            if did not in seen or rank > seen[did]["fts_rank"]:
                seen[did] = {
                    "doc_id": did,
                    "title": row["title"] or "",
                    "category": row["category"] or "",
                    **_document_metadata_from_row(row),
                    "snippet": (row["chunk_text"] or "")[:_SEARCH_SNIPPET_CHARS],
                    # The whole chunk, for the legal-reference check of _hybrid_search; not returned.
                    "chunk_text": row["chunk_text"] or "",
                    "fts_rank": rank,
                    "semantic_relevance": 0.0,
                    "match_type": "fts",
                    "fts_relaxed": relaxed,
                    **_section_metadata_from_row(row),
                }

        if document_limit and not relaxed:
            document_rows = await self._fts_document_rows(
                _strict_fts_query(query), limit=document_limit, category=category
            )
            if alternative_query:
                document_rows += await self._fts_document_rows(
                    _strict_fts_query(alternative_query), limit=document_limit, category=category
                )
            tail: dict[str, dict] = {}
            for row in document_rows:
                did = row["doc_id"]
                rank = float(row["fts_rank"])
                if did in seen or (did in tail and tail[did]["fts_rank"] >= rank):
                    continue
                tail[did] = {
                    "doc_id": did,
                    "title": row["title"] or "",
                    "category": row["category"] or "",
                    **_document_metadata_from_row(row),
                    "snippet": (row["chunk_text"] or "")[:_SEARCH_SNIPPET_CHARS],
                    "chunk_text": row["chunk_text"] or "",
                    "fts_rank": rank,
                    "semantic_relevance": 0.0,
                    "match_type": "fts",
                    "fts_relaxed": False,
                    "keyword_tail": True,
                    **_section_metadata_from_row(row),
                }
            return sorted(seen.values(), key=lambda x: x["fts_rank"], reverse=True) + sorted(
                tail.values(), key=lambda x: x["fts_rank"], reverse=True
            )

        return sorted(seen.values(), key=lambda x: x["fts_rank"], reverse=True)

    async def _fts_document_rows(self, match: str | None, *, limit: int, category: str | None) -> list[sqlite3.Row]:
        """Strict FTS matches, the best-ranked chunk of each document, best documents first."""
        if match is None:
            return []
        category_clause = "AND d.category = ?" if category else ""
        params: list = [match, *([category] if category else []), limit]
        # bm25() cannot be used in a window, so the ranks are materialised first.
        sql = f"""
            WITH matched AS MATERIALIZED (
                SELECT c.rowid AS chunk_rowid, c.doc_id AS doc_id, -bm25(chunks_fts, 2.0, 1.0) AS fts_rank
                FROM chunks_fts
                JOIN chunks AS c ON c.rowid = chunks_fts.rowid
                JOIN documents AS d ON d.document_id = c.doc_id AND d.content_hash = c.content_hash
                WHERE chunks_fts MATCH ? {category_clause}
            ),
            best AS (
                SELECT chunk_rowid, fts_rank,
                       ROW_NUMBER() OVER (PARTITION BY doc_id ORDER BY fts_rank DESC, chunk_rowid) AS position
                FROM matched
            )
            SELECT c.doc_id, d.title, d.category, d.decision_date, d.decision_number, d.source_url, c.chunk_text,
                   c.section_type, c.section_ref, c.section_start_char, c.section_end_char,
                   c.section_content_hash, best.fts_rank AS fts_rank
            FROM best
            JOIN chunks AS c ON c.rowid = best.chunk_rowid
            JOIN documents AS d ON d.document_id = c.doc_id
            WHERE best.position = 1
            ORDER BY best.fts_rank DESC
            LIMIT ?
        """
        return await self._db.run(lambda conn: conn.execute(sql, params).fetchall())

    async def _fts_rows(self, match: str | None, *, limit: int, category: str | None) -> list[sqlite3.Row]:
        if match is None:
            return []
        where = ["chunks_fts MATCH ?"]
        params: list = [match]
        if category:
            where.append("d.category = ?")
            params.append(category)
        params.append(limit)
        sql = f"""
            SELECT c.doc_id, d.title, d.category, d.decision_date, d.decision_number, d.source_url, c.chunk_text,
                   c.section_type, c.section_ref, c.section_start_char, c.section_end_char,
                   c.section_content_hash, -bm25(chunks_fts, 2.0, 1.0) AS fts_rank
            FROM chunks_fts
            JOIN chunks AS c ON c.rowid = chunks_fts.rowid
            JOIN documents AS d ON d.document_id = c.doc_id AND d.content_hash = c.content_hash
            WHERE {" AND ".join(where)}
            ORDER BY fts_rank DESC
            LIMIT ?
        """
        return await self._db.run(lambda conn: conn.execute(sql, params).fetchall())

    # -- Hybrid search (RRF fusion) -------------------------------------------

    async def _hybrid_search(
        self,
        query: str,
        limit: int = 10,
        category: str | None = None,
        *,
        embed_query: str | None = None,
        fts_alternative_query: str | None = None,
    ) -> list[dict]:
        """Hybrid search: dense + sparse retrieval fused with RRF.

        ``embed_query`` replaces the query for the embedding only, and
        ``fts_alternative_query`` adds a second strict keyword pass (both used for
        glossary abbreviations).

        Each hit carries three scores. ``semantic_relevance`` is the raw cosine
        of the embedding model; hits below the semantic threshold are dropped.
        The internal ordering score (``ordering_score``: the raw cosine plus the
        lexical and phrase boosts) orders the hits. ``relevance`` is the
        displayed value: the ordering score mapped through the model's
        calibration and, when no keyword matches at all (the FTS gate),
        multiplied by ``_FTS_GATE_PENALTY``. Every hit that passes the threshold
        is kept, up to ``limit``, however far its score lies below the top hit;
        the hits within ``_GAP_ORDER_THRESHOLD`` of the top (the only ones the
        former score-gap cut returned) keep their former place and order, and the
        others follow them.
        Documents that match the strict all-words FTS query but have no
        semantic hit follow the strong and moderate semantic hits, ahead of the weak ones, as keyword matches
        (``match_type`` "fts", no semantic score), in BM25 order.
        """
        vector_hits, fts_hits = await asyncio.gather(
            self._vector_search(
                embed_query or query,
                limit=_HYBRID_VECTOR_CANDIDATE_LIMIT,
                category=category,
                fetch_limit=_HYBRID_VECTOR_FETCH_LIMIT,
            ),
            self._fts_search(
                query,
                limit=_HYBRID_FTS_CANDIDATE_LIMIT,
                category=category,
                alternative_query=fts_alternative_query,
                # Enough documents for the keyword-only list even when every semantic hit is another one.
                document_limit=2 * limit,
            ),
        )
        # A keyword-only hit for a percentage must hold the figure itself, not only its number (the index
        # cannot tell "%12" from a 12 inside a date).
        figures = _percent_figures(query)
        keyword_candidates = [hit for hit in fts_hits if not figures or _has_percent_figures(hit, figures)]
        # The documents found only by the per-document pass serve the keyword-only list alone.
        fts_hits = [hit for hit in fts_hits if not hit.get("keyword_tail")]
        query_refs = parse_legal_refs(query)
        exact_legal_query = bool(query_refs.sections or query_refs.decision_numbers or query_refs.dates)
        vector_by_doc = {hit["doc_id"]: hit for hit in vector_hits}
        fts_by_doc = {hit["doc_id"]: hit for hit in fts_hits}

        # FTS gate: with no keyword match at all the displayed relevance is
        # penalised; the raw cosine (and so the threshold) is not.
        gate_factor = _FTS_GATE_PENALTY if len(fts_hits) == 0 else 1.0
        if gate_factor != 1.0:
            logger.debug(
                "FTS gate: 0 keyword matches, applying %.0f%% penalty to displayed relevance",
                (1 - _FTS_GATE_PENALTY) * 100,
            )

        fused = self._rrf_fuse(vector_hits, fts_hits)
        for hit in fused:
            did = hit["doc_id"]
            hit["semantic_relevance"] = round(hit.get("relevance", 0.0), _SCORE_DECIMAL_PLACES)
            if did in fts_by_doc:
                hit["fts_rank"] = fts_by_doc[did].get("fts_rank", 0.0)
            else:
                hit.setdefault("fts_rank", 0.0)

            if did in vector_by_doc and did in fts_by_doc:
                hit["match_type"] = "hybrid"
            elif (
                did in fts_by_doc
                and exact_legal_query
                and not fts_by_doc[did].get("fts_relaxed")
                and _hit_contains_reference(fts_by_doc[did], query_refs)
            ):
                # Only the strict AND query of a chunk that holds the reference itself can
                # stand in for a semantic score; relaxed OR hits face the threshold like any
                # other FTS hit, and a strict hit without the reference is a keyword-only hit.
                hit["match_type"] = "fts_exact"
            elif did in fts_by_doc:
                hit["match_type"] = "fts"
            else:
                hit["match_type"] = "vector"

        if fts_hits:
            max_fts_rank = max((hit.get("fts_rank", 0.0) for hit in fts_hits), default=0.0)
            if max_fts_rank > 0:
                for hit in fused:
                    lexical_relevance = float(hit.get("fts_rank", 0.0)) / max_fts_rank
                    hit["lexical_relevance"] = round(lexical_relevance, _SCORE_DECIMAL_PLACES)
                    if hit.get("semantic_relevance", 0.0) > 0 and lexical_relevance > 0:
                        hit["relevance"] = round(
                            min(1.0, hit.get("relevance", 0.0) + (lexical_relevance * _LEXICAL_RELEVANCE_BOOST)),
                            _SCORE_DECIMAL_PLACES,
                        )
                    phrase_matches = _phrase_match_count(query, f"{hit.get('title', '')} {hit.get('snippet', '')}")
                    hit["phrase_match_count"] = phrase_matches
                    if hit.get("semantic_relevance", 0.0) > 0 and phrase_matches > 0:
                        hit["relevance"] = round(
                            min(
                                1.0,
                                hit.get("relevance", 0.0)
                                + min(_PHRASE_RELEVANCE_MAX, phrase_matches * _PHRASE_RELEVANCE_BOOST),
                            ),
                            _SCORE_DECIMAL_PLACES,
                        )

        calibration = relevance_calibration(self._embedding_model)
        for hit in fused:
            ordering_score = round(hit.get("relevance", 0.0), _SCORE_DECIMAL_PLACES)
            hit["ordering_score"] = ordering_score
            # At the ceiling the displayed value is clamped: hits there tie at 1.0 and are
            # ordered by the ordering score, which the displayed value no longer shows.
            hit["relevance_clamped"] = gate_factor == 1.0 and ordering_score >= calibration.ceiling
            hit["relevance"] = round(
                displayed_relevance(ordering_score, calibration) * gate_factor, _SCORE_DECIMAL_PLACES
            )

        threshold = semantic_threshold(self._embedding_model)
        fused = [h for h in fused if h["semantic_relevance"] >= threshold or h["match_type"] == "fts_exact"]

        # RRF order only breaks ties. Hits with a dense score are ordered by the
        # ordering score (the displayed relevance is monotonic in it, so a tie in
        # the displayed value keeps the ordering-score order). FTS-only exact
        # legal-reference matches have none, so they follow every hybrid hit in
        # BM25 order, ahead of vector-only hits.
        exact_hits = sorted(
            (h for h in fused if h["match_type"] == "fts_exact"), key=lambda h: h["fts_rank"], reverse=True
        )
        semantic_hits = [h for h in fused if h["match_type"] != "fts_exact"]
        semantic_hits.sort(key=lambda h: (h["relevance"], h["ordering_score"]), reverse=True)

        # The hits that the former score-gap cut kept (ordering score under the FTS gate within
        # _GAP_ORDER_THRESHOLD of the top) keep their former place and order; the hits it would have
        # cut are no longer dropped but follow them, so they never push a former hit out of the limit.
        kept_hits, admitted_hits = _split_by_former_gap(semantic_hits, gate_factor)
        if exact_hits:
            hybrid_hits = [h for h in kept_hits if h["match_type"] == "hybrid"]
            vector_only_hits = [h for h in kept_hits if h["match_type"] != "hybrid"]
            # The exact legal-reference hits and the hybrid hits before them keep this place.
            head = hybrid_hits + exact_hits
            rest = vector_only_hits + admitted_hits
        else:
            head, rest = [], kept_hits + admitted_hits
        # A document that contains every query word is not hidden behind hits labelled weak: the
        # remaining strong and moderate hits come first, then the keyword-only hits, then the weak
        # semantic hits in their order.
        listed = {h["doc_id"] for h in head + rest}
        fused = (
            head
            + [h for h in rest if _shown_as_strong_or_moderate(h)]
            + _keyword_only_hits(keyword_candidates, listed)
            + [h for h in rest if not _shown_as_strong_or_moderate(h)]
        )

        for h in fused:
            if h["relevance"] >= _HIGH_CONFIDENCE_THRESHOLD:
                h["confidence"] = "high"
            elif h["relevance"] >= _MEDIUM_CONFIDENCE_THRESHOLD:
                h["confidence"] = "medium"
            else:
                h["confidence"] = "low"

        fused = fused[:limit]
        for h in fused:
            h.pop("chunk_text", None)
        return fused

    def _rrf_fuse(self, vector_hits: list[dict], fts_hits: list[dict], k: int = _RRF_K) -> list[dict]:
        """Reciprocal Rank Fusion: RRF(d) = sum over systems of 1 / (k + rank)."""
        doc_data: dict[str, dict] = {}
        rrf_scores: dict[str, float] = {}

        for rank, hit in enumerate(vector_hits, 1):
            did = hit["doc_id"]
            rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (k + rank)
            if did not in doc_data:
                doc_data[did] = hit.copy()

        for rank, hit in enumerate(fts_hits, 1):
            did = hit["doc_id"]
            rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (k + rank)
            if did not in doc_data:
                doc_data[did] = hit.copy()

        ranked_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

        results = []
        for did in ranked_ids:
            entry = doc_data[did]
            entry["rrf_score"] = round(rrf_scores[did], _RRF_SCORE_DECIMAL_PLACES)
            # FTS-only hits have no cosine: leave relevance at 0.0 so the
            # semantic threshold drops them instead of faking a score.
            entry.setdefault("relevance", 0.0)
            results.append(entry)

        return results

    # -- Bookkeeping ------------------------------------------------------------

    async def has_document(self, doc_id: str) -> bool:
        """Whether current (hash-matching) chunks exist for the document."""
        row = await self._db.run(
            lambda conn: conn.execute(
                "SELECT 1 FROM chunks AS c JOIN documents AS d "
                "ON d.document_id = c.doc_id AND d.content_hash = c.content_hash WHERE c.doc_id = ? LIMIT 1",
                (doc_id,),
            ).fetchone()
        )
        return row is not None

    async def chunk_count(self) -> int:
        return await self._db.run(
            lambda conn: conn.execute(
                "SELECT COUNT(*) FROM chunks AS c JOIN documents AS d "
                "ON d.document_id = c.doc_id AND d.content_hash = c.content_hash"
            ).fetchone()[0]
        )

    async def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks for a document."""
        deleted = await self._db.transaction(
            lambda conn: conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,)).rowcount
        )
        return deleted > 0
