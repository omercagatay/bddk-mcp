"""Build fingerprint: which inputs and which code produced a corpus file.

``bddk-mcp build`` stores one JSON object under the ``meta`` key
``build_fingerprint`` as its last step.  Each named part is a short hex digest:

- ``seed``: canonical JSON of every non-content field of each seed document
  (everything except ``markdown_content`` and ``content_hash``, which build
  already compares per document) plus the whole catalog.  Only build computes
  it, from the seed it was given; it is not comparable at serve time.
- ``parser``: the source text of ``bddk_mcp/store/section_index.py``.
- ``chunker``: only what decides chunk boundaries and chunk text: the chunking
  constants (``EMBEDDING_CHUNK_TARGET_TOKENS``, ``EMBEDDING_CHUNK_TOKEN_OVERLAP``,
  ``MAX_SECTION_CHARS``), the word-unit pattern of
  ``bddk_mcp/store/vector_store.py`` and the source text of the functions there
  that decide chunk spans and chunk text: ``_chunk_document``,
  ``_chunk_document_by_tokens``, ``_chunk_spans``, ``_token_budget_ranges``,
  ``_text_units``, ``_split_oversized_unit``, ``_overlap_units``,
  ``_count_tokens`` and ``_section_for_chunk``.
- ``model``: how a vector is produced from a chunk text: embedding model name,
  dimension and configured revision string (never a filesystem path), plus the
  passage prefix, prefix separator and vector dtype as written in
  ``bddk_mcp/store/vector_store.py``.  The seed embedding cache is keyed by chunk
  text and model name only, so a model-part difference cannot be healed from it.

The parts are derived from the source, so there is no hand-maintained parser
version to forget.  The source is read as text (line endings normalised to
``\\n``) and the vector-store constants are taken from its syntax tree, so this
module imports neither the vector store, numpy, the model nor the tokenizer.
When a source text cannot be read (unusual packaging), that part is
``"unknown"`` and is not compared; the model part is ``"unknown"`` when the
vector-store source cannot be read.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import logging

from bddk_mcp.core import config as _config
from bddk_mcp.store import section_index as _section_index

logger = logging.getLogger(__name__)

FINGERPRINT_META_KEY = "build_fingerprint"
UNKNOWN = "unknown"
PARTS = ("seed", "parser", "chunker", "model")
# The parts a running process can recompute (the seed is known to build only).
RUNTIME_PARTS = ("parser", "chunker", "model")

_PARSER_MODULE = "bddk_mcp.store.section_index"
_VECTOR_STORE_MODULE = "bddk_mcp.store.vector_store"
CHUNK_FUNCTIONS = (
    "_chunk_document",
    "_chunk_document_by_tokens",
    "_chunk_spans",
    "_token_budget_ranges",
    "_text_units",
    "_split_oversized_unit",
    "_overlap_units",
    "_count_tokens",
    "_section_for_chunk",
)
_CHUNK_CONSTANTS = ("_WORD_UNIT_RE",)
# How a vector is produced from a chunk text; part of the model part, not the chunker part.
_VECTOR_CONSTANTS = ("_PASSAGE_PREFIX", "_PREFIX_SEPARATOR", "_EMBEDDING_DTYPE")
_DIGEST_CHARS = 16


def normalise_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _loader_source(module_name: str) -> str | None:
    """The source text the module's loader gives, without importing the module."""
    spec = importlib.util.find_spec(module_name)
    return spec.loader.get_source(module_name) if spec is not None and spec.loader is not None else None


def _module_source(module_name: str) -> str | None:
    """The module's source text with ``\n`` line endings, or None when it cannot be read."""
    try:
        source = _loader_source(module_name)
    except (ImportError, OSError, UnicodeError, AttributeError):
        return None
    return normalise_line_endings(source) if isinstance(source, str) else None


def _digest(payload: object) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:_DIGEST_CHARS]


def parser_part() -> str:
    source = _module_source(_PARSER_MODULE)
    return UNKNOWN if source is None else hashlib.sha256(source.encode("utf-8")).hexdigest()[:_DIGEST_CHARS]


def _top_level_segments(source: str) -> dict[str, str]:
    """Source text of the top-level functions and of the right-hand side of top-level assignments, by name."""
    segments: dict[str, str] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            segments[node.name] = ast.get_source_segment(source, node) or ""
        elif isinstance(node, ast.Assign | ast.AnnAssign) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    segments[target.id] = ast.get_source_segment(source, node.value) or ""
    return segments


def _vector_store_segments(names: tuple[str, ...], part: str) -> dict[str, str] | None:
    """The named top-level segments of the vector-store source, or None when any cannot be read."""
    source = _module_source(_VECTOR_STORE_MODULE)
    if source is None:
        return None
    try:
        segments = _top_level_segments(source)
    except (SyntaxError, ValueError):
        return None
    if any(not segments.get(name) for name in names):
        logger.warning(
            "Build fingerprint: a function or constant of the vector store was not found; %s not compared", part
        )
        return None
    return {name: segments[name] for name in names}


def chunker_part() -> str:
    names = (*CHUNK_FUNCTIONS, *_CHUNK_CONSTANTS)
    segments = _vector_store_segments(names, "chunker")
    if segments is None:
        return UNKNOWN
    return _digest(
        {
            "target_tokens": _config.EMBEDDING_CHUNK_TARGET_TOKENS,
            "token_overlap": _config.EMBEDDING_CHUNK_TOKEN_OVERLAP,
            "max_section_chars": _section_index.MAX_SECTION_CHARS,
            "source": segments,
        }
    )


def model_part() -> str:
    segments = _vector_store_segments(_VECTOR_CONSTANTS, "model")
    if segments is None:
        return UNKNOWN
    return _digest(
        {
            "name": _config.EMBEDDING_MODEL_NAME,
            "dimension": _config.EMBEDDING_DIMENSION,
            "revision": _config.EMBEDDING_MODEL_REVISION,
            "vector": segments,
        }
    )


def seed_part(docs: list[dict], catalog: list[dict]) -> str:
    """Digest of the seed metadata and catalog; body text is compared per document by build."""
    documents = sorted(
        (
            {key: value for key, value in record.items() if key not in ("markdown_content", "content_hash")}
            for record in docs
        ),
        key=lambda record: str(record.get("document_id", "")),
    )
    entries = sorted(catalog, key=lambda record: str(record.get("document_id", "")))
    return _digest({"documents": documents, "catalog": entries})


def runtime_parts() -> dict[str, str]:
    """The parts the running code determines: parser, chunker and model."""
    return {"parser": parser_part(), "chunker": chunker_part(), "model": model_part()}


def build_fingerprint(docs: list[dict], catalog: list[dict]) -> dict[str, str]:
    return {"seed": seed_part(docs, catalog), **runtime_parts()}


def encode(fingerprint: dict[str, str]) -> str:
    return json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))


def decode(value: str | None) -> dict[str, str] | None:
    """The stored fingerprint, or None when the key is missing or not a fingerprint object."""
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict) or not all(isinstance(item, str) for item in parsed.values()):
        return None
    return parsed


def differing_parts(stored: dict[str, str], current: dict[str, str], parts: tuple[str, ...] = PARTS) -> list[str]:
    """Comparable parts whose stored value differs; a part ``unknown`` on either side is not compared."""
    return [
        part
        for part in parts
        if part in current and UNKNOWN not in (stored.get(part), current[part]) and stored.get(part) != current[part]
    ]


def unknown_parts(stored: dict[str, str], current: dict[str, str], parts: tuple[str, ...] = PARTS) -> list[str]:
    """Parts that could not be compared because one side is ``unknown``."""
    return [part for part in parts if part in current and UNKNOWN in (stored.get(part), current[part])]
