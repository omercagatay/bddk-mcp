"""Tests for the build fingerprint: named parts derived from the seed, the parser, the chunker and the model."""

from __future__ import annotations

import subprocess
import sys

import pytest

from bddk_mcp.core import config
from bddk_mcp.store import fingerprint, section_index


def _patched_source(monkeypatch, module_name: str, change):
    real = fingerprint._module_source

    def source(name: str):
        text = real(name)
        return change(text) if name == module_name else text

    monkeypatch.setattr(fingerprint, "_module_source", source)


def test_parts_are_short_hex_digests_and_stable_across_calls():
    first = fingerprint.runtime_parts()
    assert set(first) == {"parser", "chunker", "model"}
    assert all(len(value) == 16 and int(value, 16) >= 0 for value in first.values())
    assert fingerprint.runtime_parts() == first
    assert fingerprint.UNKNOWN not in first.values()  # every named chunking function exists in vector_store.py


def test_the_module_imports_neither_the_vector_store_nor_numpy():
    code = (
        "import sys; from bddk_mcp.store import fingerprint; fingerprint.runtime_parts(); "
        "print('numpy' in sys.modules, 'bddk_mcp.store.vector_store' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False False"


def test_the_parser_part_follows_the_section_index_source(monkeypatch):
    before = fingerprint.runtime_parts()
    _patched_source(monkeypatch, "bddk_mcp.store.section_index", lambda text: text + "\n# a parser change\n")
    after = fingerprint.runtime_parts()
    assert after["parser"] != before["parser"]
    assert (after["chunker"], after["model"]) == (before["chunker"], before["model"])


def test_the_chunker_part_follows_a_chunking_function_but_not_unrelated_code(monkeypatch):
    before = fingerprint.chunker_part()
    _patched_source(
        monkeypatch,
        "bddk_mcp.store.vector_store",
        lambda text: text.replace("def _chunk_spans(", "def _chunk_spans(  ", 1),
    )
    assert fingerprint.chunker_part() != before

    monkeypatch.undo()
    _patched_source(monkeypatch, "bddk_mcp.store.vector_store", lambda text: text + "\n\ndef unrelated():\n    pass\n")
    assert fingerprint.chunker_part() == before


@pytest.mark.parametrize(
    ("module", "name", "value"),
    [
        (config, "EMBEDDING_CHUNK_TARGET_TOKENS", 512),
        (config, "EMBEDDING_CHUNK_TOKEN_OVERLAP", 0),
        (section_index, "MAX_SECTION_CHARS", 10_000),
    ],
)
def test_the_chunker_part_follows_the_chunking_constants(monkeypatch, module, name, value):
    before = fingerprint.chunker_part()
    monkeypatch.setattr(module, name, value)
    assert fingerprint.chunker_part() != before


@pytest.mark.parametrize("constant", ["_PASSAGE_PREFIX", "_PREFIX_SEPARATOR", "_EMBEDDING_DTYPE"])
def test_the_prefix_and_dtype_source_change_the_model_part_and_not_the_chunker_part(monkeypatch, constant):
    before = fingerprint.runtime_parts()
    replacements = {
        "_PASSAGE_PREFIX": ('_PASSAGE_PREFIX = "passage"', '_PASSAGE_PREFIX = "doc"'),
        "_PREFIX_SEPARATOR": ('_PREFIX_SEPARATOR = ": "', '_PREFIX_SEPARATOR = " "'),
        "_EMBEDDING_DTYPE": ("_EMBEDDING_DTYPE = np.float16", "_EMBEDDING_DTYPE = np.float32"),
    }
    old, new = replacements[constant]
    _patched_source(monkeypatch, "bddk_mcp.store.vector_store", lambda text: text.replace(old, new, 1))
    after = fingerprint.runtime_parts()
    assert after["model"] != before["model"]
    assert (after["chunker"], after["parser"]) == (before["chunker"], before["parser"])


@pytest.mark.parametrize(
    ("module", "name", "value"),
    [
        (config, "EMBEDDING_CHUNK_TARGET_TOKENS", 512),
        (config, "EMBEDDING_CHUNK_TOKEN_OVERLAP", 0),
        (section_index, "MAX_SECTION_CHARS", 10_000),
    ],
)
def test_a_chunking_constant_does_not_change_the_model_part(monkeypatch, module, name, value):
    before = fingerprint.runtime_parts()
    monkeypatch.setattr(module, name, value)
    after = fingerprint.runtime_parts()
    assert after["chunker"] != before["chunker"]
    assert after["model"] == before["model"]


@pytest.mark.parametrize(
    ("name", "value"),
    [("EMBEDDING_MODEL_NAME", "other/model"), ("EMBEDDING_DIMENSION", 384), ("EMBEDDING_MODEL_REVISION", "abc")],
)
def test_the_model_part_follows_name_dimension_and_revision(monkeypatch, name, value):
    before = fingerprint.model_part()
    monkeypatch.setattr(config, name, value)
    assert fingerprint.model_part() != before


def test_the_model_part_ignores_the_local_model_path(monkeypatch):
    before = fingerprint.model_part()
    monkeypatch.setattr(config, "EMBEDDING_MODEL_PATH", "/somewhere/else")
    assert fingerprint.model_part() == before


def test_line_endings_are_normalised(monkeypatch):
    before = fingerprint.runtime_parts()
    real = fingerprint._loader_source
    monkeypatch.setattr(fingerprint, "_loader_source", lambda name: real(name).replace("\n", "\r\n"))
    assert fingerprint.runtime_parts() == before
    monkeypatch.setattr(fingerprint, "_loader_source", lambda name: real(name).replace("\n", "\r"))
    assert fingerprint.runtime_parts() == before


def test_a_loader_error_is_unknown(monkeypatch):
    def broken(name):
        raise OSError("no source in this packaging")

    monkeypatch.setattr(fingerprint, "_loader_source", broken)
    assert fingerprint.parser_part() == fingerprint.UNKNOWN
    assert fingerprint.chunker_part() == fingerprint.UNKNOWN


def test_a_source_that_cannot_be_read_is_unknown_and_not_compared(monkeypatch):
    current = fingerprint.runtime_parts()
    monkeypatch.setattr(fingerprint, "_module_source", lambda name: None)

    unknown = fingerprint.runtime_parts()
    assert unknown == {"parser": fingerprint.UNKNOWN, "chunker": fingerprint.UNKNOWN, "model": fingerprint.UNKNOWN}
    assert fingerprint.differing_parts(current, unknown) == []
    assert fingerprint.unknown_parts(current, unknown) == ["parser", "chunker", "model"]
    assert fingerprint.differing_parts(unknown, current) == []


def test_a_module_that_cannot_be_found_is_unknown():
    assert fingerprint._module_source("bddk_mcp.store.no_such_module") is None


def test_the_seed_part_covers_metadata_and_catalog_but_not_body_text():
    docs = [{"document_id": "1", "title": "T", "markdown_content": "A", "content_hash": "h1"}]
    catalog = [{"document_id": "1", "title": "T"}]
    base = fingerprint.seed_part(docs, catalog)

    assert fingerprint.seed_part(list(reversed(docs)), catalog) == base
    assert fingerprint.seed_part([{**docs[0], "markdown_content": "B", "content_hash": "h2"}], catalog) == base
    assert fingerprint.seed_part([{**docs[0], "title": "U"}], catalog) != base
    assert fingerprint.seed_part([{**docs[0], "decision_date": "01.01.2026"}], catalog) != base
    assert fingerprint.seed_part(docs, [{"document_id": "1", "title": "U"}]) != base


def test_differing_parts_counts_a_missing_stored_part():
    current = {"seed": "a", "parser": "b", "chunker": "c", "model": "d"}
    assert fingerprint.differing_parts(current, current) == []
    assert fingerprint.differing_parts({**current, "parser": "x"}, current) == ["parser"]
    assert fingerprint.differing_parts({"seed": "a"}, current) == ["parser", "chunker", "model"]
    assert fingerprint.differing_parts({**current, "seed": "z"}, current, fingerprint.RUNTIME_PARTS) == []


def test_encode_and_decode_round_trip_and_reject_other_values():
    value = {"seed": "a", "parser": "b"}
    assert fingerprint.decode(fingerprint.encode(value)) == value
    for broken in (None, "", "not json", "[1, 2]", '{"seed": 1}'):
        assert fingerprint.decode(broken) is None


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("def _section_rows(", "def _section_rows(  "),
        ("def normalize_search_text(", "def normalize_search_text(  "),
    ],
)
def test_the_parser_part_follows_how_section_rows_are_derived_and_nothing_else_does(monkeypatch, old, new):
    before = fingerprint.runtime_parts()
    _patched_source(monkeypatch, "bddk_mcp.store.doc_store", lambda text: text.replace(old, new, 1))
    after = fingerprint.runtime_parts()
    assert after["parser"] != before["parser"]
    assert (after["chunker"], after["model"]) == (before["chunker"], before["model"])


def test_unrelated_document_store_code_does_not_change_the_parser_part(monkeypatch):
    before = fingerprint.parser_part()
    _patched_source(
        monkeypatch,
        "bddk_mcp.store.doc_store",
        lambda text: text.replace("async def load_catalog(", "async def load_catalog(  ", 1) + "\n# unrelated\n",
    )
    assert fingerprint.parser_part() == before


def test_the_parser_part_is_unknown_when_a_section_row_function_is_missing(monkeypatch):
    _patched_source(
        monkeypatch, "bddk_mcp.store.doc_store", lambda text: text.replace("def _section_rows(", "def _rows(", 1)
    )
    assert fingerprint.parser_part() == fingerprint.UNKNOWN


def test_the_module_does_not_import_the_document_store():
    code = (
        "import sys; from bddk_mcp.store import fingerprint; fingerprint.runtime_parts(); "
        "print('bddk_mcp.store.doc_store' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


def _module_level_names(tree) -> dict[str, object]:
    """The module-level functions and constants of a parsed module (classes are left out), by name."""
    import ast

    names: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            names[node.name] = node
        elif isinstance(node, ast.Assign | ast.AnnAssign) and node.value is not None:
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                if isinstance(target, ast.Name):
                    names[target.id] = node.value
    return names


def _referenced_closure(module_name: str, roots: list[str]) -> set[str]:
    """Every module-level function and constant of the module that the roots reference, transitively."""
    import ast

    tree = ast.parse(fingerprint._module_source(module_name))
    defined = _module_level_names(tree)
    found: set[str] = set()
    pending = list(roots)
    while pending:
        name = pending.pop()
        if name in found or name not in defined:
            continue
        found.add(name)
        pending.extend(node.id for node in ast.walk(defined[name]) if isinstance(node, ast.Name))
    return found


def test_every_function_and_constant_that_decides_stored_rows_is_in_the_fingerprint():
    import ast

    chunker = _referenced_closure("bddk_mcp.store.vector_store", ["_chunk_spans", *fingerprint.CHUNK_FUNCTIONS])
    missing = chunker - {*fingerprint.CHUNK_FUNCTIONS, *fingerprint._CHUNK_CONSTANTS}
    assert not missing, f"chunker part misses {sorted(missing)}"

    # The helpers the section parser imports from legal_ref, and what they use there.
    section_index_tree = ast.parse(fingerprint._module_source("bddk_mcp.store.section_index"))
    imported = [
        alias.name
        for node in section_index_tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "bddk_mcp.store.legal_ref"
        for alias in node.names
    ]
    assert imported
    parser = _referenced_closure("bddk_mcp.store.legal_ref", imported)
    missing = parser - set(fingerprint.SECTION_REF_HELPERS)
    assert not missing, f"parser part misses {sorted(missing)}"


def test_the_chunker_part_follows_the_caption_merge_constants(monkeypatch):
    before = fingerprint.runtime_parts()
    _patched_source(
        monkeypatch,
        "bddk_mcp.store.vector_store",
        lambda text: text.replace("_CAPTION_MAX_CHARS = 120", "_CAPTION_MAX_CHARS = 140", 1),
    )
    after = fingerprint.runtime_parts()
    assert after["chunker"] != before["chunker"]
    assert (after["parser"], after["model"]) == (before["parser"], before["model"])


def test_the_parser_part_follows_the_canonical_section_ref_of_legal_ref(monkeypatch):
    before = fingerprint.runtime_parts()
    _patched_source(
        monkeypatch,
        "bddk_mcp.store.legal_ref",
        lambda text: text.replace("def canonical_section_ref(", "def canonical_section_ref(  ", 1),
    )
    after = fingerprint.runtime_parts()
    assert after["parser"] != before["parser"]
    assert (after["chunker"], after["model"]) == (before["chunker"], before["model"])

    monkeypatch.undo()
    _patched_source(monkeypatch, "bddk_mcp.store.legal_ref", lambda text: text + "\n\ndef unrelated():\n    pass\n")
    assert fingerprint.parser_part() == before["parser"]
