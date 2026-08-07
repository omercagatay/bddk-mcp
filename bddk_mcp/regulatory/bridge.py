"""Resolution bridge: document_sections ↔ regulatory_provisions.

Resolution path (spec §3.2):
document_sections.doc_id → regulatory_source_artifacts.repository_document_id
→ regulatory_legal_version_artifacts → regulatory_legal_versions.instrument_id
→ regulatory_provisions (canonical_path match).
"""

from __future__ import annotations

import re
from typing import Any

from bddk_mcp.store.legal_ref import turkish_casefold

_PATH_SEPARATOR_RE = re.compile(r"[/\s]+")
# document_sections stores lettered refs fused ("MADDE 9A" → ref "9a"), while
# bundle producers may write "9/A"; inserting '-' at every digit→non-digit
# boundary makes both spellings land on "9-a". Explicit [0-9] (not \d) keeps
# the Python and PostgreSQL expressions identical by inspection.
_DIGIT_SUFFIX_BOUNDARY_RE = re.compile(r"(?<=[0-9])(?=[^0-9-])")
# Canonical paths are ASCII: the section index itself stores ASCII kinds
# ("fikra", "gecici_madde"), and ASCII-degraded input like "ILKE" casefolds
# to "ılke", which must still land on "ilke". Applied AFTER turkish_casefold.
# Mirrored 1:1 by translate(...) in SECTION_PROVISION_MAP_DDL.
_TURKISH_ASCII_FOLD = str.maketrans("ıçğöşüâîû", "icgosuaiu")


def canonical_provision_path(section_type: str, section_ref: str) -> str:
    """Single normalization shared by provision import and SQL joins.

    ("Madde", "9/A") → "madde/9-a", and the fused form stored by the section
    index normalizes identically: ("madde", "9a") → "madde/9-a". Turkish
    casefold then ASCII fold ("İlke", "ILKE" → "ilke"); slashes and spaces in
    the ref collapse to '-', and a '-' is inserted between a number and its
    letter suffix, so the path itself stays unambiguous on '/'.
    """
    kind = turkish_casefold(section_type.strip()).translate(_TURKISH_ASCII_FOLD)
    ref = turkish_casefold(section_ref.strip()).translate(_TURKISH_ASCII_FOLD)
    ref = _PATH_SEPARATOR_RE.sub("-", ref)
    ref = _DIGIT_SUFFIX_BOUNDARY_RE.sub("-", ref)
    return f"{kind}/{ref}"


def sql_canonical_provision_path(type_expr: str, ref_expr: str) -> str:
    """Render the SQL twin of canonical_provision_path() for two SQL fragments.

    MUST stay in lockstep with canonical_provision_path(): btrim → casefold →
    ASCII fold → (ref only) separators → digit-suffix dash. The lockstep is
    enforced by a SQL↔Python parity test in tests/test_regulatory_bridge.py,
    which feeds this exact expression ('$1', '$2') back through PostgreSQL.
    lower(translate(x, 'Iİ', 'ıi')) is defensive only: document_sections
    values are already Python-lowercased by section_index._normalize_ref, so
    C-locale lower() behavior on non-ASCII never actually matters here.
    Note: '\\s' inside a bracket expression requires PostgreSQL >= 14.
    """
    fold = "translate(lower(translate(btrim({0}), 'Iİ', 'ıi')), 'ıçğöşüâîû', 'icgosuaiu')"
    return (
        f"{fold.format(type_expr)}\n"
        "       || '/'\n"
        "       || regexp_replace(\n"
        f"              regexp_replace({fold.format(ref_expr)},\n"
        r"                  '[/\s]+', '-', 'g'),"
        "\n"
        r"              '([0-9])([^0-9-])', '\1-\2', 'g')"
    )


SECTION_PROVISION_MAP_DDL = f"""
CREATE MATERIALIZED VIEW IF NOT EXISTS regulatory_section_provision_map AS
SELECT DISTINCT
    ds.doc_id,
    ds.section_type,
    ds.section_ref,
    p.provision_id,
    p.instrument_id
FROM document_sections ds
JOIN regulatory_source_artifacts a
    ON a.repository_document_id = ds.doc_id
JOIN regulatory_legal_version_artifacts lva
    ON lva.artifact_id = a.artifact_id
JOIN regulatory_legal_versions lv
    ON lv.legal_version_id = lva.legal_version_id
JOIN regulatory_provisions p
    ON p.instrument_id = lv.instrument_id
   AND p.canonical_path =
       {sql_canonical_provision_path("ds.section_type", "ds.section_ref")}
WITH DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_section_provision_map_key
    ON regulatory_section_provision_map (doc_id, section_type, section_ref, provision_id);
"""


async def ensure_section_provision_map(pool: Any) -> None:
    """Idempotently create the materialized bridge view."""
    async with pool.acquire() as connection:
        await connection.execute(SECTION_PROVISION_MAP_DDL)


async def refresh_section_provision_map(pool: Any) -> None:
    """Refresh after document reindex or bundle import."""
    async with pool.acquire() as connection:
        await connection.execute("REFRESH MATERIALIZED VIEW regulatory_section_provision_map")
