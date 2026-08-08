"""Frozen text-boundary data shared by legal persistence and Citation v1."""

from typing import Final

PROVISION_BOUNDARY_CODEPOINTS_V1: Final[tuple[int, ...]] = (
    9,
    10,
    11,
    12,
    13,
    28,
    29,
    30,
    31,
    32,
    133,
    160,
    5760,
    8192,
    8193,
    8194,
    8195,
    8196,
    8197,
    8198,
    8199,
    8200,
    8201,
    8202,
    8232,
    8233,
    8239,
    8287,
    12288,
)

PROVISION_BOUNDARY_WHITESPACE_V1: Final[str] = "".join(chr(codepoint) for codepoint in PROVISION_BOUNDARY_CODEPOINTS_V1)
POSTGRES_PROVISION_BOUNDARY_WHITESPACE_V1: Final[str] = " || ".join(
    f"pg_catalog.chr({codepoint})" for codepoint in PROVISION_BOUNDARY_CODEPOINTS_V1
)
