"""DDL for the canonical regulatory legal-version and relation tables."""

from __future__ import annotations

from typing import Any

_VALIDATION_COLUMNS = """
    validation_state      TEXT NOT NULL,
    validated_by          TEXT,
    validated_at          TIMESTAMPTZ,
    validation_method     TEXT,
    review_record_sha256  TEXT
"""

REGULATORY_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS regulatory_instruments (
    instrument_id     TEXT PRIMARY KEY,
    jurisdiction      TEXT NOT NULL,
    authority_code    TEXT NOT NULL,
    identity_key      TEXT NOT NULL UNIQUE,
    canonical_title   TEXT NOT NULL,
    instrument_type   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS regulatory_source_artifacts (
    artifact_id             TEXT PRIMARY KEY,
    content_sha256          TEXT NOT NULL,
    canonical_uri           TEXT NOT NULL,
    source_authority        TEXT NOT NULL,
    media_type              TEXT NOT NULL,
    retrieved_at            TIMESTAMPTZ NOT NULL,
    repository_document_id  TEXT,
    fixture_only            BOOLEAN NOT NULL DEFAULT false
);
CREATE INDEX IF NOT EXISTS idx_reg_artifacts_repo_doc
    ON regulatory_source_artifacts (repository_document_id);

CREATE TABLE IF NOT EXISTS regulatory_evidence (
    evidence_id       TEXT PRIMARY KEY,
    artifact_id       TEXT NOT NULL REFERENCES regulatory_source_artifacts(artifact_id),
    locator           TEXT NOT NULL,
    statement_sha256  TEXT NOT NULL,
    authority_level   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS regulatory_legal_versions (
    legal_version_id        TEXT PRIMARY KEY,
    instrument_id           TEXT NOT NULL REFERENCES regulatory_instruments(instrument_id),
    version_key             TEXT NOT NULL,
    legal_text_sha256       TEXT NOT NULL,
    predecessor_version_id  TEXT REFERENCES regulatory_legal_versions(legal_version_id),
    consolidation_state     TEXT NOT NULL,
    {_VALIDATION_COLUMNS}
);
CREATE INDEX IF NOT EXISTS idx_reg_versions_instrument
    ON regulatory_legal_versions (instrument_id);
CREATE INDEX IF NOT EXISTS idx_reg_versions_predecessor
    ON regulatory_legal_versions (predecessor_version_id);

CREATE TABLE IF NOT EXISTS regulatory_legal_version_artifacts (
    legal_version_id  TEXT NOT NULL REFERENCES regulatory_legal_versions(legal_version_id),
    artifact_id       TEXT NOT NULL REFERENCES regulatory_source_artifacts(artifact_id),
    source_role       TEXT NOT NULL,
    PRIMARY KEY (legal_version_id, artifact_id, source_role)
);

CREATE TABLE IF NOT EXISTS regulatory_provisions (
    provision_id    TEXT PRIMARY KEY,
    instrument_id   TEXT NOT NULL REFERENCES regulatory_instruments(instrument_id),
    provision_kind  TEXT NOT NULL,
    canonical_path  TEXT NOT NULL,
    UNIQUE (instrument_id, canonical_path)
);

CREATE TABLE IF NOT EXISTS regulatory_legal_events (
    event_id                 TEXT PRIMARY KEY,
    legal_version_id         TEXT NOT NULL REFERENCES regulatory_legal_versions(legal_version_id),
    event_type               TEXT NOT NULL,
    event_date               DATE,
    evidence_id              TEXT NOT NULL REFERENCES regulatory_evidence(evidence_id),
    target_legal_version_id  TEXT REFERENCES regulatory_legal_versions(legal_version_id),
    {_VALIDATION_COLUMNS}
);

CREATE TABLE IF NOT EXISTS regulatory_legal_status_assertions (
    assertion_id      TEXT PRIMARY KEY,
    legal_version_id  TEXT NOT NULL REFERENCES regulatory_legal_versions(legal_version_id),
    legal_status      TEXT NOT NULL,
    valid_from        DATE,
    valid_through     DATE,
    evidence_id       TEXT NOT NULL REFERENCES regulatory_evidence(evidence_id),
    {_VALIDATION_COLUMNS}
);

CREATE TABLE IF NOT EXISTS regulatory_legal_version_provisions (
    legal_version_id        TEXT NOT NULL REFERENCES regulatory_legal_versions(legal_version_id),
    provision_id            TEXT NOT NULL REFERENCES regulatory_provisions(provision_id),
    normalized_text_sha256  TEXT NOT NULL,
    evidence_id             TEXT NOT NULL REFERENCES regulatory_evidence(evidence_id),
    PRIMARY KEY (legal_version_id, provision_id)
);

CREATE TABLE IF NOT EXISTS regulatory_family_imports (
    bundle_id       TEXT NOT NULL,
    bundle_sha256   TEXT NOT NULL,
    instrument_id   TEXT NOT NULL REFERENCES regulatory_instruments(instrument_id),
    schema_version  TEXT NOT NULL,
    fixture_only    BOOLEAN NOT NULL DEFAULT false,
    imported_by     TEXT NOT NULL,
    imported_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (bundle_id, bundle_sha256)
);

CREATE TABLE IF NOT EXISTS regulatory_relations (
    relation_id           TEXT PRIMARY KEY,
    relation_type         TEXT NOT NULL CHECK (relation_type IN (
                              'amends','repeals','replaces','consolidates',
                              'implements','cites','defines','exception_to')),
    source_instrument_id  TEXT NOT NULL REFERENCES regulatory_instruments(instrument_id),
    source_provision_id   TEXT REFERENCES regulatory_provisions(provision_id),
    target_instrument_id  TEXT REFERENCES regulatory_instruments(instrument_id),
    target_provision_id   TEXT REFERENCES regulatory_provisions(provision_id),
    target_external_ref   TEXT,
    evidence_id           TEXT NOT NULL REFERENCES regulatory_evidence(evidence_id),
    extraction_method     TEXT NOT NULL,
    confidence            REAL NOT NULL,
    {_VALIDATION_COLUMNS},
    CHECK (target_instrument_id IS NOT NULL OR target_external_ref IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS idx_reg_relations_source
    ON regulatory_relations (source_instrument_id, relation_type);
CREATE INDEX IF NOT EXISTS idx_reg_relations_target
    ON regulatory_relations (target_instrument_id, relation_type);
"""


async def apply_regulatory_schema(pool: Any) -> None:
    """Idempotently create all regulatory tables and indexes."""
    async with pool.acquire() as connection:
        await connection.execute(REGULATORY_SCHEMA)
