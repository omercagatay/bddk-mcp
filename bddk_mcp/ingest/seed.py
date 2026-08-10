"""
Export / import BDDK data as JSON seed files.

Lets you populate PostgreSQL locally, export the data, bake it into the
Docker image, and deploy to Railway with zero BDDK requests.

Usage:
    python seed.py export          # dump DB → seed_data/
    python seed.py import          # seed_data/ → DB (skips if DB already has data)
    python seed.py import --force  # overwrite existing DB data
"""

import argparse
import asyncio
import hashlib
import json
import logging
import math
import time
from functools import partial
from pathlib import Path

import asyncpg

from bddk_mcp.core.config import require_database_url
from bddk_mcp.corpus_coordination import acquire_corpus_mutation_lock
from bddk_mcp.corpus_manifest import (
    CORPUS_MANIFEST_FILENAME,
    CorpusArtifact,
    CorpusManifestError,
    CorpusManifestValidation,
    load_and_validate_corpus_manifest,
)
from bddk_mcp.corpus_publication import (
    assert_release_publication_ready,
    is_strict_release_request,
    publish_strict_corpus_release,
)
from bddk_mcp.db_identity import (
    assert_database_connection_identity,
    assert_database_identity,
    assert_release_publication_connection_identity,
    assert_release_publication_identity,
)
from bddk_mcp.db_transport import assert_database_transport
from bddk_mcp.store.bulk_write import (
    insert_document_chunk_rows,
    insert_document_version_rows,
    upsert_decision_cache_rows,
    upsert_document_rows,
)
from bddk_mcp.store.section_index import DocumentSection, extract_document_sections

logger = logging.getLogger(__name__)

SEED_DIR = Path(__file__).resolve().parents[2] / "seed_data"
_RESERVED_SEED_ARTIFACTS = frozenset({"documents.json", "chunks.json", "decision_cache.json"})
_CHUNK_ARTIFACT_FIELDS = frozenset(
    {
        "doc_id",
        "chunk_index",
        "title",
        "category",
        "decision_date",
        "decision_number",
        "source_url",
        "total_chunks",
        "total_pages",
        "content_hash",
        "chunk_start_char",
        "chunk_end_char",
        "section_type",
        "section_ref",
        "section_start_char",
        "section_end_char",
        "section_content_hash",
        "chunk_text",
    }
)
_CHUNK_STRING_FIELDS = _CHUNK_ARTIFACT_FIELDS - {
    "chunk_index",
    "total_chunks",
    "total_pages",
    "chunk_start_char",
    "chunk_end_char",
    "section_start_char",
    "section_end_char",
}
_CHUNK_REQUIRED_INTEGER_FIELDS = frozenset(
    {
        "chunk_index",
        "total_chunks",
        "total_pages",
        "chunk_start_char",
        "chunk_end_char",
    }
)
_CHUNK_OPTIONAL_INTEGER_FIELDS = frozenset({"section_start_char", "section_end_char"})
_DOCUMENT_STORAGE_FIELDS = frozenset(
    {
        "document_id",
        "title",
        "category",
        "decision_date",
        "decision_number",
        "source_url",
        "markdown_content",
        "content_hash",
        "downloaded_at",
        "extracted_at",
        "extraction_method",
        "total_pages",
        "file_size",
    }
)
_DOCUMENT_FRESHNESS_FIELDS = frozenset(
    {
        "authoritative_published_at",
        "source_detected_at",
        "retrieval_published_at",
    }
)
_DOCUMENT_ARTIFACT_FIELDS = _DOCUMENT_STORAGE_FIELDS | _DOCUMENT_FRESHNESS_FIELDS
_CACHE_ARTIFACT_FIELDS = frozenset(
    {
        "document_id",
        "title",
        "content",
        "decision_date",
        "decision_number",
        "category",
        "source_url",
    }
)


def _load_manifest_bound_records(root: Path, artifact: CorpusArtifact) -> list[dict]:
    """Read the exact bounded JSON bytes whose role and digest were reviewed."""

    path = (root / artifact.path).resolve()
    try:
        with path.open("rb") as handle:
            payload = handle.read(artifact.bytes + 1)
    except OSError as exc:
        raise RuntimeError(f"Reviewed seed artifact could not be read: {artifact.role}.") from exc
    if len(payload) != artifact.bytes or hashlib.sha256(payload).hexdigest() != artifact.sha256:
        raise RuntimeError(f"Reviewed seed artifact changed after manifest validation: {artifact.role}.")
    try:
        records = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Reviewed seed artifact is not valid JSON: {artifact.role}.") from exc
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise RuntimeError(f"Reviewed seed artifact must contain object records: {artifact.role}.")
    return records


def _manifest_seed_artifacts(
    seed_root: Path,
    *,
    require_quantified_freshness: bool,
    require_measured_freshness: bool,
    require_verified_signature: bool,
    trusted_signing_key: Path | None,
) -> tuple[CorpusManifestValidation | None, dict[str, CorpusArtifact]]:
    """Validate the corpus declaration and close reserved filename bypasses."""

    manifest_path = seed_root / CORPUS_MANIFEST_FILENAME
    reserved_present = {name for name in _RESERVED_SEED_ARTIFACTS if (seed_root / name).exists()}
    if not manifest_path.exists():
        if reserved_present or any(
            (
                require_quantified_freshness,
                require_measured_freshness,
                require_verified_signature,
            )
        ):
            raise RuntimeError(
                "Reviewed seed corpus manifest validation failed: required corpus manifest is missing: "
                f"{CORPUS_MANIFEST_FILENAME}"
            )
        return None, {}
    try:
        validation = load_and_validate_corpus_manifest(
            manifest_path,
            corpus_root=seed_root,
            require_quantified_freshness=require_quantified_freshness,
            require_measured_freshness=require_measured_freshness,
            require_verified_signature=require_verified_signature,
            trusted_signing_key=trusted_signing_key,
        )
    except CorpusManifestError as exc:
        raise RuntimeError(f"Reviewed seed corpus manifest validation failed: {exc}") from exc
    declared_paths = {artifact.path for artifact in validation.manifest.artifacts}
    undeclared_reserved = reserved_present - declared_paths
    if undeclared_reserved:
        raise RuntimeError("Reviewed seed corpus contains an undeclared reserved seed artifact.")
    by_role = {artifact.role: artifact for artifact in validation.manifest.artifacts if artifact.role != "other"}
    return validation, by_role


async def _activate_strict_release(
    pool,
    *,
    validation: CorpusManifestValidation | None,
    retrieval_profile_sha256: str,
    expected_documents: list[dict],
    expected_cache: list[dict],
    expected_chunks: list[dict],
    expected_embeddings: list[list[float]],
    expected_sections: dict[str, list[DocumentSection]],
    require_quantified_freshness: bool,
    require_measured_freshness: bool,
    require_verified_signature: bool,
) -> dict | None:
    """Activate only after import, reindex, and ordinary corpus readiness pass."""

    if not is_strict_release_request(
        require_quantified_freshness=require_quantified_freshness,
        require_measured_freshness=require_measured_freshness,
        require_verified_signature=require_verified_signature,
    ):
        return None
    if validation is None:
        raise RuntimeError("Strict corpus release activation requires a verified manifest.")

    # This pre-activation check validates the corpus itself without accepting
    # an older active identity as evidence for the replacement in progress.
    # Unlike serving readiness, it deliberately supports exact schema v5/v6 for
    # the one canonical-republication remediation required by the v7 guard.
    await assert_release_publication_ready(
        pool,
        retrieval_profile_sha256=retrieval_profile_sha256,
        require_active_release=False,
    )
    try:
        async with pool.acquire() as connection, connection.transaction():
            identity = await publish_strict_corpus_release(
                connection,
                validation,
                retrieval_profile_sha256=retrieval_profile_sha256,
                require_quantified_freshness=require_quantified_freshness,
                require_measured_freshness=require_measured_freshness,
                require_verified_signature=require_verified_signature,
            )
            # The SECURITY DEFINER publisher holds SHARE locks over every
            # release-state table until this transaction ends.  Prove exact
            # signed-artifact membership only after those locks are held; any
            # mismatch rolls the activation back with the surrounding
            # transaction.  Performing this check on a separate connection
            # would leave a race between comparison and publication.
            await _assert_strict_seed_membership(
                connection,
                expected_documents=expected_documents,
                expected_cache=expected_cache,
                expected_chunks=expected_chunks,
                expected_embeddings=expected_embeddings,
                expected_sections=expected_sections,
                retrieval_profile_sha256=retrieval_profile_sha256,
            )
    except RuntimeError:
        raise
    except Exception:
        raise RuntimeError("Strict corpus release evidence could not be activated.") from None

    post_activation = await assert_release_publication_ready(
        pool,
        retrieval_profile_sha256=retrieval_profile_sha256,
        require_active_release=True,
    )
    if post_activation is None or post_activation.release_id != identity.release_id:
        raise RuntimeError("Strict corpus release activation could not be verified.")
    return identity.safe_dict()


def _validate_seed_documents(docs: list[dict]) -> dict[str, str]:
    """Validate canonical seed documents before model or database work."""

    if not isinstance(docs, list) or any(not isinstance(item, dict) for item in docs):
        raise RuntimeError("Seed payload contains an invalid record.")
    doc_ids = [item.get("document_id") for item in docs]
    if any(not isinstance(doc_id, str) or not doc_id for doc_id in doc_ids) or len(set(doc_ids)) != len(doc_ids):
        raise RuntimeError("Seed document identities are missing or duplicated.")

    doc_hashes: dict[str, str] = {}
    for document in docs:
        content = document.get("markdown_content", "")
        content_hash = document.get("content_hash", "")
        if not isinstance(content, str) or hashlib.sha256(content.encode()).hexdigest() != content_hash:
            raise RuntimeError("Seed document content hash validation failed.")
        doc_hashes[document["document_id"]] = content_hash

    return doc_hashes


def _validate_strict_seed_artifact_shapes(docs: list[dict], cache: list[dict]) -> None:
    """Reject any signed field that bootstrap would default, coerce, or ignore."""

    document_ids: set[str] = set()
    for document in docs:
        if set(document) != _DOCUMENT_ARTIFACT_FIELDS:
            raise RuntimeError("Strict corpus document artifact schema does not match the bootstrap contract.")
        string_fields = _DOCUMENT_ARTIFACT_FIELDS - {
            "downloaded_at",
            "extracted_at",
            *_DOCUMENT_FRESHNESS_FIELDS,
            "total_pages",
            "file_size",
        }
        if any(not isinstance(document[field], str) for field in string_fields):
            raise RuntimeError("Strict corpus document artifact contains an invalid field type.")
        timestamps = (
            document["downloaded_at"],
            document["extracted_at"],
            *(document[field] for field in sorted(_DOCUMENT_FRESHNESS_FIELDS)),
        )
        if any(
            value is not None
            and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value))
            for value in timestamps
        ):
            raise RuntimeError("Strict corpus document artifact contains an invalid timestamp.")
        if (
            isinstance(document["total_pages"], bool)
            or not isinstance(document["total_pages"], int)
            or document["total_pages"] < 1
            or isinstance(document["file_size"], bool)
            or not isinstance(document["file_size"], int)
            or document["file_size"] < 0
            or not document["document_id"]
            or document["document_id"] in document_ids
        ):
            raise RuntimeError("Strict corpus document artifact contains invalid or duplicate metadata.")
        document_ids.add(document["document_id"])

    cache_ids: set[str] = set()
    for item in cache:
        if set(item) != _CACHE_ARTIFACT_FIELDS or any(
            not isinstance(item[field], str) for field in _CACHE_ARTIFACT_FIELDS
        ):
            raise RuntimeError("Strict corpus decision-cache artifact schema does not match the bootstrap contract.")
        if not item["document_id"] or item["document_id"] in cache_ids:
            raise RuntimeError("Strict corpus decision-cache artifact contains a missing or duplicate identity.")
        cache_ids.add(item["document_id"])


def _chunk_artifact_matches_generated(reviewed: list[dict], generated: list[dict]) -> bool:
    """Compare the complete declared chunk artifact to current profile output.

    A manifest checksum authenticates bytes, but the runtime historically
    ignored those bytes and regenerated unrelated chunks.  This comparison is
    deliberately exact after sorting by the unique logical identity; malformed,
    partial, duplicate, extra, or stale rows are a mismatch.
    """

    def normalized_rows(rows: list[dict]) -> list[dict] | None:
        normalized: list[dict] = []
        identities: set[tuple[str, int]] = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != _CHUNK_ARTIFACT_FIELDS:
                return None
            if any(not isinstance(row[field], str) for field in _CHUNK_STRING_FIELDS):
                return None
            if any(
                isinstance(row[field], bool) or not isinstance(row[field], int)
                for field in _CHUNK_REQUIRED_INTEGER_FIELDS
            ):
                return None
            if any(
                row[field] is not None and (isinstance(row[field], bool) or not isinstance(row[field], int))
                for field in _CHUNK_OPTIONAL_INTEGER_FIELDS
            ):
                return None
            identity = (row["doc_id"], row["chunk_index"])
            if (
                not identity[0]
                or identity in identities
                or row["chunk_index"] < 0
                or row["total_chunks"] < 1
                or row["chunk_index"] >= row["total_chunks"]
                or row["total_pages"] < 1
                or row["chunk_start_char"] < 0
                or row["chunk_end_char"] <= row["chunk_start_char"]
            ):
                return None
            identities.add(identity)
            normalized.append({field: row[field] for field in sorted(_CHUNK_ARTIFACT_FIELDS)})
        return sorted(normalized, key=lambda item: (item["doc_id"], item["chunk_index"]))

    reviewed_rows = normalized_rows(reviewed)
    generated_rows = normalized_rows(generated)
    return reviewed_rows is not None and generated_rows is not None and reviewed_rows == generated_rows


def _record_chunk_artifact_match(
    result: dict,
    *,
    reviewed_chunks: list[dict] | None,
    generated_chunks: list[dict],
    strict_release: bool,
) -> None:
    """Record drift for local use and reject it for a strict release."""

    matches = reviewed_chunks is not None and _chunk_artifact_matches_generated(
        reviewed_chunks,
        generated_chunks,
    )
    result["chunk_artifact_match"] = matches
    if matches:
        return
    warning = "chunk_artifact_does_not_match_current_retrieval_profile"
    if warning not in result["corpus_scope_warnings"]:
        result["corpus_scope_warnings"].append(warning)
    if strict_release:
        raise RuntimeError(
            "Strict corpus release refused because the signed chunk artifact does not exactly match "
            "the current retrieval profile. Regenerate and separately review/re-sign the corpus artifacts."
        )
    logger.warning(
        "Seed chunk artifact does not match the current retrieval profile; "
        "runtime-regenerated chunks are not signed artifact evidence"
    )


def _embedding_vector(value, *, dimension: int) -> tuple[float, ...] | None:
    """Parse one expected or pgvector-text value under strict numeric rules."""

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, list | tuple) or len(value) != dimension:
        return None
    parsed: list[float] = []
    for component in value:
        if isinstance(component, bool):
            return None
        try:
            number = float(component)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(number):
            return None
        parsed.append(number)
    if math.fsum(component * component for component in parsed) <= 0.0:
        return None
    return tuple(parsed)


def _stored_embeddings_match_regeneration(
    expected_chunks: list[dict],
    expected_embeddings: list[list[float]],
    stored_chunks: list[dict],
) -> bool:
    """Compare every stored vector with an independently regenerated vector."""

    from bddk_mcp.store.vector_store import (
        EMBEDDING_DIMENSION,
        PUBLICATION_EMBEDDING_MAX_ABS_ERROR,
        PUBLICATION_EMBEDDING_MIN_COSINE_SIMILARITY,
    )

    if len(expected_chunks) != len(expected_embeddings) or len(stored_chunks) != len(expected_chunks):
        return False
    expected_by_key: dict[tuple[str, int], tuple[float, ...]] = {}
    for chunk, embedding in zip(expected_chunks, expected_embeddings, strict=True):
        try:
            key = (str(chunk["doc_id"]), int(chunk["chunk_index"]))
        except (KeyError, TypeError, ValueError):
            return False
        vector = _embedding_vector(embedding, dimension=EMBEDDING_DIMENSION)
        if not key[0] or vector is None or key in expected_by_key:
            return False
        expected_by_key[key] = vector

    actual_keys: set[tuple[str, int]] = set()
    for chunk in stored_chunks:
        try:
            key = (str(chunk["doc_id"]), int(chunk["chunk_index"]))
            stored = _embedding_vector(chunk.get("embedding"), dimension=EMBEDDING_DIMENSION)
        except (KeyError, TypeError, ValueError):
            return False
        expected = expected_by_key.get(key)
        if stored is None or expected is None or key in actual_keys:
            return False
        actual_keys.add(key)
        if max(abs(left - right) for left, right in zip(stored, expected, strict=True)) > (
            PUBLICATION_EMBEDDING_MAX_ABS_ERROR
        ):
            return False
        stored_norm = math.sqrt(math.fsum(value * value for value in stored))
        expected_norm = math.sqrt(math.fsum(value * value for value in expected))
        cosine = math.fsum(left * right for left, right in zip(stored, expected, strict=True)) / (
            stored_norm * expected_norm
        )
        if not math.isfinite(cosine) or cosine < PUBLICATION_EMBEDDING_MIN_COSINE_SIMILARITY:
            return False
    return actual_keys == set(expected_by_key)


async def _assert_strict_seed_membership(
    connection,
    *,
    expected_documents: list[dict],
    expected_cache: list[dict],
    expected_chunks: list[dict],
    expected_embeddings: list[list[float]],
    expected_sections: dict[str, list[DocumentSection]],
    retrieval_profile_sha256: str,
) -> None:
    """Require the release database to be exactly derived from signed seed rows.

    Strict publication is intentionally a staged/fresh-database workflow.  It
    never deletes an unexpected row to make an existing database fit a signed
    manifest, and it does not represent legacy version snapshots or retained
    PDF bytes that are absent from the seed artifact contract.
    """

    document_rows = await connection.fetch(
        """
        SELECT document_id, title, category, decision_date,
               decision_number, source_url, markdown_content, content_hash,
               downloaded_at, extracted_at, extraction_method, total_pages,
               file_size
        FROM public.documents
        ORDER BY document_id
        """
    )
    cache_rows = await connection.fetch(
        """
        SELECT document_id, title, content, decision_date, decision_number,
               category, source_url
        FROM public.decision_cache
        ORDER BY document_id
        """
    )
    chunk_rows = await connection.fetch(
        """
        SELECT doc_id, chunk_index, title, category, decision_date,
               decision_number, source_url, total_chunks, total_pages,
               content_hash, chunk_start_char, chunk_end_char, section_type,
               section_ref, section_start_char, section_end_char,
               section_content_hash, chunk_text,
               embedding::pg_catalog.text AS embedding
        FROM public.document_chunks
        ORDER BY doc_id, chunk_index
        """
    )
    section_rows = await connection.fetch(
        """
        SELECT doc_id, section_type, section_ref, heading, start_char,
               end_char, content, content_hash, page_start, page_end,
               source_content_hash
        FROM public.document_sections
        ORDER BY doc_id, start_char, end_char, section_type, section_ref,
                 content_hash
        """
    )
    publication_rows = await connection.fetch(
        """
        SELECT doc_id, content_hash, retrieval_profile_hash, expected_chunks
        FROM public.document_retrieval_publications
        ORDER BY doc_id
        """
    )
    unrepresented_versions = int(await connection.fetchval("SELECT COUNT(*) FROM public.document_versions") or 0)
    retained_pdf_blobs = int(
        await connection.fetchval("SELECT COUNT(*) FROM public.documents WHERE COALESCE(octet_length(pdf_blob), 0) > 0")
        or 0
    )
    unrepresented_legal_rows = int(
        await connection.fetchval(
            """
            SELECT
                (SELECT COUNT(*) FROM public.regulatory_instruments)
              + (SELECT COUNT(*) FROM public.regulatory_family_imports)
              + (SELECT COUNT(*) FROM public.regulatory_source_blobs)
              + (SELECT COUNT(*) FROM public.regulatory_source_artifacts)
              + (SELECT COUNT(*) FROM public.regulatory_evidence)
              + (SELECT COUNT(*) FROM public.regulatory_legal_versions)
              + (SELECT COUNT(*) FROM public.regulatory_legal_version_artifacts)
              + (SELECT COUNT(*) FROM public.regulatory_legal_events)
              + (SELECT COUNT(*) FROM public.regulatory_legal_status_assertions)
              + (SELECT COUNT(*) FROM public.regulatory_provisions)
              + (SELECT COUNT(*) FROM public.regulatory_legal_version_provisions)
              + (SELECT COUNT(*) FROM public.regulatory_relations)
            """
        )
        or 0
    )

    canonical_documents = sorted(
        ({field: item[field] for field in sorted(_DOCUMENT_STORAGE_FIELDS)} for item in expected_documents),
        key=lambda item: item["document_id"],
    )
    canonical_cache = sorted(
        ({field: item[field] for field in sorted(_CACHE_ARTIFACT_FIELDS)} for item in expected_cache),
        key=lambda item: item["document_id"],
    )
    document_hashes = {item["document_id"]: item["content_hash"] for item in expected_documents}
    canonical_sections = sorted(
        (
            {
                "doc_id": doc_id,
                "section_type": section.section_type,
                "section_ref": section.section_ref,
                "heading": section.heading,
                "start_char": section.start_char,
                "end_char": section.end_char,
                "content": section.content,
                "content_hash": section.content_hash,
                "page_start": section.page_start,
                "page_end": section.page_end,
                "source_content_hash": document_hashes[doc_id],
            }
            for doc_id, sections in expected_sections.items()
            for section in sections
        ),
        key=lambda item: (
            item["doc_id"],
            item["start_char"],
            item["end_char"],
            item["section_type"],
            item["section_ref"],
            item["content_hash"],
        ),
    )
    chunk_counts: dict[str, int] = {}
    for chunk in expected_chunks:
        chunk_counts[chunk["doc_id"]] = chunk_counts.get(chunk["doc_id"], 0) + 1
    canonical_publications = sorted(
        (
            {
                "doc_id": document_id,
                "content_hash": content_hash,
                "retrieval_profile_hash": retrieval_profile_sha256,
                "expected_chunks": chunk_counts.get(document_id, 0),
            }
            for document_id, content_hash in document_hashes.items()
        ),
        key=lambda item: item["doc_id"],
    )
    actual_chunks = [
        {field: row[field] for field in _CHUNK_ARTIFACT_FIELDS}
        for row in chunk_rows
        if _CHUNK_ARTIFACT_FIELDS.issubset(set(row.keys()))
    ]
    if (
        [dict(row) for row in document_rows] != canonical_documents
        or [dict(row) for row in cache_rows] != canonical_cache
        or not _chunk_artifact_matches_generated(expected_chunks, actual_chunks)
        or not _stored_embeddings_match_regeneration(
            expected_chunks,
            expected_embeddings,
            [dict(row) for row in chunk_rows],
        )
        or [dict(row) for row in section_rows] != canonical_sections
        or [dict(row) for row in publication_rows] != canonical_publications
        or unrepresented_versions
        or retained_pdf_blobs
        or unrepresented_legal_rows
    ):
        raise RuntimeError(
            "Strict corpus release database membership is not exactly represented by the signed seed artifacts. "
            "Use a fresh staged database or add separately reviewed artifact coverage."
        )


def _generate_seed_chunks(vs, docs: list[dict]) -> tuple[list[dict], dict[str, list[dict]]]:
    """Regenerate chunks from canonical text under the current pinned profile."""

    from bddk_mcp.core.config import PAGE_SIZE
    from bddk_mcp.store.vector_store import _chunk_document

    tokenizer = vs._chunk_tokenizer()
    generated: list[dict] = []
    grouped: dict[str, list[dict]] = {}
    for document in docs:
        doc_id = document["document_id"]
        content = document["markdown_content"]
        chunks = _chunk_document(doc_id, content, tokenizer=tokenizer)
        if not chunks:
            raise RuntimeError("Seed document could not produce a retrieval chunk.")
        total_pages = max(1, math.ceil(len(content) / PAGE_SIZE))
        rows: list[dict] = []
        for index, chunk in enumerate(chunks):
            row = {
                "doc_id": doc_id,
                "chunk_index": index,
                "title": document.get("title", ""),
                "category": document.get("category", ""),
                "decision_date": document.get("decision_date", ""),
                "decision_number": document.get("decision_number", ""),
                "source_url": document.get("source_url", ""),
                "total_chunks": len(chunks),
                "total_pages": total_pages,
                "content_hash": document["content_hash"],
                "chunk_start_char": chunk.start_char,
                "chunk_end_char": chunk.end_char,
                "section_type": chunk.section_type,
                "section_ref": chunk.section_ref,
                "section_start_char": chunk.section_start_char,
                "section_end_char": chunk.section_end_char,
                "section_content_hash": chunk.section_content_hash,
                "chunk_text": chunk.chunk_text,
            }
            rows.append(row)
            generated.append(row)
        grouped[doc_id] = rows
    return generated, grouped


async def _regenerate_seed_embedding_vectors(
    vs,
    chunks: list[dict],
    *,
    batch_size: int = 32,
) -> list[list[float]]:
    """Regenerate and validate every vector under the store's pinned profile."""

    from bddk_mcp.store.vector_store import EMBEDDING_DIMENSION

    vectors: list[list[float]] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        embedded = await vs._embed([chunk["chunk_text"] for chunk in batch], prefix="passage")
        for vector in embedded:
            parsed = _embedding_vector(vector, dimension=EMBEDDING_DIMENSION)
            if parsed is None:
                raise RuntimeError("Seed embedding output was malformed.")
            vectors.append(list(parsed))
    if len(vectors) != len(chunks):
        raise RuntimeError("Seed embedding output was incomplete.")
    return vectors


async def _embed_seed_chunks(vs, chunks: list[dict], *, batch_size: int = 32) -> list[str]:
    """Precompute every seed vector before opening the publication transaction."""

    vectors = await _regenerate_seed_embedding_vectors(vs, chunks, batch_size=batch_size)
    return ["[" + ",".join(str(value) for value in vector) + "]" for vector in vectors]


def _expected_seed_sections(docs: list[dict]) -> dict[str, list[DocumentSection]]:
    """Parse the deterministic section index expected from seed documents."""
    return {
        doc["document_id"]: extract_document_sections(
            doc["document_id"],
            doc.get("markdown_content", ""),
        )
        for doc in docs
        if doc.get("document_id")
    }


async def _section_index_matches(
    conn,
    expected_sections: dict[str, list[DocumentSection]],
) -> bool:
    """Return whether seeded documents have the exact expected section index."""
    if not expected_sections:
        return True
    rows = await conn.fetch(
        """
        SELECT doc_id, section_type, section_ref, content_hash,
               start_char, end_char
        FROM document_sections
        WHERE doc_id = ANY($1::text[])
        """,
        sorted(expected_sections),
    )
    actual = {
        (
            row["doc_id"],
            row["section_type"],
            row["section_ref"],
            row["content_hash"],
            row["start_char"],
            row["end_char"],
        )
        for row in rows
    }
    expected = {
        (
            doc_id,
            section.section_type,
            section.section_ref,
            section.content_hash,
            section.start_char,
            section.end_char,
        )
        for doc_id, sections in expected_sections.items()
        for section in sections
    }
    return actual == expected


def _strip_docs_dump_header(text: str) -> str:
    """Return the body of a docs_dump-style markdown file.

    docs_dump files follow the shape ``# Title\n- Document ID: ...\n---\n<body>``.
    Split on the first ``\\n---\\n`` so only the article body is retained. Returns
    ``text`` unchanged when no separator is present — the caller can pass either
    raw markdown or a dump file without branching.
    """
    parts = text.split("\n---\n", 1)
    return parts[1].lstrip() if len(parts) == 2 else text


# ── Export ───────────────────────────────────────────────────────────────────


async def export_seed(dsn: str | None = None, pool: asyncpg.Pool | None = None) -> None:
    """Export decision_cache and documents tables to seed_data/ as JSON."""
    owns_pool = pool is None
    if owns_pool:
        selected_dsn = assert_database_transport(dsn) if dsn else require_database_url("public")
        pool = await asyncpg.create_pool(
            selected_dsn,
            min_size=1,
            max_size=3,
            init=partial(assert_database_connection_identity, profile="public"),
        )
    SEED_DIR.mkdir(exist_ok=True)

    try:
        if owns_pool:
            await assert_database_identity(pool, "public")
        async with pool.acquire() as conn:
            # 1. Decision cache
            rows = await conn.fetch(
                "SELECT document_id, title, content, decision_date, "
                "decision_number, category, source_url FROM decision_cache"
            )
            cache_data = [dict(r) for r in rows]
            cache_path = SEED_DIR / "decision_cache.json"
            cache_path.write_text(
                json.dumps(cache_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Exported {len(cache_data)} decision cache entries → {cache_path}")

            # 2. Documents (without pdf_blob to keep size manageable)
            rows = await conn.fetch(
                "SELECT document_id, title, category, decision_date, "
                "decision_number, source_url, markdown_content, content_hash, "
                "downloaded_at, extracted_at, extraction_method, total_pages, "
                "file_size FROM documents"
            )
            docs_data = [dict(r) for r in rows]
            docs_path = SEED_DIR / "documents.json"
            docs_path.write_text(
                json.dumps(docs_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Exported {len(docs_data)} documents → {docs_path}")

            # 3. Document chunks (embeddings as lists for JSON serialization)
            rows = await conn.fetch(
                "SELECT doc_id, chunk_index, title, category, decision_date, "
                "decision_number, source_url, total_chunks, total_pages, "
                "content_hash, chunk_start_char, chunk_end_char, "
                "section_type, section_ref, section_start_char, "
                "section_end_char, section_content_hash, chunk_text FROM document_chunks"
            )
            chunks_data = [dict(r) for r in rows]
            chunks_path = SEED_DIR / "chunks.json"
            chunks_path.write_text(
                json.dumps(chunks_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Exported {len(chunks_data)} chunks → {chunks_path}")

    finally:
        if owns_pool:
            await pool.close()

    print(f"\nSeed data written to {SEED_DIR}/")
    print("Commit this directory and rebuild your Docker image.")


# ── Import ───────────────────────────────────────────────────────────────────


async def import_seed(
    dsn: str | None = None,
    force: bool = False,
    pool: asyncpg.Pool | None = None,
    *,
    reindex_existing: bool = False,
    require_quantified_freshness: bool = False,
    require_measured_freshness: bool = False,
    require_verified_signature: bool = False,
    trusted_signing_key: Path | None = None,
) -> dict:
    """Import seed data from seed_data/ into PostgreSQL.

    Returns dict with counts of imported items.
    Skips import if tables already have data (unless force=True).
    """
    result = {
        "decision_cache": 0,
        "documents": 0,
        "sections": 0,
        "chunks": 0,
        "embedded": 0,
        "reindex_scanned": 0,
        "reindex_published": 0,
        "reindex_current": 0,
        "skipped": False,
        "corpus_manifest_id": None,
        "corpus_manifest_sha256": None,
        "corpus_scope_warnings": [],
        "chunk_artifact_match": None,
        "active_corpus_release": None,
        "release_publication_required": False,
    }

    if not SEED_DIR.exists():
        logger.info("No seed_data/ directory found — skipping seed import")
        return result

    manifest_validation, artifacts_by_role = _manifest_seed_artifacts(
        SEED_DIR,
        require_quantified_freshness=require_quantified_freshness,
        require_measured_freshness=require_measured_freshness,
        require_verified_signature=require_verified_signature,
        trusted_signing_key=trusted_signing_key,
    )
    if manifest_validation is not None:
        result.update(
            corpus_manifest_id=manifest_validation.manifest.manifest_id,
            corpus_manifest_sha256=manifest_validation.manifest_sha256,
            corpus_scope_warnings=list(manifest_validation.warnings),
        )
        for warning in manifest_validation.warnings:
            logger.warning("Seed corpus scope: %s", warning)
    strict_release = is_strict_release_request(
        require_quantified_freshness=require_quantified_freshness,
        require_measured_freshness=require_measured_freshness,
        require_verified_signature=require_verified_signature,
    )
    result["release_publication_required"] = strict_release
    if strict_release:
        missing_roles = {"documents", "chunks", "decision_cache"} - set(artifacts_by_role)
        if missing_roles:
            raise RuntimeError(
                "Strict corpus release requires manifest-bound documents, chunks, and decision-cache artifacts."
            )
    documents_artifact = artifacts_by_role.get("documents")
    chunks_artifact = artifacts_by_role.get("chunks")
    cache_artifact = artifacts_by_role.get("decision_cache")
    docs_data = _load_manifest_bound_records(SEED_DIR, documents_artifact) if documents_artifact is not None else []
    reviewed_chunks = _load_manifest_bound_records(SEED_DIR, chunks_artifact) if chunks_artifact is not None else None
    cache_data = _load_manifest_bound_records(SEED_DIR, cache_artifact) if cache_artifact is not None else []

    owns_pool = pool is None
    if owns_pool:
        selected_dsn = assert_database_transport(dsn) if dsn else require_database_url("ingestion")
        pool = await asyncpg.create_pool(
            selected_dsn,
            min_size=1,
            max_size=3,
            init=partial(assert_database_connection_identity, profile="ingestion"),
        )

    try:
        # Bootstrap is a data-loading operation, not a migration path.  Fail
        # before the first DML statement unless an operator has explicitly
        # prepared the complete schema with ``bddk-mcp migrate``.
        from bddk_mcp.db_lifecycle import assert_database_ready

        await assert_database_ready(pool=pool, require_corpus=False)
        if owns_pool:
            await assert_database_identity(pool, "ingestion")

        from bddk_mcp.store.doc_store import DocumentStore

        store = DocumentStore(pool)

        from bddk_mcp.store.vector_store import VectorStore

        vs = VectorStore(pool)

        seed_hashes = _validate_seed_documents(docs_data)
        if strict_release:
            _validate_strict_seed_artifact_shapes(docs_data, cache_data)
        expected_sections = _expected_seed_sections(docs_data)
        seed_doc_ids = sorted(seed_hashes)
        seed_cache_ids = sorted(item["document_id"] for item in cache_data)
        prepared_chunks: tuple[list[dict], dict[str, list[dict]]] | None = None
        if strict_release:
            prepared_chunks = _generate_seed_chunks(vs, docs_data)
            _record_chunk_artifact_match(
                result,
                reviewed_chunks=reviewed_chunks,
                generated_chunks=prepared_chunks[0],
                strict_release=True,
            )

        if not cache_data and not docs_data and not reindex_existing:
            logger.info("Reviewed seed files are empty; no data was imported")
            return result

        async with pool.acquire() as conn:
            db_rows = await conn.fetch(
                "SELECT document_id, content_hash FROM public.documents WHERE document_id = ANY($1::text[])",
                seed_doc_ids,
            )
            db_hashes = {row["document_id"]: row["content_hash"] or "" for row in db_rows}
            changed = [doc_id for doc_id, value in db_hashes.items() if value != seed_hashes[doc_id]]
            if changed and not force:
                raise RuntimeError(
                    "Seed content differs from existing documents; rerun with --force only after reviewing the change."
                )

            cache_count = await conn.fetchval(
                "SELECT COUNT(*) FROM public.decision_cache WHERE document_id = ANY($1::text[])",
                seed_cache_ids,
            )
            null_embedding_count = await conn.fetchval(
                "SELECT COUNT(*) FROM public.document_chunks WHERE doc_id = ANY($1::text[]) AND embedding IS NULL",
                seed_doc_ids,
            )
            publication_count = await conn.fetchval(
                "SELECT COUNT(*) FROM public.document_retrieval_publications "
                "WHERE doc_id = ANY($1::text[]) AND retrieval_profile_hash = $2",
                seed_doc_ids,
                vs.retrieval_profile_hash,
            )
            sections_ok = await _section_index_matches(conn, expected_sections)
            if (
                not force
                and len(db_hashes) == len(docs_data)
                and cache_count == len(cache_data)
                and null_embedding_count == 0
                and publication_count == len(docs_data)
                and sections_ok
            ):
                logger.info("Reviewed seed publication is already current; import skipped")
                result["skipped"] = True
                if reindex_existing:
                    result.update(await reindex_existing_documents(pool, vs=vs))
                return result

        # Expensive model work happens before the database transaction.  No
        # canonical or retrieval row can become visible if embedding fails.
        if prepared_chunks is None:
            prepared_chunks = _generate_seed_chunks(vs, docs_data)
            _record_chunk_artifact_match(
                result,
                reviewed_chunks=reviewed_chunks,
                generated_chunks=prepared_chunks[0],
                strict_release=False,
            )
        chunks_data, grouped_chunks = prepared_chunks
        vector_values = await _embed_seed_chunks(vs, chunks_data)

        async with pool.acquire() as conn, conn.transaction():
            await acquire_corpus_mutation_lock(conn)
            locked_rows = await conn.fetch(
                "SELECT document_id, content_hash, markdown_content FROM public.documents "
                "WHERE document_id = ANY($1::text[]) FOR UPDATE",
                seed_doc_ids,
            )
            locked_documents = {row["document_id"]: row for row in locked_rows}
            changed = [
                doc_id for doc_id, row in locked_documents.items() if (row["content_hash"] or "") != seed_hashes[doc_id]
            ]
            if changed and not force:
                raise RuntimeError(
                    "Seed content changed during preparation; import refused without a reviewed --force operation."
                )

            now = time.time()
            if cache_data:
                await conn.execute(
                    "DELETE FROM public.decision_cache WHERE document_id = ANY($1::text[])",
                    seed_cache_ids,
                )
                await upsert_decision_cache_rows(
                    conn,
                    [
                        (
                            item["document_id"],
                            item.get("title", ""),
                            item.get("content", ""),
                            item.get("decision_date", ""),
                            item.get("decision_number", ""),
                            item.get("category", ""),
                            item.get("source_url", ""),
                            now,
                        )
                        for item in cache_data
                    ],
                )

            version_sources = {
                doc_id: existing
                for doc_id, existing in locked_documents.items()
                if existing["content_hash"] and existing["content_hash"] != seed_hashes[doc_id]
            }
            max_versions: dict[str, int] = {}
            if version_sources:
                version_rows = await conn.fetch(
                    """
                    SELECT requested.document_id,
                           COALESCE(pg_catalog.max(version.version), 0)::pg_catalog.int4 AS max_version
                    FROM pg_catalog.unnest($1::pg_catalog.text[]) AS requested(document_id)
                    LEFT JOIN public.document_versions AS version
                      ON version.document_id = requested.document_id
                    GROUP BY requested.document_id
                    """,
                    sorted(version_sources),
                )
                max_versions = {str(row["document_id"]): int(row["max_version"]) for row in version_rows}
            await insert_document_version_rows(
                conn,
                [
                    (
                        doc_id,
                        max_versions[doc_id] + 1,
                        version_sources[doc_id]["content_hash"],
                        version_sources[doc_id]["markdown_content"],
                        now,
                    )
                    for doc_id in sorted(version_sources)
                ],
            )
            await upsert_document_rows(
                conn,
                [
                    (
                        document["document_id"],
                        document.get("title", ""),
                        document.get("category", ""),
                        document.get("decision_date", ""),
                        document.get("decision_number", ""),
                        document.get("source_url", ""),
                        document.get("markdown_content", ""),
                        document["content_hash"],
                        document.get("downloaded_at"),
                        document.get("extracted_at"),
                        document.get("extraction_method", "markitdown"),
                        document.get("total_pages", 1),
                        document.get("file_size", 0),
                    )
                    for document in docs_data
                ],
            )
            indexed_sections = await store._replace_document_sections_many_on_connection(
                conn,
                {doc_id: expected_sections[doc_id] for doc_id in seed_doc_ids},
                source_content_hashes=seed_hashes,
            )

            if seed_doc_ids:
                await conn.execute(
                    "DELETE FROM public.document_chunks WHERE doc_id = ANY($1::text[])",
                    seed_doc_ids,
                )
            await insert_document_chunk_rows(
                conn,
                [
                    (
                        chunk["doc_id"],
                        chunk["chunk_index"],
                        chunk.get("title", ""),
                        chunk.get("category", ""),
                        chunk.get("decision_date", ""),
                        chunk.get("decision_number", ""),
                        chunk.get("source_url", ""),
                        chunk["total_chunks"],
                        chunk.get("total_pages", 1),
                        chunk["content_hash"],
                        chunk.get("chunk_start_char"),
                        chunk.get("chunk_end_char"),
                        chunk.get("section_type", ""),
                        chunk.get("section_ref", ""),
                        chunk.get("section_start_char"),
                        chunk.get("section_end_char"),
                        chunk.get("section_content_hash", ""),
                        chunk["chunk_text"],
                        vector,
                    )
                    for chunk, vector in zip(chunks_data, vector_values, strict=True)
                ],
            )
            await vs._publish_documents_on_connection(
                conn,
                [(doc_id, seed_hashes[doc_id], len(grouped_chunks[doc_id])) for doc_id in sorted(grouped_chunks)],
            )

            result.update(
                decision_cache=len(cache_data),
                documents=len(docs_data),
                sections=indexed_sections,
                chunks=len(chunks_data),
                embedded=len(chunks_data),
            )
            logger.info(
                "Atomically published reviewed seed (%d cache, %d documents, %d sections, %d chunks)",
                len(cache_data),
                len(docs_data),
                indexed_sections,
                len(chunks_data),
            )

        if reindex_existing:
            result.update(await reindex_existing_documents(pool, vs=vs))

    finally:
        if owns_pool:
            await pool.close()

    return result


async def publish_seed_release(
    dsn: str | None = None,
    pool: asyncpg.Pool | None = None,
    *,
    trusted_signing_key: Path,
) -> dict:
    """Verify and activate an already imported strict corpus with a publisher identity.

    Import and publication intentionally use different PostgreSQL LOGINs.  This
    command performs no corpus mutation: it revalidates the signed artifacts,
    regenerates every deterministic derivative, proves exact database
    membership while the SECURITY DEFINER publisher holds state-table locks,
    and only then commits the append-only activation.
    """

    manifest_validation, artifacts_by_role = _manifest_seed_artifacts(
        SEED_DIR,
        require_quantified_freshness=True,
        require_measured_freshness=True,
        require_verified_signature=True,
        trusted_signing_key=trusted_signing_key,
    )
    if manifest_validation is None:
        raise RuntimeError("Strict corpus release publication requires a verified manifest.")
    missing_roles = {"documents", "chunks", "decision_cache"} - set(artifacts_by_role)
    if missing_roles:
        raise RuntimeError(
            "Strict corpus release requires manifest-bound documents, chunks, and decision-cache artifacts."
        )

    documents = _load_manifest_bound_records(SEED_DIR, artifacts_by_role["documents"])
    reviewed_chunks = _load_manifest_bound_records(SEED_DIR, artifacts_by_role["chunks"])
    decision_cache = _load_manifest_bound_records(SEED_DIR, artifacts_by_role["decision_cache"])
    _validate_seed_documents(documents)
    _validate_strict_seed_artifact_shapes(documents, decision_cache)
    expected_sections = _expected_seed_sections(documents)

    owns_pool = pool is None
    if owns_pool:
        selected_dsn = assert_database_transport(dsn) if dsn else require_database_url("release-publisher")
        pool = await asyncpg.create_pool(
            selected_dsn,
            min_size=1,
            max_size=3,
            init=assert_release_publication_connection_identity,
        )
    if pool is None:
        raise RuntimeError("Release-publisher database pool is unavailable.")

    try:
        if owns_pool:
            await assert_release_publication_identity(pool)
        from bddk_mcp.store.vector_store import VectorStore

        vector_store = VectorStore(pool)
        generated_chunks, _grouped = _generate_seed_chunks(vector_store, documents)
        comparison = {
            "chunk_artifact_match": None,
            "corpus_scope_warnings": list(manifest_validation.warnings),
        }
        _record_chunk_artifact_match(
            comparison,
            reviewed_chunks=reviewed_chunks,
            generated_chunks=generated_chunks,
            strict_release=True,
        )
        # This is deliberately independent of the vectors already stored in
        # PostgreSQL.  Loading/encoding through VectorStore pins the exact model
        # and runtime descriptor whose hash is activated below.
        expected_embeddings = await _regenerate_seed_embedding_vectors(vector_store, generated_chunks)
        active_release = await _activate_strict_release(
            pool,
            validation=manifest_validation,
            retrieval_profile_sha256=vector_store.retrieval_profile_hash,
            expected_documents=documents,
            expected_cache=decision_cache,
            expected_chunks=generated_chunks,
            expected_embeddings=expected_embeddings,
            expected_sections=expected_sections,
            require_quantified_freshness=True,
            require_measured_freshness=True,
            require_verified_signature=True,
        )
        if active_release is None:
            raise RuntimeError("Strict corpus release publication did not produce an active identity.")
        return {
            "active_corpus_release": active_release,
            "chunk_artifact_match": comparison["chunk_artifact_match"],
            "corpus_manifest_id": manifest_validation.manifest.manifest_id,
            "corpus_manifest_sha256": manifest_validation.manifest_sha256,
            "corpus_scope_warnings": comparison["corpus_scope_warnings"],
            "documents": len(documents),
            "chunks": len(generated_chunks),
        }
    finally:
        if owns_pool:
            await pool.close()


async def reindex_existing_documents(pool: asyncpg.Pool, *, vs=None) -> dict[str, int]:
    """Resumably publish every canonical document under the current profile."""

    if vs is None:
        from bddk_mcp.store.vector_store import VectorStore

        vs = VectorStore(pool)
    scanned = published = current = 0
    async with pool.acquire() as connection:
        rows = await connection.fetch(
            """
            SELECT document_id, title, markdown_content, category, decision_date,
                   decision_number, source_url
            FROM public.documents
            WHERE COALESCE(markdown_content, '') <> ''
            ORDER BY document_id
            """
        )
    # Each publication has its own short transaction-scoped mutation lock.
    # Operator jobs already hold the distinct session admission lease; taking
    # the mutation key on the reader connection here would make add_document's
    # second connection wait forever on the same job.
    for row in rows:
        scanned += 1
        if await vs.has_document(row["document_id"]):
            current += 1
            continue
        await vs.add_document(
            doc_id=row["document_id"],
            title=row["title"] or row["document_id"],
            content=row["markdown_content"],
            category=row["category"] or "",
            decision_date=row["decision_date"] or "",
            decision_number=row["decision_number"] or "",
            source_url=row["source_url"] or "",
        )
        published += 1

    logger.info(
        "Existing-document vector reconciliation complete (scanned=%d, published=%d, current=%d)",
        scanned,
        published,
        current,
    )
    return {
        "reindex_scanned": scanned,
        "reindex_published": published,
        "reindex_current": current,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="BDDK seed data export/import")
    parser.add_argument("--db", help="PostgreSQL DSN", default=None)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("export", help="Export DB → seed_data/")

    imp = sub.add_parser("import", help="Import seed_data/ → DB")
    imp.add_argument("--force", action="store_true", help="Overwrite existing data")

    args = parser.parse_args()

    if args.command == "export":
        asyncio.run(export_seed(args.db))
    elif args.command == "import":
        result = asyncio.run(import_seed(args.db, force=args.force))
        if result["skipped"]:
            print("Skipped — DB already has data (use --force to overwrite)")
        else:
            embedded = result.get("embedded", 0)
            print(
                f"\nImported: {result['decision_cache']} cache, {result['documents']} docs, "
                f"{result['chunks']} chunks ({embedded} embedded)"
            )
    else:
        parser.print_help()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
