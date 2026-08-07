"""Read-only traversal over legal versions and relation edges."""

from __future__ import annotations

from typing import Any

MAX_CHAIN_DEPTH = 50
MAX_XREF_DEPTH = 3

_DIRECTIONS = ("both", "incoming", "outgoing")


async def _instrument_for_doc(pool: Any, doc_id: str) -> str | None:
    return await pool.fetchval(
        """
        SELECT lv.instrument_id
        FROM regulatory_source_artifacts a
        JOIN regulatory_legal_version_artifacts lva ON lva.artifact_id = a.artifact_id
        JOIN regulatory_legal_versions lv ON lv.legal_version_id = lva.legal_version_id
        WHERE a.repository_document_id = $1
        LIMIT 1
        """,
        doc_id,
    )


async def amendment_chain(
    pool: Any,
    *,
    instrument_id: str | None = None,
    doc_id: str | None = None,
    include_unvalidated: bool = False,
) -> list[dict]:
    """Version chain oldest→newest with events and cross-instrument edges.

    Chain edges are human-validated only by default (spec §2); with
    include_unvalidated=True machine-inferred edges are included as well, but
    human-rejected edges are never returned on either path.
    """
    if instrument_id is None:
        if doc_id is None:
            raise ValueError("amendment_chain needs instrument_id or doc_id")
        instrument_id = await _instrument_for_doc(pool, doc_id)
        if instrument_id is None:
            return []
    rows = await pool.fetch(
        f"""
        WITH RECURSIVE chain AS (
            SELECT lv.*, 0 AS depth
            FROM regulatory_legal_versions lv
            WHERE lv.instrument_id = $1 AND lv.predecessor_version_id IS NULL
            UNION ALL
            SELECT lv.*, chain.depth + 1
            FROM regulatory_legal_versions lv
            JOIN chain ON lv.predecessor_version_id = chain.legal_version_id
            WHERE chain.depth < {MAX_CHAIN_DEPTH}
        )
        SELECT legal_version_id, version_key, predecessor_version_id,
               consolidation_state, depth
        FROM chain ORDER BY depth
        """,
        instrument_id,
    )
    if not rows:
        return []
    version_ids = [row["legal_version_id"] for row in rows]
    events = await pool.fetch(
        """
        SELECT legal_version_id, event_type, event_date, evidence_id
        FROM regulatory_legal_events
        WHERE legal_version_id = ANY($1::text[])
        ORDER BY event_date NULLS LAST, event_id
        """,
        version_ids,
    )
    validation_filter = (
        "validation_state <> 'rejected'" if include_unvalidated else "validation_state = 'human_validated'"
    )
    edges = await pool.fetch(
        f"""
        SELECT relation_type, source_instrument_id, evidence_id, validation_state
        FROM regulatory_relations
        WHERE target_instrument_id = $1
          AND relation_type IN ('amends', 'repeals', 'replaces')
          AND {validation_filter}
        ORDER BY relation_id
        """,
        instrument_id,
    )
    events_by_version: dict[str, list[dict]] = {}
    for event in events:
        events_by_version.setdefault(event["legal_version_id"], []).append(
            {
                "event_type": event["event_type"],
                "event_date": event["event_date"],
                "evidence_id": event["evidence_id"],
            }
        )
    edge_dicts = [dict(edge) for edge in edges]
    return [
        {
            "legal_version_id": row["legal_version_id"],
            "version_key": row["version_key"],
            "predecessor_version_id": row["predecessor_version_id"],
            "consolidation_state": row["consolidation_state"],
            "events": events_by_version.get(row["legal_version_id"], []),
            "edges": edge_dicts,
        }
        for row in rows
    ]


async def _fetch_hop_edges(
    pool: Any,
    *,
    frontier: list[str],
    provision_id: str | None,
    types: list[str] | None,
    include_unvalidated: bool,
) -> list[Any]:
    """One hop of the neighborhood: edges touching any frontier instrument.

    The SQL is rebuilt from scratch per hop, so placeholder numbering is
    always derived from the params list built right here (clarity over
    cleverness — see the Task 6 brief).
    """
    params: list[Any] = [frontier]
    conditions = [
        "(r.source_instrument_id = ANY($1::text[]) OR r.target_instrument_id = ANY($1::text[]))"
    ]
    if provision_id is not None:
        params.append(provision_id)
        placeholder = f"${len(params)}"
        conditions.append(
            f"(r.source_provision_id = {placeholder} OR r.target_provision_id = {placeholder}"
            " OR (r.source_provision_id IS NULL AND r.target_provision_id IS NULL))"
        )
    if not include_unvalidated:
        conditions.append("r.validation_state = 'human_validated'")
    else:
        # Human-rejected edges never ride along, even when the caller opts
        # into machine-inferred edges (spec §2).
        conditions.append("r.validation_state <> 'rejected'")
    if types:
        params.append(types)
        conditions.append(f"r.relation_type = ANY(${len(params)}::text[])")
    query = f"""
        SELECT r.relation_id, r.relation_type, r.source_instrument_id, r.target_instrument_id,
               r.target_external_ref, r.evidence_id, r.confidence, r.validation_state
        FROM regulatory_relations r
        WHERE {" AND ".join(conditions)}
        ORDER BY r.relation_id
        """
    return await pool.fetch(query, *params)


async def cross_references(
    pool: Any,
    *,
    doc_id: str,
    section_type: str | None,
    section_ref: str | None,
    direction: str = "both",
    types: list[str] | None = None,
    include_unvalidated: bool = False,
    depth: int = 1,
) -> list[dict]:
    """Relation neighborhood of a document (optionally narrowed to one section)."""
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction must be one of {_DIRECTIONS}")
    depth = max(1, min(depth, MAX_XREF_DEPTH))
    instrument_id = await _instrument_for_doc(pool, doc_id)
    if instrument_id is None:
        return []

    provision_id: str | None = None
    if section_type and section_ref:
        provision_id = await pool.fetchval(
            """
            SELECT provision_id FROM regulatory_section_provision_map
            WHERE doc_id = $1 AND section_type = $2 AND section_ref = $3
            LIMIT 1
            """,
            doc_id,
            section_type,
            section_ref,
        )
        if provision_id is None:
            return []

    results: list[dict] = []
    seen_relation_ids: set[str] = set()
    frontier = {instrument_id}
    visited: set[str] = set()
    for hop in range(1, depth + 1):
        if not frontier:
            break
        rows = await _fetch_hop_edges(
            pool,
            frontier=sorted(frontier),
            # Section narrowing only applies to the first hop.
            provision_id=provision_id if hop == 1 else None,
            types=types,
            include_unvalidated=include_unvalidated,
        )
        visited |= frontier
        next_frontier: set[str] = set()
        for row in rows:
            if row["relation_id"] in seen_relation_ids:
                continue
            edge_direction = "outgoing" if row["source_instrument_id"] in frontier else "incoming"
            if direction != "both" and edge_direction != direction:
                continue
            seen_relation_ids.add(row["relation_id"])
            results.append(
                {
                    "relation_type": row["relation_type"],
                    "direction": edge_direction,
                    "source_instrument_id": row["source_instrument_id"],
                    "target_instrument_id": row["target_instrument_id"],
                    "target_external_ref": row["target_external_ref"],
                    "evidence_id": row["evidence_id"],
                    "confidence": row["confidence"],
                    "validation_state": row["validation_state"],
                    "depth": hop,
                }
            )
            for neighbor in (row["source_instrument_id"], row["target_instrument_id"]):
                if neighbor and neighbor not in visited:
                    next_frontier.add(neighbor)
        frontier = next_frontier
    return results


async def one_hop_section_refs(
    pool: Any,
    *,
    doc_id: str,
    section_type: str,
    section_ref: str,
    limit: int = 3,
) -> list[dict]:
    """Validated one-hop neighbors resolved back to concrete sections."""
    rows = await pool.fetch(
        """
        SELECT DISTINCT m2.doc_id, m2.section_type, m2.section_ref, r.relation_type
        FROM regulatory_section_provision_map m1
        JOIN regulatory_relations r
          ON (r.source_provision_id = m1.provision_id OR r.target_provision_id = m1.provision_id)
         AND r.validation_state = 'human_validated'
        JOIN regulatory_section_provision_map m2
          ON m2.provision_id = CASE WHEN r.source_provision_id = m1.provision_id
                                    THEN r.target_provision_id ELSE r.source_provision_id END
        WHERE m1.doc_id = $1 AND m1.section_type = $2 AND m1.section_ref = $3
          AND NOT (m2.doc_id = m1.doc_id AND m2.section_type = m1.section_type
                   AND m2.section_ref = m1.section_ref)
        ORDER BY m2.doc_id, m2.section_type, m2.section_ref, r.relation_type
        LIMIT $4
        """,
        doc_id,
        section_type,
        section_ref,
        limit,
    )
    return [dict(row) for row in rows]
