"""Read-only traversal over validated legal versions and relation edges.

All queries go through the ``regulatory_validated_*`` views, which expose
only reviewer-validated rows backed by non-fixture artifacts.  Serving
workload LOGINs hold SELECT on those views and nothing on the base tables,
so unvalidated or rejected claims are unreachable here by construction.
"""

from __future__ import annotations

from typing import Any

MAX_CHAIN_DEPTH = 50
MAX_XREF_DEPTH = 3

_DIRECTIONS = ("both", "incoming", "outgoing")

_CHAIN_EDGE_TYPES = ("amends", "repeals", "replaces")


async def _instrument_for_doc(pool: Any, doc_id: str) -> str | None:
    return await pool.fetchval(
        """
        SELECT instrument_id
        FROM public.regulatory_validated_legal_versions
        WHERE repository_document_id = $1
        ORDER BY legal_version_id
        LIMIT 1
        """,
        doc_id,
    )


async def amendment_chain(
    pool: Any,
    *,
    instrument_id: str | None = None,
    doc_id: str | None = None,
) -> list[dict]:
    """Validated version chain oldest→newest with events and incoming edges.

    A predecessor that is itself unvalidated breaks the visible chain: its
    successors are reported as chain roots rather than silently reattached.
    """
    if instrument_id is None:
        if doc_id is None:
            raise ValueError("amendment_chain needs instrument_id or doc_id")
        instrument_id = await _instrument_for_doc(pool, doc_id)
        if instrument_id is None:
            return []
    version_rows = await pool.fetch(
        """
        SELECT DISTINCT legal_version_id, version_key, predecessor_version_id, consolidation_state
        FROM public.regulatory_validated_legal_versions
        WHERE instrument_id = $1
        ORDER BY version_key, legal_version_id
        """,
        instrument_id,
    )
    if not version_rows:
        return []
    by_id = {row["legal_version_id"]: row for row in version_rows}
    successors: dict[str, list[str]] = {}
    roots: list[str] = []
    for row in version_rows:
        predecessor = row["predecessor_version_id"]
        if predecessor is not None and predecessor in by_id:
            successors.setdefault(predecessor, []).append(row["legal_version_id"])
        else:
            roots.append(row["legal_version_id"])
    ordered: list[tuple[str, int]] = []
    seen: set[str] = set()
    frontier = [(version_id, 0) for version_id in roots]
    while frontier:
        version_id, depth = frontier.pop(0)
        if version_id in seen or depth > MAX_CHAIN_DEPTH:
            continue
        seen.add(version_id)
        ordered.append((version_id, depth))
        frontier.extend((successor, depth + 1) for successor in successors.get(version_id, []))

    version_ids = [version_id for version_id, _ in ordered]
    events = await pool.fetch(
        """
        SELECT legal_version_id, event_type, event_date, evidence_id
        FROM public.regulatory_validated_legal_events
        WHERE legal_version_id = ANY($1::text[])
        ORDER BY event_date NULLS LAST, event_id
        """,
        version_ids,
    )
    edges = await pool.fetch(
        """
        SELECT relation_type, source_instrument_id, evidence_id, confidence
        FROM public.regulatory_validated_relations
        WHERE target_instrument_id = $1
          AND relation_type = ANY($2::text[])
        ORDER BY relation_id
        """,
        instrument_id,
        list(_CHAIN_EDGE_TYPES),
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
            "legal_version_id": version_id,
            "version_key": by_id[version_id]["version_key"],
            "predecessor_version_id": by_id[version_id]["predecessor_version_id"],
            "consolidation_state": by_id[version_id]["consolidation_state"],
            "depth": depth,
            "events": events_by_version.get(version_id, []),
            "edges": edge_dicts,
        }
        for version_id, depth in ordered
    ]


async def _provision_for_section(
    pool: Any,
    *,
    doc_id: str,
    section_type: str,
    section_ref: str,
) -> str | None:
    """Resolve a stored section to its validated provision, if any."""
    return await pool.fetchval(
        """
        SELECT citation.provision_id
        FROM public.regulatory_validated_section_citations AS citation
        JOIN public.document_sections AS section
          ON section.id = citation.document_section_id
        WHERE section.doc_id = $1
          AND section.section_type = $2
          AND section.section_ref = $3
        ORDER BY citation.provision_id
        LIMIT 1
        """,
        doc_id,
        section_type,
        section_ref,
    )


async def _fetch_hop_edges(
    pool: Any,
    *,
    frontier: list[str],
    provision_id: str | None,
    types: list[str] | None,
) -> list[Any]:
    """One hop of the neighborhood: validated edges touching any frontier instrument.

    The SQL is rebuilt from scratch per hop, so placeholder numbering is
    always derived from the params list built right here (clarity over
    cleverness).
    """
    params: list[Any] = [frontier]
    conditions = [
        "(source_instrument_id = ANY($1::text[]) OR target_instrument_id = ANY($1::text[]))"
    ]
    if provision_id is not None:
        params.append(provision_id)
        placeholder = f"${len(params)}"
        conditions.append(
            f"(source_provision_id = {placeholder} OR target_provision_id = {placeholder}"
            " OR (source_provision_id IS NULL AND target_provision_id IS NULL))"
        )
    if types:
        params.append(types)
        conditions.append(f"relation_type = ANY(${len(params)}::text[])")
    query = f"""
        SELECT relation_id, relation_type, source_instrument_id, target_instrument_id,
               target_external_ref, evidence_id, confidence
        FROM public.regulatory_validated_relations
        WHERE {" AND ".join(conditions)}
        ORDER BY relation_id
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
    depth: int = 1,
) -> list[dict]:
    """Validated relation neighborhood of a document (optionally one section)."""
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction must be one of {_DIRECTIONS}")
    depth = max(1, min(depth, MAX_XREF_DEPTH))
    instrument_id = await _instrument_for_doc(pool, doc_id)
    if instrument_id is None:
        return []

    provision_id: str | None = None
    if section_type and section_ref:
        provision_id = await _provision_for_section(
            pool, doc_id=doc_id, section_type=section_type, section_ref=section_ref
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
    """Validated one-hop neighbors resolved back to concrete stored sections."""
    rows = await pool.fetch(
        """
        WITH section_provisions AS (
            SELECT section.doc_id, section.section_type, section.section_ref, citation.provision_id
            FROM public.regulatory_validated_section_citations AS citation
            JOIN public.document_sections AS section
              ON section.id = citation.document_section_id
        )
        SELECT DISTINCT m2.doc_id, m2.section_type, m2.section_ref, r.relation_type
        FROM section_provisions m1
        JOIN public.regulatory_validated_relations r
          ON (r.source_provision_id = m1.provision_id OR r.target_provision_id = m1.provision_id)
        JOIN section_provisions m2
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
