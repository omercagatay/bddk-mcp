"""Regulatory graph tools: validated amendment chains and cross-references.

Both tools read only the ``regulatory_validated_*`` views, so every returned
version, event, and edge is reviewer-validated and backed by a non-fixture
artifact.  Unvalidated machine-extracted candidates are never served.
"""

from __future__ import annotations

import logging
import time

import asyncpg

from bddk_mcp.core.deps import Dependencies
from bddk_mcp.observability.telemetry import elapsed_ms, record_tool_call_trace
from bddk_mcp.regulatory.graph_queries import (
    _instrument_for_doc,
    amendment_chain,
    cross_references,
)
from bddk_mcp.store.legal_ref import turkish_casefold
from bddk_mcp.tools.contract_types import DocumentId, EdgeDirection, SectionRef, SectionType
from bddk_mcp.tools.structured_outputs import (
    AmendmentChainResponse,
    AmendmentChainToolResult,
    AmendmentChainVersion,
    CrossReferencesResponse,
    CrossReferencesToolResult,
    LegalEventItem,
    RelationEdgeItem,
    structured_tool_result,
)
from bddk_mcp.tools.tool_logging import logged_tool

logger = logging.getLogger(__name__)

_NO_COVERAGE = (
    "Bu doküman için düzenleyici graf kapsamı bulunmuyor: doküman henüz "
    "doğrulanmış canonical legal-version modeline bağlanmamış. Bölüm araması "
    "ve doküman araçları kullanılabilir durumda."
)

_NO_EDGES_MATCH_FILTERS = (
    "Bu doküman düzenleyici graf kapsamında, ancak mevcut filtrelerle (yön, "
    "bölüm) eşleşen doğrulanmış çapraz referans kenarı bulunamadı. Yalnızca "
    "insan onaylı kenarlar sunulur; filtreleri gevşetmeyi deneyebilirsiniz."
)

_VALIDATED_ONLY_NOTE = "Yalnızca insan onaylı (validated) kenarlar ve sürümler listelenir."


def _normalize_section_arg(value: str | int | None) -> str | None:
    """Match user section args to the stored normalized forms.

    document_sections values are Python-lowercased by
    section_index._normalize_ref; turkish_casefold is the existing shared
    primitive that additionally lands Turkish dotted 'İ' on plain 'i'
    ("İlke" → "ilke") and is a no-op on the already-normalized stored forms.
    """
    if value is None:
        return None
    normalized = turkish_casefold(str(value).strip())
    return normalized or None


def _edge_line(edge: dict) -> str:
    """Render a cross-reference edge.

    Outgoing edges point at their target; for incoming edges the acting
    instrument is the *source*, so the arrow is flipped and the source shown —
    otherwise "amends → 943" would read as if this document amends itself.
    """
    if edge.get("direction") == "incoming":
        head = f"- `{edge['relation_type']}` ← kaynak: {edge.get('source_instrument_id') or '-'}"
    else:
        target = edge.get("target_external_ref") or edge.get("target_instrument_id") or "-"
        head = f"- `{edge['relation_type']}` → {target}"
    return f"{head} (kanıt: {edge['evidence_id']}, güven: {edge.get('confidence', '-')})"


def _chain_edge_line(edge: dict) -> str:
    """Render an incoming amendment-chain edge (source instrument acts on this one)."""
    return f"- `{edge['relation_type']}` ← kaynak: {edge['source_instrument_id']} (kanıt: {edge['evidence_id']})"


def register(mcp, deps: Dependencies) -> None:
    """Register regulatory graph tools on the given MCP instance."""

    @mcp.tool()
    @logged_tool(logger)
    async def get_amendment_chain(document_id: DocumentId) -> AmendmentChainToolResult:
        """
        Bir düzenlemenin doğrulanmış sürüm zincirini (değişiklik geçmişini) getirir.

        Bir dokümanın hangi doğrulanmış sürümlerden geçtiğini, hangi olaylarla
        (yayım, yürürlük, yerine geçme) değiştiğini ve onu etkileyen doğrulanmış
        düzenlemeleri kanıt kayıtlarıyla birlikte listeler. Yalnızca insan
        onaylı sürümler ve kenarlar döner; makine çıkarımı adaylar hiçbir zaman
        sunulmaz.

        Args:
            document_id: Doküman ID, örn. `943` veya `mevzuat_22599`
        """
        start = time.perf_counter()
        args = {"document_id": document_id}
        pool = getattr(deps, "pool", None)
        if pool is None:
            return _no_coverage_chain(document_id)
        try:
            chain = await amendment_chain(pool, doc_id=document_id)
        except asyncpg.exceptions.UndefinedTableError:
            # Deployment where migration v0009 was never applied: explicit
            # no-coverage marker instead of an error dump.
            return _no_coverage_chain(document_id)
        if not chain:
            await record_tool_call_trace(
                getattr(deps, "telemetry_pool", None),
                tool_name="get_amendment_chain",
                args=args,
                latency_ms=elapsed_ms(start),
                result_count=0,
                doc_ids=[document_id],
                relevance_stats={"status": "no_coverage"},
            )
            return _no_coverage_chain(document_id)
        lines = [f"## Doğrulanmış sürüm zinciri — {document_id}", "", _VALIDATED_ONLY_NOTE, ""]
        for entry in chain:
            lines.append(f"### {entry['version_key']} (`{entry['legal_version_id']}`)")
            if entry["predecessor_version_id"]:
                lines.append(f"- Önceki sürüm: `{entry['predecessor_version_id']}`")
            lines.append(f"- Konsolidasyon durumu: {entry['consolidation_state']}")
            for event in entry["events"]:
                lines.append(
                    f"- Olay: {event['event_type']}"
                    f" ({event['event_date'] or 'tarih bilinmiyor'},"
                    f" kanıt: {event['evidence_id']})"
                )
            lines.append("")
        edges = chain[0]["edges"]
        if edges:
            lines.append("### Bu düzenlemeyi etkileyen doğrulanmış kenarlar")
            lines.extend(_chain_edge_line(edge) for edge in edges)
        await record_tool_call_trace(
            getattr(deps, "telemetry_pool", None),
            tool_name="get_amendment_chain",
            args=args,
            latency_ms=elapsed_ms(start),
            result_count=len(chain),
            doc_ids=[document_id],
            relevance_stats={"status": "chain"},
        )
        instrument_id = await _instrument_for_doc(pool, document_id)
        return structured_tool_result(
            AmendmentChainResponse(
                status="ok",
                text="\n".join(lines).rstrip(),
                document_id=document_id,
                instrument_id=instrument_id,
                versions=[
                    AmendmentChainVersion(
                        legal_version_id=entry["legal_version_id"],
                        version_key=entry["version_key"],
                        predecessor_version_id=entry["predecessor_version_id"],
                        consolidation_state=entry["consolidation_state"],
                        depth=entry["depth"],
                        events=[LegalEventItem(**event) for event in entry["events"]],
                    )
                    for entry in chain
                ],
                incoming_edges=[
                    RelationEdgeItem(
                        relation_type=edge["relation_type"],
                        direction="incoming",
                        source_instrument_id=edge["source_instrument_id"],
                        evidence_id=edge["evidence_id"],
                        confidence=edge["confidence"],
                        depth=1,
                    )
                    for edge in edges
                ],
            )
        )

    @mcp.tool()
    @logged_tool(logger)
    async def get_cross_references(
        document_id: DocumentId,
        section_type: SectionType = None,
        section_ref: SectionRef = None,
        direction: EdgeDirection = "both",
    ) -> CrossReferencesToolResult:
        """
        Bir doküman veya bölümün doğrulanmış çapraz referans kenarlarını getirir.

        Yalnızca insan onaylı kenarlar döner; makine çıkarımı adaylar insan
        incelemesinden geçene kadar hiçbir profile sunulmaz.

        Args:
            document_id: Doküman ID, örn. `943` veya `mevzuat_22599`
            section_type: Opsiyonel bölüm türü, örn. `madde`
            section_ref: Opsiyonel bölüm referansı, örn. `9`
            direction: `both`, `incoming` veya `outgoing`
        """
        start = time.perf_counter()
        args = {
            "document_id": document_id,
            "section_type": section_type,
            "section_ref": section_ref,
            "direction": direction,
        }
        normalized_type = _normalize_section_arg(section_type)
        normalized_ref = _normalize_section_arg(section_ref)
        pool = getattr(deps, "pool", None)
        if pool is None:
            return _no_coverage_xref(document_id, normalized_type, normalized_ref, direction)
        try:
            edges = await cross_references(
                pool,
                doc_id=document_id,
                section_type=normalized_type,
                section_ref=normalized_ref,
                direction=direction,
            )
            # Distinguish "doc not in the validated graph at all" from "doc is
            # mapped but the filters excluded every edge".
            is_mapped = bool(edges) or await _instrument_for_doc(pool, document_id) is not None
        except asyncpg.exceptions.UndefinedTableError:
            return _no_coverage_xref(document_id, normalized_type, normalized_ref, direction)
        if not edges:
            status = "no_edges_matched_filters" if is_mapped else "no_coverage"
            await record_tool_call_trace(
                getattr(deps, "telemetry_pool", None),
                tool_name="get_cross_references",
                args=args,
                latency_ms=elapsed_ms(start),
                result_count=0,
                doc_ids=[document_id],
                relevance_stats={"status": status},
            )
            if is_mapped:
                return structured_tool_result(
                    CrossReferencesResponse(
                        status="no_results",
                        text=_NO_EDGES_MATCH_FILTERS,
                        document_id=document_id,
                        section_type=normalized_type,
                        section_ref=normalized_ref,
                        direction=direction,
                    )
                )
            return _no_coverage_xref(document_id, normalized_type, normalized_ref, direction)
        grouped: dict[str, list[dict]] = {}
        for edge in edges:
            grouped.setdefault(edge["relation_type"], []).append(edge)
        lines = [f"## Doğrulanmış çapraz referanslar — {document_id}", "", _VALIDATED_ONLY_NOTE, ""]
        for relation_type in sorted(grouped):
            lines.append(f"### {relation_type}")
            lines.extend(_edge_line(edge) for edge in grouped[relation_type])
            lines.append("")
        await record_tool_call_trace(
            getattr(deps, "telemetry_pool", None),
            tool_name="get_cross_references",
            args=args,
            latency_ms=elapsed_ms(start),
            result_count=len(edges),
            doc_ids=[document_id],
            relevance_stats={"status": "edges"},
        )
        return structured_tool_result(
            CrossReferencesResponse(
                status="ok",
                text="\n".join(lines).rstrip(),
                document_id=document_id,
                section_type=normalized_type,
                section_ref=normalized_ref,
                direction=direction,
                edges=[
                    RelationEdgeItem(
                        relation_type=edge["relation_type"],
                        direction=edge["direction"],
                        source_instrument_id=edge["source_instrument_id"],
                        target_instrument_id=edge["target_instrument_id"],
                        target_external_ref=edge["target_external_ref"],
                        evidence_id=edge["evidence_id"],
                        confidence=edge["confidence"],
                        depth=edge["depth"],
                    )
                    for edge in edges
                ],
            )
        )


def _no_coverage_chain(document_id: str) -> AmendmentChainToolResult:
    return structured_tool_result(
        AmendmentChainResponse(
            status="no_results",
            text=_NO_COVERAGE,
            document_id=document_id,
        )
    )


def _no_coverage_xref(
    document_id: str,
    section_type: str | None,
    section_ref: str | None,
    direction: str,
) -> CrossReferencesToolResult:
    return structured_tool_result(
        CrossReferencesResponse(
            status="no_results",
            text=_NO_COVERAGE,
            document_id=document_id,
            section_type=section_type,
            section_ref=section_ref,
            direction=direction,
        )
    )
