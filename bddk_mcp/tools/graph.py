"""Regulatory graph tools: amendment chains and cross-references."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import asyncpg

from bddk_mcp.observability.telemetry import elapsed_ms, record_tool_call_trace
from bddk_mcp.regulatory.graph_queries import (
    _instrument_for_doc,
    amendment_chain,
    cross_references,
)
from bddk_mcp.store.legal_ref import turkish_casefold

if TYPE_CHECKING:
    from bddk_mcp.core.deps import Dependencies

_NO_COVERAGE = (
    "Bu doküman için düzenleyici graf kapsamı bulunmuyor: doküman henüz "
    "canonical legal-version modeline bağlanmamış. Bölüm araması ve doküman "
    "araçları kullanılabilir durumda."
)

_NO_EDGES_MATCH_FILTERS = (
    "Bu doküman düzenleyici graf kapsamında, ancak mevcut filtrelerle "
    "(doğrulama durumu, yön, bölüm) eşleşen çapraz referans kenarı bulunamadı. "
    "`include_unvalidated=true` ile makine çıkarımı kenarları dahil edebilir "
    "veya filtreleri gevşetebilirsiniz."
)


def _normalize_section_arg(value: str | None) -> str | None:
    """Match user section args to the stored normalized forms.

    document_sections values are Python-lowercased by
    section_index._normalize_ref; turkish_casefold is the existing shared
    primitive that additionally lands Turkish dotted 'İ' on plain 'i'
    ("İlke" → "ilke") and is a no-op on the already-normalized stored forms.
    """
    if value is None:
        return None
    normalized = turkish_casefold(value.strip())
    return normalized or None


def _validation_flag(edge: dict) -> str:
    if edge.get("validation_state") != "human_validated":
        return " [doğrulanmamış — makine çıkarımı]"
    return ""


def _edge_line(edge: dict) -> str:
    """Render a cross-reference edge (full relation row).

    Outgoing edges point at their target; for incoming edges the acting
    instrument is the *source*, so the arrow is flipped and the source shown —
    otherwise "amends → 943" would read as if this document amends itself.
    """
    if edge.get("direction") == "incoming":
        head = f"- `{edge['relation_type']}` ← kaynak: {edge.get('source_instrument_id') or '-'}"
    else:
        target = edge.get("target_external_ref") or edge.get("target_instrument_id") or "-"
        head = f"- `{edge['relation_type']}` → {target}"
    return f"{head} (kanıt: {edge['evidence_id']}, güven: {edge.get('confidence', '-')}){_validation_flag(edge)}"


def _chain_edge_line(edge: dict) -> str:
    """Render an incoming amendment-chain edge (source instrument acts on this one)."""
    return (
        f"- `{edge['relation_type']}` ← kaynak: {edge['source_instrument_id']}"
        f" (kanıt: {edge['evidence_id']})"
        f"{_validation_flag(edge)}"
    )


def register(mcp, deps: Dependencies) -> None:
    """Register regulatory graph tools on the given MCP instance."""

    @mcp.tool()
    async def get_amendment_chain(document_id: str, include_unvalidated: bool = False) -> str:
        """
        Bir düzenlemenin sürüm zincirini (değişiklik/yürürlük geçmişini) getirir.

        Bir dokümanın hangi sürümlerden geçtiğini, hangi olaylarla (yayım,
        yürürlük, yerine geçme) değiştiğini ve onu etkileyen düzenlemeleri
        kanıt kayıtlarıyla birlikte listeler. Varsayılan olarak yalnızca insan
        onaylı kenarlar döner; makine çıkarımı kenarlar `include_unvalidated=true`
        ile açıkça işaretlenerek listelenir. Reddedilmiş kenarlar hiçbir zaman
        gösterilmez.

        Args:
            document_id: Doküman ID, örn. `943` veya `mevzuat_22599`
            include_unvalidated: İnsan onayı olmayan makine çıkarımı kenarları da göster
        """
        start = time.perf_counter()
        args = {"document_id": document_id, "include_unvalidated": include_unvalidated}
        pool = getattr(deps, "pool", None)
        if pool is None:
            return _NO_COVERAGE
        try:
            chain = await amendment_chain(pool, doc_id=document_id, include_unvalidated=include_unvalidated)
        except asyncpg.exceptions.UndefinedTableError:
            # Deployment where the regulatory schema was never applied:
            # explicit no-coverage marker instead of an error dump (spec §6).
            return _NO_COVERAGE
        if not chain:
            await record_tool_call_trace(
                pool,
                tool_name="get_amendment_chain",
                args=args,
                latency_ms=elapsed_ms(start),
                result_count=0,
                doc_ids=[document_id],
                relevance_stats={"status": "no_coverage"},
            )
            return _NO_COVERAGE
        lines = [f"## Sürüm zinciri — {document_id}", ""]
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
            lines.append("### Bu düzenlemeyi etkileyen kenarlar")
            lines.extend(_chain_edge_line(edge) for edge in edges)
        await record_tool_call_trace(
            pool,
            tool_name="get_amendment_chain",
            args=args,
            latency_ms=elapsed_ms(start),
            result_count=len(chain),
            doc_ids=[document_id],
            relevance_stats={"status": "chain"},
        )
        return "\n".join(lines)

    @mcp.tool()
    async def get_cross_references(
        document_id: str,
        section_type: str | None = None,
        section_ref: str | None = None,
        direction: str = "both",
        include_unvalidated: bool = False,
    ) -> str:
        """
        Bir doküman veya bölümün çapraz referans kenarlarını getirir.

        Varsayılan olarak yalnızca insan onaylı kenarlar döner; makine
        çıkarımı kenarlar `include_unvalidated=true` ile açıkça işaretlenerek
        listelenir.

        Args:
            document_id: Doküman ID, örn. `943` veya `mevzuat_22599`
            section_type: Opsiyonel bölüm türü, örn. `madde`
            section_ref: Opsiyonel bölüm referansı, örn. `9`
            direction: `both`, `incoming` veya `outgoing`
            include_unvalidated: İnsan onayı olmayan makine çıkarımı kenarları da göster
        """
        start = time.perf_counter()
        args = {
            "document_id": document_id,
            "section_type": section_type,
            "section_ref": section_ref,
            "direction": direction,
            "include_unvalidated": include_unvalidated,
        }
        pool = getattr(deps, "pool", None)
        if pool is None:
            return _NO_COVERAGE
        try:
            edges = await cross_references(
                pool,
                doc_id=document_id,
                section_type=_normalize_section_arg(section_type),
                section_ref=_normalize_section_arg(section_ref),
                direction=direction,
                include_unvalidated=include_unvalidated,
            )
            # Distinguish "doc not in the graph at all" from "doc is mapped
            # but the filters excluded every edge" before claiming no coverage.
            is_mapped = bool(edges) or await _instrument_for_doc(pool, document_id) is not None
        except asyncpg.exceptions.UndefinedTableError:
            # Deployment where the regulatory schema was never applied:
            # explicit no-coverage marker instead of an error dump (spec §6).
            return _NO_COVERAGE
        if not edges:
            status = "no_edges_matched_filters" if is_mapped else "no_coverage"
            await record_tool_call_trace(
                pool,
                tool_name="get_cross_references",
                args=args,
                latency_ms=elapsed_ms(start),
                result_count=0,
                doc_ids=[document_id],
                relevance_stats={"status": status},
            )
            return _NO_EDGES_MATCH_FILTERS if is_mapped else _NO_COVERAGE
        grouped: dict[str, list[dict]] = {}
        for edge in edges:
            grouped.setdefault(edge["relation_type"], []).append(edge)
        lines = [f"## Çapraz referanslar — {document_id}", ""]
        for relation_type in sorted(grouped):
            lines.append(f"### {relation_type}")
            lines.extend(_edge_line(edge) for edge in grouped[relation_type])
            lines.append("")
        await record_tool_call_trace(
            pool,
            tool_name="get_cross_references",
            args=args,
            latency_ms=elapsed_ms(start),
            result_count=len(edges),
            doc_ids=[document_id],
            relevance_stats={"status": "edges"},
        )
        return "\n".join(lines).rstrip()
