"""Regulatory graph tools: amendment chains and cross-references."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from bddk_mcp.observability.telemetry import elapsed_ms, record_tool_call_trace
from bddk_mcp.regulatory.graph_queries import amendment_chain, cross_references

if TYPE_CHECKING:
    from bddk_mcp.core.deps import Dependencies

_NO_COVERAGE = (
    "Bu doküman için düzenleyici graf kapsamı bulunmuyor: doküman henüz "
    "canonical legal-version modeline bağlanmamış. Bölüm araması ve doküman "
    "araçları kullanılabilir durumda."
)


def _validation_flag(edge: dict) -> str:
    if edge.get("validation_state") != "human_validated":
        return " [doğrulanmamış — makine çıkarımı]"
    return ""


def _edge_line(edge: dict) -> str:
    """Render a cross-reference edge (full relation row)."""
    target = edge.get("target_external_ref") or edge.get("target_instrument_id") or "-"
    return (
        f"- `{edge['relation_type']}` → {target}"
        f" (kanıt: {edge['evidence_id']}, güven: {edge.get('confidence', '-')})"
        f"{_validation_flag(edge)}"
    )


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
    async def get_amendment_chain(document_id: str) -> str:
        """
        Bir düzenlemenin sürüm zincirini (değişiklik/yürürlük geçmişini) getirir.

        Bir dokümanın hangi sürümlerden geçtiğini, hangi olaylarla (yayım,
        yürürlük, yerine geçme) değiştiğini ve onu etkileyen düzenlemeleri
        kanıt kayıtlarıyla birlikte listeler.

        Args:
            document_id: Doküman ID, örn. `943` veya `mevzuat_22599`
        """
        start = time.perf_counter()
        args = {"document_id": document_id}
        pool = getattr(deps, "pool", None)
        if pool is None:
            return _NO_COVERAGE
        chain = await amendment_chain(pool, doc_id=document_id)
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
        edges = await cross_references(
            pool,
            doc_id=document_id,
            section_type=section_type,
            section_ref=section_ref,
            direction=direction,
            include_unvalidated=include_unvalidated,
        )
        if not edges:
            await record_tool_call_trace(
                pool,
                tool_name="get_cross_references",
                args=args,
                latency_ms=elapsed_ms(start),
                result_count=0,
                doc_ids=[document_id],
                relevance_stats={"status": "no_coverage"},
            )
            return _NO_COVERAGE
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
