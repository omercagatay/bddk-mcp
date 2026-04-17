# benchmark/report.py
"""Console + JSON + diagnosis report generator."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from benchmark.config import PHASE1_THRESHOLDS

logger = logging.getLogger(__name__)


def console_report(all_results: dict) -> str:
    """Generate a human-readable console report."""
    lines = []
    lines.append("=" * 90)
    lines.append("BDDK BENCHMARK RESULTS")
    lines.append(f"Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("=" * 90)

    # Phase 1a summary table
    if "phase1a" in all_results:
        lines.append("\n## Phase 1a: Tool-Calling Accuracy\n")
        lines.append(f"{'Model':<30} {'Tool Acc':>10} {'Consist':>10} {'Param F1':>10} {'Latency':>10}")
        lines.append("-" * 72)
        for model_name, result in all_results["phase1a"].items():
            acc = result["tool_selection_accuracy"]
            cons = result["tool_consistency"]
            pf1 = result["avg_parameter_f1"]
            lat = result["avg_latency_s"]
            threshold_marker = " *" if acc < PHASE1_THRESHOLDS["tool_selection"] else ""
            lines.append(f"{model_name:<30} {acc:>9.1%} {cons:>9.1%} {pf1:>9.2f} {lat:>9.1f}s{threshold_marker}")

    # Phase 1b summary table
    if "phase1b" in all_results:
        lines.append("\n## Phase 1b: Banking NLI\n")
        lines.append(f"{'Model':<30} {'Accuracy':>10} {'Macro-F1':>10} {'Unknown':>10}")
        lines.append("-" * 62)
        for model_name, result in all_results["phase1b"].items():
            acc = result["accuracy"]
            f1 = result["macro_f1"]
            unk = result.get("unknown_responses", 0)
            threshold_marker = " *" if f1 < PHASE1_THRESHOLDS["nli_macro_f1"] else ""
            lines.append(f"{model_name:<30} {acc:>9.1%} {f1:>9.2f} {unk:>10}{threshold_marker}")

    # Phase 1c summary table
    if "phase1c" in all_results:
        lines.append("\n## Phase 1c: BDDK Terminology\n")
        lines.append(f"{'Model':<30} {'Accuracy':>10} {'Correct':>10} {'No Ans':>10}")
        lines.append("-" * 62)
        for model_name, result in all_results["phase1c"].items():
            acc = result["accuracy"]
            cor = result["correct"]
            na = result.get("no_answer_count", 0)
            threshold_marker = " *" if acc < PHASE1_THRESHOLDS["terminology"] else ""
            lines.append(f"{model_name:<30} {acc:>9.1%} {cor:>10} {na:>10}{threshold_marker}")

    # Phase 2 summary table
    if "phase2" in all_results:
        lines.append("\n## Phase 2: End-to-End Grounding\n")
        lines.append(f"{'Model':<30} {'Code Grd':>10} {'Model Grd':>10} {'Chain %':>10} {'Errors':>10}")
        lines.append("-" * 72)
        for model_name, result in all_results["phase2"].items():
            cg = result["avg_code_grounding"]
            mg = result["avg_model_grounding"]
            cs = result["chain_success_rate"]
            er = result["error_count"]
            lines.append(f"{model_name:<30} {cg:>9.1%} {mg:>9.1%} {cs:>9.1%} {er:>10}")

    # Threshold legend
    lines.append(
        f"\n* Below threshold (tool>{PHASE1_THRESHOLDS['tool_selection']:.0%}, "
        f"nli>{PHASE1_THRESHOLDS['nli_macro_f1']:.2f}, "
        f"term>{PHASE1_THRESHOLDS['terminology']:.0%})"
    )
    lines.append("=" * 90)

    return "\n".join(lines)


def diagnosis_report(all_results: dict) -> str:
    """Generate per-model diagnosis with recommended fixes."""
    lines = []
    lines.append("=" * 90)
    lines.append("BDDK BENCHMARK — DIAGNOSIS REPORT")
    lines.append("=" * 90)

    models = set()
    for phase_results in all_results.values():
        if isinstance(phase_results, dict):
            models.update(phase_results.keys())

    for model_name in sorted(models):
        lines.append(f"\n### {model_name}\n")
        failures = []
        recommendations = []

        # Check Phase 1a
        p1a = all_results.get("phase1a", {}).get(model_name)
        if p1a:
            acc = p1a["tool_selection_accuracy"]
            if acc < PHASE1_THRESHOLDS["tool_selection"]:
                failures.append(f"Tool selection: {acc:.1%} (threshold: {PHASE1_THRESHOLDS['tool_selection']:.0%})")
                recommendations.append("Try: few-shot examples + chain-of-thought in system prompt")

        # Check Phase 1b
        p1b = all_results.get("phase1b", {}).get(model_name)
        if p1b:
            f1 = p1b["macro_f1"]
            if f1 < PHASE1_THRESHOLDS["nli_macro_f1"]:
                failures.append(f"NLI macro-F1: {f1:.2f} (threshold: {PHASE1_THRESHOLDS['nli_macro_f1']:.2f})")
                recommendations.append("Try: glossary injection + few-shot NLI examples")

        # Check Phase 1c
        p1c = all_results.get("phase1c", {}).get(model_name)
        if p1c:
            term_acc = p1c["accuracy"]
            if term_acc < PHASE1_THRESHOLDS["terminology"]:
                failures.append(f"Terminology: {term_acc:.1%} (threshold: {PHASE1_THRESHOLDS['terminology']:.0%})")
                recommendations.append("Try: BDDK glossary injection in system prompt")

        # Check Phase 2
        p2 = all_results.get("phase2", {}).get(model_name)
        if p2:
            grounding = p2["avg_model_grounding"]
            if grounding < 0.6:
                failures.append(f"Grounding: {grounding:.1%}")
                recommendations.append("Try: RAG grounding instruction ('only cite tool results')")

        if not failures:
            lines.append("  PASS — All metrics above threshold")
            lines.append("  Recommendation: Deploy with RAG + current prompts")
        else:
            lines.append("  FAILURES:")
            for f in failures:
                lines.append(f"    - {f}")
            lines.append("  RECOMMENDATIONS:")
            for r in recommendations:
                lines.append(f"    - {r}")

            if len(failures) >= 3:
                lines.append("  NOTE: Multiple failures — consider fine-tuning if Phase 3 fixes don't help")

    return "\n".join(lines)


def save_json_results(all_results: dict, output_dir: Path) -> Path:
    """Save results to a timestamped JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"benchmark_{timestamp}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    logger.info("Results saved to %s", path)
    return path
