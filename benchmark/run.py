# benchmark/run.py
"""CLI entrypoint for BDDK benchmark.

Usage:
    python -m benchmark.run                         # Full pipeline, all models
    python -m benchmark.run --phase 1               # Phase 1 only (offline)
    python -m benchmark.run --phase 2               # Phase 2 (requires MCP server)
    python -m benchmark.run --phase 3               # Phase 3 (prompt engineering)
    python -m benchmark.run --model gemma-4-26b-a4b-it   # Single model
    python -m benchmark.run --diagnose              # Diagnosis report only (from last results)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from benchmark.config import MODELS, PHASE1_THRESHOLDS
from benchmark.phase1_nli import run_phase1b
from benchmark.phase1_terms import run_phase1c
from benchmark.phase1_tools import run_phase1a
from benchmark.phase2_e2e import run_phase2
from benchmark.phase3_prompts import run_phase3
from benchmark.report import console_report, diagnosis_report, save_json_results

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "benchmark_results"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BDDK Benchmark Harness")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], help="Run specific phase only")
    parser.add_argument("--model", type=str, help="Run single model (by name)")
    parser.add_argument("--diagnose", action="store_true", help="Print diagnosis from last results")
    parser.add_argument("--mcp-url", default="http://localhost:8000", help="MCP server URL for Phase 2")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    return parser.parse_args()


def _get_models(model_filter: str | None) -> list[dict]:
    if model_filter:
        matches = [m for m in MODELS if m["name"] == model_filter]
        if not matches:
            print(f"Model '{model_filter}' not found. Available: {[m['name'] for m in MODELS]}")
            sys.exit(1)
        return matches
    return MODELS


def _passes_phase1(results: dict, model_name: str) -> bool:
    """Check if a model passes all Phase 1 thresholds."""
    p1a = results.get("phase1a", {}).get(model_name, {})
    p1b = results.get("phase1b", {}).get(model_name, {})
    p1c = results.get("phase1c", {}).get(model_name, {})

    if not p1a or not p1b or not p1c:
        return False

    return (
        p1a.get("tool_selection_accuracy", 0) >= PHASE1_THRESHOLDS["tool_selection"]
        and p1b.get("macro_f1", 0) >= PHASE1_THRESHOLDS["nli_macro_f1"]
        and p1c.get("accuracy", 0) >= PHASE1_THRESHOLDS["terminology"]
    )


async def _main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    models = _get_models(args.model)
    all_results: dict = {}

    # Phase 1
    if args.phase is None or args.phase == 1:
        print("\n=== PHASE 1: Offline Evaluation ===\n")

        all_results["phase1a"] = {}
        all_results["phase1b"] = {}
        all_results["phase1c"] = {}

        for model in models:
            tag = model["ollama_tag"]
            name = model["name"]
            print(f"\n--- {name} ({tag}) ---\n")

            print("  Phase 1a: Tool-calling...")
            all_results["phase1a"][name] = await run_phase1a(tag)

            print("  Phase 1b: NLI...")
            all_results["phase1b"][name] = await run_phase1b(tag)

            print("  Phase 1c: Terminology...")
            all_results["phase1c"][name] = await run_phase1c(tag)

    # Phase 2 (only models that pass Phase 1)
    if args.phase is None or args.phase == 2:
        print("\n=== PHASE 2: Live End-to-End ===\n")
        all_results["phase2"] = {}

        for model in models:
            name = model["name"]
            tag = model["ollama_tag"]

            if args.phase == 2 or _passes_phase1(all_results, name):
                print(f"  {name}: running Phase 2...")
                all_results["phase2"][name] = await run_phase2(tag, args.mcp_url)
            else:
                print(f"  {name}: SKIPPED (below Phase 1 thresholds)")

    # Phase 3
    if args.phase is None or args.phase == 3:
        print("\n=== PHASE 3: Prompt Engineering ===\n")
        all_results["phase3"] = {}

        for model in models:
            name = model["name"]
            tag = model["ollama_tag"]
            print(f"  {name}: testing prompt fixes...")
            all_results["phase3"][name] = await run_phase3(tag, all_results, args.mcp_url)

    # Reports
    print("\n" + console_report(all_results))

    if args.diagnose or args.phase is None:
        print("\n" + diagnosis_report(all_results))

    json_path = save_json_results(all_results, OUTPUT_DIR)
    print(f"\nResults saved to: {json_path}")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
