#!/usr/bin/env python3
"""Real-model gate of the section-search audit set (maintainer tool; CI runs the same check in image-smoke).

Usage: uv run python scripts/eval_section_search.py --corpus PATH [--json]

Opens the corpus file read-only through the MCP server exactly as it serves (real embedding model from
BDDK_EMBEDDING_MODEL_PATH), calls ``search_document_sections`` with only the query of each answerable question of
``tests/fixtures/section_search_audit.json`` and compares the outcome with the recorded ``real_model`` outcome. A
question is found when a returned result of a gold document overlaps the span of a gold section. Prints one line per
question whose outcome differs (lost or gained) and a summary line. Exit 0 when every outcome matches the record, 1
when one differs, 2 on a usage or corpus error (including gold spans that differ from the corpus file's sections).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "section_search_audit.json"

# Shared with scripts/ci_image_smoke.py and tests/test_section_search_audit.py; keep this part free of bddk_mcp
# imports, because the corpus path must be set before bddk_mcp is imported.


def gated_questions(path: Path = DATA) -> list[dict]:
    """The answerable questions of the audit set (category 'absent' is not gated)."""
    return [q for q in json.loads(path.read_text(encoding="utf-8"))["questions"] if q["category"] != "absent"]


def is_found(results: list[dict], gold: list[dict]) -> bool:
    """Some returned result of a gold document overlaps the character span of a gold section."""
    return any(
        r.get("document_id") == g["document_id"]
        and r.get("start_char", -1) < g["end_char"]
        and r.get("end_char", -1) > g["start_char"]
        for r in results
        for g in gold
    )


def difference(question: dict, found: bool, recorded_key: str, results: list[dict]) -> str | None:
    """One line when the outcome differs from the recorded one, else None.

    A question marked ``platform_dependent`` (its outcome was measured to differ between Windows and Linux; the
    data file says why in ``note``) is never reported: one record cannot be right on both.
    """
    recorded = question[recorded_key] == "found"
    if found == recorded or question.get("platform_dependent"):
        return None
    top = ", ".join(f"{r.get('document_id')}:{r.get('section_type')}:{r.get('section_ref')}" for r in results)
    word = "GAINED" if found else "LOST"
    return f'{word} {question["id"]} (recorded {question[recorded_key]}): "{question["query"]}" -> [{top}]'


def _check_spans(corpus: Path, questions: list[dict]) -> list[str]:
    con = sqlite3.connect(f"{corpus.resolve().as_uri()}?mode=ro", uri=True)
    try:
        wrong = []
        for q in questions:
            for g in q["gold"]:
                rows = con.execute(
                    "SELECT start_char, end_char FROM document_sections"
                    " WHERE doc_id = ? AND section_type = ? AND section_ref = ?",
                    (g["document_id"], g["section_type"], g["section_ref"]),
                ).fetchall()
                if rows != [(g["start_char"], g["end_char"])]:
                    wrong.append(f"{q['id']} {g['document_id']} {g['section_type']} {g['section_ref']}: {rows}")
        return wrong
    finally:
        con.close()


async def _run(questions: list[dict]) -> list[tuple[dict, bool, list[dict]]]:
    from mcp.server.fastmcp.exceptions import ToolError

    from bddk_mcp.server import create_deps, create_mcp, teardown_deps

    deps = await create_deps()
    try:
        server = create_mcp(deps)
        rows = []
        for q in questions:
            try:
                res = await server.call_tool("search_document_sections", {"query": q["query"]})
            except ToolError:
                res = None
            structured = res[1] if isinstance(res, tuple) else getattr(res, "structuredContent", None)
            results = (structured or {}).get("results") or []
            rows.append((q, is_found(results, q["gold"]), results))
        return rows
    finally:
        await teardown_deps(deps)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True, help="Corpus file to open read-only.")
    parser.add_argument("--json", action="store_true", help="Print one JSON object with every outcome instead.")
    args = parser.parse_args(argv)
    if not args.corpus.is_file():
        print(f"Corpus file not found: {args.corpus}", file=sys.stderr)
        return 2
    questions = gated_questions()
    try:
        wrong = _check_spans(args.corpus, questions)
    except sqlite3.Error as exc:
        print(f"Corpus file could not be read: {exc}", file=sys.stderr)
        return 2
    if wrong:
        print("Gold spans of the audit set differ from this corpus file's sections:", file=sys.stderr)
        print("\n".join(wrong), file=sys.stderr)
        return 2
    os.environ["BDDK_CORPUS_PATH"] = str(args.corpus)  # read when bddk_mcp is imported, in _run
    logging.disable(logging.WARNING)
    try:
        rows = asyncio.run(_run(questions))
    except Exception as exc:  # corpus not ready, model missing, server failure
        print(f"Could not run the audit through the server: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    lines = [line for q, found, results in rows if (line := difference(q, found, "real_model", results))]
    found_count = sum(found for _, found, _ in rows)
    recorded = sum(q["real_model"] == "found" for q in questions)
    if args.json:
        print(
            json.dumps(
                {
                    "found": found_count,
                    "recorded_found": recorded,
                    "questions": len(rows),
                    "differences": lines,
                    "outcomes": {q["id"]: "found" if found else "missed" for q, found, _ in rows},
                },
                ensure_ascii=False,
                indent=1,
            )
        )
    else:
        for line in lines:
            print(line)
        print(
            f"section search audit (real model): {found_count} of {len(rows)} found, recorded {recorded}; "
            f"{len(lines)} differ{'s' if len(lines) == 1 else ''}"
        )
    return 1 if lines else 0


if __name__ == "__main__":
    raise SystemExit(main())
