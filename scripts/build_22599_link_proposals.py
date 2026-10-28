"""Write the rule-extracted link proposals for mevzuat_22599 to a reviewable file.

Reads the stored text from ``seed_data/documents.json`` and the ontology
specification, runs the three extraction rules, and writes
``docs/evidence/legal-versions/22599.proposals.json``.  The output is
deterministic for the same inputs; ``--check`` verifies that the committed
file still matches them.  This script opens no database connection: proposals
count for nothing until the review store approves them.

Usage: uv run python scripts/build_22599_link_proposals.py [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from bddk_mcp.ontology.proposals import ProposalFile, dump_proposal_file
from bddk_mcp.ontology.rule_extract import build_proposal_file
from bddk_mcp.ontology.spec import DEFAULT_SPEC_PATH

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "seed_data" / "documents.json"
OUT = ROOT / "docs" / "evidence" / "legal-versions" / "22599.proposals.json"
DOCUMENT_ID = "mevzuat_22599"


def build(*, docs_path: Path = DOCS, spec_path: Path = DEFAULT_SPEC_PATH) -> ProposalFile:
    document = next(
        item for item in json.loads(docs_path.read_text(encoding="utf-8")) if item["document_id"] == DOCUMENT_ID
    )
    return build_proposal_file(
        document_id=DOCUMENT_ID,
        content_hash=document["content_hash"],
        title=document["title"],
        text=document["markdown_content"],
        ontology_spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
    )


def summary(file: ProposalFile) -> str:
    lines = [
        f"document={file.document_id} content_hash={file.content_hash[:12]}…",
        f"rg_table: original {file.rg_table.original.rg.label_tr()}, {len(file.rg_table.amendments)} amendments",
        f"proposals={file.counts.total} links={file.counts.links} events={file.counts.events}",
        "by_rule=" + ", ".join(f"{rule}:{count}" for rule, count in file.counts.by_rule.items()),
        "by_flag=" + ", ".join(f"{flag}:{count}" for flag, count in file.counts.by_flag.items()),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if the committed file differs from a fresh build")
    parser.add_argument("--out", type=Path, default=OUT, help="output path (default: the committed evidence file)")
    args = parser.parse_args(argv)

    file = build()
    rendered = dump_proposal_file(file)
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.is_file() else None
        if current != rendered:
            print(f"{args.out} is stale or missing; rerun without --check to regenerate", file=sys.stderr)
            return 1
        print(f"{args.out} matches the stored text and rules")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.out}")
    print(summary(file))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
