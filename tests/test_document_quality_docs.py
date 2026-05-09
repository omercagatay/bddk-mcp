"""Regression checks for document quality operator documentation."""

from pathlib import Path

from scripts.backfill_quality_failures import load_fail_documents

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_readme_links_document_quality_page():
    readme = _read("README.md")

    assert "[docs/DOCUMENT_QUALITY.md](docs/DOCUMENT_QUALITY.md)" in readme


def test_document_quality_page_covers_operator_workflow():
    page = _read("docs/DOCUMENT_QUALITY.md")

    required_text = [
        "Extraction Methods",
        "Quality Labels",
        "`clean`",
        "`warning`",
        "`fail`",
        "Known Fail List",
        "Backfill Process",
        "Quality Warnings In MCP Results",
        "formula-heavy failed documents require source review",
        "uv run python scripts/scan_document_quality.py --db --out-dir quality_reports --allow-failures",
        "uv run python scripts/backfill_quality_failures.py --dry-run",
    ]
    for text in required_text:
        assert text in page


def test_document_quality_page_lists_tracked_fail_documents():
    page = _read("docs/DOCUMENT_QUALITY.md")
    candidates = load_fail_documents(ROOT / "quality_failures.yml")

    assert len(candidates) == 11
    for candidate in candidates:
        assert candidate.document_id in page
        assert candidate.reason in page
