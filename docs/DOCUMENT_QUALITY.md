# Document Quality

This page documents how BDDK MCP labels extracted Markdown quality, how warnings appear in MCP results, and how operators scan or backfill known problem documents.

## Extraction Methods

Stored documents can come from several extraction paths:

- `html_parser`: preferred when mevzuat.gov.tr or BDDK HTML preserves legal text cleanly.
- `markitdown`: general PDF/office extraction path; good for ordinary text, weaker for formulas, annex tables, and embedded images.
- `markitdown_degraded`: fallback or legacy extraction where important structure may be missing.
- `lightocr`, `chandra2`, `pp_structure`, or `manual_latex`: formula-aware paths or manual repairs used when formulas, tables, or image-based math matter.

Trade-offs:

- HTML extraction usually preserves paragraphs and headings better than PDF text extraction.
- PDF extraction can lose formula glyphs, table layout, page boundaries, or Turkish characters when fonts are encoded poorly.
- OCR can recover image-only content but may introduce recognition errors.
- Manual LaTeX repair is highest quality for formulas but should be reserved for benchmark-critical or audit-critical documents.

## Quality Labels

The quality engine reports one document-level label plus flags:

- `clean`: no material extraction signal detected. The document can be used normally, subject to ordinary legal-source caution.
- `warning`: usable text with extraction caveats such as control characters, formula references without visible formulas, moderate duplication, or layout artifacts.
- `fail`: severe extraction risk such as raw HTML/data URI leakage, many `cid:` markers, replacement characters, very long blob-like lines, or repeated corrupted blocks.

Labels are deterministic signals, not legal conclusions. A `clean` document is not a legal validation; a `fail` document means the extracted Markdown should not be treated as audit-grade evidence without source review.

## Known Fail List

The authoritative tracked fail list is packaged with the runtime at
`bddk_mcp/quality/quality_failures.yml`. Registry membership overrides content heuristics: a
listed document remains `fail` in search and retrieval results until it is repaired,
rescanned, reviewed, and then removed from this file:

| Document ID | Reason | Preferred backfill |
|---|---|---|
| `mevzuat_21192` | `raw_html_data_uri_formula_artifacts` | `html_parser_or_ocr_manual_latex` |
| `1314` | `severe_cid_formula_corruption` | `ocr_or_manual_latex` |
| `1313` | `cid_and_phi_formula_corruption` | `ocr_or_manual_latex` |
| `1043` | `repeated_account_plan_blocks` | `structural_table_cleanup` |
| `1045` | `repeated_account_plan_blocks` | `structural_table_cleanup` |
| `1334` | `repeated_account_plan_blocks` | `structural_table_cleanup` |
| `903` | `concatenation_formula_table_artifacts` | `ocr_or_manual_latex` |
| `905` | `concatenation_formula_table_artifacts` | `ocr_or_manual_latex` |
| `907` | `concatenation_formula_table_artifacts` | `ocr_or_manual_latex` |
| `mevzuat_16290` | `financial_statement_annex_table_layout_degradation` | `structural_table_cleanup` |
| `1305` | `repeated_blocks_formula_artifacts` | `ocr_or_manual_latex` |

As a standing rule, formula-heavy failed documents require source review before they are used for calculation-level answers. In particular, do not reconstruct missing formulas from memory or general standards when the extracted source text only says that a formula appears below.

## Quality Warnings In MCP Results

Public retrieval tools sanitize document content before returning it to a model. `get_bddk_document` removes unsafe context artifacts such as raw HTML tags, `data:image/...;base64,...` payloads, `cid:` markers, and pathological long blobs.

When a page is not `clean`, `get_bddk_document` includes concise metadata in the response header:

- `Quality: warning` or `Quality: fail`
- `Quality flags: ...`
- A visible quality warning before the document text

Formula-unaware extraction methods also receive an extraction warning so the assistant tells users when equations or images may be missing.

Search results and section retrieval can surface quality metadata so agents can decide whether a snippet is appropriate evidence. Vector and FTS hits label quality from the **full stored document**, not only the matching chunk, so a clean snippet cannot hide document-level formula or extraction failures. Warning or fail labels do not block retrieval by themselves; the server sanitizes and warns unless unsafe inline blobs would leak into context.

## Scan Commands

Run the canonical DB quality scan and write Markdown, CSV, JSON, and snippet outputs:

```bash
uv run python scripts/scan_document_quality.py --db --out-dir quality_reports --allow-failures
```

Scan a local Markdown export:

```bash
uv run python scripts/scan_document_quality.py \
  --md-dir ./bddk-md-export/docs \
  --manifest ./bddk-md-export/manifest.csv \
  --out-dir quality_reports \
  --allow-failures
```

Fail the command when selected severe signals appear:

```bash
uv run python scripts/scan_document_quality.py \
  --db \
  --out-dir quality_reports \
  --fail-on data_uri_image,cid_marker,raw_html_tag,replacement_char
```

Outputs:

- `quality_report.md`
- `quality_findings.csv`
- `quality_findings.json`
- `suspicious_snippets.md`

## Backfill Process

Dry-run the tracked fail list:

```bash
uv run python scripts/backfill_quality_failures.py --dry-run
```

Dry-run one document:

```bash
uv run python scripts/backfill_quality_failures.py --dry-run --doc-id mevzuat_21192
```

Execute a targeted re-extraction:

```bash
uv run python scripts/backfill_quality_failures.py --doc-id mevzuat_21192 --execute
```

Use `--database-url` for a one-off production or staging connection instead of the local `BDDK_DATABASE_URL`:

```bash
uv run python scripts/backfill_quality_failures.py \
  --doc-id mevzuat_21192 \
  --database-url "$DATABASE_PUBLIC_URL" \
  --execute
```

Recommended operator loop:

1. Run the quality scan and keep `quality_reports/` artifacts.
2. Dry-run the known fail list.
3. Execute one targeted backfill at a time.
4. Re-run the quality scan.
5. Compare before/after labels, flags, snippets, and content hashes.
6. After owner review confirms the repair, remove the document from
   `bddk_mcp/quality/quality_failures.yml` and redeploy/restart the server; until then, runtime retrieval continues
   to label it `fail`.

Backfill should not erase prior versions. It should create a fresh stored version or preserve enough hash/version history to compare old and new extraction output.
