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
- `unknown`: quality metadata is unavailable or unresolved. A locally clean section must not erase an explicitly unknown document-level assessment.

## Verbatim serving contract

Document and section tools return **exact stored characters**. There is no
summarization, paraphrase, line wrapping, whitespace normalization, case folding,
or punctuation repair on the way out. If returning a document unchanged would
leak an unsafe extraction artifact (embedded `data:` URI, `cid:` marker, raw HTML
tag or control character), the tool **refuses** with `VERBATIM_UNAVAILABLE` instead
of silently rewriting the text; repair and re-publish the source. All 318 reviewed
seed documents are servable verbatim and are covered by
`tests/test_verbatim_documents.py`, which also proves the production token chunker
emits exact slices and that PostgreSQL round-trips every page exactly.

A capped parser span stores a system truncation notice. The notice is storage
metadata, never legal text: serving strips it and reports truncation
(`content_truncated`). Catalog search results carry BDDK catalog summary/excerpt
metadata, which is labeled as such and must never be quoted as provision text;
document-store snippets are exact leading excerpts of one chunk and may be
truncated.

Returned structured responses report `schema_version` `2.0`; the stricter
verbatim/exact-quotation semantics are part of that version, and Citation v2 uses
the `exact_stored_range_v2` render transform.

Labels are deterministic signals, not legal conclusions. A `clean` document is not a legal validation; a `fail` document means the extracted Markdown should not be treated as audit-grade evidence without source review.

## Known Fail List

The authoritative tracked fail list is packaged with the runtime at
`bddk_mcp/quality/quality_failures.yml`. Registry membership overrides content heuristics: a
listed document remains `fail` in search and retrieval results until it is repaired,
rescanned, reviewed, and then removed from this file.

No documents are currently registered as extraction failures. All 11 historical
failures were source-repaired and are included with the signed corpus in this
change. Real local MCP verification found 318 clean labels, with the release
receipt in [local-release-v6.json](evidence/document-repairs/local-release-v6.json).
This is not production deployment evidence or independent legal approval.
Formula-unaware provenance warnings and legitimate source/layout flags remain.

The retired entries below are retained as review history, not active failures:

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

Formula-unaware extraction methods receive an extraction warning only when the returned text refers to a formula, so a guide with no formula mention is not flagged merely for using markitdown. Whole-document and section tools share this policy; section results expose `formula_unaware_extraction` without invalidating an otherwise exact text quotation. The retrieval quality-policy identity is `markdown-quality-assessment-v7`. It recognizes the reviewed formula/XML identifiers in documents 903 and 907 without exempting arbitrary code, math or text after unmatched delimiters; (since v6) the reviewed literal reference "Yönetmelik EK-2’de yer alan formüllerde" in document 1296, which points at the formulas of another regulation rather than at a formula missing from that text; and (since v7) ten reviewed, per-document references to a formula stated in another article, paragraph or annex of the same text (1312, 1313, `mevzuat_19498`, `mevzuat_20029`). Section-level assessment would otherwise flag those references because the formula sits in a different section; [`policy/internal-formula-references.json`](evidence/document-repairs/quality-pass-2026-10-16/policy/internal-formula-references.json) records where each referenced formula is. Section parser `turkish-regulatory-sections-v10` keeps a section's closing formula lines instead of trimming them as a next-article title, so the only section that still carries a formula note is the capped `mevzuat_21192` EK-4, whose formula lies past the 20,000-character cap. Document retrieval assesses the full canonical body, not a page fragment: page-cut tables cannot create false failures, and a clean page cannot hide failures elsewhere. An unavailable full-document assessment stays `unknown`; a formula-aware method cannot erase a missing-formula signal. This policy change requires reindexing and governed release verification; it does not remove registry entries or formula-unaware extraction warnings.

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

The DB scan counts `c1_control_char` (U+0080–U+009F, cp1254 punctuation decoded as Latin-1) in SQL; `camelcase_concat` and `formula_ref_without_formula` come from the per-document assessment, so reviewed identifiers and inline formulas are not counted.

## Quality pass 2026-10-16

All 319 documents were already engine-clean, so the 2026-10-16 pass looked for defect classes the labels do not see and repaired 297 documents against their official sources: mis-decoded glyphs, scrambled justified lines, flattened tables, page furniture, mid-sentence paragraph breaks, glued footnote markers and formula-variable sub/superscripts, missing articles, and indented paragraphs that render as code blocks. Each changed document has a SHA-256 receipt in [`evidence/document-repairs/quality-pass-2026-10-16/`](evidence/document-repairs/quality-pass-2026-10-16/), with the baseline documents, the tools and the per-document specs needed to reproduce it. Text printed in the official source is kept as printed, including its typos. Where a consolidated text failed to apply an amendment, the amending Resmî Gazete text is used and the receipt says so (mevzuat_21194 Ek-1 fıkra 5). Document 1296 now matches its current official PDF.

## Admin manual edits

Admin edits are isolated, unpublished SQLite drafts, explicitly `unverified`.
The extraction method is read-only original provenance, never a form assertion
of `manual_latex` or formula-aware quality. An editorial Ed25519 signature binds
the saved draft but does not certify extraction accuracy, remove fail-list
membership, update vectors, or publish a corpus release. Source-review the
repair before changing quality provenance or the failure registry. See the
[configuration and publication handoff](CORPUS_GOVERNANCE.md#admin-editorial-drafts-and-document-signatures).

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
