# Document Quality

This page documents how BDDK MCP labels extracted Markdown quality, how warnings appear in MCP results, and how operators scan or backfill known problem documents.

## Extraction Methods

Stored documents can come from several extraction paths:

- `html_parser`: preferred when mevzuat.gov.tr or BDDK HTML preserves legal text cleanly.
- `pdftotext_layout`: the method of most reviewed seed documents (233 of 319); formula-unaware.
- `markitdown`: general PDF/office extraction path; good for ordinary text, weaker for formulas, annex tables, and embedded images.
- `pdftotext_degraded` and `markitdown_degraded`: the CPU fallbacks of the sync OCR chain, where important structure may be missing.
- `manual_pdf` and combined names such as `pdftotext_layout+manual_latex`: manually repaired text; only a name containing `lightocr`, `pp_structure` or `manual_latex` counts as formula-aware (`lightocr` needs the `gpu` group).

Trade-offs:

- HTML extraction usually preserves paragraphs and headings better than PDF text extraction.
- PDF extraction can lose formula glyphs, table layout, page boundaries, or Turkish characters when fonts are encoded poorly.
- OCR can recover image-only content but may introduce recognition errors.
- Manual LaTeX repair is highest quality for formulas but should be reserved for benchmark-critical or audit-critical documents.

## Quality Labels

The quality engine reports one document-level label plus flags:

- `clean`: none of the signals below fired. The document can be used normally, subject to ordinary legal-source caution.
- `warning`: one of these signals fired: control characters, any `cid:` marker, double-encoded Turkish text (mojibake such as `Ã¶` or `ÄŸ`), a formula reference without a visible formula or image, a long underscore run, a camelCase concatenation, or a paragraph block repeated more than twice.
- `fail`: raw HTML tags, a `data:` URI image, 20 or more `cid:` markers, replacement characters, a very long line (over 3,000 characters) combined with raw markup or blob content, or an entry in the quality-failure registry.
- `unknown`: quality metadata is unavailable or unresolved. A locally clean section must not erase an explicitly unknown document-level assessment.

## Verbatim serving contract

Document and section tools return **exact stored characters**. There is no
summarization, paraphrase, line wrapping, whitespace normalization, case folding,
or punctuation repair on the way out. If returning a document unchanged would
leak an unsafe extraction artifact (embedded `data:` URI, `cid:` marker, raw HTML
tag or control character), the tool **refuses** with `VERBATIM_UNAVAILABLE` instead
of silently rewriting the text; repair and re-publish the source. All 319 reviewed
seed documents are servable verbatim and are covered by
`tests/test_verbatim_documents.py`, which also proves the production token chunker
emits exact slices and that the corpus file round-trips every page exactly.

A capped parser span stores a system truncation notice. The notice is storage
metadata, never legal text: serving strips it and reports truncation
(`content_truncated`).
A capped named section (stored window of 20,000 characters) also reports its true
extent: `stored_span_capped`, `original_end_char` and a text line with the stored and
the full length. A `quotation` that is not found in the stored window is
`unavailable` (check reason `reference_truncated`, gap
`quotation_reference_truncated`), not `not_found`, because the text beyond the window
was not compared; a quotation equal to the whole stored window has extent `excerpt`.
The rest of the provision is in full in `get_bddk_document`; parts of it are also
stored as later rows (nested rows or `govde` windows), which are labelled as
continuing that provision. In corpus files built with the code of this release the words of the
truncation notice are not in the section full-text index. Catalog search results carry BDDK catalog summary/excerpt
metadata, which is labeled as such and must never be quoted as provision text;
document-store snippets are exact leading excerpts of one chunk and may be
truncated.

Returned structured responses report `schema_version` `2.0`; the stricter
verbatim/exact-quotation semantics are part of that version.

Labels are deterministic signals, not legal conclusions. A `clean` document is not a legal validation; a `fail` document means the extracted Markdown should not be treated as audit-grade evidence without source review.

## Known Fail List

The authoritative tracked fail list is packaged with the runtime at
`bddk_mcp/quality/quality_failures.yml`. Registry membership overrides content heuristics: a
listed document remains `fail` in search and retrieval results until it is repaired,
rescanned, reviewed, and then removed from this file.

No documents are currently registered as extraction failures. All 11 historical
failures were source-repaired and are part of the current seed data; the last
full scan found all 319 documents clean. The repair receipts were archived with
the governance layer at Git tag `archive/pre-simplify-2026-11-09`.
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

Public retrieval tools return the exact stored characters and do not rewrite them. When returning the text unchanged would leak an unsafe artifact (embedded `data:image` base64 payloads, `cid:` markers, the raw tags `div`, `table`, `tr`, `td`, `img`, `span` and `font`, or C0 control characters), `get_bddk_document` refuses with `VERBATIM_UNAVAILABLE` instead of cleaning it.

When a page is not `clean`, `get_bddk_document` includes concise metadata in the response header:

- `Quality: warning` or `Quality: fail`
- `Quality flags: ...`
- A visible quality warning before the document text

Formula-unaware extraction methods receive an extraction warning only when the returned text refers to a formula, so a guide with no formula mention is not flagged merely for using markitdown. Whole-document and section tools share this policy; section results expose `formula_unaware_extraction` without invalidating an otherwise exact text quotation. The retrieval quality-policy identity is `markdown-quality-assessment-v8`. It recognizes the reviewed formula/XML identifiers in documents 903 and 907 without exempting arbitrary code, math or text after unmatched delimiters; (since v6) the reviewed literal reference "Yönetmelik EK-2’de yer alan formüllerde" in document 1296, which points at the formulas of another regulation rather than at a formula missing from that text; and (since v7) ten reviewed, per-document references to a formula stated in another article, paragraph or annex of the same text (1312, 1313, `mevzuat_19498`, `mevzuat_20029`); and (since v8) any `cid:` marker or double-encoded Turkish text (mojibake) makes a document at least `warning`. Section-level assessment would otherwise flag those references because the formula sits in a different section; [`tests/fixtures/internal-formula-references.json`](../tests/fixtures/internal-formula-references.json) records where each referenced formula is. The section parser keeps a section's closing formula lines instead of trimming them as a next-article title, so the only section that still carries a formula note is the capped `mevzuat_21192` EK-4, whose formula lies past the 20,000-character cap. The parser carries no version name in code (Git history is its record); since 2026-11-12 it also indexes KGK standards that print bare paragraph numbers (`kgk_tms_12`), and ends a capped body (`govde`) window at a paragraph, line or word break instead of mid-word. Document retrieval assesses the full canonical body, not a page fragment: page-cut tables cannot create false failures, and a clean page cannot hide failures elsewhere. An unavailable full-document assessment stays `unknown`; a formula-aware method cannot erase a missing-formula signal. A parser change is invisible to `bddk-mcp build`, which skips by content hash: delete the local corpus file and run `bddk-mcp build` so sections and chunks are regenerated (the shipped vectors are reused while the chunk texts still match; `--force` would re-embed every document), then `bddk-mcp export`. `scripts/reindex_document_sections.py` rewrites `document_sections` only and leaves the chunks' section fields stale. A rebuild does not remove registry entries or formula-unaware extraction warnings.

Search results and section retrieval can surface quality metadata so agents can decide whether a snippet is appropriate evidence. Vector and FTS hits label quality from the **full stored document**, not only the matching chunk, so a clean snippet cannot hide document-level formula or extraction failures. Warning or fail labels do not block retrieval by themselves; the server returns the exact text with a warning, and refuses only when unsafe artifacts would leak into context.

## Scan Commands

Run the canonical corpus quality scan and write Markdown, CSV, JSON, and snippet outputs (`--db` reads the corpus file at `BDDK_CORPUS_PATH`, default `corpus/bddk.sqlite`; `--corpus PATH` names another file):

```bash
uv run python scripts/scan_document_quality.py --db --out-dir quality_reports --allow-failures
```

Scan a local Markdown export:

```bash
uv run python scripts/scan_document_quality.py \
  --md-dir ./bddk-md-export/docs \
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

`--fail-on` takes the signal names the report shows (scan anomalies and assessment flags, also on documents whose label is clean); an unknown name is rejected with exit code 2.

Outputs:

- `quality_report.md`
- `quality_findings.csv`
- `quality_findings.json`
- `suspicious_snippets.md`

The corpus scan counts `c1_control_char` (U+0080–U+009F, cp1254 punctuation decoded as Latin-1) per document in Python; `camelcase_concat` and `formula_ref_without_formula` come from the per-document assessment, so reviewed identifiers and inline formulas are not counted.

## Quality pass 2026-10-16

All 319 documents were already engine-clean, so the 2026-10-16 pass looked for defect classes the labels do not see and repaired 297 documents against their official sources: mis-decoded glyphs, scrambled justified lines, flattened tables, page furniture, mid-sentence paragraph breaks, glued footnote markers and formula-variable sub/superscripts, missing articles, and indented paragraphs that render as code blocks. Each changed document had a SHA-256 receipt, baseline document, tooling and per-document spec under `docs/evidence/document-repairs/quality-pass-2026-10-16/`; that evidence was archived at Git tag `archive/pre-simplify-2026-11-09`. Text printed in the official source is kept as printed, including its typos. Where a consolidated text failed to apply an amendment, the amending Resmî Gazete text is used and the receipt says so (mevzuat_21194 Ek-1 fıkra 5). Document 1296 now matches its current official PDF.

## Backfill Process

The tracked fail list is empty, so `scripts/backfill_quality_failures.py` with its default `--config` has no candidates and re-extracts nothing. The workflow that works today re-syncs one document into a copy of the corpus file:

1. Run the quality scan and keep the `quality_reports/` artifacts.
2. Copy the corpus file. Never repair the file a serving process has open:

   ```bash
   cp corpus/bddk.sqlite corpus/bddk-repair.sqlite
   ```

3. Re-extract one document into the copy, one at a time (the owner approves each `sync`; it downloads from the allowlisted BDDK and mevzuat hosts and re-embeds the document):

   ```bash
   uv run bddk-mcp sync --doc-id mevzuat_21192 --force --corpus corpus/bddk-repair.sqlite
   ```

   `--force` re-extracts the document; a mevzuat PDF that is already stored is reused unless HTML-first is on. The previous body is kept in `document_versions`.
4. Re-run the quality scan against the copy (`scan_document_quality.py --db --corpus corpus/bddk-repair.sqlite`) and compare labels, flags, snippets and content hashes with the first scan.
5. After owner review confirms the repair, remove the document from `bddk_mcp/quality/quality_failures.yml` if it was listed, export the repaired copy back to `seed_data/` (`bddk-mcp export --corpus corpus/bddk-repair.sqlite`) so the reviewed source matches, rebuild the corpus file from the seed, redeploy it and restart the server. Until then runtime retrieval keeps the old text and labels.

`scripts/backfill_quality_failures.py` runs the same forced re-extraction for a batch: `--config quality_reports/quality_findings.csv --dry-run` lists the documents labelled `fail` in a scan, and `--execute` (with `--corpus` on a copy) re-extracts them. `reindex_document_sections.py`, `scan_document_quality.py`, `document_quality_score.py` and `patch_doc.py` also read or write the file at `BDDK_CORPUS_PATH`; `--corpus PATH` targets another file. `patch_doc.py` refuses a replacement text that the corpus would rewrite on storage (it names the markers, also with `--dry-run`) and writes the seed in the canonical format; it updates the repository `seed_data/` only for the default corpus file, and `--seed-dir PATH` names another seed directory.

Backfill should not erase prior versions. It should create a fresh stored version or preserve enough hash/version history to compare old and new extraction output.
