# Source-reviewed document repairs

## Current local status (2026-09-21)

The 50 repairs, including the final native-source rebuilds of `mevzuat_16290`
and `mevzuat_21192`, are in the **signed corpus committed with this PR**. Local
verification/activation used a fresh database, separate verifier/publisher roles,
and the unchanged trust anchor. The complete receipt is
[`local-release-v6.json`](local-release-v6.json). Production was not deployed.

Real MCP checks cover all 318 documents and six second-page comparisons: **318
clean labels, zero warning/fail labels**. All 11 historical registry entries are
retired together with their repaired corpus. **310 formula-unaware provenance
warnings and legitimate source/layout flags remain**; source repetition, table
width and long tokens were not deleted to satisfy heuristics.

The final `16290` repair uses 112 native tables with exact cell/span coverage,
not approximate PDF fragments. The `21192` repair includes the main document and
all four annexes, 23 tables, and recovered embedded equations. Ordered native
Word/body text matches across all five parts. See the `*-native.json` patches
and compressed official sources under `native-sources/`. This is assistant
technical review, not independent human/legal approval.

Quality policy v5 assesses the canonical body rather than page-cut fragments and
recognizes only reviewed source identifiers. A missing full assessment stays
`unknown`; code/math delimiters do not hide concatenation errors. The local
launcher is `bash quality_reports/repair-audit/serve-local.sh`, pinned to an
immutable image with no mutable source/empty-registry mounts. Reproduce the
read-only MCP check with `.venv/bin/python quality_reports/repair-audit/check_local_release.py`.
The checked-in `seed_data/` and failure registry now match that reviewed runtime
profile. Deployment must still use fresh staging → strict bootstrap → independent
verification → separate activation. Do not deploy the registry or code alone.

**History incident:** an earlier attempt deleted 50 version rows to pass strict
membership. That was not valid remediation. Original document text/hashes are
preserved in the old signed corpus and an explicit local archive; deleted
per-version IDs/timestamps have not been recovered. The final v6 release used a
new database and deleted no versions; the prior database was left untouched.
Future content changes must likewise use fresh staging, not delete unexpected
rows. Historical details remain in `local-release-v5.json`.

## Historical candidate review records

The descriptions and `publication-handoff.json` below record the **pre-signature
candidate phase**, not current runtime state. Their original statuses remain for
reconstruction/audit. They are assistant source reviews, not independent
human/legal approvals. Current local publication state is recorded above.

## 903 — General Circular 2016/1

- Source: <https://www.bddk.org.tr/Mevzuat/DokumanGetir/903>
- PDF SHA-256: `42a35ee66926db24bf0ff14caf8170a9cc5eb38dde3e074e982c2e13f349e0b1`
- Scope: all nine PDF pages visually compared; display equations are on physical
  pages 3–6 (printed pages 2–5). This is assistant source review, not independent
  human/legal approval or verification of current applicability.
- Repair: transcribe 11 display equations to LaTeX, restoring summation bounds,
  subscripts and operators. Removes all 90 private-use font glyphs from the
  extraction. Every character outside the 11 replacement spans is unchanged.
- Reproducible exact replacements and before/after hashes: [`903.json`](903.json).
- Local candidate: `quality_reports/repair-audit/candidates/903.md`; PDF and
  rendered review pages are under the same audit directory.

The printed source itself contains inconsistencies. They are **preserved**, not
silently corrected: physical page 5 uses an upper summation bound of `2` in the
third-bank example; page 6 prints `-3,5 TL`, unusual `TL` factors, and `ikinci
derece` in the third-bank prose despite `n=3`. Calculations must not be certified
from this transcription alone.

Residual heuristic flags are `duplicate_paragraphs` and `camelcase_concat`:
repeated bank headings/equations and the source identifier `İpotekTutarı` are not
reasons to delete or rewrite source text. The canonical document remains `fail`
because its registry entry is deliberately retained pending owner review and
publication of the repaired version.

### Check and reproduce

```bash
uv run pytest tests/test_reviewed_document_repairs.py -q
```

The test verifies the exact signed-baseline Markdown hash, ordered non-overlapping
replacement spans, candidate hash, 11 LaTeX blocks, absence of private-use glyphs,
retention of source inconsistencies, and the unchanged registry protection.
To reconstruct the candidate, read document `903` from `seed_data/documents.json`,
verify `original_markdown_sha256`, and apply `replacements` in reverse offset
order, requiring each original slice to equal `old`. Verify
`candidate_markdown_sha256` before using the output. Offsets count Unicode
characters, not UTF-8 bytes.

## 907 — General Circular 2011/1

- Source: <https://www.bddk.org.tr/Mevzuat/DokumanGetir/907>
- PDF SHA-256: `3c8e587ab275bc97de5e17b276a20e2b1abd5332fcb116e2a498833aaa4a330c`
- Scope: all eight PDF pages visually compared. No mathematical display equations
  occur in this circular; its structured examples are XML and XSD, not formulas.
- Repair: restore 21 dotted-I characters, six list bullets and the reading order
  of the maximum-charge paragraph and `01.07.2011 tarihinden itibaren` sentence.
  Fence the three printed XML/XSD page blocks and quote inline XML tags so
  Markdown consumers do not interpret them as HTML. Repeated running footers
  remain outside code blocks; no examples, dates, values or identifiers deleted.
- Exact patch and hashes: [`907.json`](907.json). Local candidate:
  `quality_reports/repair-audit/candidates/907.md`.
- Tests parse the combined XML example and XSD with stdlib ElementTree and compare
  canonicalized trees against the old extraction after only the reviewed encoding
  repair and removal of running footers from the comparison. Framing changes do
  not silently change schema/example semantics.

The source's lowercase `<deger>` in the instructions, uppercase `<Deger>` in the
example/schema, `uniqueKalem` selector with `@Tur`, and literal dot placeholders
are preserved. Camel-case XML identifiers, long URL/tag tokens and duplicate
footers still trigger heuristics. They must not be renamed, split or deleted to
satisfy the scanner. This candidate is not an assertion that the illustrative
XML satisfies its printed schema, nor an owner-approved release.

## 1314 — Source-reviewed draft KDA regulation candidate

The existing optional GPU group was installed (`uv sync --frozen --group gpu`),
without adding dependencies or changing the lockfile. Chandra 0.2.0 produced all
26 page candidates using cached model revision
`af93b47dba1b47b6640c86ccf487ed2260ab9a09`. The full pass ran offline, kept each
page's source/model/output identity, and rejected empty, errored or token-limited
results. After a tool timeout, verified page receipts allowed resuming rather
than discarding or silently skipping pages. All 26 receipts and hashes pass.

All **26 physical pages have now been visually compared**, then lexically
reconciled against the PDF text and signed baseline. Higher-resolution crops
resolved disputed readings. The assembled candidate contains all 37 articles,
14 tables and 20 display equations. It fixes missing formulas, incorrect OCR
`veya` in place of `vega`, wrongly assigned credit-quality table cells, broken
reading order, Turkish characters and duplicated list numbering. Cross-page
sentences and tables are joined without deleting source content.

The source's inconsistencies remain: page 5 defines alpha but prints denominator
`a`; page 13 uses lowercase `s_b`/`s_c` despite uppercase definitions; page 14
uses uppercase/lowercase `S` differently. Printed wording such as `ihtiyarlarına`,
`dışından` and `değişiklerle`, conflicting table-group labels and draft date
placeholders are retained, not silently corrected.

Exact baseline-to-candidate patch: [`1314.json`](1314.json). Per-page raw/reviewed
hashes and correction diffs: [`1314-review-progress.json`](1314-review-progress.json).
Raw OCR remains unchanged. Local candidate: `quality_reports/repair-audit/candidates/1314.md`.
Tests reconstruct it without local OCR files and check article/table/formula
coverage, repaired table assignments and retained registry protection.

This is assistant source review, **not independent legal approval or a published
repair**. Review the old metadata's 16-page count against the current 26-page PDF
before publication. Remaining heuristic flags concern wide/dense tables, long
paragraphs/tokens and repeated source text; they are not grounds for deleting
content. The configured failure warning remains until governed publication.

## 905 — Source-reviewed font, table and footnote candidate

A document-specific encoding repair restores 67 `Ġ`→`İ`, 43 `Ģ`→`ş`,
10 `ġ`→`Ş`, and 82 private-use `U+F0B7`→`•` characters. Rendered pages
1, 2 and 37 confirm the glyph mappings. Every other character is unchanged;
reconstruction tests bind the partial output to the signed baseline hash.
See [`905-review-progress.json`](905-review-progress.json).

The cross-page bank risk-weight table on physical pages 20–22 is now reconstructed
as two Markdown grids, retaining the merged-cell meanings, four maturity/currency
columns and printed footnotes. Exact patch: [`905-bank-table.json`](905-bank-table.json).
Four numerical tables on pages 32–35 are also reconstructed. The source's inconsistent
`31.111.111,11` row value and `411.827.746,45` total remain unchanged; no printed
values were recalculated. Font-size inspection of all 42 PDF pages located 16 body
superscript references (including the repeated 7). Visual inspection of their pages
confirmed the references: four are restored in table cells and twelve in surrounding
prose, including `%350⁸`, `%1.250⁹`, `1,95¹¹` and `1,46¹³`, which were previously
misleadingly glued to numeric values. Patch: [`905-numeric-tables.json`](905-numeric-tables.json).

All 42 pages have now been textually reconciled. The 35 non-table pages match the
source token-for-token after reviewed font/whitespace normalization and relocating
PDF-internal footer order to its printed position. Two PDF noncharacters are visibly
hyphens on pages 6 and 26; the baseline already preserves them. Seven table-bearing
pages have individual-page visual comparisons; the remaining layouts were inspected
in contact sheets. No image-only content was found. This is **not** a claim of
line-by-line visual prose review or independent legal approval.

Coverage receipt: [`905-textual-coverage.json`](905-textual-coverage.json).
Complete baseline-to-candidate patch: [`905.json`](905.json). Historical institution
names, source arithmetic inconsistencies and URL remain as printed. Tests reconstruct
all stages, check per-page coverage and retain registry protection. Local assembled
candidate: `quality_reports/repair-audit/candidates/905.md`.
Owner approval, provenance review and governed publication remain outstanding.

## 1313 — Source-reviewed formula/table and reading-order candidate

Pinned offline Chandra extracted 24 selected formula/table-bearing pages from the
54-page PDF; all receipts pass source/model/output hash and error/token-limit checks.
These OCR files are unreviewed drafts, not replacement documents. Source review
has restored 44 display equations, two inline expressions and Tables 1–4, including
the damaged renewal-cost, effective-maturity, portfolio-haircut and aggregation
expressions. Twelve prose/definition spans were restored to the printed reading
order with exact token-multiset equality; a stray standalone `D` absent from the
source paragraph was removed. Every character outside the 63 repaired spans remains
unchanged. Complete patch: [`1313.json`](1313.json); OCR receipts and review scope:
[`1313-review-progress.json`](1313-review-progress.json).

Table 2 preserves the printed slash/multiplication order and unmatched ordinary
parentheses rather than silently substituting a standard Black–Scholes formula.
Its squared sigma lacks an `i` subscript in the source numerator. The source PDF
encodes the printed CDF symbol as Cyrillic `ф` (U+0444), including its definition;
that source symbol is retained. Mixed decimal punctuation also remains unchanged.
OCR prose changes were not accepted automatically. High-resolution crops also
confirm upper-case `K`/`J` in the commodity `T` terms, lower-case `k` in `D_k`, and
a printed euro sign in the page-26 summation bound. These are preserved rather
than silently replaced with more conventional notation. Table 3 retains blank
correlation cells for interest-rate and FX classes; blank is not a fabricated zero.

All 54 pages were textually reconciled; all 166 source/baseline token differences
are accounted for by reviewed repairs. Coverage receipt:
[`1313-textual-coverage.json`](1313-textual-coverage.json). Equations and tables on
25 pages were individually inspected, including high-resolution symbol crops;
the other 29 layouts were inspected in contact sheets. No image-only content was
found. This is not a claim of line-by-line visual review of every prose page or
independent legal approval.

Local assembled candidate: `quality_reports/repair-audit/candidates/1313.md`.
All 69 articles, the general rationale and draft date remain. No private-use glyphs
or malformed Markdown table rows remain. The missing article number in the printed
BRT definition and unindexed `SMKT`/`RTi` in the page-46 equation are preserved.
Long LaTeX tokens and repeated source text still trigger heuristic flags; source
content must not be deleted to hide them. Owner approval, provenance review and
governed publication remain outstanding; registry protection is unchanged.

## 1305 — Source-reviewed equations, classification tables and reading order

The source PDF has eleven embedded equation images on pages 10, 11, 13 and 22;
all eleven were absent from the stored Markdown despite a clean content-only scan.
They have now been source-transcribed to LaTeX. Two existing risk-weight equations
were also framed as LaTeX, and four further display equations plus five risk-weight,
expected-loss and collateral tables on pages 12, 16, 18–19 and 24 were reconstructed. Exact patches and ten pinned offline OCR
receipts: [`1305-review-progress.json`](1305-review-progress.json).

Rendered source crops resolved OCR errors: `TO` is not `T0`, maturity uses `V`, and
the maturity coefficient is `b`, not delta. Corporate and retail formulas retain
their different maturity factors and correlation parameters. Page 18's printed
`THK_S` in prose versus `THK_T` in its formula/definition is preserved. No OCR prose
was automatically adopted and no source calculations were silently corrected.

Complete candidate: `quality_reports/repair-audit/candidates/1305.md`; reproducible
baseline patch: [`1305.json`](1305.json). All characters outside the exact repair
spans remain unchanged. The installed
`pdfplumber` dependency also produced 23 source-bound grid drafts for the classification
annex on pages 46–67. Shaded header/text rectangles were falsely detected as borders;
excluding those backgrounds from grid detection (without removing text) restores
the four-column rating map and five-column classification grids. The extraction
receipt remains an **unreviewed intermediate**; the separate evidence below records
subsequent source review and cross-page reconstruction. Local extraction receipt:
`quality_reports/repair-audit/extracted/1305-annex-grid-draft.json`.

The Annex 3 rating map and project-finance table on pages 46–53 are now separately
source-reviewed: [`1305-annex-project.json`](1305-annex-project.json). All eight
rendered pages were compared. Reconstruction restores an open-bottom activity-risk
row that default grid extraction dropped on page 49; page 50's leading `karşılık
hesapları.` is joined back to that row rather than the next criterion. Three unruled
criteria are separated, and cross-page continuations are joined into a 36-row grid.
Column-by-column non-whitespace equality proves that no text was removed or shifted
between rating categories. Only repeated table headers and explicitly reviewed
intra-word whitespace breaks were normalized—not repeated substantive passages.
The other three classification groups on pages 54–67 are now also source-reviewed:
[`1305-annex-remaining.json`](1305-annex-remaining.json). Their reconstructed grids
contain 19, 25 and 15 rows. The otherwise dropped design/maintenance row on page 56
is restored and joined to page 57. Column-preservation checks pass for every group;
legitimately repeated criteria remain repeated.

All 68 pages were textually reconciled. Ten prose spans were repaired with exact
token-multiset preservation; all 262 source/baseline differences are accounted for
in [`1305-textual-coverage.json`](1305-textual-coverage.json). The eleven image
formulas were separately visually checked because text-only comparison cannot
verify them. Thirty-two pages were individually inspected; other layouts were
inspected in contact sheets, not line-by-line visual prose review. The candidate
contains seventeen LaTeX display expressions and ten reconstructed tables, retaining
the eleven main articles, annexes and general rationale. Readable plain-text
arithmetic remains plain text.

Owner approval, provenance review and governed publication remain outstanding.
Registry, provenance and signed corpus are unchanged.

## 1045 — Source-reviewed account-catalogue and explanation candidate

The first 56 PDF pages contain 2,708 distinct account/group codes. The stored
extraction separates codes from their names, creating misleading repeated-name
blocks. Source-line reconstruction pairs every code with its name, preserves all
leading zeros and exact source order, joins 42 explicitly enumerated continuation
lines, and retains eight uncoded headings separately. Every code/name pair passes a
PDF glyph-position check: code in the left column, name in the right, aligned vertically.
The ordered code inventory matches the signed baseline exactly.

All 42 continuation contexts and eight uncoded-heading contexts were visually checked
in source crops. Codes 944/945 have unusually left-aligned currency suffixes; dedicated
crops confirm that `T.P.`/`Y.P.` belong to those headings rather than new rows.
Catalogue token-multiset equality holds after removing only 55 identical running headers.
Repeated substantive names remain paired with distinct codes.

The explanation section on pages 56–76 restores misplaced heading codes and currency
labels. Two hundred headings were verified against glyph positions. Exact substantive
token inventory is preserved after removing 20 identical running headers. The ten
numbered section titles appear in both the catalogue and the explanation in the source;
those ten duplicate headings are retained. No embedded images. All 76 pages were
inspected in contact sheets, with the 50 special catalogue contexts checked individually.
This is assistant source review, **not independent legal approval or a published repair**.

Exact baseline-to-candidate patch: [`1045.json`](1045.json). Receipts:
[`1045-review-progress.json`](1045-review-progress.json),
[`1045-explanation-progress.json`](1045-explanation-progress.json).
Local candidate: `quality_reports/repair-audit/candidates/1045.md`.
Heuristic `repeated_para_blocks_gt2` is gone; remaining `duplicate_paragraphs` are the
ten source section titles. Registry protection is unchanged until governed publication.

## 1043 — Source-reviewed account-catalogue and explanation candidate

Native PDF lines pair 6,776 distinct codes with names (including 8-digit codes).
The stored extraction split codes from names and mixed in page numbers. Source-line
reconstruction pairs every code/name with a glyph-position check, joins 64 wrapped
name tails, and keeps 78 uncoded subheadings (`ALIM SATIM`, mevduat groups,
`SATICI`/`ALICI TARAF`, `CAYILAMAZ`/`CAYILABİLİR`). The printed
`ALICI (LEHDAR) TARAF=` equals sign is preserved. All 177 pages were inspected in
contact sheets. This is assistant source review, **not independent legal approval**.

The explanation on pages 139–177 restores source order with 281 geometry-verified
headings. Non-digit token inventory is preserved after removing 38 running headers.
Ten numbered section titles appear in both catalogue and explanation in the source
and are retained. Exact patch: [`1043.json`](1043.json). Local candidate:
`quality_reports/repair-audit/candidates/1043.md`. Heuristic `repeated_para_blocks_gt2`
is gone; remaining `duplicate_paragraphs` are the ten source section titles. Registry
protection is unchanged until governed publication.

## 1334 — Source-reviewed account-catalogue and explanation candidate

Native PDF lines pair 9,381 account codes (9,380 distinct; code `138114` is printed
twice on page 37 under different parents and is preserved). 150 wrapped name tails
are joined; 51 uncoded subheadings are kept (`ALIM SATIM`, GERÇEĞE seçeneği,
mevduat groups, `SATICI`/`ALICI`, `CAYILAMAZ`/`CAYILABİLİR`). Two-line running
headers were removed (249). Explanation pages 250–298 restore source order with 273
geometry-verified headings. Catalogue special contexts and sample layouts plus all
49 explanation pages were inspected in contact sheets. This is assistant source
review, **not independent legal approval**.

Exact patch: [`1334.json`](1334.json). Local candidate:
`quality_reports/repair-audit/candidates/1334.md`. Heuristic `repeated_para_blocks_gt2`
is gone; remaining `duplicate_paragraphs` are source section titles and repeated
option/katılma labels. Registry protection is unchanged until governed publication.

## mevzuat_16290 — Financial-statement table drafts

The stored Markdown has no pipe tables; financial-statement grids were flattened into
repeated labels (`Cari Dönem`, `TP`, `YP`). A page-by-page pdfplumber rebuild renders 152 non-empty tables as Markdown and keeps
non-table prose. Blank template cells stay blank. Merged headers are approximate grids.
Contact sheets of representative pages (including 4, 15–20, 40–45, 70–75) were inspected.
This is assistant source review, **not independent legal approval**. Scanner without ID is
`clean` aside from wide-table pipe density and three source duplicate paragraphs.
`repeated_para_blocks_gt2` is gone. Patch: [`mevzuat_16290.json`](mevzuat_16290.json).
Registry protection is unchanged until governed publication.

## Word private-use bullets (U+F0B7)

Twenty-two documents contain only U+F0B7 private-use glyphs. Page 5 of 954
shows they are list bullets. Each is replaced with `•`; no other character
changes. Eleven further documents mix those bullets with `U+F0A7` squares, `U+F0BE` em-dashes,
and `U+F0D8` arrowheads (source PDFs 934, 787, 1191). Mapped glyphs only; `U+F0E0` on 1131
remains. Unpublished.
Unpublished.

## 1296 — 2025/1 interest-shock tables

Four CID-garbled $\pm R$ shock tables on pages 3–4 are restored from the source PDF.
Printed $\bar{R}_{paralel/kısa/uzun}$ and `Para Birimi p` are kept. Patch: [`1296.json`](1296.json).
Unpublished.

## 1312 — Kredi riski azaltım draft formulas

Stored extraction dropped operands in MADDE 33’s $K^{*}$ formula and MADDE 39’s
$H=\sum a_i H_i$. Both were restored from the 30-page source PDF (pages 16 and 20).
Page 18’s $K^{*}=\mathrm{maksimum}\{0, [(\Sigma K - \Sigma T) + (RMD)]\}$ was already
intact. Printed `maksimum` and `0,4` are kept. Patch: [`1312.json`](1312.json).
Not fail-listed; scanner `malformed_table_rows` 3→0. Unpublished.

## Historical publication handoff (unsigned at preparation)

Fifty source-reviewed candidates are applied only to a local copy at
`quality_reports/repair-audit/seed-repaired/documents.json`. Signed `seed_data/`,
`quality_failures.yml`, chunks/embeddings, and production are unchanged. There is
no private corpus-signing key in this repository. Receipt:
[`publication-handoff.json`](publication-handoff.json).

## Remaining review inventory

All 11 main PDFs were downloaded using verified TLS and validated with `pdfinfo`;
source hashes are retained in `quality_reports/repair-audit/sources/manifest.json`.
A new `pdftotext -layout` extraction was generated for each under `extracted/`.
Those text files are **diagnostic extractions, not approved replacements**.

| Document | PDF pages | Remaining source-review work |
|---|---:|---|
| 1043 | 177 | Source-reviewed candidate assembled; owner approval, provenance review and governed publication remain. |
| 1045 | 76 | Source-reviewed candidate assembled; owner approval, provenance review and governed publication remain. |
| 1334 | 298 | Source-reviewed candidate assembled; owner approval, provenance review and governed publication remain. |
| 1305 | 68 | Source-reviewed candidate complete; owner approval, provenance review and governed publication remain. |
| 1313 | 54 | Source-reviewed candidate complete; owner approval, provenance review and governed publication remain. |
| 1314 | 26 | Source-reviewed candidate complete; owner approval, metadata review and governed publication remain. |
| 905 | 42 | Source-reviewed candidate assembled; owner approval, provenance review and governed publication remain. |
| mevzuat_16290 | 77 | Source-reviewed table-rebuild candidate assembled; owner approval and governed publication remain. |
| mevzuat_21192 | 36+annex | Two abs-value formulas in stored EK-2 framed as LaTeX. Annex zip still unreachable; main PDF still cannot replace the 70-page merge. |

**Important false-clean case:** fresh `pdftotext` output for 1314 has no heuristic
flags when assessed without its document ID, but the first formulas have missing
operands. Its registry protection is essential. A clean scanner result is not
source-completeness evidence. Do not use blanket deduplication, relabel ordinary
extraction as `manual_latex`, or replace the annex-merged document with its shorter
main PDF just to remove flags.

For subsequent promotion, incorporate approved repairs into a new version/corpus and
review any corresponding failure-registry removals as part of the same retrieval-
profile change. Regenerate chunks and embeddings, review and sign the changed
artifacts, then verify/stage/activate using separate roles. Do not deploy registry
removals separately from the reviewed repaired release. This candidate does not
perform or authorize those operations.
