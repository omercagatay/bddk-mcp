# Legal-answer evidence readiness

This engineering increment improves source checks and answer evaluation, not just
retrieval ranking. It is **not legal certification, a measured overall readiness
score, or deployment approval**. Formal certification and deployment remain deferred.

## Runtime evidence checks

`get_document_section` accepts optional `quotation` (at most 2,000 characters) and
`as_of` (ISO date). Existing callers supplying neither retain their response shape
and issue no status query. No tool, dependency, migration, live-fetch route, or legal
approval import was added.

```json
{
  "document_id": "1040",
  "section_type": "paragraf",
  "section_ref": "134",
  "quotation": "Bankalar, tahmin edilen zarar karşılıkları ile gerçekleşen zararları geriye dönük testler uygulayarak test etmelidir.",
  "as_of": "2024-06-30"
}
```

This quotation is in the local seed. The example date is a requested check, **not**
evidence that the paragraph was effective then. With the uncurated seed mapping,
the result remains `local_text`, with an explicit missing-citation/status gap.

The `answer_assessment` evidence states are:

| State | Established by the check |
|---|---|
| `insufficient` | Missing/ambiguous/unparsed, truncated, failed/unknown-quality or unreconstructable section; or the proposed quotation is absent. |
| `local_text` | Reconstructed local source text, without validated Citation v1. |
| `validated_citation` | Reconstructable, independently mapped source occurrence; not yet verified as the effective cited version on the requested date. |
| `dated_version` | The canonical status resolver selected the citation's instrument, legal version, normalized text hash and validation-review record for the requested date. |

Quotation matching tolerates whitespace only—not case, accents, punctuation,
negation, numbers or semantic changes. The date is supplied by the task, never
inferred from today, a catalog date, download time or a quotation match. Source
ranges, hashes and sanitized content must agree. Document-level failures and
explicitly unknown quality cannot be hidden by a clean excerpt. Formula-unaware
extraction receives the existing warning in both section and whole-document tools.

The shared status repository also checks the requested instrument/date, evidence
intervals, canonical version identity, acquisition/blob/evidence identities, and the
required publication/effective/status claim identities. Duplicate roles are rejected.
Optional relationship targets remain validated by the database-owned resolver;
the response projection does not contain all inputs to reconstruct those claim IDs.
These are consistency checks, not proof that an independent legal review is genuine.

Gaps and dated evidence are visible in structured output **and** text, inside the
untrusted-data frame. Another effective version is identified, never silently
substituted. Missing status retains usable source evidence without authorizing
current-law conclusions. `scope_and_entailment` always remains `not_assessed`:
exact words plus a valid date do not approve a paraphrase, application to a bank,
omitted exception, unstated formula, frequency, data window or threshold.

## Answer workflow

Server instructions now require:

1. Decompose the question into material claims and the required applicability date.
2. Retrieve full exact provisions and review relevant cross-references/exceptions.
3. Check each proposed quotation against its own document/unit/reference; check the
   requested date before asserting current applicability.
4. Keep a claim-to-provision list distinguishing quotation, interpretation, context
   and missing evidence. Do not transfer IRB capital requirements into TFRS 9 accounting
   or confuse them with problem-loan resolution.
5. Answer supported parts and explicitly withhold unsupported conclusions.

The live Phase 2 client now forwards the server's initialization instructions to
the answering model and records the effective system-prompt hash. Previously it
used only its own generic prompt. This is tested through model-request capture and
actual MCP initialization; instruction following is not thereby guaranteed.

## Answer-level evaluation

The existing optional model-grader path accepts a question-aware legal rubric.
Six development cases cover:

- exact local quotation (1040 paragraph 134);
- complete validation elements, including outsourcing and corrective measures (943 paragraph 43);
- accounting versus IRB scope (943 paragraph 43 and 935 paragraph 30);
- invented frequency, history window, threshold and formula;
- dated applicability with conditional abstention;
- a nonexistent exact provision, without substitution from neighbors.

These are **development rubrics**, not independently approved expert ground truth.
Their content, expected abstention and date participate in the dataset identity.

Review output reports separately:

- **Claim support:** model verdict per answer span, with source/quotation/citation
  linkage checks. Every non-whitespace answer character must be covered in order.
- **Completeness:** coverage for every configured rubric point, each anchored to an
  actual answer span. Duplicate, missing or invented point reviews fail validation.
- **Abstention:** correctness against configured expectations; no invented grounding
  score of 1.0 for answers containing no factual claims.

Evidence stays attached to its document and provision. Wrong hyperlink destinations,
borrowed quotations, failed source-integrity assessments, missing citation labels,
truncation, unknown/failed quality and unverified currentness veto full support.
Requested filters, failed assessments and dated evidence survive evidence reduction.
A rejected proposed quotation does not invalidate a corrected local quotation.
Numeric overlap remains visible diagnostically but does not override claim-level
legal review: mentioning a threshold to deny that the paragraph specifies it is
not an assertion of that threshold.

Both JSON and human reports expose review availability, support, completeness and
abstention. Incomplete or unavailable reviews cannot promote the exploratory audit
success flag. Supported-but-incomplete and fully reviewed outcomes have separate
positive/negative harness regressions. No report authorizes deployment.

### Limits and execution

The grader remains explicitly opt-in through `BDDK_BENCHMARK_ALLOW_EXTERNAL_GRADER`
and the existing provider credentials. Structured review permits up to 6,000 output
tokens per call (rather than the legacy scalar grader's 10), so approved runs can
cost more. Oversized evidence/answers are **unavailable**, not truncated and scored.

```bash
# Requires the configured answering-model endpoint and an approved live MCP release.
# Do not enable external grading without the required egress authorization.
uv run python -m benchmark.run --phase 2 --model <configured-model-name>
```

No live generative-model performance run or human calibration was performed here.
Provider responses in contract tests are simulated. A model can still misclassify
semantics, non-factual spans, currentness or coverage; linkage checks cannot prove
those judgments correct. Each reviewed claim currently has one primary source;
multi-source arguments must be decomposed. Link checking handles strict inline
Markdown URLs; reference-style/HTML citations are explicitly unverified rather than
guessed. These limitations are not hidden behind an overall readiness score.

## Verification and review

- **DB-less suite:** 1,731 passed, 38 skipped in 39.12 seconds, after all review fixes.
- **PostgreSQL/offline suite:** 222 passed, 5 skipped in 571.67 seconds, covering the
  final runtime/status/quality implementation. Later changes affect benchmark code only.
- **Retrieval guard:** 3 tests passed, covering all 10 full-seed development hit@1
  queries, exact source reconstruction and the eight mechanical ranking cases.
- **Real corpus:** actual MCP calls verify quotations from 1040/134, 943/43 and 935/30;
  invented duties fail, and uncurated text never acquires a validated legal status.
- **Combined PostgreSQL path:** actual MCP section retrieval, Citation v1 reconstruction,
  canonical dated status and out-of-period abstention work through a restricted reader
  role; synthetic records and grants are transaction-rolled back, never published.
- **Public contract:** required metadata/descriptions, bounds, invalid inputs and
  backward compatibility are tested in public and operator profiles.
- **Static guards:** Ruff lint, formatting (322 Python files) and `git diff --check` pass.
- GPU tests were excluded. Skips and warnings are not represented as passes.

Two independent read-only code reviews (`a4d0ecaf-4079-477` and
`4812a7c7-7dbf-402`) supplied concrete counterexamples. Runtime
review findings (canonical identities, unknown quality and missing extraction
warnings) and grader findings (section-key contract, wrong links and lost assessment
context) were reproduced before fixes. Grader counterexamples initially produced
five failures; their corrected tests pass. A further positive-answer regression
exposed numeric-overlap rejection of negated values and now uses claim-level support.
These reviews are code reviews, **not independent legal approval**.

### Bounded autoresearch experiment

Fixed metric: unsafe acceptances in eight malformed status-identity cases; lower is
better. Baseline **8/8**, corrected **0/8**. This measures only the stated malformed
backend boundary, not legal accuracy or evidence of a production incident.

- `b1a2f97`: metric improved, but the lint guard failed; reverted by `4acec13`.
- `e7b1f90`: same focused fix with import ordering corrected; eight metric cases,
  57 status/version tests including PostgreSQL, and Ruff pass.
- The baseline guard was not separately measured; the log does not claim it passed.
- At experiment completion, only the status repository and its tests were committed;
  other task changes were still in the working tree. This historical checkpoint predates
  the dedicated branch/PR packaging. No corpus was published or activated.

Ledger: [`legal_answer_autoresearch.tsv`](legal_answer_autoresearch.tsv). The unrelated
pre-existing root `autoresearch-results.tsv` was preserved byte-for-byte. Graphify was
used for dependency exploration, followed by direct source/caller inspection; its
existing graph is not authoritative for newly added files.

Logs are under `/tmp/bddk-answer-readiness-*`, `/tmp/bddk-identity-*`,
`/tmp/bddk-grading-review-*` and `/tmp/bddk-negative-numeric-baseline.log`.
Signed seed/trust files and the expert-evaluation draft remain unchanged. Quality
policy `markdown-quality-assessment-v3` changes the retrieval-profile identity;
older candidate/release evidence is not silently promoted to match this code.

The earlier retrieval-only evidence is recorded in
[`CREDIT_LOSS_RETRIEVAL_REVIEW.md`](CREDIT_LOSS_RETRIEVAL_REVIEW.md).
