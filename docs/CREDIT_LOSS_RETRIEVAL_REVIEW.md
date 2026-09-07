# Credit-loss retrieval and legal-answer readiness review

Initial review: 2026-09-06 (UTC). Original objective: improve retrieval quality and legal-answer readiness to 10/10.
**Current scope: engineering improvements and regression verification only.** The user deferred formal
certification and deployment; Goal mode is inactive. The historical release blockers below remain
unresolved but are not prerequisites for this engineering pass. No overall perfect score or
production/legal readiness is certified. Verification, Goal-mode and delivery states below
are historical checkpoints; the completed readiness engineering increment is documented in
[`LEGAL_ANSWER_READINESS.md`](LEGAL_ANSWER_READINESS.md).

## Reproduction and implementation

Baseline: `c4e631423bd58f4e171d02b2926b75bd7e6b5c9f`, branch `fix/railway-schema-v11`.
The unrelated, pre-existing untracked `autoresearch-results.tsv` was not modified.

The PostgreSQL regression loads all 318 reviewed seed documents into a rolled-back
transaction and calls the registered section tools. It reproduced the user's
pasted first ten results in the same order, including the two mislabeled principle
spans. It does not load synthetic legal-validation mappings or publish a corpus.

Implemented:

- Parse unique, near-contiguous paragraph inventories inside principle-based guides;
  retain printed/source order, refuse restarting numbering, and close principles at
  part/decimal-outline headings. Do not manufacture paragraph IDs for those headings.
- Make `943 paragraf 43` an exact reference; exclude provision numbers from inferred
  document IDs.
- Widen bounded token-fallback candidate retrieval before ranking, weight distinctive
  terms by candidate frequency, and penalize oversized spans without rewarding tiny
  fragments. Existing eight mechanical ranking cases remain passing.
- Return document title, catalog URL and extraction method in the existing evidence
  schema. Label catalog URLs as unvalidated, not as Citation v1.
- Propagate whole-document extraction failures to section results and citation gates.
- Expose structured retrieval warnings in text-only results inside the untrusted-data
  frame, including absent citation mappings and legal-applicability limitations.
- Preserve normalized excerpt offsets across Turkish capital letters/leading whitespace;
  prefer complete matching phrases and report parser-capped bodies as partial.
- Add grounding instructions requiring query decomposition, exact provisions, separation
  of TFRS 9, problem-loan resolution and IRB scopes, and dated legal-status validation.

Parser profile: `turkish-regulatory-sections-v8`.
Current search profile: `document-section-simple-fts-length-normalized-v6` (initial pass: v5).
These changes require a newly verified release; bypassing the active-profile guard is
not an acceptable rollout.

## Initial-pass retrieval results (v5; historical)

| Query | Baseline | Revised |
|---|---|---|
| `gerçekleşen kayıp oranı geriye dönük test LGD tarihsel veri stres testi` | 1040 paragraph 80 absent from top 10; unrelated market-risk article first | 1040 paragraph 80 first; 935 paragraphs 29/30 third/fourth; 1040 paragraph 134 fifth |
| `beklenen kredi zararı geriye dönük test tarihsel kayıp deneyimi ileriye yönelik bilgi stres senaryosu`, document 1040 | Annex 3 first; paragraph 80 third; paragraph 134 absent from top 10 | Paragraph 80 second; paragraph 134 sixth; Annex 3 absent from top 10 |
| `tahmin edilen zarar karşılıkları gerçekleşen zararları geriye dönük test`, document 1040 | Paragraph 134 first | Paragraph 134 first |
| `BKZ model validasyonu`, document 943 | Principle 5 first; no exact paragraph 43 index entry | Principle 5 first, paragraph 43 second; paragraph 43 separately retrievable |

The four targeted hit-at-k checks improved from 3/4 to 4/4. Their cutoffs are
5, 3, 1 and 3 respectively. **This is a small regression set, not a precision/recall
benchmark or proof of 10/10.** In that initial pass, the second broad query ranked a contextual
cross-reference (paragraph 44) above the substantive paragraph 80. Independent
relevance judgments, held-out questions and an authoritative completeness inventory
remain necessary before an overall perfect retrieval rating.

Exact read checks cover 943 paragraph 43, 1040 paragraphs 80/134 and 935 paragraph 29.
They require titles, source URLs, extraction methods, visible citation-unavailability
warnings and no fabricated citations. The principle boundary check covers 943
Principle 7 / Third Part and 935 Principle 6 / outline 2.2.

## Resumed engineering pass (v6)

Kept the existing parser, SQL search, tool contracts and release controls; added no dependencies.
Changes since the initial pass:

- Remove squared character-length weighting: candidate rarity supplies specificity without
  giving long title words an extra advantage over short terms such as `test`, `veri`, and `THK`.
  Keep the existing length penalty and temporary-provision handling.
- Search both LGD/THK spellings in loose fallback, count the acronym once in scoring, and
  split slash/hyphen-separated query words. Source evidence is never rewritten. This is
  retrieval expansion, not a statement that accounting and capital requirements are equivalent.
- Resolve equal fallback scores deterministically by source position and section identity;
  do not expose incomparable per-term FTS ranks.
- If an exact reference is absent but lexical alternatives are returned, report `partial`
  and warn in both structured/text results that those alternatives do not answer the exact reference.

The full-seed development regression now has ten queries; all ten require a relevant first
result (hit@1). These cases and relevance sets were developed during implementation, not
independently adjudicated or held out. Their 10/10 pass count is **not** an overall quality rating.

| Query focus | Document filter | First result |
|---|---|---|
| Broad realized loss / LGD / historical data / stress test | None | 946 paragraph 120 |
| Broad BKZ / backtest / historical loss / forward-looking information | 1040 | Paragraph 80 |
| Estimated provisions versus realized losses | 1040 | Paragraph 134 |
| BKZ model validation | 943 | Principle 5 |
| LGD backtesting, Turkish query | 935 | Paragraph 30 |
| THK backtesting, hyphenated Turkish query | 935 | Paragraph 30 |
| Historical loss experience / forward-looking macroeconomic information | 943 | Principle 6 |
| Model outputs / performance thresholds / recalibration | 943 | Paragraph 43 |
| Backtesting collateral valuations | 1040 | Paragraph 139 |
| Real-estate cash flows under adverse economic conditions | 1040 | Paragraph 157 |

946 paragraph 120 was added to the first query's relevance set after reading its stored
text: it explicitly discusses credit pools, TO/THK, net loss projections, and calibration/
backtesting with the bank's own loss data in a stress-testing context. It is not automatically
an accounting obligation. Under v6, 1040 paragraph 80 ranks third for that unfiltered query,
not first as under v5. A market-risk result still ranks second: the top ten are not all relevant.
For the 1040-filtered broad query, paragraph 80 improves from second to first and paragraph
134 from sixth to fourth, but contextual paragraph 44 still precedes paragraph 134.

Exact retrieval checks now cover seven provisions: 943/43, 1040/80, 1040/134, 935/29,
946/120, 1040/139 and 1040/157. They reconstruct each span from stored Markdown, verify
its SHA-256 and rendered content, require an untrusted-source marker and extraction quality,
and retain the visible missing-citation/currentness warnings. These checks verify fidelity
to the local corpus, not fidelity to an independently reacquired official PDF or current law.

## Verification

```sh
BDDK_REQUIRE_TEST_DATABASE=1 uv run --frozen pytest \
  tests/test_credit_loss_retrieval.py tests/section_rank_score.py \
  tests/test_tools_sections.py tests/test_legal_ref.py tests/test_section_index.py \
  tests/test_structured_retrieval_outputs.py tests/test_retrieval_profile.py -q
uv run --frozen pytest tests/ -m 'not postgres and not gpu' -q
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  BDDK_REQUIRE_TEST_DATABASE=1 \
  BDDK_EMBEDDING_MODEL_PATH=/tmp/bddk-review-embedding \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run --frozen pytest tests/ -m 'postgres and not gpu' -q
```

Final resumed-pass verification (2026-09-06):

- Full DB-less suite: **1,647 passed, 38 skipped**, 37.93 seconds.
- Full PostgreSQL suite with required database/offline local assets: **221 passed,
  5 skipped**, 555.78 seconds. No PostgreSQL-connection skips were accepted.
- Final focused parser/reference/section/structured-output/profile and full-seed
  suite, explicitly including `tests/section_rank_score.py`: **121 passed**.
- Ten development queries satisfy hit@1; eight existing mechanical ranking cases pass.
- Ruff lint, formatting (316 files) and `git diff --check` pass.
- `autoresearch-results.tsv`, signed seed files and trust files remain unchanged;
  nothing is staged, committed, published or activated.

The final focused command uses `tests/test_structured_retrieval_outputs.py`;
initially selecting the nonexistent `tests/test_structured_outputs.py` exited 4
without running tests, then the corrected command above passed. Routine verification
was sufficient for this resumed pass; no new autoresearch experiment loop was started.

Earlier full runs: 1644 passed / 38 skipped (initial-pass final DB-less run);
221 passed / 5 skipped (PostgreSQL run before the final small wrapper changes).
After the final changes, the focused parser/reference/section/structured-output and
full-seed suite passed 83 tests; Ruff and `git diff --check` also passed.
DB-less skips comprise seven unavailable Chandra tests and 31 missing Kubernetes/
OpenShift renderer tests. GPU lanes were excluded. Skips are not passes. PostgreSQL validation required the
existing local model assets: the initial unconfigured run failed seven seed tests
because the default Hugging Face cache was unavailable. An initial offline run hit
a 180-second command timeout; the complete rerun finished in 563.56 seconds.

## Derived corpus artifact audit

Canonical `documents.json` SHA-256:
`4a469fc11bd9848d2acd540296d794126ab9c506dc68d929fb72f580ebe5dba6`.
No signed seed files or signatures were changed.

Initial-pass candidate artifact and detailed delta (unreviewed, not activated;
these snapshots were not refreshed for the final v6 implementation):
`/tmp/bddk-credit-loss-candidate/chunks.json`, `delta.json`,
`baseline-comparison.json`.
Candidate SHA-256: `949dcaa12b8377beb7bca364f8aa614ddbdd2b984be4082948a3bb62b68456d9`.

Using the same available local tokenizer, HEAD regenerates 12,925 chunks versus the
committed 10,483; all 318 documents already differ in derived metadata. The revised
parser generates 13,240. Comparing HEAD regeneration with revised regeneration
isolates this implementation's delta to six documents:

| Document | HEAD chunks | Revised chunks |
|---|---:|---:|
| 1282 | 27 | 67 |
| 1291 | 33 | 98 |
| 934 | 63 | 63 |
| 935 | 45 | 160 |
| 943 | 107 | 202 |
| 947 | 41 | 41 |

Same-count changes still alter section metadata. Do not attribute all committed-seed
drift to this patch or silently re-sign it. The release owner must review the
regeneration and approve the exact production model/profile and corpus artifacts.

## Deferred formal requirements / historical legal-readiness blockers

1. **Authoritative source and applicability verification:** HTTPS acquisitions of
   official documents 943, 1040 and 935 failed certificate validation, with both the
   normal trust configuration and the system CA store. The server presents a leaf
   issued by `GlobalSign RSA OV SSL CA 2018`; the local client cannot build its chain.
   The bare `bddk.org.tr` hostname also failed public DNS resolution. TLS verification
   and the exact-host allowlist were not disabled. Search-engine snippets are not a
   substitute for retained authoritative artifacts and provision-level review.
2. **Dated legal evidence:** the reviewed manifest explicitly says legal effective,
   amendment, repeal, supersession and consolidation status is not authoritatively
   modeled; source-event freshness is `not_measured`, and completeness has not been
   independently reconciled. Neither clean extraction nor a URL resolves these gaps.
   The relevant as-of date and independently validated legal-status/occurrence mappings
   are still required for a definitive legal answer.
3. **Live verification:** the configured MCP server cannot connect:
   `RuntimeError: BDDK_DATABASE_URL is not set for the public process profile.`
   Tests exercise actual PostgreSQL/tool code, not the unavailable deployed MCP session.
4. **Reviewed publication:** the changed parser/search profile must be regenerated,
   independently reviewed, signed by the corpus owner and verified/activated through
   the separate release-verifier/publisher workflow in `docs/CORPUS_GOVERNANCE.md`.
   The candidate was not passed off as an approved release.
5. **Overall 10/10 evaluation:** independent relevance/coverage judgments and legal
   review are outstanding. Passing regression checks does not establish a perfect
   score. No goal completion has been claimed.

### Second goal-turn verification

The read-only evaluation scout completed successfully. Its result handle had been
cleaned up (`Agent not found`), so the final response was recovered from the supplied
transcript artifact, not by switching execution modes or launching a replacement.

The repository's actual expert-release validator was executed against the tracked
pilot and separately provisioned project public key. It reports `release_ready=false`:

- 20 draft/unapproved cases; 40 pending independent annotations; 20 pending adjudications.
- 21 pending Citation v1 mappings; all 21 evidence entries have unverified currentness.
- Missing dataset approval/signature, legal citation attestation, legal source-release
  evidence/latest checkpoint, and two required query classes.
- Quantified freshness is present, but measured source-event freshness is absent.

`verify-corpus --require-quantified-freshness --require-verified-signature` passes with
`deploy/trust/corpus-signing-public-key.pem`. Adding `--require-measured-freshness`
fails with `corpus freshness SLO compliance is not measured`. A valid signature
proves artifact integrity, not the missing legal facts or retrieval-profile match.

The live MCP connection was retried and again failed because `BDDK_DATABASE_URL`
is unset. A separate official-domain source check for 1040 paragraphs 80/134 again
returned `missing-evidence` (response ID `mtps9sj3hbtenf`). No exact authoritative
passages were obtained.

To complete the remaining gates, the owner must supply the intended public-runtime
connection through its secret configuration, an approved authoritative-source/trust
path, the required dated independent legal reviews, and reviewed/signed publication
and evaluation artifacts. These approvals cannot truthfully be supplied by relabeling
this implementation's tests or draft annotations.

### Third consecutive goal-turn blocker audit

Current HEAD/branch remain `c4e631423bd58f4e171d02b2926b75bd7e6b5c9f` /
`fix/railway-schema-v11`; implementation changes are uncommitted. No deployment,
source-authority review, corpus signing or activation is claimed.

- The public, ingestion, release-verifier and release-publisher DSN environment
  variables are all unset. The checkout contains only `.env.example`, not an
  alternative configured project dotenv file. The third live MCP connection
  attempt failed with the same missing-public-DSN error.
- The expert-release validator was rerun from the current files: still 20 draft
  cases, 40 pending annotations, 20 pending adjudications, 21 pending citations,
  missing signed legal evidence and `release_ready=false`.
- Strict measured-freshness verification again exited 2 with
  `corpus freshness SLO compliance is not measured`.
- A fresh bounded HTTPS retrieval of official document 943 again failed with
  `CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate`.

The implementation and unit/regression evidence are preserved. The full 10/10
objective is blocked, not complete: owner-controlled runtime access, source trust,
independent dated legal evidence and signed release approvals are required. No
trust policy was weakened and no pending approval was relabeled as completed.
