# Corpus regeneration delta — section parser v3 → v5

Measured **2026-08-26** against tracked `seed_data/` at branch base `df5bf34`.
This is the owner's review input for gap-register **CUR-018**: it must be read
and accepted before `scripts/sign_corpus_manifest.py` is run, because signing is
an owner attestation over exactly this change.

## Why the corpus changed

`bddk_mcp/store/section_index.py` moved from `turkish-regulatory-sections-v3` to
`v5` in #135/#136. That constant is part of `retrieval_profile_descriptor()`
(`bddk_mcp/store/vector_store.py`), so it is part of the retrieval-profile
identity a governed release is bound to. Chunking is section-aware, so a parser
change is a corpus change. `seed_data/` was not regenerated at the time, leaving
the signed `bddk-job-corpus-2026-08-14` artifact bound to v3 metadata while the
code computes a v5 profile.

Regenerated with `uv run python scripts/regen_chunks_seed.py` (the same chunker
the production path uses; embeddings are not stored in the seed and regenerate
on first search).

## Summary

| Measure | Before (v3) | After (v5) |
|---|---:|---:|
| Documents | 318 | 318 |
| Chunks | 9,675 | 10,483 |
| Chunk artifact bytes | 16,488,940 | 17,024,862 |

- **18 of 318 documents** changed chunk count. All 18 changed by **growth only**.
- **300 of 300** same-count documents are **bit-identical** — no silent rewrites.
- No document lost chunks; no document dropped to zero.

## Section-type distribution

| Section type | Before | After |
|---|---:|---:|
| `paragraf` | 0 | 1,282 |
| (none) | 2,749 | 2,275 |
| `fikra` | 4,002 | 4,002 |
| `madde` | 2,354 | 2,354 |
| `bent` | 201 | 201 |
| `ilke` | 159 | 159 |
| `gecici_madde` | 138 | 138 |
| `ek` | 72 | 72 |

The entire delta is the new `paragraf` recognition: 1,282 chunks gain that
section type while previously untyped text (`none`, −474) becomes addressable.
Every pre-existing structural type is unchanged, which is the expected shape for
"recognize dash-numbered paragraphs and index bodies behind late annex headings"
and is evidence the change is additive rather than a re-interpretation of
already-recognized structure.

## Documents with changed chunk counts

| Document | Before | After | Δ |
|---|---:|---:|---:|
| 954 | 70 | 219 | +149 |
| 1040 | 69 | 199 | +130 |
| 946 | 63 | 170 | +107 |
| 1167 | 29 | 111 | +82 |
| 1311 | 24 | 74 | +50 |
| 950 | 23 | 71 | +48 |
| 948 | 40 | 86 | +46 |
| 956 | 20 | 59 | +39 |
| 955 | 17 | 52 | +35 |
| 953 | 16 | 42 | +26 |
| 945 | 27 | 52 | +25 |
| 952 | 9 | 32 | +23 |
| 944 | 13 | 28 | +15 |
| 1135 | 50 | 62 | +12 |
| 903 | 15 | 22 | +7 |
| 1285 | 44 | 50 | +6 |
| 1296 | 5 | 9 | +4 |
| 43 | 3 | 7 | +4 |

These are the numbered-paragraph Rehber and Genelge documents #135/#136
targeted; the largest movers are guides whose bodies were previously indexed as
long untyped spans.

## Owner review checklist before signing

1. Confirm the 18 changed documents are the intended numbered-paragraph
   population, and spot-check two or three against their source PDFs for
   section boundaries.
2. Confirm no regulatory content was lost: growth-only plus 300 bit-identical
   documents support this, but the acceptance is the owner's.
3. Run `uv run bddk-mcp verify-corpus --seed-dir seed_data` — it must report the
   staged manifest id and warn only that no signature is configured.
4. Run `uv run python scripts/sign_corpus_manifest.py --private-key <path>`.
5. Commit `seed_data/` and re-run the 26 corpus-bound contract tests.

## Verification state at the time of measurement

`bddk-mcp verify-corpus --seed-dir seed_data` accepts the staged manifest
(`bddk-job-corpus-2026-08-26`, canonical sha256
`49bbd025bdab1547f27f0e5c48c486a13abfa50a762ef3f9ef7fab030c73e2be`) with the
expected three warnings: non-exhaustive selection, unmeasured freshness, and no
configured signature. The staged manifest is intentionally
`signature_status: not_configured`; artifact sha256/bytes/records rows were
recomputed from disk and validated against the strict manifest model.
