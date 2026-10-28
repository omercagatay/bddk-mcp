# Ontology plan: one regulation, end to end

**Progress: 0 / 6 steps done.** Next: Step 0.

## Goal

Run one regulation through the ontology, from text to reviewed links to answers the MCP tools can give:
the loan classification regulation (`mevzuat_22599`, *Kredilerin Sınıflandırılması ve Bunlar İçin Ayrılacak Karşılıklara İlişkin Usul ve Esaslar Hakkında Yönetmelik*).

**The pilot is done when:** `get_amendment_chain` shows its 7 amending regulations, `get_cross_references` shows the
Madde 22 repeal, and `resolve_regulation_status` resolves inside my review window but abstains before 1/1/2018.

## Ground rules for Claude Code

1. Follow `AGENTS.md`: explain in plain Turkish, plan before code, small steps, no commits or push unless I ask.
2. **No database, schema or corpus change without my explicit approval.** Approval to edit code is not approval to run
   migrations or imports.
3. **Do not use `resolve_and_import` in `bddk_mcp/regulatory/relation_extract.py` for this pilot.** It writes unreviewed
   edges, and relation rows are write-once. Candidates go to a proposals file; only approved items reach the corpus,
   through a bundle.
4. **Every link records how it was made:** `rule` (parsed from the text), `ai_approved` (proposed by AI, approved by me),
   or `action` (recorded through an action). A proposal counts for nothing until it is approved.
5. **Unknown is not false.** If something can't be confirmed, the tools abstain.
6. Public regulatory text only. Bank data, findings and anything confidential never enter this repo.
7. **Make my review fast, not long.** Rules and auto-checks clear the certain items; I only see the doubtful ones.

## Reference

`lcr-irrbb-ontology-demo` (my separate demo, made-up data) shows the intended shape: `ontology.yml` as the single
source of truth, link sources, actions with an audit log, and a review queue that previews what approving changes.
Use it as a design reference, not as code to copy.

## Facts to verify, from the stored text of mevzuat_22599

- The Resmî Gazete table at the end lists the original (22/6/2016, No. 29750) and 7 amending regulations, from
  14/12/2016 to 24/5/2022.
- Articles carry tags such as `(Değişik:RG-14/12/2016-29918)` and `(Mülga:RG-18/10/2018-30569)`.
- Madde 23 sets entry into force (1/1/2018). Madde 22 repeals the 2006 regulation.

## Steps

Each step is one session. Do only the first unchecked step.

- [ ] **Step 0: Read and plan (no code).**
  Read `AGENTS.md`, `docs/STATUS.md`, `docs/CORPUS_GOVERNANCE.md`, `bddk_mcp/regulatory/` (`relations.py`,
  `relation_extract.py`, `graph_queries.py`, `repository.py`, `legal_versions.py`),
  `bddk_mcp/migrations/v0009_regulatory_relations.py`, `v0011_graph_corpus_state.py`, `bddk_mcp/admin/drafts.py`,
  `scripts/build_943_citation_bundle.py` and `docs/evidence/legal-versions/943.json`.
  Write a one-page plan in Turkish mapping Steps 1–5 to files, with risks and rollback.
  *Done when:* I approve the plan.

- [ ] **Step 1: Write the ontology spec.**
  `ontology/ontology.yml` with object types (instrument, legal version, provision, obligation), link types (reuse the
  existing relation kinds: amends, repeals, replaces, consolidates, implements, cites, defines, exception_to; add
  contains and creates), link sources (rule, ai_approved, action), proposal statuses (proposed, approved, rejected) and
  action types (review link proposal, confirm obligation) with their rules. Add a test that fails if the relation kinds
  in code and in the spec drift apart, like the tool registry check.
  *Done when:* the test passes, and I can explain every object and link from the file without notes.

- [ ] **Step 2: Rule extractor, writing to a file only.**
  Parse (a) the Resmî Gazete table, (b) every inline tag `(Değişik|Ek|Mülga:RG-DD/MM/YYYY-NNNNN)` with its article, and
  (c) Madde 22 and Madde 23 (flag these two as `needs_review`). Write
  `docs/evidence/legal-versions/22599.proposals.json`: each proposal has source `rule`, the exact quoted text with
  character offsets, and a confidence. Cross-check tags against the table: a tag whose RG date and number appear in the
  table is `cross_confirmed`. Flag any RG number that is not 5 digits (this catches artifacts like `294599`).
  Unit tests on fixture text.
  *Done when:* the file lists the 7 amending regulations and every tagged article, tests pass, and nothing was written
  to the database.

- [ ] **Step 3: Review screen (the fun part).**
  A loopback admin page, "Link proposals": one card per proposal with the quote, source, confidence and what approving
  adds (for example "adds RG 29918 to the amendment chain"). Approve and Reject buttons with an optional reason,
  keyboard shortcuts (A, R, next), and a counter ("34 of 52 reviewed"). `cross_confirmed` rule proposals are
  auto-accepted, except a random 1 in 5 shown to me as a spot check. Decisions go to a separate review store outside the
  corpus (like `drafts.py`), each with reviewer, time and reason. CSRF protection like the existing forms.
  *Done when:* I can review all proposals in one sitting in the browser, the decisions are saved, and PostgreSQL is
  unchanged.

- [ ] **Step 4: Bundle builder and dry-run import.**
  Following `scripts/build_943_citation_bundle.py`, build a validated bundle from **approved decisions only**: one
  validated version (today's consolidated text), publication 22/6/2016 RG 29750, effective 1/1/2018 per Madde 23, one
  status assertion for the window I give, minimal instruments for the 7 amending regulations, and the accepted edges.
  Add an operator import command with `--dry-run` that prints exactly what would be inserted.
  *Done when:* the dry-run output matches my approved decisions, and tests pass.

- [ ] **Step 5: Import into the local test database (only with my explicit approval) and verify.**
  Check through the MCP tools: the amendment chain shows the 7 amendments, cross-references show the repeal, and status
  resolves inside my window but abstains before 1/1/2018. Add these as 3 benchmark cases.
  *Done when:* all three pass on the local test database.

- [ ] **Step 6: Obligations (only after Step 5).**
  The AI proposes provision → obligation links for Madde 9 into the same review screen. Then a coverage-gap function
  over made-up sample controls.

## Backlog (do not start until the pilot is done)

- Searches for absent topics return "strong match" results anyway: "yapay zeka" returned 8 heading-only snippets at
  83–87%. Add a no-match threshold and filter heading-only chunks.
- The stored text of LCR Madde 32(1) shows `RG-20/1/2016-294599`, a six-digit number, probably a glued footnote.
  Add a quality check for RG numbers.
- "destek hizmeti" misses the Bilgi Sistemleri Yönetmeliği, which says "dış hizmet". Add a synonym map.

## Session prompt (paste this to start each session)

> Read `docs/ONTOLOGY_PLAN.md` and `AGENTS.md`. Do only the first unchecked step. Explain your plan in Turkish first
> and wait for my approval. When you finish, show me the result and the "Done when" check, then tick the box and update
> the progress line.
