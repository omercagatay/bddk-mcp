@AGENTS.md

# Claude Code only

AGENTS.md above is the canonical shared instruction file; maintain shared rules there rather than duplicating them here.

## Owner's mandate

The owner has given the orchestrating Claude Code session that maintains this repository the standing described here. For that session, and for no other agent, it changes two rules of "Working with the owner" in AGENTS.md and adds one exception. A subagent has none of this by loading this file: it has what its brief gives it, and a brief never passes on the merge exception.

- Instead of the plan-approval rule: the owner decides what is built and what counts as done; the session decides how.
  - "What" includes everything that changes what an inspector sees or can rely on: search ranking and the order of results, the text and conditions of warnings and notices, tool names, arguments and outputs, what is in the corpus and how a document is labelled. Present such a change in plain Turkish before implementing it (what would change for the inspector, why, and how it will be verified) and wait for the owner's yes. A bug fix that changes one of these is such a change.
  - Adding a source or widening the outbound allowlist, and restoring anything AGENTS.md lists as removed, are also the owner's decisions.
  - Everything else is "how" and is the session's to decide without a plan approval: code structure, tests, tooling, CI, refactoring, and documentation of existing behaviour. Implement, verify, then report what was decided and why.
- Instead of "Do not create commits unless requested": a commit on a local topic branch needs no approval. Never commit to `main`.
- Unchanged, and needed each time with no approval carried over to the next action: an explicit yes before `git push`, merging a pull request, deployment, a schema change, any destructive operation, and any mutation of `seed_data/` or of a corpus file a checkout serves (`BDDK_CORPUS_PATH`, default `corpus/bddk.sqlite`, in the owner's checkout or any worktree): `bddk-mcp build`, `sync`, `export`, `scripts/patch_doc.py` and every other maintainer script that writes them.
- The one corpus command that needs no approval, as an exception to "obtain approval before `build`, `sync` or `export`" in AGENTS.md: `bddk-mcp build --corpus PATH` with a path outside every checkout, to build a throwaway file for a measurement. `sync` is never throwaway (it downloads from the live site), and `export` always writes `seed_data/`.
- Exception to the approval rule, for merging only: a pull request whose every commit is by `dependabot[bot]` and that changes nothing but version or digest pins in `uv.lock`, in a workflow under `.github/workflows/` or in the base-image line of the `Dockerfile` may be merged without asking, once it has been reviewed (the security review of AGENTS.md, written into the description) and every CI check is green. The exception covers the merge and the deployment that the merge triggers, which the session then verifies and reports. It covers no push to the branch, no manual deployment or rollback and no corpus command.
- Ask first, even for such a pull request: any change of a package of the embedding stack (sentence-transformers, transformers, tokenizers, torch, numpy), bringing the owner a comparison of the embeddings of the same sentences under the locked and the proposed versions, because the CI audit of 71 questions does not prove the vectors are unchanged; a major version of a GitHub Action or of a package on the HTTP or authentication boundary (mcp, starlette, uvicorn, httpx, pyjwt, anyio); a base image that is not a patch release of the same Python minor; a red or flaky CI; a changed outcome or a newly ungated question in the section-search audit set; a need to rebuild the corpus.

## Subagent models

The tiers of "Delegation and model choice" in AGENTS.md map to Claude models as follows: smallest model is Haiku, middle model is Sonnet, strongest subagent model is Opus. The top tier is the model class above Opus (Fable, or Mythos when the owner runs the session on it). Pass `model` explicitly on every Agent call and on every `agent()` call of a workflow, because an omitted `model` falls back to the agent definition's model or the session's model. For the top-tier review that AGENTS.md describes, pass `fable`. The Agent tool has no Mythos value: a subagent could reach Mythos only by inheriting the model of a session the owner runs on it, and that path has not been tested.

## Advisor

The advisor is the model the owner chose with `/advisor`, stored as `advisorModel` in the owner's user settings. It is the owner's setting: an agent does not change it.

- The agent that does the work calls it. Subagents inherit the configured advisor and receive the `advisor` tool; a cheaper model working with the stronger one advising is the intended pairing.
- Each call sends the caller's whole transcript to the advisor uncached, so a subagent's focused transcript is the cheaper place to ask from. Every brief for substantial delegated work therefore tells the subagent to consult the advisor at the points below.
- When the advisor tool is available, use it whenever the work needs it (the owner's standing instruction); do not wait to be told. It is needed before committing to an approach or a design on substantial work, when results do not fit or an approach is not converging, before changing approach, and before reporting substantial work as done. A routine step that the last tool output already dictates needs no call.
- An orchestrator that runs on the advisor's own model gets a second look rather than a stronger one and pays for its whole transcript on each call, so it calls sparingly: at a change of approach and before reporting substantial work as done.
- Weigh its advice against primary evidence: where the code or a measurement contradicts it, say so and reconcile instead of switching silently.
