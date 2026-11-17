@AGENTS.md

# Claude Code only

AGENTS.md above is the canonical shared instruction file; maintain shared rules there rather than duplicating them here.

## Owner's mandate

The owner made the Claude Code agent that maintains this repository its owner and long-term maintainer: "its design is up to you". For that agent, and for no other agent, this changes two rules of "Working with the owner" in AGENTS.md and adds one exception.

- Instead of the plan-approval rule: decide the design, implement, verify, and then report what was decided and why. Do not ask the owner to approve a plan or a design choice.
- Instead of "Do not create commits unless requested": a commit on a local topic branch needs no approval. Never commit to `main`.
- Unchanged, and needed each time with no approval carried over to the next action: an explicit yes before `git push`, merging a pull request, deployment, any mutation of the owner's corpus file or of `seed_data/` (`bddk-mcp build`, `sync`, `export`, `scripts/patch_doc.py` and every other maintainer script that writes them), a schema change, and any destructive operation. A throwaway corpus file built in a temporary directory for a measurement is not such a mutation.
- Exception, for merging only: a pull request authored by `dependabot[bot]` that changes nothing but `uv.lock`, a workflow under `.github/workflows/` or the base-image line of the `Dockerfile` may be merged without asking, once it has been reviewed (the security review of AGENTS.md, written into the description) and every CI check is green. When a package of the embedding stack changes (sentence-transformers, transformers, tokenizers, torch, numpy), first show that the embeddings of fixed sentences are equal, within float16 rounding, between the locked and the proposed versions; the CI audit of 71 questions does not prove that. The exception covers the merge and the deployment that the merge triggers, which the agent then verifies and reports. It covers no push to the branch, no manual deployment or rollback and no corpus command.
- Ask first, even for a Dependabot pull request: a major version of a GitHub Action, of a package on the HTTP or authentication boundary (mcp, starlette, uvicorn, httpx, pyjwt, anyio) or of a package of the embedding stack; a base image that is not a patch release of the same Python minor; a red or flaky CI; a changed outcome in the section-search audit set; a need to rebuild the corpus.

## Subagent models

The tiers of "Delegation and model choice" in AGENTS.md map to Claude models as follows: smallest model is Haiku, middle model is Sonnet, strongest subagent model is Opus. The top tier is the model class above Opus (Fable, or Mythos when the owner runs the session on it). Pass `model` explicitly on every Agent call and on every `agent()` call of a workflow, because an omitted `model` falls back to the agent definition's model or the session's model. For the top-tier review that AGENTS.md describes, pass `fable`. The Agent tool has no Mythos value: a subagent could reach Mythos only by inheriting the model of a session the owner runs on it, and that path has not been tested.

## Advisor

The advisor is the model the owner chose with `/advisor`, stored as `advisorModel` in the owner's user settings. It is the owner's setting: an agent does not change it.

- The agent that does the work calls it. Subagents inherit the configured advisor and receive the `advisor` tool; a cheaper model working with the stronger one advising is the intended pairing.
- Each call sends the caller's whole transcript to the advisor uncached, so a subagent's focused transcript is the cheaper place to ask from. Every brief for substantial delegated work therefore tells the subagent to consult the advisor at the points below.
- When the advisor tool is available, use it whenever the work needs it (the owner's standing instruction); do not wait to be told. It is needed before committing to an approach or a design on substantial work, when results do not fit or an approach is not converging, before changing approach, and before reporting substantial work as done. A routine step that the last tool output already dictates needs no call.
- An orchestrator that runs on the advisor's own model gets a second look rather than a stronger one and pays for its whole transcript on each call, so it calls sparingly: at a change of approach and before reporting substantial work as done.
- Weigh its advice against primary evidence: where the code or a measurement contradicts it, say so and reconcile instead of switching silently.
