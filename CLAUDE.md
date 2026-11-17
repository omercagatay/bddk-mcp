@AGENTS.md

# Claude Code only

AGENTS.md above is the canonical shared instruction file; maintain shared rules there rather than duplicating them here.

## Subagent models

The tiers of "Delegation and model choice" in AGENTS.md map to Claude models as follows: smallest model is Haiku, middle model is Sonnet, strongest subagent model is Opus. The top tier is the session's own model (Fable, or Mythos when the owner runs the session on it). Pass `model` explicitly on every Agent call and on every `agent()` call of a workflow, because an omitted `model` inherits the session's model; omit it only for a job AGENTS.md gives to the top tier.

## Advisor

The advisor is the model the owner chose with `/advisor`, stored as `advisorModel` in the owner's user settings. It is the owner's setting: an agent does not change it.

- The agent that does the work calls it. Subagents inherit the configured advisor and receive the `advisor` tool; a cheaper model working with the stronger one advising is the intended pairing.
- Each call sends the caller's whole transcript to the advisor uncached, so a subagent's focused transcript is the cheaper place to ask from. Every brief for substantial delegated work therefore tells the subagent to consult the advisor at the points below.
- Use it whenever the work needs it; do not wait to be told. It is needed before committing to an approach or a design on substantial work, when results do not fit or an approach is not converging, before changing approach, and before reporting substantial work as done. A routine step that the last tool output already dictates needs no call.
- An orchestrator that runs on the advisor's own model gets a second look rather than a stronger one and pays for its whole transcript on each call, so it calls sparingly: at a change of approach and before reporting substantial work as done.
- Weigh its advice against primary evidence: where the code or a measurement contradicts it, say so and reconcile instead of switching silently.
