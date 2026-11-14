# Project Instructions

Read and follow [AGENTS.md](AGENTS.md), the canonical shared instructions for this repository, before working on the project. Maintain shared rules there rather than duplicating them here.

## Claude Code only: subagent models

The tiers of "Delegation and model choice" in AGENTS.md map to the current Claude models as follows: smallest model is Haiku, middle model is Sonnet, strongest subagent model is Opus. The session's own model (Fable or another model above Opus) orchestrates and reviews and does not run subagents. Pass `model` explicitly on every Agent call and on every `agent()` call of a workflow, because an omitted `model` inherits the session's model.

## Claude Code only: advisor

The advisor is Fable 5.2; the owner switches it on for a session with `/advisor fable`, and an agent cannot switch it on itself. The agent that does the work calls it: subagents started in a session with an advisor receive the `advisor` tool too (checked on 2026-11-15 with a Haiku subagent), and that is the intended pairing, a cheaper model working and the stronger one advising. Every brief for substantial delegated work therefore tells the subagent to consult the advisor at the points below. The orchestrator, when it runs on the advisor's own model, gets a second look rather than a stronger one and pays for its whole transcript on each call, so it calls sparingly: at a change of approach and before reporting substantial work as done.

When the advisor tool is available, use it whenever the work needs it (the owner's standing instruction of 2026-11-15); do not wait to be told. It is needed before committing to an approach or a design on substantial work, when results do not fit or an approach is not converging, before changing approach, and before reporting substantial work as done. A routine step that the last tool output already dictates needs no call. Weigh its advice against primary evidence: where the code or a measurement contradicts it, say so and reconcile instead of switching silently.

@AGENTS.md
