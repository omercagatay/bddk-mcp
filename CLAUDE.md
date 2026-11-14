# Project Instructions

Read and follow [AGENTS.md](AGENTS.md), the canonical shared instructions for this repository, before working on the project. Maintain shared rules there rather than duplicating them here.

## Claude Code only: subagent models

The tiers of "Delegation and model choice" in AGENTS.md map to the current Claude models as follows: smallest model is Haiku, middle model is Sonnet, strongest subagent model is Opus. The session's own model (Fable or another model above Opus) orchestrates and reviews and does not run subagents. Pass `model` explicitly on every Agent call and on every `agent()` call of a workflow, because an omitted `model` inherits the session's model.

@AGENTS.md
