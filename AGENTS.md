# Project instructions

## Use the Polypack MCP

Use the Polypack MCP as the project’s durable, cross-session memory when it is
available. It is supplementary context; the repository, tests, and current
task instructions remain authoritative.

- Before substantive work, call `mcp__polypack__memory_context` for the current
  task/repository context. Use `mcp__polypack__memory_recall` when a targeted
  question, prior decision, or historical constraint would help.
- After meaningful work, store durable and reusable facts with
  `mcp__polypack__memory_store`: architectural decisions, invariants, tricky
  compatibility constraints, proven workflows, and notable failure modes.
  Include a concise context and provenance when known. Do not store secrets,
  credentials, large code dumps, or transient conversational details.
- If recalled memory materially helped or was misleading, record that with
  `mcp__polypack__memory_feedback`. Treat memories as hypotheses until checked
  against the current checkout.
- Use `mcp__polypack__memory_consolidate` only to turn several related,
  evidence-backed memories into a durable summary. Use
  `mcp__polypack__memory_supersede` for changed facts and
  `mcp__polypack__memory_suppress` for stale or unhelpful memories; preserve
  history rather than deleting it.
- Use `mcp__polypack__graph_query` only for narrowly scoped graph inspection or
  escape-hatch operations when the memory tools do not provide what is needed.

If the MCP is unavailable or returns no useful context, continue using local
repository evidence and do not block the task.

## Repository conventions

- Preserve unrelated working-tree changes.
- Prefer `rg` for repository searches and `apply_patch` for edits.
- Run focused tests or checks relevant to the change before handing off.
- Keep TypeScript, Python, and Rust behavior aligned where the same public
  contract is implemented in multiple languages.
