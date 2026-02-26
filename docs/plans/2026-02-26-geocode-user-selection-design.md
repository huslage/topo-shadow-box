# Geocode User Selection Design

**Date:** 2026-02-26

## Problem

`geocode_place` returns multiple candidates with text instructions to ask the
user before proceeding, but the LLM ignores this and immediately calls
`select_geocode_result(1)`. Text instructions inside successful tool results
compete against the model's trained drive to continue making progress in an
active agentic loop.

## Root Cause

Claude Code's agentic loop runs as long as `stop_reason == "tool_use"`. It only
exits to the user on `end_turn`. There is no MCP elicitation support in Claude
Code (open issue #2799). Instructions embedded in tool result text are
unreliable — the model treats them as guidance, not gates.

## Design

### Single result → auto-select

When `geocode_place` returns exactly 1 candidate, set the area automatically
using that candidate's bounding box. Clear `pending_geocode_candidates`. Return
a success message indicating the result was auto-selected.

No user interaction required. This is the common case.

### Multiple results → `isError: true`

When `geocode_place` returns 2+ candidates, store them in
`state.pending_geocode_candidates` and **raise an exception**. FastMCP catches
exceptions from tool handlers and returns `CallToolResult(isError=True, ...)`.

Error results cause Claude to stop the tool-calling pipeline and report to the
user rather than continue — the behavioral difference we need. The exception
message contains the numbered candidate list plus the instruction to reply with
a number so `select_geocode_result` can be called.

### `select_geocode_result` — no logic changes

Docstring updated to clarify it is only needed when geocode returned multiple
results. All validation and area-setting logic stays the same.

## What Is Not Changing

- The `select_geocode_result` tool logic
- `set_area_from_coordinates` (still usable for direct coordinate input)
- Behavior when geocode returns 0 results

## Reliability

This is "most of the time" reliable, not guaranteed. Error results strongly
bias Claude toward stopping, but in edge cases (e.g., very directive system
prompts) it could still auto-proceed. The correct long-term fix is MCP
elicitation support in Claude Code (issue #2799).
