# Permissions Documentation Design

**Date:** 2026-02-25
**Status:** Approved

## Problem

Users who install the plugin from the marketplace get permission prompts when tools are called for the first time. One-time prompts are acceptable; the friction is not knowing to click "Allow always" vs "Allow once", or how to pre-approve everything upfront.

## Design

Add a **Permissions** section to `README.md`, between Installation and Example Conversations, covering:

1. A one-sentence explanation of why the prompts appear (Claude asks before calling new MCP tools)
2. Instruction to click **"Allow always"** (not "Allow") so approvals persist
3. A wildcard snippet for users who want to pre-approve all tools at once via `~/.claude/settings.json`

The wildcard `mcp__plugin_topo-shadow-box_topo-shadow-box__*` covers all current and future tools in a single entry — no need to update it when new tools are added.

## Non-Goals

- No code changes
- No undocumented settings.json hacking
- No plugin manifest changes
