# MCP Improvements Design

**Date:** 2026-02-24
**Approach:** B — Tool-layer progress orchestration

## Summary

Five improvements to the topo-shadow-box MCP server: a passive state resource, a `validate_area` tool, fine-grained progress notifications during mesh generation, richer tool docstrings, and session serialization. All implemented with TDD.

---

## Section 1: State Resource + validate_area Tool

### State Resource (`state://session`)

Register a FastMCP resource using `@mcp.resource("state://session")`. Returns `json.dumps(state.summary())` — the same data the `get_status` tool already produces. Clients can read it passively without consuming a tool call.

**Files:** `src/topo_shadow_box/server.py` (register resource), or a new `src/topo_shadow_box/resources.py`.

**Testing:** Verify the registered URI returns valid JSON matching `state.summary()`.

### `validate_area` Tool

New tool added to `src/topo_shadow_box/tools/area.py`. Called after `set_area_*`. Checks:
- Area span < 100m → error ("area too small for meaningful terrain")
- Area span > 500km → warning ("very large area — fetching may be slow and detail will be low")
- If elevation already fetched: relief < 20m → warning ("model will print nearly flat; consider a more mountainous area")

Returns structured string with warnings, or "Area looks good." Does not block the pipeline.

**Testing (TDD):**
- Small area → error string
- Large area → warning string
- Flat relief → warning string
- Valid area, no elevation → "Area looks good"
- Valid area, good relief → "Area looks good"

---

## Section 2: Progress Notifications

### `generate_model` becomes async with `ctx: Context`

`generate_model` in `src/topo_shadow_box/tools/generate.py` gains `ctx: Context` parameter and becomes `async def`. FastMCP passes Context automatically when declared.

### Progress granularity

Total units counted upfront:
- 1 for terrain mesh
- 1 per road (up to 200)
- 1 per water body (up to 50)
- 1 per building (up to 150)
- 1 for GPX track (if present)

`await ctx.report_progress(current, total)` called after each unit completes.

### Tool-layer per-feature iteration

Features iterated one-by-one in the tool layer. A new function `generate_single_feature_mesh(feature, feature_type, elevation, bounds, transform, vertical_scale, shape)` added to `src/topo_shadow_box/core/mesh.py`. The existing `generate_feature_meshes` batch function stays for backwards compatibility and test use.

**Non-changes:** Core mesh generation functions (`generate_terrain_mesh`, individual mesh generators) stay synchronous and unchanged.

### Testing (TDD)

Mock `ctx.report_progress`. Run `generate_model` with a known feature count. Assert:
- Called exactly `1 + n_roads + n_water + n_buildings + (1 if gpx else 0)` times
- Progress values are non-decreasing
- Final progress equals total

---

## Section 3: Richer Tool Docstrings

All 12 tools get updated docstrings following the pattern:
> [Purpose]. **Requires:** [prerequisite tools]. **Next:** [what to call after].

Tools updated: `set_area_from_coordinates`, `set_area_from_gpx`, `fetch_elevation`, `fetch_features`, `set_model_params`, `set_colors`, `generate_model`, `generate_map_insert`, `export_3mf`, `export_openscad`, `export_svg`, `preview`.

**Testing:** No unit tests — verified by inspection during code review.

---

## Section 4: State Serialization

### Two new tools: `save_session` and `load_session`

Added to a new file `src/topo_shadow_box/tools/session.py`, registered in `server.py`.

**`save_session(path: str | None = None)`**
- Default path: `~/.cache/topo-shadow-box/session.json`
- Serializes: bounds, model params, colors, elevation metadata (min/max/resolution — not the grid), GPX track coordinates
- Does NOT serialize meshes or elevation grid (regeneratable, too large)
- Creates directory if needed

**`load_session(path: str | None = None)`**
- Restores: bounds, model params, colors, GPX tracks
- Sets `elevation.is_set = False` and clears features/meshes
- Returns summary of what was restored and what still needs fetching (elevation, features, model)

Session files are < 5KB and human-readable JSON.

### Testing (TDD)

- Save configured session → load into fresh state → assert bounds/params/colors match
- After load: elevation.is_set is False, terrain_mesh is None
- Default path directory is created if missing
- Load nonexistent file → clear error message
- Round-trip for GPX track coordinates

---

## Implementation Order

1. `validate_area` tool (standalone, no dependencies)
2. State resource (standalone, one decorator)
3. Richer docstrings (no logic changes)
4. State serialization (`save_session` / `load_session`)
5. Progress notifications (most invasive — last)
