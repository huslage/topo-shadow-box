# Codebase Improvements Design

**Date:** 2026-02-24
**Approach:** Prioritized Incremental (B)

## Summary

Five phases of improvements addressing reliability, test coverage, performance, security, and architecture — in that order of priority. Each phase is an independent, reviewable unit.

---

## Phase 1: Reliability & Error Handling

**Goal:** Replace silent failures with visible, actionable logging.

**Changes:**
- `elevation.py:111-112` — Replace `except: pass` with `logger.warning()` recording the failed tile URL and exception. Zeros-for-failed-tiles behavior is preserved.
- `osm.py:24-25` — Replace `except Exception: continue` with explicit catches:
  - `httpx.TimeoutException` → warn and try next server
  - `httpx.HTTPStatusError` (429, 503) → log status and try next server
  - Other exceptions → log and try next server
  - Return a typed signal distinguishing "no features found" from "all servers failed"
- `tools/data.py` and other tools — Surface the "all servers failed" signal in the tool return string.
- Add module-level `logging.getLogger(__name__)` to `elevation.py`, `osm.py`, and tools layer.

**Non-changes:** Tile-zeros fallback stays; no new dependencies; no async/sync boundary changes.

---

## Phase 2: Test Coverage

**Goal:** Close gaps in exporters, network failures, integration, and edge cases.

**Changes:**
- `tests/test_exporters.py` (new):
  - 3MF: valid ZIP structure, required XML files, multi-material, entity escaping
  - OpenSCAD: contains `polyhedron()`, valid structure
  - SVG: valid XML, expected elements present
- Mock `httpx.AsyncClient` in elevation and OSM tests:
  - `test_elevation.py`: partial tile failure, full failure, timeout
  - `test_osm.py`: all servers timeout, HTTP 429, mixed success/failure
- `tests/test_integration.py` (new): one end-to-end test with mocked HTTP running full pipeline (set area → fetch elevation → generate model → export 3MF). Verifies output file exists and is non-zero.
- Edge cases added to existing test files: very small area (< 100m), all-zero elevation grid, empty feature list after clipping.

**Non-changes:** pytest only; mocking via `unittest.mock`. No new test frameworks.

---

## Phase 3: Performance

**Goal:** Remove avoidable sequential I/O and redundant computation.

**Changes:**
- `elevation.py:95-110` — Collect all tile URLs, fetch concurrently with `asyncio.gather()`, stitch results. Replaces sequential nested loop.
- `mesh.py` — Calculate `_elevation_normalization()` once in `generate_model()`, pass as parameter to all mesh generators. Removes repeated O(n log n) percentile calculations.

**Non-changes:** Tile sources, interpolation logic, and output are identical. Pure performance refactor.

---

## Phase 4: Security & Correctness

**Goal:** Prevent abuse vectors and fix known correctness gaps.

**Changes:**
- `osm.py` — Add `asyncio.sleep(1.0)` between Overpass requests to respect OSM's 1 req/sec rate limit.
- `threemf.py:54-56` — Add `>` → `&gt;` to existing XML entity escaping.
- `tools/export.py` — Validate resolved absolute output path before `os.makedirs()`; raise a clear error if path escapes the user's home directory.
- `elevation.py` and `osm.py` HTTP clients — Add `User-Agent: topo-shadow-box/1.0` header.

**Non-changes:** No new dependencies; no behavior changes for valid inputs.

---

## Phase 5: Architecture

**Goal:** Improve internal consistency and document known constraints.

**Changes:**
- Add a `require_state()` helper for uniform tool prerequisite validation, replacing ad-hoc string error returns.
- `export.py` — Log a warning (instead of silently falling back) when a mesh feature type has no corresponding color in `SessionState`. Add a test asserting all feature types have colors.
- `state.py` — Add a prominent comment documenting the single-client limitation of the global singleton and what would need to change for multi-client support.

**Non-changes:** Global singleton stays — refactoring it requires touching every tool and test with minimal practical benefit for a single-user MCP server.

---

## Sequencing Rationale

1. **Reliability first** — Silent failures cause data quality issues today and are cheap to fix.
2. **Tests second** — Test infrastructure catches regressions introduced by Phases 3–5.
3. **Performance third** — Pure refactors with measurable benefit, low risk once tests are in place.
4. **Security fourth** — Important but not blocking current usage.
5. **Architecture last** — Lowest urgency; mostly documentation and minor internal consistency.
