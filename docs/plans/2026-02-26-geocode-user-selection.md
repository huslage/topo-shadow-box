# Geocode User Selection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `geocode_place` auto-select single results and return `isError: true` for multiple results so the LLM stops and asks the user instead of auto-proceeding.

**Architecture:** Two behavior branches in `geocode_place`: single result sets the area directly and returns success; multiple results store candidates and raise an exception (FastMCP converts unhandled exceptions to `isError: true` in the MCP response, which breaks Claude's tool-chaining loop). `select_geocode_result` is unchanged except its docstring.

**Tech Stack:** Python, FastMCP (MCP SDK 1.26.0), Pydantic, pytest

---

### Task 1: Single result auto-selects area

**Files:**
- Modify: `src/topo_shadow_box/tools/area.py:190-221`
- Test: `tests/test_area_tools.py`

**Step 1: Write the failing tests**

Add to `tests/test_area_tools.py`:

```python
FAKE_SINGLE_RESULT = [
    {
        "display_name": "Mount Hood, Hood River County, Oregon, United States",
        "lat": "45.3736",
        "lon": "-121.6959",
        "type": "peak",
        "boundingbox": ["45.3536", "45.3936", "-121.7159", "-121.6759"],
    }
]


def test_geocode_place_single_result_auto_selects_area(monkeypatch):
    import httpx
    from topo_shadow_box.state import Bounds

    geocode_place = _register_and_get("geocode_place")
    state.bounds = Bounds()  # reset
    state.pending_geocode_candidates = []

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _make_fake_geocode_response(FAKE_SINGLE_RESULT))

    result = geocode_place(query="Mount Hood")

    assert state.bounds.is_set
    assert abs(state.bounds.north - 45.3936) < 0.001
    assert abs(state.bounds.south - 45.3536) < 0.001
    assert "auto" in result.lower() or "1 result" in result.lower() or "mount hood" in result.lower()


def test_geocode_place_single_result_clears_pending_candidates(monkeypatch):
    import httpx

    geocode_place = _register_and_get("geocode_place")
    state.pending_geocode_candidates = []

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _make_fake_geocode_response(FAKE_SINGLE_RESULT))

    geocode_place(query="Mount Hood")

    assert state.pending_geocode_candidates == []
```

**Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_area_tools.py::test_geocode_place_single_result_auto_selects_area tests/test_area_tools.py::test_geocode_place_single_result_clears_pending_candidates -v
```

Expected: FAIL — `state.bounds.is_set` is False (current code doesn't auto-select).

**Step 3: Implement single-result auto-select**

In `src/topo_shadow_box/tools/area.py`, replace the block after candidates are built (after the closing `)`  of the candidates loop, currently line ~207) so that when `len(candidates) == 1` it sets the area and returns:

```python
        # Single result: auto-select without requiring user input
        if len(candidates) == 1:
            c = candidates[0]
            bounds = Bounds(
                north=c.bbox_north,
                south=c.bbox_south,
                east=c.bbox_east,
                west=c.bbox_west,
                is_set=True,
            )
            state.bounds = bounds
            state.pending_geocode_candidates = []
            state.elevation = ElevationData()
            state.features = OsmFeatureSet()
            state.terrain_mesh = None
            state.feature_meshes = []
            state.gpx_mesh = None
            return (
                f"Found 1 result: '{c.display_name}' (auto-selected). "
                f"Area set: N={bounds.north:.6f}, S={bounds.south:.6f}, "
                f"E={bounds.east:.6f}, W={bounds.west:.6f} "
                f"(~{bounds.lat_range * 111_000:.0f}m x {bounds.lon_range * 111_000:.0f}m)"
            )

        state.pending_geocode_candidates = candidates
        # ... rest of existing multiple-results code unchanged
```

Place this block between the candidate list construction and the existing `state.pending_geocode_candidates = candidates` line.

**Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_area_tools.py::test_geocode_place_single_result_auto_selects_area tests/test_area_tools.py::test_geocode_place_single_result_clears_pending_candidates -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/topo_shadow_box/tools/area.py tests/test_area_tools.py
git commit -m "feat: auto-select area when geocode returns single result"
```

---

### Task 2: Multiple results raise error (isError: true)

**Files:**
- Modify: `src/topo_shadow_box/tools/area.py:207-221`
- Test: `tests/test_area_tools.py`

**Step 1: Write the failing test**

Add to `tests/test_area_tools.py`:

```python
def test_geocode_place_multiple_results_raises_exception(monkeypatch):
    import httpx
    import pytest

    geocode_place = _register_and_get("geocode_place")
    state.pending_geocode_candidates = []

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _make_fake_geocode_response())

    with pytest.raises(Exception) as exc_info:
        geocode_place(query="Mount Hood")

    msg = str(exc_info.value)
    assert "1." in msg
    assert "2." in msg
    assert "select_geocode_result" in msg


def test_geocode_place_multiple_results_still_stores_candidates(monkeypatch):
    import httpx
    import pytest

    geocode_place = _register_and_get("geocode_place")
    state.pending_geocode_candidates = []

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _make_fake_geocode_response())

    with pytest.raises(Exception):
        geocode_place(query="Mount Hood")

    assert len(state.pending_geocode_candidates) == 2
```

**Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_area_tools.py::test_geocode_place_multiple_results_raises_exception tests/test_area_tools.py::test_geocode_place_multiple_results_still_stores_candidates -v
```

Expected: FAIL — current code returns a string, not an exception.

**Step 3: Replace the multiple-results return with a raise**

In `src/topo_shadow_box/tools/area.py`, replace the final `lines.append(...)` and `return "\n".join(lines)` block for the multiple-results path:

```python
        lines = [f"Found {len(candidates)} location(s) for '{query}':\n"]
        for i, c in enumerate(candidates, 1):
            lines.append(
                f"{i}. {c.display_name}\n"
                f"   Type: {c.place_type} | Center: {c.lat:.5f}, {c.lon:.5f}\n"
                f"   Bbox: N={c.bbox_north:.5f}, S={c.bbox_south:.5f}, "
                f"E={c.bbox_east:.5f}, W={c.bbox_west:.5f}"
            )
        lines.append(
            f"\nUser input required: ask the user which number (1–{len(candidates)}) "
            "they want, then call select_geocode_result with that number."
        )
        raise ValueError("\n".join(lines))
```

**Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_area_tools.py::test_geocode_place_multiple_results_raises_exception tests/test_area_tools.py::test_geocode_place_multiple_results_still_stores_candidates -v
```

Expected: PASS

**Step 5: Run the full test suite**

```bash
.venv/bin/python -m pytest --tb=short
```

Expected: All tests pass. Note: `test_geocode_place_returns_candidates` and `test_geocode_place_stores_candidates_in_state` now need updating — they test behavior that has changed. See Task 3.

**Step 6: Commit**

```bash
git add src/topo_shadow_box/tools/area.py tests/test_area_tools.py
git commit -m "feat: raise error for multiple geocode results to force user selection"
```

---

### Task 3: Fix affected existing tests

The tests `test_geocode_place_returns_candidates` and `test_geocode_place_stores_candidates_in_state` were written against the old behavior (returns string). They now need to use `pytest.raises`.

**Files:**
- Modify: `tests/test_area_tools.py`

**Step 1: Update the two tests**

`test_geocode_place_returns_candidates` — currently asserts on the return value. Change to assert on the exception message:

```python
def test_geocode_place_returns_candidates(monkeypatch):
    import httpx
    import pytest

    geocode_place = _register_and_get("geocode_place")

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _make_fake_geocode_response())

    with pytest.raises(Exception) as exc_info:
        geocode_place(query="Mount Hood")

    msg = str(exc_info.value)
    assert "1." in msg
    assert "2." in msg
    assert "45.3736" in msg
    assert "peak" in msg
```

`test_geocode_place_stores_candidates_in_state` — add `pytest.raises` wrapper:

```python
def test_geocode_place_stores_candidates_in_state(monkeypatch):
    import httpx
    import pytest

    geocode_place = _register_and_get("geocode_place")
    state.pending_geocode_candidates = []

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _make_fake_geocode_response())

    with pytest.raises(Exception):
        geocode_place(query="Mount Hood")

    assert len(state.pending_geocode_candidates) == 2
    assert state.pending_geocode_candidates[0].lat == 45.3736
    assert state.pending_geocode_candidates[1].place_type == "resort"
```

**Step 2: Run full test suite**

```bash
.venv/bin/python -m pytest --tb=short
```

Expected: All tests pass, no warnings.

**Step 3: Commit**

```bash
git add tests/test_area_tools.py
git commit -m "test: update geocode tests for new error-based multiple-results behavior"
```

---

### Task 4: Update select_geocode_result docstring

**Files:**
- Modify: `src/topo_shadow_box/tools/area.py:224-234`

**Step 1: Update the docstring**

Replace the existing `select_geocode_result` docstring with:

```python
        """Select a geocode candidate by number and set it as the area of interest.

        Only call this after geocode_place returned multiple candidates AND the user
        has replied with their chosen number. Do not call this without a user-provided
        number — geocode_place handles single results automatically.

        **Requires:** geocode_place called first with multiple results (candidates stored in session).
        **Next:** fetch_elevation, then fetch_features (optional), then generate_model.

        Args:
            number: 1-based index of the candidate the user selected.
        """
```

**Step 2: Run full test suite to confirm nothing broke**

```bash
.venv/bin/python -m pytest --tb=short
```

Expected: All tests pass.

**Step 3: Commit and push, open PR**

```bash
git add src/topo_shadow_box/tools/area.py
git commit -m "docs: clarify select_geocode_result docstring"
git push -u origin HEAD
gh pr create --title "fix: auto-select single geocode results, raise error for multiple" \
  --body "$(cat <<'EOF'
## Summary

- Single geocode result → area is set automatically, no user input needed
- Multiple geocode results → raises exception (FastMCP returns isError: true), breaking Claude's tool-chaining loop so it stops and asks the user which candidate to use
- Updated select_geocode_result docstring to clarify when to call it

## Test plan
- [ ] Single result auto-selects and sets bounds
- [ ] Single result clears pending candidates
- [ ] Multiple results raises exception with numbered list
- [ ] Multiple results still stores candidates in state
- [ ] All 267 existing tests pass

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
