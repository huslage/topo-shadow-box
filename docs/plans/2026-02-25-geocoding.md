# Geocoding Feature Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `geocode_place` MCP tool that resolves a place name to a list of candidate locations so the model can present options to the user before calling `set_area_from_coordinates`.

**Architecture:** New read-only tool in `tools/area.py` registered alongside existing area tools. Uses the Nominatim (OpenStreetMap) geocoding API via synchronous `httpx`. Returns a formatted candidate list string; does not modify session state. A `GeocodeCandidate` Pydantic model in `models.py` holds each result.

**Tech Stack:** Python 3.11+, Pydantic v2, httpx (sync), Nominatim API (free, no key required)

---

### Task 1: Add `GeocodeCandidate` model

**Files:**
- Modify: `src/topo_shadow_box/models.py`

**Step 1: Write the failing test**

Add to `tests/test_area_tools.py`:

```python
def test_geocode_candidate_model():
    from topo_shadow_box.models import GeocodeCandidate
    c = GeocodeCandidate(
        display_name="Mount Hood, Hood River County, Oregon, United States",
        lat=45.3736,
        lon=-121.6959,
        place_type="peak",
        bbox_north=45.3936,
        bbox_south=45.3536,
        bbox_east=-121.6759,
        bbox_west=-121.7159,
    )
    assert c.lat == 45.3736
    assert c.bbox_north == 45.3936
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/huslage/work/topo-shadow-box
uv run pytest tests/test_area_tools.py::test_geocode_candidate_model -v
```

Expected: FAIL with `ImportError: cannot import name 'GeocodeCandidate'`

**Step 3: Add `GeocodeCandidate` to models.py**

Append to the end of `src/topo_shadow_box/models.py`:

```python


class GeocodeCandidate(BaseModel):
    display_name: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    place_type: str
    bbox_north: float
    bbox_south: float
    bbox_east: float
    bbox_west: float
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_area_tools.py::test_geocode_candidate_model -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/topo_shadow_box/models.py tests/test_area_tools.py
git commit -m "feat: add GeocodeCandidate model"
```

---

### Task 2: Implement `geocode_place` tool

**Files:**
- Modify: `src/topo_shadow_box/tools/area.py`

**Step 1: Write the failing tests**

Add to `tests/test_area_tools.py`:

```python
def test_geocode_place_returns_candidates(monkeypatch):
    import httpx
    from unittest.mock import MagicMock

    geocode_place = _register_and_get("geocode_place")

    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = [
        {
            "display_name": "Mount Hood, Hood River County, Oregon, United States",
            "lat": "45.3736",
            "lon": "-121.6959",
            "type": "peak",
            "boundingbox": ["45.3536", "45.3936", "-121.7159", "-121.6759"],
        },
        {
            "display_name": "Mount Hood Meadows, Clackamas County, Oregon, United States",
            "lat": "45.3300",
            "lon": "-121.6660",
            "type": "resort",
            "boundingbox": ["45.3200", "45.3400", "-121.6760", "-121.6560"],
        },
    ]

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: fake_response)

    result = geocode_place(query="Mount Hood")
    assert "1." in result
    assert "2." in result
    assert "45.3736" in result
    assert "peak" in result


def test_geocode_place_no_results(monkeypatch):
    import httpx
    from unittest.mock import MagicMock

    geocode_place = _register_and_get("geocode_place")

    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = []

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: fake_response)

    result = geocode_place(query="xyzzy_nowhere_12345")
    assert "no locations found" in result.lower()


def test_geocode_place_network_error(monkeypatch):
    import httpx

    geocode_place = _register_and_get("geocode_place")

    def raise_error(*a, **kw):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", raise_error)

    result = geocode_place(query="Mount Hood")
    assert "error" in result.lower()


def test_geocode_place_limit_clamped(monkeypatch):
    import httpx
    from unittest.mock import MagicMock

    geocode_place = _register_and_get("geocode_place")

    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = []

    captured = {}

    def fake_get(url, **kwargs):
        captured["params"] = kwargs.get("params", {})
        return fake_response

    monkeypatch.setattr(httpx, "get", fake_get)

    geocode_place(query="test", limit=99)
    assert captured["params"]["limit"] <= 10

    geocode_place(query="test", limit=0)
    assert captured["params"]["limit"] >= 1
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_area_tools.py::test_geocode_place_returns_candidates tests/test_area_tools.py::test_geocode_place_no_results tests/test_area_tools.py::test_geocode_place_network_error tests/test_area_tools.py::test_geocode_place_limit_clamped -v
```

Expected: FAIL with `KeyError: 'geocode_place'`

**Step 3: Implement `geocode_place` in `tools/area.py`**

Add `import httpx` at the top of `tools/area.py` alongside existing imports. Add `GeocodeCandidate` to the import from `..models`. Then add the following inside `register_area_tools`, after the `validate_area` function:

```python
    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def geocode_place(query: str, limit: int = 5) -> str:
        """Search for a place by name and return candidate locations with coordinates.

        Use this when the user provides a place name but no coordinates or GPX file.
        If the user provides a GPX file, use set_area_from_gpx instead — no geocoding needed.

        Returns a numbered list of candidates. Present them to the user, let them pick one,
        then call set_area_from_coordinates with the chosen lat/lon or bounding box.

        **Next:** set_area_from_coordinates with the chosen candidate's coordinates.

        Args:
            query: Place name to search for (e.g., "Mount Hood", "Grand Canyon", "Portland, Oregon").
            limit: Maximum number of candidates to return (1–10, default 5).
        """
        limit = max(1, min(10, limit))

        try:
            response = httpx.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": query, "format": "json", "limit": limit},
                headers={"User-Agent": "topo-shadow-box/1.0"},
                timeout=10.0,
            )
            response.raise_for_status()
            results = response.json()
        except httpx.HTTPStatusError as exc:
            return f"Error: Nominatim returned HTTP {exc.response.status_code}."
        except Exception as exc:
            return f"Error contacting geocoding service: {exc}"

        if not results:
            return f"No locations found for '{query}'. Try a more specific name or add a region (e.g., 'Portland, Oregon')."

        from ..models import GeocodeCandidate

        candidates = []
        for item in results:
            bbox = item.get("boundingbox", [])
            # Nominatim boundingbox order: [south, north, west, east]
            candidates.append(
                GeocodeCandidate(
                    display_name=item["display_name"],
                    lat=float(item["lat"]),
                    lon=float(item["lon"]),
                    place_type=item.get("type", "unknown"),
                    bbox_south=float(bbox[0]) if len(bbox) >= 4 else float(item["lat"]),
                    bbox_north=float(bbox[1]) if len(bbox) >= 4 else float(item["lat"]),
                    bbox_west=float(bbox[2]) if len(bbox) >= 4 else float(item["lon"]),
                    bbox_east=float(bbox[3]) if len(bbox) >= 4 else float(item["lon"]),
                )
            )

        lines = [f"Found {len(candidates)} location(s) for '{query}':\n"]
        for i, c in enumerate(candidates, 1):
            lines.append(
                f"{i}. {c.display_name}\n"
                f"   Type: {c.place_type} | Center: {c.lat:.5f}, {c.lon:.5f}\n"
                f"   Bbox: N={c.bbox_north:.5f}, S={c.bbox_south:.5f}, "
                f"E={c.bbox_east:.5f}, W={c.bbox_west:.5f}"
            )
        lines.append(
            "\nAsk the user which location to use, then call set_area_from_coordinates "
            "with the chosen lat/lon (and a radius_m) or the bounding box coordinates."
        )
        return "\n".join(lines)
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_area_tools.py::test_geocode_place_returns_candidates tests/test_area_tools.py::test_geocode_place_no_results tests/test_area_tools.py::test_geocode_place_network_error tests/test_area_tools.py::test_geocode_place_limit_clamped -v
```

Expected: all 4 PASS

**Step 5: Run the full test suite to check for regressions**

```bash
uv run pytest tests/test_area_tools.py -v
```

Expected: all tests PASS

**Step 6: Commit**

```bash
git add src/topo_shadow_box/tools/area.py tests/test_area_tools.py
git commit -m "feat: add geocode_place tool using Nominatim API"
```

---

### Task 3: Add permission entry and update docstring

**Files:**
- Modify: `.claude/settings.local.json`
- Modify: `src/topo_shadow_box/tools/area.py`

**Step 1: Add `geocode_place` to the allow list in `.claude/settings.local.json`**

Add this line to the `"allow"` array alongside the other `mcp__plugin_topo-shadow-box_topo-shadow-box__*` entries:

```json
"mcp__plugin_topo-shadow-box_topo-shadow-box__geocode_place",
```

**Step 2: Update `set_area_from_coordinates` docstring to mention geocoding**

In `tools/area.py`, update the docstring of `set_area_from_coordinates` to add a Prior step hint:

```
        **Prior:** If you only have a place name, call geocode_place first to get coordinates.
```

Add it right before the existing `**Next:**` line.

**Step 3: Run the full test suite one final time**

```bash
uv run pytest -v
```

Expected: all tests PASS

**Step 4: Commit**

```bash
git add .claude/settings.local.json src/topo_shadow_box/tools/area.py
git commit -m "feat: wire geocode_place permissions and docstring cross-references"
```

---

### Done

The feature is complete when:
- `geocode_place("Mount Hood")` returns a numbered list of candidates
- `geocode_place("xyzzy_nowhere")` returns a friendly no-results message
- Network failures return a descriptive error string
- `limit` outside 1–10 is silently clamped
- The tool is in the permissions allow list (no prompts during generation)
- All existing tests still pass
