# MCP Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add five MCP improvements: `validate_area` tool, `state://session` resource, richer tool docstrings, session save/load, and fine-grained progress notifications in `generate_model`.

**Architecture:** All changes are additive except `generate_model`, which becomes `async def` with a `ctx: Context` parameter for progress reporting. Features are iterated one-by-one in the tool layer; core mesh functions stay unchanged. Session serialization uses JSON to `~/.cache/topo-shadow-box/session.json`.

**Tech Stack:** Python 3.11+, FastMCP (mcp[cli]>=1.0), pydantic v2, pytest, anyio

---

## Task 1: validate_area tool

**Files:**
- Modify: `src/topo_shadow_box/tools/area.py`
- Test: `tests/test_area_tools.py` (new)

**Step 1: Write the failing tests**

Create `tests/test_area_tools.py`:

```python
"""Tests for validate_area tool."""
import pytest
from unittest.mock import MagicMock
from topo_shadow_box.state import state, Bounds, ElevationData
import numpy as np


def _register_and_get(tool_name: str):
    """Register area tools against a mock MCP and extract the named tool."""
    from topo_shadow_box.tools.area import register_area_tools
    tools = {}
    mock_mcp = MagicMock()
    def capture():
        def decorator(fn):
            tools[fn.__name__] = fn
            return fn
        return decorator
    mock_mcp.tool = capture
    register_area_tools(mock_mcp)
    return tools[tool_name]


def test_validate_area_small_area_returns_error():
    validate_area = _register_and_get("validate_area")
    # Set an area that is ~50m x 50m (way too small)
    state.bounds = Bounds(
        north=37.7501, south=37.7500, east=-122.4499, west=-122.4500, is_set=True
    )
    result = validate_area()
    assert "too small" in result.lower() or "error" in result.lower()


def test_validate_area_large_area_returns_warning():
    validate_area = _register_and_get("validate_area")
    # Set an area ~600km x 600km
    state.bounds = Bounds(
        north=47.75, south=37.75, east=-112.45, west=-122.45, is_set=True
    )
    result = validate_area()
    assert "large" in result.lower() or "warning" in result.lower()


def test_validate_area_flat_relief_warns_when_elevation_set():
    validate_area = _register_and_get("validate_area")
    state.bounds = Bounds(
        north=37.80, south=37.75, east=-122.40, west=-122.45, is_set=True
    )
    # Flat elevation (5m relief)
    state.elevation = ElevationData(
        grid=np.full((10, 10), 100.0),
        lats=np.linspace(37.75, 37.80, 10),
        lons=np.linspace(-122.45, -122.40, 10),
        resolution=10,
        min_elevation=98.0,
        max_elevation=103.0,
        is_set=True,
    )
    result = validate_area()
    assert "flat" in result.lower() or "relief" in result.lower()


def test_validate_area_good_area_no_elevation_passes():
    validate_area = _register_and_get("validate_area")
    from topo_shadow_box.state import ElevationData
    state.bounds = Bounds(
        north=37.80, south=37.75, east=-122.40, west=-122.45, is_set=True
    )
    state.elevation = ElevationData()  # not set
    result = validate_area()
    assert "good" in result.lower() or "ok" in result.lower()


def test_validate_area_good_area_good_relief_passes():
    validate_area = _register_and_get("validate_area")
    state.bounds = Bounds(
        north=37.80, south=37.75, east=-122.40, west=-122.45, is_set=True
    )
    state.elevation = ElevationData(
        grid=np.linspace(0, 500, 100).reshape(10, 10),
        lats=np.linspace(37.75, 37.80, 10),
        lons=np.linspace(-122.45, -122.40, 10),
        resolution=10,
        min_elevation=0.0,
        max_elevation=500.0,
        is_set=True,
    )
    result = validate_area()
    assert "good" in result.lower() or "ok" in result.lower()


def test_validate_area_requires_bounds_set():
    validate_area = _register_and_get("validate_area")
    from topo_shadow_box.state import Bounds
    state.bounds = Bounds()  # not set
    result = validate_area()
    assert "error" in result.lower() or "area" in result.lower()
```

**Step 2: Run to verify failure**

```bash
.venv/bin/pytest tests/test_area_tools.py -v 2>&1 | head -20
```
Expected: FAIL — `KeyError: 'validate_area'`

**Step 3: Implement validate_area in area.py**

Add after `set_area_from_gpx` inside `register_area_tools`:

```python
    @mcp.tool()
    def validate_area() -> str:
        """Check the current area for potential problems before fetching data.

        **Requires:** set_area_from_coordinates or set_area_from_gpx first.
        **Next:** fetch_elevation, then fetch_features (optional), then generate_model.

        Checks area size, and elevation relief if elevation has been fetched.
        Returns warnings but does not block the pipeline.
        """
        if not state.bounds.is_set:
            return "Error: Set an area first with set_area_from_coordinates or set_area_from_gpx."

        b = state.bounds
        # Approximate span in meters (1 degree lat ≈ 111km)
        lat_m = b.lat_range * 111_000
        lon_m = b.lon_range * 111_000 * abs(math.cos(math.radians(b.center_lat)))
        min_span_m = min(lat_m, lon_m)
        max_span_m = max(lat_m, lon_m)

        warnings = []

        if min_span_m < 100:
            return (
                f"Error: Area too small ({min_span_m:.0f}m minimum span). "
                "Use a larger area for meaningful terrain detail."
            )
        if max_span_m > 500_000:
            warnings.append(
                f"Very large area ({max_span_m / 1000:.0f}km span) — "
                "fetching will be slow and terrain detail will be low."
            )

        if state.elevation.is_set:
            relief = state.elevation.max_elevation - state.elevation.min_elevation
            if relief < 20:
                warnings.append(
                    f"Low elevation relief ({relief:.0f}m) — "
                    "model will print nearly flat. Consider a more mountainous area."
                )

        if warnings:
            return "Warnings: " + " | ".join(warnings)
        return "Area looks good."
```

Also add `import math` at the top of `area.py` (after existing imports).

**Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_area_tools.py -v
```
Expected: All PASS

**Step 5: Full suite**

```bash
.venv/bin/pytest --tb=short -q 2>&1 | tail -5
```

**Step 6: Commit**

```bash
git add src/topo_shadow_box/tools/area.py tests/test_area_tools.py
git commit -m "feat: add validate_area tool with size and relief checks"
```

---

## Task 2: State resource

**Files:**
- Modify: `src/topo_shadow_box/server.py`
- Test: `tests/test_state_resource.py` (new)

**Step 1: Write the failing test**

Create `tests/test_state_resource.py`:

```python
"""Tests for state://session MCP resource."""
import json
import pytest


def test_state_resource_returns_valid_json():
    """state://session resource should return JSON matching state.summary()."""
    from topo_shadow_box.server import mcp
    from topo_shadow_box.state import state

    # Find the registered resource
    resources = {r.uri: r for r in mcp._resource_manager._resources.values()}
    assert "state://session" in resources, (
        f"state://session not registered. Registered: {list(resources.keys())}"
    )


def test_state_resource_content_matches_summary():
    """Resource content should match state.summary() output."""
    from topo_shadow_box.server import mcp
    from topo_shadow_box.state import state
    import asyncio

    # Get the resource function directly
    resources = mcp._resource_manager._resources
    resource = next(
        (r for r in resources.values() if "session" in str(r.uri)), None
    )
    assert resource is not None

    # Call the resource function
    result = asyncio.get_event_loop().run_until_complete(resource.fn()) if asyncio.iscoroutinefunction(resource.fn) else resource.fn()
    parsed = json.loads(result)
    expected = state.summary()
    assert parsed.keys() == expected.keys()
```

**Step 2: Run to verify failure**

```bash
.venv/bin/pytest tests/test_state_resource.py::test_state_resource_returns_valid_json -v 2>&1
```
Expected: FAIL — `AssertionError: state://session not registered`

**Step 3: Register the resource in server.py**

Add after `mcp = FastMCP(...)` and before `register_area_tools(mcp)`:

```python
import json as _json
from .state import state as _state


@mcp.resource("state://session")
def session_state_resource() -> str:
    """Current session state: bounds, elevation, features, model params, mesh status."""
    return _json.dumps(_state.summary(), indent=2)
```

**Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_state_resource.py -v
```

If the FastMCP resource manager API differs from `_resource_manager._resources`, adjust the test to use whatever introspection the installed version supports. The key assertion is that calling the resource function returns valid JSON with the expected keys.

**Step 5: Full suite**

```bash
.venv/bin/pytest --tb=short -q 2>&1 | tail -5
```

**Step 6: Commit**

```bash
git add src/topo_shadow_box/server.py tests/test_state_resource.py
git commit -m "feat: register state://session MCP resource for passive state reads"
```

---

## Task 3: Richer tool docstrings

**Files:**
- Modify: `src/topo_shadow_box/tools/area.py`
- Modify: `src/topo_shadow_box/tools/data.py`
- Modify: `src/topo_shadow_box/tools/model.py`
- Modify: `src/topo_shadow_box/tools/generate.py`
- Modify: `src/topo_shadow_box/tools/preview.py`
- Modify: `src/topo_shadow_box/tools/export.py`

**No tests** — verified by inspection.

**Step 1: Update all tool docstrings**

Replace existing docstrings with the versions below. Keep all other code unchanged.

**`set_area_from_coordinates`** (`tools/area.py`):
```python
        """Define the area of interest by center+radius or explicit bounding box.

        Either provide (lat, lon, radius_m) for a circular area,
        or (north, south, east, west) for a rectangular bounding box.

        **Next:** Optionally call validate_area to check for problems,
        then fetch_elevation, then fetch_features (optional), then generate_model.

        Args:
            lat: Center latitude (degrees). Use with lon and radius_m.
            lon: Center longitude (degrees). Use with lat and radius_m.
            radius_m: Radius in meters around the center point.
            north/south/east/west: Explicit bounding box (degrees).
        """
```

**`set_area_from_gpx`** (`tools/area.py`):
```python
        """Load a GPX file and use its bounds (plus padding) as the area of interest.

        Also stores the GPX tracks for rendering as a raised strip on the terrain.
        **Next:** fetch_elevation, then fetch_features (optional), then generate_model.

        Args:
            file_path: Absolute path to a .gpx file.
            padding_m: Padding in meters around the GPX bounds (default 500m).
        """
```

**`fetch_elevation`** (`tools/data.py`):
```python
        """Fetch terrain elevation data for the current area of interest.

        Uses AWS Terrain-RGB tiles (free, globally available).
        **Requires:** set_area_from_coordinates or set_area_from_gpx first.
        **Next:** Optionally fetch_features, then generate_model.

        Args:
            resolution: Grid points per axis (default 200). Higher = more detail
                but slower generate_model. Use 100 for quick previews.
        """
```

**`fetch_features`** (`tools/data.py`):
```python
        """Fetch OpenStreetMap roads, water, and buildings for the current area.

        **Requires:** set_area_from_coordinates or set_area_from_gpx first.
        **Next:** generate_model (features are optional — skip if not needed).

        Args:
            include: Feature types to fetch. Options: 'roads', 'water', 'buildings'.
                     Default: all three. Omit types you don't want in the model.
        """
```

**`set_model_params`** (`tools/model.py`):
```python
        """Set model geometry parameters.

        Can be called any time before generate_model.
        **Next:** generate_model (re-run after changing params to update meshes).

        Args:
            width_mm: Model width in mm (default 200). The larger geographic
                dimension maps to this value.
            vertical_scale: Elevation exaggeration multiplier (default 1.5).
                Use 2-3 for flat terrain, 1 for mountains.
            base_height_mm: Thickness of the solid base (default 10mm).
            shape: Model outline shape — 'square', 'circle', 'hexagon', or 'rectangle'.
        """
```

**`set_colors`** (`tools/model.py`):
```python
        """Set material colors for each feature type (hex #RRGGBB).

        Can be called any time before export.
        **Next:** generate_model or export (colors are applied at export time).

        Args:
            terrain/water/roads/buildings/gpx_track/map_insert: Hex color strings.
        """
```

**`generate_model`** (`tools/generate.py`):
```python
        """Generate the full 3D model from current state.

        **Requires:** set_area_from_coordinates/gpx + fetch_elevation.
        Features and GPX tracks are optional but must be fetched before calling this.
        **Next:** preview (optional), then export_3mf / export_openscad / export_svg.

        Re-run this after changing model params or colors to update the meshes.
        Reports fine-grained progress as each feature is processed.
        """
```

**`generate_map_insert`** (`tools/generate.py`):
```python
        """Generate a background map insert (SVG for paper printing and/or 3D plate).

        **Requires:** set_area_from_coordinates or set_area_from_gpx first.
        **Next:** export_svg (for paper) or export_3mf (includes the plate).

        Args:
            format: 'svg' for paper map only, 'plate' for 3D-printable flat plate,
                    'both' for both (default).
        """
```

**`preview`** (`tools/preview.py`) — read current docstring and prepend:
```
        **Requires:** generate_model first.
        **Next:** export when satisfied with the preview.

```

**`export_3mf`** (`tools/export.py`):
```python
        """Export the model as a multi-material 3MF file for 3D printing.

        Each feature type gets its own material color (set via set_colors).
        **Requires:** generate_model first.

        Args:
            output_path: Absolute path for the .3mf file (must be within home directory).
        """
```

**`export_openscad`** (`tools/export.py`):
```python
        """Export the model as a parametric OpenSCAD .scad file.

        Includes editable parameters at the top. Open in OpenSCAD to render
        or customize dimensions.
        **Requires:** generate_model first.

        Args:
            output_path: Absolute path for the .scad file (must be within home directory).
        """
```

**`export_svg`** (`tools/export.py`):
```python
        """Export the map insert as an SVG file for paper printing.

        Shows streets, water, and GPX tracks styled for printing as a background
        insert behind the 3D terrain model.
        **Requires:** set_area_from_coordinates or set_area_from_gpx first
        (does not require generate_model).

        Args:
            output_path: Absolute path for the .svg file (must be within home directory).
        """
```

**Step 2: Full suite to confirm nothing broke**

```bash
.venv/bin/pytest --tb=short -q 2>&1 | tail -5
```
Expected: All PASS (docstrings only, no logic changed)

**Step 3: Commit**

```bash
git add src/topo_shadow_box/tools/
git commit -m "docs: add sequencing hints and richer descriptions to all tool docstrings"
```

---

## Task 4: Session serialization (save_session / load_session)

**Files:**
- Create: `src/topo_shadow_box/tools/session.py`
- Modify: `src/topo_shadow_box/server.py`
- Test: `tests/test_session_tools.py` (new)

**Step 1: Write the failing tests**

Create `tests/test_session_tools.py`:

```python
"""Tests for save_session and load_session tools."""
import json
import os
import pytest
import numpy as np
from unittest.mock import MagicMock
from pathlib import Path


def _get_session_tools():
    from topo_shadow_box.tools.session import register_session_tools
    tools = {}
    mock_mcp = MagicMock()
    def capture():
        def decorator(fn):
            tools[fn.__name__] = fn
            return fn
        return decorator
    mock_mcp.tool = capture
    register_session_tools(mock_mcp)
    return tools


def _configure_state():
    """Put known data into state for round-trip tests."""
    from topo_shadow_box.state import state, Bounds, ElevationData
    state.bounds = Bounds(north=37.80, south=37.75, east=-122.40, west=-122.45, is_set=True)
    state.model_params.width_mm = 150.0
    state.model_params.vertical_scale = 2.0
    state.colors.terrain = "#AABBCC"
    state.elevation = ElevationData(
        grid=np.zeros((10, 10)),
        lats=np.zeros(10),
        lons=np.zeros(10),
        resolution=10,
        min_elevation=0.0,
        max_elevation=100.0,
        is_set=True,
    )


def test_save_session_creates_file(tmp_path):
    tools = _get_session_tools()
    _configure_state()
    path = str(tmp_path / "session.json")
    result = tools["save_session"](path=path)
    assert os.path.exists(path), f"File not created. Result: {result}"
    assert "saved" in result.lower()


def test_save_session_file_is_valid_json(tmp_path):
    tools = _get_session_tools()
    _configure_state()
    path = str(tmp_path / "session.json")
    tools["save_session"](path=path)
    with open(path) as f:
        data = json.load(f)
    assert "bounds" in data
    assert "model_params" in data
    assert "colors" in data


def test_save_session_does_not_include_elevation_grid(tmp_path):
    tools = _get_session_tools()
    _configure_state()
    path = str(tmp_path / "session.json")
    tools["save_session"](path=path)
    with open(path) as f:
        data = json.load(f)
    # Elevation metadata (min/max) is fine, but not the full grid array
    if "elevation" in data:
        assert "grid" not in data["elevation"], "Should not serialize numpy grid"


def test_load_session_restores_bounds_and_params(tmp_path):
    tools = _get_session_tools()
    _configure_state()
    path = str(tmp_path / "session.json")
    tools["save_session"](path=path)

    # Reset state
    from topo_shadow_box.state import state, Bounds, ElevationData
    state.bounds = Bounds()
    state.elevation = ElevationData()
    state.model_params.width_mm = 200.0

    result = tools["load_session"](path=path)
    assert state.bounds.north == pytest.approx(37.80)
    assert state.bounds.south == pytest.approx(37.75)
    assert state.model_params.width_mm == pytest.approx(150.0)
    assert "restored" in result.lower() or "loaded" in result.lower()


def test_load_session_clears_elevation_and_meshes(tmp_path):
    tools = _get_session_tools()
    _configure_state()
    path = str(tmp_path / "session.json")
    tools["save_session"](path=path)

    from topo_shadow_box.state import state, MeshData
    # Pretend meshes exist
    state.terrain_mesh = MeshData(vertices=[[0,0,0]], faces=[[0,0,0]], name="T", feature_type="terrain")

    tools["load_session"](path=path)
    assert state.elevation.is_set is False, "Elevation should need re-fetching after load"
    assert state.terrain_mesh is None, "Meshes should be cleared after load"


def test_load_session_missing_file_returns_error(tmp_path):
    tools = _get_session_tools()
    result = tools["load_session"](path=str(tmp_path / "nonexistent.json"))
    assert "error" in result.lower() or "not found" in result.lower()


def test_save_session_default_path_creates_directory(tmp_path, monkeypatch):
    """Default path (~/.cache/topo-shadow-box/session.json) directory is created."""
    tools = _get_session_tools()
    _configure_state()
    fake_cache = tmp_path / ".cache" / "topo-shadow-box"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    result = tools["save_session"]()
    expected = tmp_path / ".cache" / "topo-shadow-box" / "session.json"
    assert expected.exists(), f"Default file not created. Result: {result}"
```

**Step 2: Run to verify failure**

```bash
.venv/bin/pytest tests/test_session_tools.py -v 2>&1 | head -20
```
Expected: FAIL — module not found

**Step 3: Create src/topo_shadow_box/tools/session.py**

```python
"""Session persistence tools: save_session, load_session."""

import json
import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from ..state import state, Bounds, ElevationData, ModelParams, Colors
from ._prereqs import require_state

logger = logging.getLogger(__name__)

_DEFAULT_SESSION_PATH = Path.home() / ".cache" / "topo-shadow-box" / "session.json"


def _default_path() -> Path:
    return Path.home() / ".cache" / "topo-shadow-box" / "session.json"


def register_session_tools(mcp: FastMCP):

    @mcp.tool()
    def save_session(path: str | None = None) -> str:
        """Save the current session to a JSON file for later resumption.

        Saves bounds, model params, colors, and GPX tracks.
        Does NOT save the elevation grid or meshes (regenerate after loading).
        **Next:** load_session in a future session to restore this configuration.

        Args:
            path: Where to save. Default: ~/.cache/topo-shadow-box/session.json
        """
        save_path = Path(path) if path else _default_path()
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data: dict = {
            "bounds": None,
            "model_params": state.model_params.model_dump(),
            "colors": state.colors.model_dump(),
            "elevation_metadata": None,
            "gpx_tracks": [],
        }

        if state.bounds.is_set:
            data["bounds"] = {
                "north": state.bounds.north,
                "south": state.bounds.south,
                "east": state.bounds.east,
                "west": state.bounds.west,
            }

        if state.elevation.is_set:
            data["elevation_metadata"] = {
                "resolution": state.elevation.resolution,
                "min_elevation": state.elevation.min_elevation,
                "max_elevation": state.elevation.max_elevation,
            }

        if state.gpx_tracks:
            data["gpx_tracks"] = [
                {
                    "name": t.name,
                    "points": [{"lat": p.lat, "lon": p.lon, "elevation": p.elevation}
                               for p in t.points],
                }
                for t in state.gpx_tracks
            ]

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Session saved to %s", save_path)
        return f"Session saved to {save_path}"

    @mcp.tool()
    def load_session(path: str | None = None) -> str:
        """Load a previously saved session from a JSON file.

        Restores bounds, model params, colors, and GPX tracks.
        Clears elevation and meshes — you will need to re-run fetch_elevation
        and generate_model after loading.
        **Next:** fetch_elevation, then optionally fetch_features, then generate_model.

        Args:
            path: Path to load from. Default: ~/.cache/topo-shadow-box/session.json
        """
        from ..models import GpxTrack, GpxPoint

        load_path = Path(path) if path else _default_path()

        if not load_path.exists():
            return f"Error: Session file not found at {load_path}"

        try:
            with open(load_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return f"Error: Invalid session file — {e}"

        # Restore bounds
        if data.get("bounds"):
            b = data["bounds"]
            state.bounds = Bounds(
                north=b["north"], south=b["south"],
                east=b["east"], west=b["west"],
                is_set=True,
            )

        # Restore model params
        if data.get("model_params"):
            state.model_params = ModelParams(**data["model_params"])

        # Restore colors
        if data.get("colors"):
            state.colors = Colors(**data["colors"])

        # Restore GPX tracks
        if data.get("gpx_tracks"):
            state.gpx_tracks = [
                GpxTrack(
                    name=t["name"],
                    points=[GpxPoint(**p) for p in t["points"]],
                )
                for t in data["gpx_tracks"]
            ]
        else:
            state.gpx_tracks = []

        # Clear things that need regeneration
        state.elevation = ElevationData()
        state.features_reset()  # use standard reset
        state.terrain_mesh = None
        state.feature_meshes = []
        state.gpx_mesh = None

        restored = []
        if state.bounds.is_set:
            restored.append("bounds")
        restored.append("model_params")
        restored.append("colors")
        if state.gpx_tracks:
            restored.append(f"{len(state.gpx_tracks)} GPX track(s)")

        return (
            f"Session restored from {load_path}. "
            f"Restored: {', '.join(restored)}. "
            "Still needed: fetch_elevation, then generate_model."
        )
```

**Note:** `state.features_reset()` doesn't exist yet — use this instead:
```python
        from ..core.models import OsmFeatureSet
        state.features = OsmFeatureSet()
```

**Step 4: Register session tools in server.py**

Add to `server.py`:
```python
from .tools.session import register_session_tools
```
And after the other `register_*_tools(mcp)` calls:
```python
register_session_tools(mcp)
```

**Step 5: Run tests**

```bash
.venv/bin/pytest tests/test_session_tools.py -v
```
Expected: All PASS

**Step 6: Full suite**

```bash
.venv/bin/pytest --tb=short -q 2>&1 | tail -5
```

**Step 7: Commit**

```bash
git add src/topo_shadow_box/tools/session.py src/topo_shadow_box/server.py tests/test_session_tools.py
git commit -m "feat: add save_session and load_session tools for session persistence"
```

---

## Task 5: Progress notifications in generate_model

**Files:**
- Modify: `src/topo_shadow_box/core/mesh.py`
- Modify: `src/topo_shadow_box/tools/generate.py`
- Test: `tests/test_generate_progress.py` (new)

**Step 1: Add generate_single_feature_mesh to mesh.py**

Read the bottom of `generate_feature_meshes` in `src/topo_shadow_box/core/mesh.py` to understand the internal helpers `_generate_road_mesh`, `_generate_water_mesh`, `_generate_building_mesh`.

Add this function after `generate_feature_meshes`:

```python
def generate_single_feature_mesh(
    feature,
    feature_type: str,
    elevation: ElevationData,
    bounds: Bounds,
    transform: GeoToModelTransform,
    vertical_scale: float = 1.5,
    shape: str = "square",
    _norm: tuple[float, float] | None = None,
) -> "MeshResult | None":
    """Generate a mesh for a single OSM feature.

    Used by generate_model for per-feature progress reporting.
    feature_type: 'road', 'water', or 'building'
    """
    min_elev, elev_range = _norm if _norm is not None else _elevation_normalization(elevation.grid)
    model_width = max(transform.model_width_x, transform.model_width_z)
    size_scale = model_width / 200.0
    clipper = _create_shape_clipper(shape, transform)

    if feature_type == "road":
        result = _generate_road_mesh(
            feature, elevation, transform, min_elev, elev_range,
            vertical_scale, size_scale, shape_clipper=clipper,
        )
        if result:
            return MeshResult(
                vertices=result["vertices"], faces=result["faces"],
                name=result.get("name", "Road"), feature_type="road",
            )
    elif feature_type == "water":
        result = _generate_water_mesh(
            feature, elevation, transform, min_elev, elev_range,
            vertical_scale, size_scale, shape_clipper=clipper,
        )
        if result:
            return MeshResult(
                vertices=result["vertices"], faces=result["faces"],
                name=result.get("name", "Water"), feature_type="water",
            )
    elif feature_type == "building":
        building_gen = BuildingShapeGenerator()
        result = _generate_building_mesh(
            feature, elevation, transform, min_elev, elev_range,
            vertical_scale, size_scale, shape_clipper=clipper,
            building_shape_gen=building_gen,
        )
        if result:
            return MeshResult(
                vertices=result["vertices"], faces=result["faces"],
                name=result.get("name", "Building"), feature_type="building",
            )
    return None
```

**Step 2: Write the failing progress tests**

Create `tests/test_generate_progress.py`:

```python
"""Tests for generate_model progress notifications."""
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch


def _make_state_with_features(n_roads=2, n_water=1, n_buildings=1, has_gpx=False):
    """Configure state with known feature counts."""
    from topo_shadow_box.state import state, Bounds, ElevationData, MeshData
    from topo_shadow_box.core.models import OsmFeatureSet
    from topo_shadow_box.models import RoadFeature, WaterFeature, BuildingFeature, Coordinate, GpxTrack, GpxPoint

    state.bounds = Bounds(north=37.80, south=37.75, east=-122.40, west=-122.45, is_set=True)
    grid = np.linspace(0, 200, 20 * 20).reshape(20, 20)
    state.elevation = ElevationData(
        grid=grid,
        lats=np.linspace(37.75, 37.80, 20),
        lons=np.linspace(-122.45, -122.40, 20),
        resolution=20,
        min_elevation=0.0,
        max_elevation=200.0,
        is_set=True,
    )

    coord_pair = [
        Coordinate(lat=37.76, lon=-122.44),
        Coordinate(lat=37.77, lon=-122.43),
    ]
    coord_triple = [
        Coordinate(lat=37.76, lon=-122.44),
        Coordinate(lat=37.77, lon=-122.43),
        Coordinate(lat=37.78, lon=-122.42),
    ]

    roads = [RoadFeature(id=i, coordinates=coord_pair, name=f"Road{i}", road_type="primary")
             for i in range(n_roads)]
    waters = [WaterFeature(id=i+100, coordinates=coord_triple, name=f"Water{i}")
              for i in range(n_water)]
    buildings = [BuildingFeature(id=i+200, coordinates=coord_triple, name=f"Bldg{i}", height=10.0)
                 for i in range(n_buildings)]
    state.features = OsmFeatureSet(roads=roads, water=waters, buildings=buildings)

    if has_gpx:
        state.gpx_tracks = [GpxTrack(name="Track", points=[
            GpxPoint(lat=37.76, lon=-122.44, elevation=100),
            GpxPoint(lat=37.77, lon=-122.43, elevation=110),
        ])]
    else:
        state.gpx_tracks = []

    return state


def _get_generate_model_fn():
    from topo_shadow_box.tools.generate import register_generate_tools
    tools = {}
    mock_mcp = MagicMock()
    def capture():
        def decorator(fn):
            tools[fn.__name__] = fn
            return fn
        return decorator
    mock_mcp.tool = capture
    register_generate_tools(mock_mcp)
    return tools["generate_model"]


@pytest.mark.anyio
async def test_generate_model_reports_progress():
    """generate_model should call ctx.report_progress for each work unit."""
    _make_state_with_features(n_roads=2, n_water=1, n_buildings=1, has_gpx=False)
    generate_model = _get_generate_model_fn()

    mock_ctx = AsyncMock()
    mock_ctx.report_progress = AsyncMock()

    result = await generate_model(ctx=mock_ctx)

    assert mock_ctx.report_progress.called, "report_progress should have been called"
    calls = mock_ctx.report_progress.call_args_list
    # 1 terrain + 2 roads + 1 water + 1 building = 5 total
    assert len(calls) >= 4, f"Expected at least 4 progress calls, got {len(calls)}"


@pytest.mark.anyio
async def test_generate_model_progress_is_nondecreasing():
    """Progress values should never decrease."""
    _make_state_with_features(n_roads=3, n_water=0, n_buildings=0, has_gpx=False)
    generate_model = _get_generate_model_fn()

    mock_ctx = AsyncMock()
    progress_values = []

    async def capture_progress(current, total):
        progress_values.append((current, total))

    mock_ctx.report_progress = capture_progress

    await generate_model(ctx=mock_ctx)

    assert len(progress_values) >= 2
    currents = [p[0] for p in progress_values]
    assert currents == sorted(currents), f"Progress not non-decreasing: {currents}"


@pytest.mark.anyio
async def test_generate_model_final_progress_equals_total():
    """Final progress call should have current == total."""
    _make_state_with_features(n_roads=1, n_water=1, n_buildings=0, has_gpx=False)
    generate_model = _get_generate_model_fn()

    mock_ctx = AsyncMock()
    progress_values = []

    async def capture_progress(current, total):
        progress_values.append((current, total))

    mock_ctx.report_progress = capture_progress

    await generate_model(ctx=mock_ctx)

    last_current, last_total = progress_values[-1]
    assert last_current == last_total, (
        f"Final progress {last_current} should equal total {last_total}"
    )


@pytest.mark.anyio
async def test_generate_model_with_gpx_includes_gpx_in_total():
    """GPX track counts as one work unit."""
    _make_state_with_features(n_roads=0, n_water=0, n_buildings=0, has_gpx=True)
    generate_model = _get_generate_model_fn()

    mock_ctx = AsyncMock()
    progress_values = []

    async def capture_progress(current, total):
        progress_values.append((current, total))

    mock_ctx.report_progress = capture_progress

    await generate_model(ctx=mock_ctx)

    # 1 terrain + 1 gpx = 2 total
    assert len(progress_values) >= 2
    totals = [p[1] for p in progress_values]
    assert all(t == totals[0] for t in totals), "Total should be consistent across calls"
    assert totals[0] == 2, f"Expected total=2 (terrain+gpx), got {totals[0]}"
```

**Step 3: Run to verify failure**

```bash
.venv/bin/pytest tests/test_generate_progress.py -v 2>&1 | head -20
```
Expected: FAIL — `generate_model` is not async / doesn't accept `ctx`

**Step 4: Refactor generate_model to async with progress**

Replace `generate_model` in `src/topo_shadow_box/tools/generate.py` with:

```python
    @mcp.tool()
    async def generate_model(ctx: Context) -> str:
        """Generate the full 3D model from current state.

        **Requires:** set_area_from_coordinates/gpx + fetch_elevation.
        Features and GPX tracks are optional but must be fetched before calling this.
        **Next:** preview (optional), then export_3mf / export_openscad / export_svg.

        Re-run this after changing model params or colors to update the meshes.
        Reports fine-grained progress as each feature is processed.
        """
        try:
            require_state(state, bounds=True, elevation=True)
        except ValueError as e:
            return f"Error: {e}"

        b = state.bounds
        mp = state.model_params

        transform = GeoToModelTransform(bounds=b, model_width_mm=mp.width_mm)
        norm = _elevation_normalization(state.elevation.grid)

        # Count total work units upfront
        features = state.features
        roads = features.roads[:200] if features else []
        waters = features.water[:50] if features else []
        buildings = features.buildings[:150] if features else []
        has_gpx = bool(state.gpx_tracks)
        total = 1 + len(roads) + len(waters) + len(buildings) + (1 if has_gpx else 0)
        current = 0

        # Generate terrain
        terrain = generate_terrain_mesh(
            elevation=state.elevation, bounds=b, transform=transform,
            vertical_scale=mp.vertical_scale, base_height_mm=mp.base_height_mm,
            shape=mp.shape, _norm=norm,
        )
        state.terrain_mesh = MeshData(
            vertices=terrain.vertices, faces=terrain.faces,
            name=terrain.name, feature_type=terrain.feature_type,
        )
        current += 1
        await ctx.report_progress(current, total)

        # Generate feature meshes one-by-one for progress
        state.feature_meshes = []
        for road in roads:
            fm = generate_single_feature_mesh(
                road, "road", state.elevation, b, transform,
                mp.vertical_scale, mp.shape, norm,
            )
            if fm:
                state.feature_meshes.append(MeshData(
                    vertices=fm.vertices, faces=fm.faces,
                    name=fm.name, feature_type=fm.feature_type,
                ))
            current += 1
            await ctx.report_progress(current, total)

        for water in waters:
            fm = generate_single_feature_mesh(
                water, "water", state.elevation, b, transform,
                mp.vertical_scale, mp.shape, norm,
            )
            if fm:
                state.feature_meshes.append(MeshData(
                    vertices=fm.vertices, faces=fm.faces,
                    name=fm.name, feature_type=fm.feature_type,
                ))
            current += 1
            await ctx.report_progress(current, total)

        for building in buildings:
            fm = generate_single_feature_mesh(
                building, "building", state.elevation, b, transform,
                mp.vertical_scale, mp.shape, norm,
            )
            if fm:
                state.feature_meshes.append(MeshData(
                    vertices=fm.vertices, faces=fm.faces,
                    name=fm.name, feature_type=fm.feature_type,
                ))
            current += 1
            await ctx.report_progress(current, total)

        # Generate GPX track mesh
        state.gpx_mesh = None
        if has_gpx:
            gpx = generate_gpx_track_mesh(
                tracks=state.gpx_tracks, elevation=state.elevation,
                bounds=b, transform=transform,
                vertical_scale=mp.vertical_scale, shape=mp.shape, _norm=norm,
            )
            if gpx:
                state.gpx_mesh = MeshData(
                    vertices=gpx.vertices, faces=gpx.faces,
                    name=gpx.name, feature_type=gpx.feature_type,
                )
            current += 1
            await ctx.report_progress(current, total)

        terrain_verts = len(state.terrain_mesh.vertices)
        terrain_faces = len(state.terrain_mesh.faces)
        feature_count = len(state.feature_meshes)
        total_verts = terrain_verts + sum(len(m.vertices) for m in state.feature_meshes)
        if state.gpx_mesh:
            total_verts += len(state.gpx_mesh.vertices)

        return (
            f"Model generated: {terrain_verts} terrain vertices, {terrain_faces} faces, "
            f"{feature_count} feature meshes, "
            f"GPX: {'yes' if state.gpx_mesh else 'no'}. "
            f"Total vertices: {total_verts}"
        )
```

Add to imports at top of `generate.py`:
```python
from mcp.server.fastmcp import Context
from ..core.mesh import (
    generate_terrain_mesh, generate_feature_meshes,
    generate_gpx_track_mesh, generate_single_feature_mesh,
    _elevation_normalization,
)
```

**Step 5: Run progress tests**

```bash
.venv/bin/pytest tests/test_generate_progress.py -v
```
Expected: All PASS

**Step 6: Full suite**

```bash
.venv/bin/pytest --tb=short -q 2>&1 | tail -5
```
Expected: All PASS

**Step 7: Commit**

```bash
git add src/topo_shadow_box/core/mesh.py src/topo_shadow_box/tools/generate.py tests/test_generate_progress.py
git commit -m "feat: add fine-grained progress notifications to generate_model via ctx.report_progress"
```

---

## Final Verification

```bash
.venv/bin/pytest -v --tb=short 2>&1 | tail -20
```
Expected: All tests pass, no regressions.
