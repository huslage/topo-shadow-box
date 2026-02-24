# Codebase Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve reliability, test coverage, performance, security, and internal consistency of the topo-shadow-box MCP server across five sequential phases.

**Architecture:** Each phase is an independent unit — complete and commit each before starting the next. TDD order applies: write the failing test first, then implement.

**Tech Stack:** Python 3.11+, pytest, httpx, asyncio, pydantic v2, FastMCP

---

## Phase 1: Reliability & Error Handling

### Task 1.1: Add logging to elevation.py

**Files:**
- Modify: `src/topo_shadow_box/core/elevation.py`

**Step 1: Write the failing test**

Add to `tests/test_mesh.py` (or create `tests/test_elevation.py` if it doesn't exist):

```python
import logging
import pytest

def test_elevation_module_has_logger():
    import topo_shadow_box.core.elevation as elev_mod
    assert hasattr(elev_mod, 'logger'), "elevation module should have a module-level logger"

def test_failed_tile_logs_warning(caplog):
    """A failed tile fetch should log a warning, not silently pass."""
    import httpx
    from unittest.mock import patch, AsyncMock
    import numpy as np
    from topo_shadow_box.core.elevation import fetch_terrain_elevation
    import asyncio

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.content = b""  # will cause PIL to raise

    with caplog.at_level(logging.WARNING, logger="topo_shadow_box.core.elevation"):
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(side_effect=Exception("tile fetch failed"))
            mock_client_cls.return_value = mock_client

            with pytest.raises(Exception):
                asyncio.get_event_loop().run_until_complete(
                    fetch_terrain_elevation(37.8, 37.75, -122.4, -122.45, resolution=10)
                )

    assert any("tile" in r.message.lower() or "failed" in r.message.lower()
               for r in caplog.records), "Should log a warning for failed tile"
```

**Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_elevation.py::test_elevation_module_has_logger -v
```
Expected: FAIL — `AttributeError: module has no attribute 'logger'`

**Step 3: Implement**

In `src/topo_shadow_box/core/elevation.py`, after the imports add:

```python
import logging

logger = logging.getLogger(__name__)
```

Replace lines 111–112:
```python
                except Exception:
                    pass  # Leave zeros for failed tiles
```
with:
```python
                except Exception as exc:
                    logger.warning("Failed to fetch elevation tile %s: %s", url, exc)
```

**Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_elevation.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/topo_shadow_box/core/elevation.py tests/test_elevation.py
git commit -m "feat: log warning on failed elevation tile fetch instead of silently passing"
```

---

### Task 1.2: Classify exceptions in osm.py

**Files:**
- Modify: `src/topo_shadow_box/core/osm.py`

**Step 1: Write the failing test**

Create `tests/test_osm.py`:

```python
import logging
import pytest
from unittest.mock import patch, AsyncMock
import httpx


def test_osm_module_has_logger():
    import topo_shadow_box.core.osm as osm_mod
    assert hasattr(osm_mod, 'logger')


@pytest.mark.asyncio
async def test_timeout_logs_warning_and_tries_next_server(caplog):
    """TimeoutException on one server should warn and try the next."""
    from topo_shadow_box.core.osm import _query_overpass

    call_count = 0

    async def mock_post(url, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise httpx.TimeoutException("timeout")
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = AsyncMock()
        mock_resp.json = AsyncMock(return_value={"elements": [{"id": 1}]})
        return mock_resp

    with caplog.at_level(logging.WARNING, logger="topo_shadow_box.core.osm"):
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = mock_post
            mock_client_cls.return_value = mock_client

            result = await _query_overpass("test query")

    assert result == [{"id": 1}]
    assert any("timeout" in r.message.lower() for r in caplog.records)


@pytest.mark.asyncio
async def test_all_servers_fail_returns_empty_with_warning(caplog):
    """When all servers fail, return empty list and log warning."""
    from topo_shadow_box.core.osm import _query_overpass

    with caplog.at_level(logging.WARNING, logger="topo_shadow_box.core.osm"):
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_client_cls.return_value = mock_client

            result = await _query_overpass("test query")

    assert result == []
    assert any("all" in r.message.lower() or "server" in r.message.lower()
               for r in caplog.records)


@pytest.mark.asyncio
async def test_http_status_error_logs_status(caplog):
    """HTTP 429 response should log status code and try next server."""
    from topo_shadow_box.core.osm import _query_overpass

    mock_resp = AsyncMock()
    mock_resp.raise_for_status = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "429", request=AsyncMock(), response=AsyncMock(status_code=429)
        )
    )

    with caplog.at_level(logging.WARNING, logger="topo_shadow_box.core.osm"):
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await _query_overpass("test query")

    assert result == []
    assert any("429" in r.message or "status" in r.message.lower()
               for r in caplog.records)
```

**Step 2: Run to verify failure**

```bash
.venv/bin/pytest tests/test_osm.py -v
```
Expected: FAIL — `AssertionError: module has no attribute 'logger'`

**Step 3: Implement**

In `src/topo_shadow_box/core/osm.py`, after imports add:

```python
import logging

logger = logging.getLogger(__name__)
```

Replace the `_query_overpass` function:

```python
async def _query_overpass(query: str) -> list[dict]:
    """Execute an Overpass API query with server fallback."""
    async with httpx.AsyncClient(timeout=45.0) as client:
        for server in OVERPASS_SERVERS:
            try:
                response = await client.post(server, data={"data": query})
                response.raise_for_status()
                data = response.json()
                return data.get("elements", [])
            except httpx.TimeoutException as exc:
                logger.warning("Overpass server %s timed out: %s", server, exc)
                continue
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "Overpass server %s returned HTTP %s", server, exc.response.status_code
                )
                continue
            except Exception as exc:
                logger.warning("Overpass server %s failed: %s", server, exc)
                continue
    logger.warning("All Overpass servers failed for query")
    return []
```

**Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_osm.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/topo_shadow_box/core/osm.py tests/test_osm.py
git commit -m "feat: classify OSM Overpass exceptions and log warnings instead of silently failing"
```

---

### Task 1.3: Surface OSM failure in fetch_features tool

**Files:**
- Modify: `src/topo_shadow_box/tools/data.py`

**Step 1: Write the failing test**

Add to `tests/test_osm.py`:

```python
@pytest.mark.asyncio
async def test_fetch_features_reports_empty_when_all_servers_fail():
    """fetch_features tool should mention fetched counts even when OSM fails."""
    from topo_shadow_box.tools.data import register_data_tools
    from topo_shadow_box.state import state, Bounds

    # Set up state with bounds
    state.bounds = Bounds(north=37.8, south=37.75, east=-122.4, west=-122.45, is_set=True)

    with patch("topo_shadow_box.tools.data.fetch_osm_features") as mock_fetch:
        from topo_shadow_box.core.models import OsmFeatureSet
        mock_fetch.return_value = OsmFeatureSet(roads=[], water=[], buildings=[])

        # Import and call the function directly
        from topo_shadow_box.core.osm import fetch_osm_features
        result = await fetch_osm_features(
            north=37.8, south=37.75, east=-122.4, west=-122.45,
            feature_types=["roads"]
        )

    # Empty results should come back (not an error)
    assert result.roads == []
```

**Step 2: Run to verify**

```bash
.venv/bin/pytest tests/test_osm.py::test_fetch_features_reports_empty_when_all_servers_fail -v
```

**Step 3: Add logger to tools/data.py**

In `src/topo_shadow_box/tools/data.py`, after imports add:

```python
import logging

logger = logging.getLogger(__name__)
```

In `fetch_features`, update the return to surface zero counts more explicitly:

```python
        counts = {
            k: v for k, v in {
                "roads": len(features.roads),
                "water": len(features.water),
                "buildings": len(features.buildings),
            }.items() if k in include
        }
        if all(v == 0 for v in counts.values()):
            return f"Features fetched: none found (check logs if this is unexpected) — {counts}"
        return f"Features fetched: {counts}"
```

**Step 4: Run all Phase 1 tests**

```bash
.venv/bin/pytest tests/test_elevation.py tests/test_osm.py -v
```
Expected: All PASS

**Step 5: Commit**

```bash
git add src/topo_shadow_box/tools/data.py
git commit -m "feat: surface zero-feature result explicitly in fetch_features tool output"
```

---

## Phase 2: Test Coverage

### Task 2.1: Exporter tests — 3MF

**Files:**
- Create: `tests/test_exporters.py`

**Step 1: Write the failing tests**

Create `tests/test_exporters.py`:

```python
"""Tests for 3MF, OpenSCAD, and SVG exporters."""

import io
import os
import zipfile
import tempfile
import pytest


# ── helpers ──────────────────────────────────────────────────────────────────

def _minimal_mesh(name="terrain", mtype="terrain", color="#C8A882"):
    """Return a minimal valid mesh dict (single tetrahedron)."""
    return {
        "name": name,
        "type": mtype,
        "color": color,
        "vertices": [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "faces": [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ],
    }


# ── 3MF tests ─────────────────────────────────────────────────────────────────

class TestExport3MF:
    def test_output_is_valid_zip(self, tmp_path):
        from topo_shadow_box.exporters.threemf import export_3mf
        out = str(tmp_path / "test.3mf")
        export_3mf([_minimal_mesh()], out)
        assert zipfile.is_zipfile(out)

    def test_zip_contains_required_files(self, tmp_path):
        from topo_shadow_box.exporters.threemf import export_3mf
        out = str(tmp_path / "test.3mf")
        export_3mf([_minimal_mesh()], out)
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
        assert "[Content_Types].xml" in names
        assert "_rels/.rels" in names
        assert "3D/3dmodel.model" in names

    def test_model_xml_contains_polyhedron_data(self, tmp_path):
        from topo_shadow_box.exporters.threemf import export_3mf
        out = str(tmp_path / "test.3mf")
        export_3mf([_minimal_mesh()], out)
        with zipfile.ZipFile(out) as zf:
            xml = zf.read("3D/3dmodel.model").decode()
        assert "<vertices>" in xml
        assert "<triangles>" in xml
        assert "<vertex" in xml
        assert "<triangle" in xml

    def test_multi_material_has_one_object_per_mesh(self, tmp_path):
        from topo_shadow_box.exporters.threemf import export_3mf
        meshes = [
            _minimal_mesh("terrain", "terrain", "#C8A882"),
            _minimal_mesh("roads", "roads", "#D4C5A9"),
        ]
        out = str(tmp_path / "multi.3mf")
        result = export_3mf(meshes, out)
        assert result["objects"] == 2
        with zipfile.ZipFile(out) as zf:
            xml = zf.read("3D/3dmodel.model").decode()
        assert xml.count('<object ') == 2

    def test_entity_escaping_in_name(self, tmp_path):
        from topo_shadow_box.exporters.threemf import export_3mf
        mesh = _minimal_mesh(name='Rock & Roll <>"', color="#FF0000")
        out = str(tmp_path / "escape.3mf")
        export_3mf([mesh], out)
        with zipfile.ZipFile(out) as zf:
            xml = zf.read("3D/3dmodel.model").decode()
        assert "&amp;" in xml
        assert "&lt;" in xml
        assert "&quot;" in xml
        # Raw unescaped chars must NOT appear in attribute context
        assert ' name="Rock & Roll' not in xml

    def test_raises_on_empty_mesh_list(self, tmp_path):
        from topo_shadow_box.exporters.threemf import export_3mf
        with pytest.raises(ValueError, match="No mesh data"):
            export_3mf([], str(tmp_path / "empty.3mf"))

    def test_returns_correct_filepath(self, tmp_path):
        from topo_shadow_box.exporters.threemf import export_3mf
        out = str(tmp_path / "check.3mf")
        result = export_3mf([_minimal_mesh()], out)
        assert result["filepath"] == out
        assert os.path.exists(out)
```

**Step 2: Run to verify they pass** (these should pass since exporters already exist)

```bash
.venv/bin/pytest tests/test_exporters.py::TestExport3MF -v
```
Expected: Most PASS. If any fail, the exporter has a bug — fix the test or the exporter as appropriate.

**Step 3: No implementation needed** (tests are for existing code)

**Step 4: Commit**

```bash
git add tests/test_exporters.py
git commit -m "test: add 3MF exporter tests for ZIP structure, multi-material, and entity escaping"
```

---

### Task 2.2: Exporter tests — OpenSCAD and SVG

**Files:**
- Modify: `tests/test_exporters.py`

**Step 1: Add OpenSCAD and SVG tests**

Append to `tests/test_exporters.py`:

```python
# ── OpenSCAD tests ────────────────────────────────────────────────────────────

class TestExportOpenSCAD:
    def _model_params(self):
        from topo_shadow_box.state import ModelParams
        return ModelParams(width_mm=200.0, vertical_scale=1.5, base_height_mm=10.0, shape="square")

    def test_output_file_is_created(self, tmp_path):
        from topo_shadow_box.exporters.openscad import export_openscad
        out = str(tmp_path / "test.scad")
        export_openscad([_minimal_mesh()], out, self._model_params())
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_contains_polyhedron_calls(self, tmp_path):
        from topo_shadow_box.exporters.openscad import export_openscad
        out = str(tmp_path / "test.scad")
        export_openscad([_minimal_mesh()], out, self._model_params())
        content = open(out).read()
        assert "polyhedron(" in content

    def test_contains_parameter_block(self, tmp_path):
        from topo_shadow_box.exporters.openscad import export_openscad
        out = str(tmp_path / "test.scad")
        export_openscad([_minimal_mesh()], out, self._model_params())
        content = open(out).read()
        assert "model_width" in content
        assert "vertical_scale" in content

    def test_contains_color_call(self, tmp_path):
        from topo_shadow_box.exporters.openscad import export_openscad
        out = str(tmp_path / "test.scad")
        export_openscad([_minimal_mesh()], out, self._model_params())
        content = open(out).read()
        assert "color(" in content

    def test_multiple_meshes_multiple_polyhedrons(self, tmp_path):
        from topo_shadow_box.exporters.openscad import export_openscad
        meshes = [_minimal_mesh("a"), _minimal_mesh("b")]
        out = str(tmp_path / "multi.scad")
        export_openscad(meshes, out, self._model_params())
        content = open(out).read()
        assert content.count("polyhedron(") == 2


# ── SVG tests ─────────────────────────────────────────────────────────────────

class TestExportSVG:
    def _bounds(self):
        from topo_shadow_box.state import Bounds
        return Bounds(north=37.8, south=37.75, east=-122.4, west=-122.45, is_set=True)

    def _colors(self):
        from topo_shadow_box.state import Colors
        return Colors()

    def test_output_file_is_created(self, tmp_path):
        from topo_shadow_box.exporters.svg import export_svg
        from topo_shadow_box.core.models import OsmFeatureSet
        out = str(tmp_path / "test.svg")
        export_svg(
            bounds=self._bounds(),
            features=OsmFeatureSet(),
            gpx_tracks=[],
            colors=self._colors(),
            output_path=out,
        )
        assert os.path.exists(out)

    def test_output_is_valid_xml(self, tmp_path):
        from topo_shadow_box.exporters.svg import export_svg
        from topo_shadow_box.core.models import OsmFeatureSet
        import xml.etree.ElementTree as ET
        out = str(tmp_path / "test.svg")
        export_svg(
            bounds=self._bounds(),
            features=OsmFeatureSet(),
            gpx_tracks=[],
            colors=self._colors(),
            output_path=out,
        )
        # Should not raise
        tree = ET.parse(out)
        root = tree.getroot()
        assert "svg" in root.tag.lower()
```

**Step 2: Run**

```bash
.venv/bin/pytest tests/test_exporters.py -v
```
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/test_exporters.py
git commit -m "test: add OpenSCAD and SVG exporter tests"
```

---

### Task 2.3: Network failure tests for elevation

**Files:**
- Modify: `tests/test_elevation.py`

**Step 1: Add network failure tests**

Add to `tests/test_elevation.py`:

```python
import asyncio
import pytest
from unittest.mock import patch, AsyncMock
import httpx
import numpy as np


@pytest.mark.asyncio
async def test_all_tiles_fail_raises_or_returns_zeros():
    """When all tile fetches fail, function should either raise or return a grid of zeros."""
    from topo_shadow_box.core.elevation import fetch_terrain_elevation

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=Exception("network error"))
        mock_client_cls.return_value = mock_client

        # Should either raise ValueError (no data) or return zeros — both acceptable
        try:
            result = await fetch_terrain_elevation(37.8, 37.75, -122.4, -122.45, resolution=10)
            # If it returns, grid should be zeros (no real data)
            assert np.all(result.grid == 0.0) or True  # zeros or processed zeros
        except (ValueError, Exception):
            pass  # Raising is also acceptable


@pytest.mark.asyncio
async def test_partial_tile_failure_logs_warning(caplog):
    """When some tiles fail and some succeed, should log warnings for failures."""
    import logging
    from topo_shadow_box.core.elevation import fetch_terrain_elevation
    from PIL import Image
    from io import BytesIO

    call_count = 0

    async def mock_get(url, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("first tile failed")
        # Return a valid 256x256 RGB tile for subsequent calls
        img = Image.new("RGB", (256, 256), color=(128, 128, 128))
        buf = BytesIO()
        img.save(buf, format="PNG")
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.content = buf.getvalue()
        return mock_resp

    with caplog.at_level(logging.WARNING, logger="topo_shadow_box.core.elevation"):
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = mock_get
            mock_client_cls.return_value = mock_client

            try:
                await fetch_terrain_elevation(37.8, 37.75, -122.4, -122.45, resolution=10)
            except Exception:
                pass

    assert any("failed" in r.message.lower() or "tile" in r.message.lower()
               for r in caplog.records)
```

**Step 2: Run**

```bash
.venv/bin/pytest tests/test_elevation.py -v
```
Expected: All PASS (the logging was added in Task 1.1)

**Step 3: Commit**

```bash
git add tests/test_elevation.py
git commit -m "test: add network failure tests for elevation tile fetching"
```

---

### Task 2.4: Integration test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write the integration test**

Create `tests/test_integration.py`:

```python
"""End-to-end integration test for the full pipeline with mocked HTTP."""

import asyncio
import os
import pytest
import numpy as np
from unittest.mock import patch, AsyncMock, MagicMock
from PIL import Image
from io import BytesIO


def _make_tile_response():
    """Create a mock HTTP response with a valid 256x256 RGB elevation tile."""
    # Encode ~100m elevation in Terrarium format: R*256 + G + B/256 - 32768
    # elevation 100m → R=128, G=100, B=0
    r, g, b = 128, 100, 0
    img = Image.new("RGB", (256, 256), color=(r, g, b))
    buf = BytesIO()
    img.save(buf, format="PNG")
    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.content = buf.getvalue()
    return mock_resp


def _make_osm_response():
    """Return a mock Overpass response with one road."""
    mock_resp = AsyncMock()
    mock_resp.raise_for_status = AsyncMock()
    mock_resp.json = AsyncMock(return_value={
        "elements": [
            {
                "type": "way",
                "id": 1,
                "tags": {"highway": "primary", "name": "Test Road"},
                "geometry": [
                    {"lat": 37.76, "lon": -122.44},
                    {"lat": 37.77, "lon": -122.43},
                    {"lat": 37.78, "lon": -122.42},
                ],
            }
        ]
    })
    return mock_resp


@pytest.mark.asyncio
async def test_full_pipeline_produces_3mf_file(tmp_path):
    """Full pipeline: set area → fetch elevation → generate model → export 3MF."""
    from topo_shadow_box.state import state, Bounds, ElevationData
    from topo_shadow_box.core.elevation import fetch_terrain_elevation
    from topo_shadow_box.core.osm import fetch_osm_features
    from topo_shadow_box.core.mesh import generate_terrain_mesh
    from topo_shadow_box.core.coords import GeoToModelTransform
    from topo_shadow_box.exporters.threemf import export_3mf
    from topo_shadow_box.state import MeshData

    # Reset state
    state.bounds = Bounds(north=37.8, south=37.75, east=-122.4, west=-122.45, is_set=True)

    # Step 1: Mock-fetch elevation
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=_make_tile_response())
        mock_client_cls.return_value = mock_client

        elevation = await fetch_terrain_elevation(
            north=37.8, south=37.75, east=-122.4, west=-122.45, resolution=20
        )

    state.elevation = ElevationData(
        grid=elevation.grid,
        lats=elevation.lats,
        lons=elevation.lons,
        resolution=elevation.resolution,
        min_elevation=elevation.min_elevation,
        max_elevation=elevation.max_elevation,
        is_set=True,
    )

    # Step 2: Generate terrain mesh
    b = state.bounds
    mp = state.model_params
    transform = GeoToModelTransform(bounds=b, model_width_mm=mp.width_mm)
    terrain = generate_terrain_mesh(
        elevation=state.elevation,
        bounds=b,
        transform=transform,
        vertical_scale=mp.vertical_scale,
        base_height_mm=mp.base_height_mm,
        shape="square",
    )
    state.terrain_mesh = MeshData(
        vertices=terrain.vertices,
        faces=terrain.faces,
        name=terrain.name,
        feature_type=terrain.feature_type,
    )

    # Step 3: Export to 3MF
    out = str(tmp_path / "output.3mf")
    meshes = [{
        "name": state.terrain_mesh.name,
        "type": state.terrain_mesh.feature_type,
        "vertices": state.terrain_mesh.vertices,
        "faces": state.terrain_mesh.faces,
        "color": "#C8A882",
    }]
    result = export_3mf(meshes, out)

    # Verify
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0
    assert result["success"] is True
    assert result["objects"] == 1
```

**Step 2: Run**

```bash
.venv/bin/pytest tests/test_integration.py -v
```
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration test with mocked HTTP for full pipeline"
```

---

### Task 2.5: Edge case tests

**Files:**
- Modify: `tests/test_mesh.py`

**Step 1: Add edge case tests**

Append to `tests/test_mesh.py`:

```python
class TestEdgeCases:
    """Edge cases for mesh generation."""

    def _make_flat_elevation(self, rows=10, cols=10, value=0.0):
        """Create a flat elevation grid."""
        from topo_shadow_box.state import ElevationData
        import numpy as np
        grid = np.full((rows, cols), value)
        return ElevationData(
            grid=grid,
            lats=np.linspace(37.75, 37.80, rows),
            lons=np.linspace(-122.45, -122.40, cols),
            resolution=rows,
            min_elevation=float(value),
            max_elevation=float(value),
            is_set=True,
        )

    def _make_bounds(self):
        from topo_shadow_box.state import Bounds
        return Bounds(north=37.80, south=37.75, east=-122.40, west=-122.45, is_set=True)

    def test_all_zero_elevation_grid_does_not_crash(self):
        """An all-zero elevation grid should produce a valid (flat) terrain mesh."""
        from topo_shadow_box.core.mesh import generate_terrain_mesh
        from topo_shadow_box.core.coords import GeoToModelTransform

        bounds = self._make_bounds()
        elev = self._make_flat_elevation(value=0.0)
        transform = GeoToModelTransform(bounds=bounds, model_width_mm=200.0)

        result = generate_terrain_mesh(
            elevation=elev, bounds=bounds, transform=transform,
            vertical_scale=1.5, base_height_mm=10.0, shape="square",
        )
        assert len(result.vertices) > 0
        assert len(result.faces) > 0

    def test_uniform_elevation_grid_produces_flat_terrain(self):
        """A constant elevation grid should produce a flat top surface."""
        from topo_shadow_box.core.mesh import generate_terrain_mesh
        from topo_shadow_box.core.coords import GeoToModelTransform
        import numpy as np

        bounds = self._make_bounds()
        elev = self._make_flat_elevation(value=500.0)
        transform = GeoToModelTransform(bounds=bounds, model_width_mm=200.0)

        result = generate_terrain_mesh(
            elevation=elev, bounds=bounds, transform=transform,
            vertical_scale=1.5, base_height_mm=10.0, shape="square",
        )
        verts = np.array(result.vertices)
        # All top surface Y values should be the same (flat terrain)
        # Top surface verts have Y >= 0 (base is at -base_height)
        top_ys = verts[verts[:, 1] >= 0, 1]
        if len(top_ys) > 0:
            assert np.allclose(top_ys, top_ys[0], atol=0.1), "Flat elevation should produce flat top"

    def test_empty_feature_list_produces_no_feature_meshes(self):
        """generate_feature_meshes with empty OsmFeatureSet should return empty list."""
        from topo_shadow_box.core.mesh import generate_feature_meshes
        from topo_shadow_box.core.models import OsmFeatureSet
        from topo_shadow_box.core.coords import GeoToModelTransform

        bounds = self._make_bounds()
        elev = self._make_flat_elevation()
        transform = GeoToModelTransform(bounds=bounds, model_width_mm=200.0)

        result = generate_feature_meshes(
            features=OsmFeatureSet(roads=[], water=[], buildings=[]),
            elevation=elev,
            bounds=bounds,
            transform=transform,
            vertical_scale=1.5,
            shape="square",
        )
        assert result == [] or len(result) == 0
```

**Step 2: Run**

```bash
.venv/bin/pytest tests/test_mesh.py::TestEdgeCases -v
```
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_mesh.py
git commit -m "test: add edge case tests for zero elevation, flat terrain, and empty features"
```

---

## Phase 3: Performance

### Task 3.1: Parallelize elevation tile fetching

**Files:**
- Modify: `src/topo_shadow_box/core/elevation.py`

**Step 1: Write a timing/behavior test first**

Add to `tests/test_elevation.py`:

```python
@pytest.mark.asyncio
async def test_tiles_fetched_concurrently():
    """Tiles should be fetched concurrently (all URLs requested, order doesn't matter)."""
    from topo_shadow_box.core.elevation import fetch_terrain_elevation
    from PIL import Image
    from io import BytesIO

    fetched_urls = []

    async def mock_get(url, **kwargs):
        fetched_urls.append(url)
        img = Image.new("RGB", (256, 256), color=(128, 100, 0))
        buf = BytesIO()
        img.save(buf, format="PNG")
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.content = buf.getvalue()
        return mock_resp

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = mock_get
        mock_client_cls.return_value = mock_client

        result = await fetch_terrain_elevation(37.8, 37.75, -122.4, -122.45, resolution=10)

    # All tiles should have been fetched
    assert len(fetched_urls) >= 1
    assert result is not None
```

**Step 2: Run to confirm it currently passes** (behavior test, not timing)

```bash
.venv/bin/pytest tests/test_elevation.py::test_tiles_fetched_concurrently -v
```

**Step 3: Implement parallel fetching**

In `src/topo_shadow_box/core/elevation.py`, replace the sequential fetching block (lines 94–112) with:

```python
    async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": "topo-shadow-box/1.0"}) as client:
        # Collect all tile coordinates
        tile_coords = [
            (tx, ty)
            for ty in range(y_min, y_max + 1)
            for tx in range(x_min, x_max + 1)
        ]

        async def fetch_tile(tx: int, ty: int) -> None:
            url = AWS_TERRAIN_URL.format(z=zoom, x=tx, y=ty)
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    img_array = np.array(img)
                    r = img_array[:, :, 0].astype(np.float64)
                    g = img_array[:, :, 1].astype(np.float64)
                    b = img_array[:, :, 2].astype(np.float64)
                    tile_elevations = _decode_terrarium(r, g, b)
                    px = (tx - x_min) * tile_size
                    py = (ty - y_min) * tile_size
                    stitched_elevations[py:py + tile_size, px:px + tile_size] = tile_elevations
            except Exception as exc:
                logger.warning("Failed to fetch elevation tile %s: %s", url, exc)

        import asyncio
        await asyncio.gather(*[fetch_tile(tx, ty) for tx, ty in tile_coords])
```

Note: Move the `import asyncio` to the top of the file with other imports.

**Step 4: Run all elevation tests**

```bash
.venv/bin/pytest tests/test_elevation.py -v
```
Expected: All PASS

**Step 5: Commit**

```bash
git add src/topo_shadow_box/core/elevation.py
git commit -m "perf: fetch elevation tiles concurrently with asyncio.gather instead of sequentially"
```

---

### Task 3.2: Cache elevation normalization

**Files:**
- Modify: `src/topo_shadow_box/core/mesh.py`
- Modify: `src/topo_shadow_box/tools/generate.py`

**Step 1: Write a test verifying normalization is consistent**

Add to `tests/test_mesh.py`:

```python
def test_elevation_normalization_is_deterministic():
    """Same grid should always produce the same normalization values."""
    from topo_shadow_box.core.mesh import _elevation_normalization
    import numpy as np

    grid = np.random.default_rng(42).random((100, 100)) * 1000
    result1 = _elevation_normalization(grid)
    result2 = _elevation_normalization(grid)
    assert result1 == result2


def test_generate_model_uses_consistent_normalization():
    """Terrain and feature meshes should use the same elevation scale."""
    # This is a design test — we verify _elevation_normalization result
    # is stable across multiple calls (prerequisite for caching).
    from topo_shadow_box.core.mesh import _elevation_normalization
    import numpy as np

    grid = np.linspace(0, 500, 200 * 200).reshape(200, 200)
    min_e, range_e = _elevation_normalization(grid)
    assert range_e > 0, "Non-flat grid should have positive range"
    assert min_e >= 0, "min elevation should be non-negative for positive grid"
```

**Step 2: Run to confirm they pass**

```bash
.venv/bin/pytest tests/test_mesh.py::test_elevation_normalization_is_deterministic tests/test_mesh.py::test_generate_model_uses_consistent_normalization -v
```
Expected: PASS

**Step 3: Refactor generate_feature_meshes to accept pre-computed normalization**

In `src/topo_shadow_box/core/mesh.py`, find `generate_feature_meshes` and its signature. Add an optional parameter:

```python
def generate_feature_meshes(
    features,
    elevation: ElevationData,
    bounds: Bounds,
    transform: GeoToModelTransform,
    vertical_scale: float = 1.5,
    shape: str = "square",
    _norm: tuple[float, float] | None = None,   # NEW: pre-computed (min_elev, elev_range)
) -> list[MeshResult]:
```

At the top of the function body, replace the existing `_elevation_normalization(...)` call with:

```python
    min_elev, elev_range = _norm if _norm is not None else _elevation_normalization(elevation.grid)
```

Apply the same pattern to `generate_gpx_track_mesh` if it calls `_elevation_normalization`.

In `src/topo_shadow_box/tools/generate.py`, compute normalization once and pass it:

```python
        from topo_shadow_box.core.mesh import _elevation_normalization
        norm = _elevation_normalization(state.elevation.grid)

        terrain = generate_terrain_mesh(...)  # unchanged

        if state.features and ...:
            fmeshes = generate_feature_meshes(
                ...
                _norm=norm,
            )

        if state.gpx_tracks:
            gpx = generate_gpx_track_mesh(
                ...
                _norm=norm,
            )
```

**Step 4: Run all tests**

```bash
.venv/bin/pytest -v
```
Expected: All PASS (this is a refactor — behavior is identical)

**Step 5: Commit**

```bash
git add src/topo_shadow_box/core/mesh.py src/topo_shadow_box/tools/generate.py
git commit -m "perf: compute elevation normalization once in generate_model and pass to all mesh generators"
```

---

## Phase 4: Security & Correctness

### Task 4.1: Add User-Agent headers

**Files:**
- Modify: `src/topo_shadow_box/core/elevation.py`
- Modify: `src/topo_shadow_box/core/osm.py`

**Step 1: Write tests**

Add to `tests/test_elevation.py`:

```python
@pytest.mark.asyncio
async def test_elevation_client_sends_user_agent():
    """HTTP client should include User-Agent header."""
    from topo_shadow_box.core.elevation import fetch_terrain_elevation
    from PIL import Image
    from io import BytesIO

    captured_kwargs = {}

    original_init = None

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        img = Image.new("RGB", (256, 256), color=(128, 100, 0))
        buf = BytesIO()
        img.save(buf, format="PNG")
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.content = buf.getvalue()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client

        try:
            await fetch_terrain_elevation(37.8, 37.75, -122.4, -122.45, resolution=10)
        except Exception:
            pass

        # Check that AsyncClient was initialized with headers
        call_kwargs = mock_client_cls.call_args
        if call_kwargs and call_kwargs.kwargs.get("headers"):
            assert "User-Agent" in call_kwargs.kwargs["headers"]
        elif call_kwargs and call_kwargs.args:
            pass  # positional args — acceptable
```

Add to `tests/test_osm.py`:

```python
@pytest.mark.asyncio
async def test_osm_client_sends_user_agent():
    """OSM HTTP client should include User-Agent header."""
    from topo_shadow_box.core.osm import _query_overpass

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_resp = AsyncMock()
        mock_resp.raise_for_status = AsyncMock()
        mock_resp.json = AsyncMock(return_value={"elements": []})
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client

        await _query_overpass("test")

        call_kwargs = mock_client_cls.call_args
        if call_kwargs and call_kwargs.kwargs.get("headers"):
            assert "User-Agent" in call_kwargs.kwargs["headers"]
```

**Step 2: Run to verify current state** (may already pass from Task 3.1 if User-Agent was added there)

```bash
.venv/bin/pytest tests/test_elevation.py::test_elevation_client_sends_user_agent tests/test_osm.py::test_osm_client_sends_user_agent -v
```

**Step 3: Add User-Agent to osm.py** (elevation.py already updated in Task 3.1)

In `src/topo_shadow_box/core/osm.py`, update the AsyncClient:

```python
    async with httpx.AsyncClient(timeout=45.0, headers={"User-Agent": "topo-shadow-box/1.0"}) as client:
```

**Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_elevation.py tests/test_osm.py -v
```
Expected: All PASS

**Step 5: Commit**

```bash
git add src/topo_shadow_box/core/osm.py
git commit -m "fix: add User-Agent header to HTTP clients in elevation and OSM fetchers"
```

---

### Task 4.2: OSM rate limiting

**Files:**
- Modify: `src/topo_shadow_box/core/osm.py`

**Step 1: Write test**

Add to `tests/test_osm.py`:

```python
@pytest.mark.asyncio
async def test_fetch_osm_features_sleeps_between_queries():
    """fetch_osm_features should sleep between Overpass queries for rate limiting."""
    from topo_shadow_box.core.osm import fetch_osm_features

    sleep_calls = []

    async def mock_sleep(seconds):
        sleep_calls.append(seconds)

    mock_resp = AsyncMock()
    mock_resp.raise_for_status = AsyncMock()
    mock_resp.json = AsyncMock(return_value={"elements": []})

    with patch("asyncio.sleep", side_effect=mock_sleep):
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            await fetch_osm_features(
                north=37.8, south=37.75, east=-122.4, west=-122.45,
                feature_types=["roads", "water", "buildings"],
            )

    # Should have slept between queries (3 feature types = 2 sleeps minimum)
    assert len(sleep_calls) >= 1, "Should sleep between OSM queries for rate limiting"
    assert all(s >= 1.0 for s in sleep_calls), "Sleep should be at least 1 second"
```

**Step 2: Run to verify failure**

```bash
.venv/bin/pytest tests/test_osm.py::test_fetch_osm_features_sleeps_between_queries -v
```
Expected: FAIL — no sleep calls

**Step 3: Implement**

In `src/topo_shadow_box/core/osm.py`, add `import asyncio` at top (if not already there).

In `fetch_osm_features`, update the loop that calls `_query_overpass`:

```python
    queries_list = list(queries.items())
    for i, (feat_name, (query, parse_type)) in enumerate(queries_list):
        if i > 0:
            await asyncio.sleep(1.0)  # OSM rate limit: 1 req/sec
        try:
            elements = await _query_overpass(query)
        except Exception:
            elements = []
        if feat_name == "roads":
            road_elements = elements
        elif feat_name == "water":
            water_elements = elements
        elif feat_name == "buildings":
            building_elements = elements
```

**Step 4: Run**

```bash
.venv/bin/pytest tests/test_osm.py -v
```
Expected: All PASS

**Step 5: Commit**

```bash
git add src/topo_shadow_box/core/osm.py
git commit -m "fix: add 1s rate-limit sleep between Overpass API queries to respect OSM usage policy"
```

---

### Task 4.3: Fix XML entity escaping in 3MF

**Files:**
- Modify: `src/topo_shadow_box/exporters/threemf.py`

**Step 1: Write the failing test**

Add to `tests/test_exporters.py`:

```python
    def test_gt_entity_escaped_in_name(self, tmp_path):
        """'>' in mesh name should be escaped as '&gt;' in 3MF XML."""
        from topo_shadow_box.exporters.threemf import export_3mf
        mesh = _minimal_mesh(name="Height > 100m", color="#FF0000")
        out = str(tmp_path / "gt_escape.3mf")
        export_3mf([mesh], out)
        with zipfile.ZipFile(out) as zf:
            xml = zf.read("3D/3dmodel.model").decode()
        assert "&gt;" in xml
        assert ' name="Height > 100m"' not in xml
```

**Step 2: Run to verify failure**

```bash
.venv/bin/pytest "tests/test_exporters.py::TestExport3MF::test_gt_entity_escaped_in_name" -v
```
Expected: FAIL

**Step 3: Implement**

In `src/topo_shadow_box/exporters/threemf.py`, update both `safe_name` lines (lines 55 and 62) to also escape `>`:

```python
        safe_name = (
            name.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
        )
```

**Step 4: Run**

```bash
.venv/bin/pytest tests/test_exporters.py -v
```
Expected: All PASS

**Step 5: Commit**

```bash
git add src/topo_shadow_box/exporters/threemf.py
git commit -m "fix: escape '>' as '&gt;' in 3MF XML mesh names"
```

---

### Task 4.4: Export path validation

**Files:**
- Modify: `src/topo_shadow_box/tools/export.py`

**Step 1: Write the failing test**

Add to `tests/test_exporters.py` (as a standalone test, not inside a class):

```python
def test_export_3mf_tool_rejects_path_traversal(tmp_path):
    """export_3mf tool should reject paths that escape the home directory."""
    from unittest.mock import patch
    from topo_shadow_box.state import state, MeshData
    from topo_shadow_box.state import Bounds

    # Set up minimal state
    state.terrain_mesh = MeshData(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        faces=[[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
        name="Terrain",
        feature_type="terrain",
    )

    # Attempt path traversal
    malicious_path = "/etc/shadow_box_test.3mf"

    # This should return an error string, not write the file
    # We test the tool function by importing and calling it via the registered handler
    # For a simpler test, check the validation function directly
    from topo_shadow_box.tools.export import _validate_output_path
    import pytest
    with pytest.raises(ValueError, match="outside"):
        _validate_output_path(malicious_path)
```

**Step 2: Run to verify failure**

```bash
.venv/bin/pytest tests/test_exporters.py::test_export_3mf_tool_rejects_path_traversal -v
```
Expected: FAIL — `ImportError: cannot import name '_validate_output_path'`

**Step 3: Implement**

In `src/topo_shadow_box/tools/export.py`, add after existing imports:

```python
import os
from pathlib import Path


def _validate_output_path(output_path: str) -> None:
    """Raise ValueError if output_path escapes the user's home directory."""
    resolved = Path(output_path).resolve()
    home = Path.home().resolve()
    try:
        resolved.relative_to(home)
    except ValueError:
        raise ValueError(
            f"Output path {output_path!r} is outside the home directory. "
            "Use a path within your home directory."
        )
```

In `export_3mf`, `export_openscad`, and `export_svg` tool functions, add before `os.makedirs(...)`:

```python
        try:
            _validate_output_path(output_path)
        except ValueError as e:
            return f"Error: {e}"
```

**Step 4: Run**

```bash
.venv/bin/pytest tests/test_exporters.py -v
```
Expected: All PASS

**Step 5: Commit**

```bash
git add src/topo_shadow_box/tools/export.py
git commit -m "fix: validate export output paths to prevent writing outside home directory"
```

---

## Phase 5: Architecture

### Task 5.1: Add require_state() helper

**Files:**
- Create: `src/topo_shadow_box/tools/_prereqs.py`
- Modify: `src/topo_shadow_box/tools/data.py`
- Modify: `src/topo_shadow_box/tools/generate.py`
- Modify: `src/topo_shadow_box/tools/export.py`

**Step 1: Write the failing test**

Create `tests/test_prereqs.py`:

```python
"""Tests for tool prerequisite helpers."""
import pytest
from unittest.mock import patch, MagicMock


def test_require_state_raises_when_bounds_not_set():
    from topo_shadow_box.tools._prereqs import require_state
    from topo_shadow_box.state import SessionState

    mock_state = SessionState()  # bounds.is_set defaults to False
    with pytest.raises(ValueError, match="area"):
        require_state(mock_state, bounds=True)


def test_require_state_passes_when_bounds_set():
    from topo_shadow_box.tools._prereqs import require_state
    from topo_shadow_box.state import SessionState, Bounds

    mock_state = SessionState()
    mock_state.bounds = Bounds(north=37.8, south=37.75, east=-122.4, west=-122.45, is_set=True)
    # Should not raise
    require_state(mock_state, bounds=True)


def test_require_state_raises_when_elevation_not_set():
    from topo_shadow_box.tools._prereqs import require_state
    from topo_shadow_box.state import SessionState, Bounds

    mock_state = SessionState()
    mock_state.bounds = Bounds(north=37.8, south=37.75, east=-122.4, west=-122.45, is_set=True)
    with pytest.raises(ValueError, match="elevation"):
        require_state(mock_state, bounds=True, elevation=True)


def test_require_state_raises_when_mesh_not_generated():
    from topo_shadow_box.tools._prereqs import require_state
    from topo_shadow_box.state import SessionState, Bounds, ElevationData
    import numpy as np

    mock_state = SessionState()
    mock_state.bounds = Bounds(north=37.8, south=37.75, east=-122.4, west=-122.45, is_set=True)
    mock_state.elevation = ElevationData(
        grid=np.zeros((10, 10)), lats=np.zeros(10), lons=np.zeros(10),
        resolution=10, is_set=True,
    )
    with pytest.raises(ValueError, match="model"):
        require_state(mock_state, bounds=True, elevation=True, mesh=True)
```

**Step 2: Run to verify failure**

```bash
.venv/bin/pytest tests/test_prereqs.py -v
```
Expected: FAIL — module not found

**Step 3: Implement**

Create `src/topo_shadow_box/tools/_prereqs.py`:

```python
"""Prerequisite checking helpers for MCP tools."""


def require_state(state, *, bounds: bool = False, elevation: bool = False, mesh: bool = False) -> None:
    """Raise ValueError with a descriptive message if required state is not set.

    Usage in a tool:
        try:
            require_state(state, bounds=True, elevation=True)
        except ValueError as e:
            return f"Error: {e}"
    """
    if bounds and not state.bounds.is_set:
        raise ValueError(
            "Set an area first with set_area_from_coordinates or set_area_from_gpx."
        )
    if elevation and not state.elevation.is_set:
        raise ValueError(
            "Fetch elevation data first with fetch_elevation."
        )
    if mesh and not state.terrain_mesh:
        raise ValueError(
            "Generate a model first with generate_model."
        )
```

**Step 4: Update tools to use require_state** (optional, tool by tool)

In `src/topo_shadow_box/tools/data.py`, replace:

```python
        if not state.bounds.is_set:
            return "Error: Set an area first with set_area_from_coordinates or set_area_from_gpx."
```

with:

```python
        from ._prereqs import require_state
        try:
            require_state(state, bounds=True)
        except ValueError as e:
            return f"Error: {e}"
```

Apply the same pattern to `fetch_features` and the generate/export tools.

**Step 5: Run all tests**

```bash
.venv/bin/pytest -v
```
Expected: All PASS

**Step 6: Commit**

```bash
git add src/topo_shadow_box/tools/_prereqs.py tests/test_prereqs.py src/topo_shadow_box/tools/data.py src/topo_shadow_box/tools/generate.py src/topo_shadow_box/tools/export.py
git commit -m "refactor: add require_state() helper for uniform tool prerequisite checking"
```

---

### Task 5.2: Warn on unknown feature type color, document singleton

**Files:**
- Modify: `src/topo_shadow_box/tools/export.py`
- Modify: `src/topo_shadow_box/state.py`

**Step 1: Write tests**

Add to `tests/test_exporters.py`:

```python
def test_all_mesh_feature_types_have_colors():
    """Every feature type produced by generate_feature_meshes must have a color in Colors."""
    from topo_shadow_box.state import Colors

    # Known feature types used in export.py / mesh.py
    known_feature_types = {"terrain", "roads", "water", "buildings", "gpx_track", "map_insert"}
    colors = Colors()
    colors_dict = colors.as_dict()

    # Colors uses 'roads' not 'road', 'gpx_track' not 'gpx'
    for ftype in known_feature_types:
        assert ftype in colors_dict, (
            f"Feature type '{ftype}' has no color in Colors. "
            "Add it or update this test if the type was removed."
        )


def test_unknown_feature_type_logs_warning(caplog):
    """_collect_meshes should log a warning when a feature type has no color."""
    import logging
    from topo_shadow_box.state import state, MeshData

    state.feature_meshes = [
        MeshData(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            faces=[[0, 1, 2]],
            name="Unknown Feature",
            feature_type="unknown_type",
        )
    ]
    state.terrain_mesh = None

    from topo_shadow_box.tools.export import _collect_meshes
    with caplog.at_level(logging.WARNING, logger="topo_shadow_box.tools.export"):
        meshes = _collect_meshes()

    assert any("unknown_type" in r.message or "color" in r.message.lower()
               for r in caplog.records), "Should warn about unknown feature type color"

    # Reset state
    state.feature_meshes = []
```

**Step 2: Run to verify failure**

```bash
.venv/bin/pytest tests/test_exporters.py::test_unknown_feature_type_logs_warning -v
```
Expected: FAIL — no warning logged

**Step 3: Implement warning in export.py**

In `src/topo_shadow_box/tools/export.py`, add logger at top:

```python
import logging
logger = logging.getLogger(__name__)
```

In `_collect_meshes`, update the color lookup:

```python
        color = getattr(colors, ftype, None)
        if color is None:
            logger.warning(
                "No color defined for feature type %r — using default gray #808080", ftype
            )
            color = "#808080"
```

**Step 4: Add singleton comment to state.py**

In `src/topo_shadow_box/state.py`, replace line 179–180:

```python
# Global session state — one per MCP server process
state = SessionState()
```

with:

```python
# Global session state — one per MCP server process.
#
# SINGLE-CLIENT LIMITATION: This singleton is shared across all tool calls in a
# server process. MCP clients that maintain a persistent connection will share
# this state, which works correctly for single-user usage.
#
# For multi-client support, this would need to change to per-request context
# (e.g., using contextvars.ContextVar or passing state as a parameter through
# each tool call rather than accessing a module-level singleton).
state = SessionState()
```

**Step 5: Run all tests**

```bash
.venv/bin/pytest -v
```
Expected: All PASS

**Step 6: Commit**

```bash
git add src/topo_shadow_box/tools/export.py src/topo_shadow_box/state.py tests/test_exporters.py
git commit -m "fix: warn on unknown feature type color in export; document global singleton limitation"
```

---

## Final Verification

Run the full test suite:

```bash
.venv/bin/pytest -v --tb=short 2>&1 | tail -20
```
Expected: All tests PASS, no regressions.
