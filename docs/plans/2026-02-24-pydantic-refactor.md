# Pydantic Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace all `@dataclass` and plain `dict` data structures with Pydantic `BaseModel` throughout the codebase, adding thorough validation at every boundary.

**Architecture:** Bottom-up, three layers: (1) core return models in `core/models.py`, (2) domain/feature models in `models.py`, (3) state models in `state.py`. Each layer is built and tested before the next depends on it.

**Tech Stack:** Python 3.11+, Pydantic v2 (`BaseModel`, `Field`, `model_validator`, `field_validator`, `ConfigDict`), pytest

---

## Task 1: Add pydantic dependency

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add pydantic to dependencies**

Edit `pyproject.toml` dependencies list:
```toml
dependencies = [
    "mcp[cli]>=1.0",
    "numpy>=1.24",
    "scipy>=1.10",
    "gpxpy>=1.6",
    "httpx>=0.24",
    "Pillow>=10.0",
    "websockets>=12.0",
    "pydantic>=2.0",
]
```

**Step 2: Sync dependencies**

```bash
uv sync
```

Expected: installs pydantic 2.x, no errors.

**Step 3: Verify import**

```bash
.venv/bin/python -c "import pydantic; print(pydantic.__version__)"
```

Expected: `2.x.x`

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add pydantic>=2.0 dependency"
```

---

## Task 2: Create `MeshResult` (Layer 1, TDD)

**Files:**
- Create: `tests/test_core_models.py`
- Create: `src/topo_shadow_box/core/models.py`

**Step 1: Write the failing tests**

Create `tests/test_core_models.py`:
```python
"""Tests for core return models."""
import pytest
from pydantic import ValidationError


class TestMeshResult:
    def test_valid_mesh(self):
        from topo_shadow_box.core.models import MeshResult
        m = MeshResult(
            vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            faces=[[0, 1, 2]],
        )
        assert len(m.vertices) == 3
        assert len(m.faces) == 1

    def test_vertex_must_have_3_components(self):
        from topo_shadow_box.core.models import MeshResult
        with pytest.raises(ValidationError):
            MeshResult(
                vertices=[[0.0, 0.0]],  # only 2 components
                faces=[[0, 0, 0]],
            )

    def test_face_must_have_3_indices(self):
        from topo_shadow_box.core.models import MeshResult
        with pytest.raises(ValidationError):
            MeshResult(
                vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                faces=[[0, 1]],  # only 2 indices
            )

    def test_face_index_must_be_non_negative(self):
        from topo_shadow_box.core.models import MeshResult
        with pytest.raises(ValidationError):
            MeshResult(
                vertices=[[0.0, 0.0, 0.0]],
                faces=[[-1, 0, 0]],
            )

    def test_face_index_must_be_valid_vertex_index(self):
        from topo_shadow_box.core.models import MeshResult
        with pytest.raises(ValidationError):
            MeshResult(
                vertices=[[0.0, 0.0, 0.0]],  # only 1 vertex (index 0)
                faces=[[0, 1, 2]],  # indices 1 and 2 out of range
            )

    def test_name_and_feature_type_default_empty(self):
        from topo_shadow_box.core.models import MeshResult
        m = MeshResult(vertices=[], faces=[])
        assert m.name == ""
        assert m.feature_type == ""

    def test_empty_mesh_is_valid(self):
        from topo_shadow_box.core.models import MeshResult
        m = MeshResult(vertices=[], faces=[])
        assert m.vertices == []
        assert m.faces == []
```

**Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_core_models.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `core/models.py` doesn't exist yet.

**Step 3: Create `src/topo_shadow_box/core/models.py` with `MeshResult`**

```python
"""Pydantic return models for core computation functions."""

from typing import Optional
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class MeshResult(BaseModel):
    """Return type for all mesh generation functions."""
    vertices: list[list[float]]
    faces: list[list[int]]
    name: str = ""
    feature_type: str = ""

    @field_validator("vertices")
    @classmethod
    def vertices_must_be_3d(cls, v: list[list[float]]) -> list[list[float]]:
        for i, vertex in enumerate(v):
            if len(vertex) != 3:
                raise ValueError(f"Vertex {i} must have exactly 3 components, got {len(vertex)}")
        return v

    @field_validator("faces")
    @classmethod
    def faces_must_be_triangles_with_non_negative_indices(
        cls, v: list[list[int]]
    ) -> list[list[int]]:
        for i, face in enumerate(v):
            if len(face) != 3:
                raise ValueError(f"Face {i} must have exactly 3 indices, got {len(face)}")
            for idx in face:
                if idx < 0:
                    raise ValueError(f"Face {i} has negative index {idx}")
        return v

    @model_validator(mode="after")
    def face_indices_must_be_valid(self) -> "MeshResult":
        n_verts = len(self.vertices)
        if n_verts == 0:
            return self
        for i, face in enumerate(self.faces):
            for idx in face:
                if idx >= n_verts:
                    raise ValueError(
                        f"Face {i} references vertex {idx} but only {n_verts} vertices exist"
                    )
        return self
```

**Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_core_models.py::TestMeshResult -v
```

Expected: all 7 tests PASS.

**Step 5: Commit**

```bash
git add tests/test_core_models.py src/topo_shadow_box/core/models.py
git commit -m "feat: add MeshResult pydantic model with geometry validation"
```

---

## Task 3: Add `ElevationResult` to `core/models.py` (TDD)

**Files:**
- Modify: `tests/test_core_models.py`
- Modify: `src/topo_shadow_box/core/models.py`

**Step 1: Write failing tests — append to `tests/test_core_models.py`**

```python
class TestElevationResult:
    def test_valid_elevation_result(self):
        import numpy as np
        from topo_shadow_box.core.models import ElevationResult
        e = ElevationResult(
            grid=np.zeros((10, 10)),
            lats=np.linspace(47.0, 47.1, 10),
            lons=np.linspace(-122.0, -121.9, 10),
            resolution=10,
            min_elevation=100.0,
            max_elevation=500.0,
        )
        assert e.resolution == 10

    def test_resolution_must_be_positive(self):
        import numpy as np
        from pydantic import ValidationError
        from topo_shadow_box.core.models import ElevationResult
        with pytest.raises(ValidationError):
            ElevationResult(
                grid=np.zeros((10, 10)),
                lats=np.zeros(10),
                lons=np.zeros(10),
                resolution=0,
                min_elevation=0.0,
                max_elevation=0.0,
            )

    def test_resolution_must_not_exceed_1000(self):
        import numpy as np
        from pydantic import ValidationError
        from topo_shadow_box.core.models import ElevationResult
        with pytest.raises(ValidationError):
            ElevationResult(
                grid=np.zeros((10, 10)),
                lats=np.zeros(10),
                lons=np.zeros(10),
                resolution=1001,
                min_elevation=0.0,
                max_elevation=0.0,
            )
```

**Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_core_models.py::TestElevationResult -v
```

Expected: FAIL — `ElevationResult` not defined.

**Step 3: Add `ElevationResult` to `core/models.py`**

Add after `MeshResult`:
```python
class ElevationResult(BaseModel):
    """Return type for fetch_terrain_elevation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    grid: np.ndarray
    lats: np.ndarray
    lons: np.ndarray
    resolution: int = Field(gt=0, le=1000)
    min_elevation: float
    max_elevation: float
```

**Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_core_models.py::TestElevationResult -v
```

Expected: all 3 tests PASS.

**Step 5: Commit**

```bash
git add tests/test_core_models.py src/topo_shadow_box/core/models.py
git commit -m "feat: add ElevationResult pydantic model"
```

---

## Task 4: Add placeholder `OsmFeatureSet` to `core/models.py` (TDD)

This placeholder uses untyped `list[dict]` for now. Task 7 will upgrade to typed lists once Layer 2 models exist.

**Files:**
- Modify: `tests/test_core_models.py`
- Modify: `src/topo_shadow_box/core/models.py`

**Step 1: Write failing tests — append to `tests/test_core_models.py`**

```python
class TestOsmFeatureSet:
    def test_empty_feature_set(self):
        from topo_shadow_box.core.models import OsmFeatureSet
        fs = OsmFeatureSet()
        assert fs.roads == []
        assert fs.water == []
        assert fs.buildings == []

    def test_feature_set_with_data(self):
        from topo_shadow_box.core.models import OsmFeatureSet
        fs = OsmFeatureSet(
            roads=[{"id": 1}],
            water=[],
            buildings=[{"id": 2}, {"id": 3}],
        )
        assert len(fs.roads) == 1
        assert len(fs.buildings) == 2
```

**Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_core_models.py::TestOsmFeatureSet -v
```

Expected: FAIL — `OsmFeatureSet` not defined.

**Step 3: Add `OsmFeatureSet` to `core/models.py`**

```python
class OsmFeatureSet(BaseModel):
    """Return type for fetch_osm_features. Feature lists are typed in Layer 2."""
    roads: list = []
    water: list = []
    buildings: list = []
```

**Step 4: Run all core model tests**

```bash
.venv/bin/pytest tests/test_core_models.py -v
```

Expected: all tests PASS.

**Step 5: Commit**

```bash
git add tests/test_core_models.py src/topo_shadow_box/core/models.py
git commit -m "feat: add OsmFeatureSet placeholder pydantic model"
```

---

## Task 5: Create `Coordinate`, `GpxPoint`, `GpxWaypoint`, `GpxTrack` (Layer 2, TDD)

**Files:**
- Create: `tests/test_models.py`
- Create: `src/topo_shadow_box/models.py`

**Step 1: Write failing tests**

Create `tests/test_models.py`:
```python
"""Tests for domain/feature Pydantic models."""
import pytest
from pydantic import ValidationError


class TestCoordinate:
    def test_valid_coordinate(self):
        from topo_shadow_box.models import Coordinate
        c = Coordinate(lat=47.6, lon=-122.3)
        assert c.lat == 47.6

    def test_lat_out_of_range(self):
        from topo_shadow_box.models import Coordinate
        with pytest.raises(ValidationError):
            Coordinate(lat=91.0, lon=0.0)

    def test_lat_negative_out_of_range(self):
        from topo_shadow_box.models import Coordinate
        with pytest.raises(ValidationError):
            Coordinate(lat=-91.0, lon=0.0)

    def test_lon_out_of_range(self):
        from topo_shadow_box.models import Coordinate
        with pytest.raises(ValidationError):
            Coordinate(lat=0.0, lon=181.0)

    def test_lon_negative_out_of_range(self):
        from topo_shadow_box.models import Coordinate
        with pytest.raises(ValidationError):
            Coordinate(lat=0.0, lon=-181.0)


class TestGpxPoint:
    def test_valid_gpx_point(self):
        from topo_shadow_box.models import GpxPoint
        p = GpxPoint(lat=47.6, lon=-122.3, elevation=50.0)
        assert p.elevation == 50.0

    def test_lat_out_of_range(self):
        from topo_shadow_box.models import GpxPoint
        with pytest.raises(ValidationError):
            GpxPoint(lat=95.0, lon=0.0, elevation=0.0)


class TestGpxWaypoint:
    def test_valid_waypoint(self):
        from topo_shadow_box.models import GpxWaypoint
        wp = GpxWaypoint(name="Summit", lat=47.6, lon=-122.3, elevation=1200.0)
        assert wp.description == ""

    def test_description_defaults_empty(self):
        from topo_shadow_box.models import GpxWaypoint
        wp = GpxWaypoint(name="X", lat=0.0, lon=0.0, elevation=0.0)
        assert wp.description == ""


class TestGpxTrack:
    def test_valid_track(self):
        from topo_shadow_box.models import GpxTrack, GpxPoint
        t = GpxTrack(
            name="Morning Ride",
            points=[
                GpxPoint(lat=47.6, lon=-122.3, elevation=50.0),
                GpxPoint(lat=47.7, lon=-122.2, elevation=60.0),
            ],
        )
        assert len(t.points) == 2

    def test_track_requires_at_least_2_points(self):
        from topo_shadow_box.models import GpxTrack, GpxPoint
        with pytest.raises(ValidationError):
            GpxTrack(
                name="Bad",
                points=[GpxPoint(lat=47.6, lon=-122.3, elevation=0.0)],
            )

    def test_empty_points_fails(self):
        from topo_shadow_box.models import GpxTrack
        with pytest.raises(ValidationError):
            GpxTrack(name="Empty", points=[])
```

**Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_models.py -v
```

Expected: FAIL — `topo_shadow_box.models` doesn't exist.

**Step 3: Create `src/topo_shadow_box/models.py`**

```python
"""Pydantic domain models for geographic features and GPX data."""

from pydantic import BaseModel, Field


class Coordinate(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class GpxPoint(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    elevation: float


class GpxWaypoint(BaseModel):
    name: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    elevation: float
    description: str = ""


class GpxTrack(BaseModel):
    name: str
    points: list[GpxPoint] = Field(min_length=2)
```

**Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_models.py::TestCoordinate tests/test_models.py::TestGpxPoint tests/test_models.py::TestGpxWaypoint tests/test_models.py::TestGpxTrack -v
```

Expected: all tests PASS.

**Step 5: Commit**

```bash
git add tests/test_models.py src/topo_shadow_box/models.py
git commit -m "feat: add Coordinate, GpxPoint, GpxWaypoint, GpxTrack pydantic models"
```

---

## Task 6: Add `RoadFeature`, `WaterFeature`, `BuildingFeature` (TDD)

**Files:**
- Modify: `tests/test_models.py`
- Modify: `src/topo_shadow_box/models.py`

**Step 1: Write failing tests — append to `tests/test_models.py`**

```python
class TestRoadFeature:
    def test_valid_road(self):
        from topo_shadow_box.models import RoadFeature, Coordinate
        r = RoadFeature(
            id=1,
            coordinates=[
                Coordinate(lat=47.6, lon=-122.3),
                Coordinate(lat=47.7, lon=-122.2),
            ],
        )
        assert r.type == "road"
        assert r.road_type == ""

    def test_road_requires_at_least_2_coordinates(self):
        from topo_shadow_box.models import RoadFeature, Coordinate
        with pytest.raises(ValidationError):
            RoadFeature(id=1, coordinates=[Coordinate(lat=47.6, lon=-122.3)])

    def test_road_type_literal(self):
        from topo_shadow_box.models import RoadFeature, Coordinate
        coords = [Coordinate(lat=47.6, lon=-122.3), Coordinate(lat=47.7, lon=-122.2)]
        with pytest.raises(ValidationError):
            RoadFeature(id=1, coordinates=coords, type="building")


class TestWaterFeature:
    def test_valid_water(self):
        from topo_shadow_box.models import WaterFeature, Coordinate
        w = WaterFeature(
            id=2,
            coordinates=[
                Coordinate(lat=47.6, lon=-122.3),
                Coordinate(lat=47.7, lon=-122.2),
                Coordinate(lat=47.65, lon=-122.1),
            ],
        )
        assert w.type == "water"

    def test_water_requires_at_least_3_coordinates(self):
        from topo_shadow_box.models import WaterFeature, Coordinate
        with pytest.raises(ValidationError):
            WaterFeature(id=2, coordinates=[
                Coordinate(lat=47.6, lon=-122.3),
                Coordinate(lat=47.7, lon=-122.2),
            ])


class TestBuildingFeature:
    def test_valid_building(self):
        from topo_shadow_box.models import BuildingFeature, Coordinate
        b = BuildingFeature(
            id=3,
            coordinates=[
                Coordinate(lat=47.6, lon=-122.3),
                Coordinate(lat=47.61, lon=-122.3),
                Coordinate(lat=47.61, lon=-122.29),
            ],
        )
        assert b.height == 10.0
        assert b.type == "building"

    def test_building_height_must_be_positive(self):
        from topo_shadow_box.models import BuildingFeature, Coordinate
        coords = [
            Coordinate(lat=47.6, lon=-122.3),
            Coordinate(lat=47.61, lon=-122.3),
            Coordinate(lat=47.61, lon=-122.29),
        ]
        with pytest.raises(ValidationError):
            BuildingFeature(id=3, coordinates=coords, height=0.0)

    def test_building_requires_at_least_3_coordinates(self):
        from topo_shadow_box.models import BuildingFeature, Coordinate
        with pytest.raises(ValidationError):
            BuildingFeature(id=3, coordinates=[
                Coordinate(lat=47.6, lon=-122.3),
                Coordinate(lat=47.61, lon=-122.3),
            ])
```

**Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_models.py::TestRoadFeature tests/test_models.py::TestWaterFeature tests/test_models.py::TestBuildingFeature -v
```

Expected: FAIL — models not defined.

**Step 3: Add feature models to `src/topo_shadow_box/models.py`**

Add after `GpxTrack`:
```python
from typing import Literal


class RoadFeature(BaseModel):
    id: int
    type: Literal["road"] = "road"
    coordinates: list[Coordinate] = Field(min_length=2)
    tags: dict = Field(default_factory=dict)
    name: str = ""
    road_type: str = ""


class WaterFeature(BaseModel):
    id: int
    type: Literal["water"] = "water"
    coordinates: list[Coordinate] = Field(min_length=3)
    tags: dict = Field(default_factory=dict)
    name: str = ""


class BuildingFeature(BaseModel):
    id: int
    type: Literal["building"] = "building"
    coordinates: list[Coordinate] = Field(min_length=3)
    tags: dict = Field(default_factory=dict)
    name: str = ""
    height: float = Field(default=10.0, gt=0)
```

Also add `from typing import Literal` at the top of the file.

**Step 4: Run all model tests**

```bash
.venv/bin/pytest tests/test_models.py -v
```

Expected: all tests PASS.

**Step 5: Commit**

```bash
git add tests/test_models.py src/topo_shadow_box/models.py
git commit -m "feat: add RoadFeature, WaterFeature, BuildingFeature pydantic models"
```

---

## Task 7: Upgrade `OsmFeatureSet` to typed feature lists

**Files:**
- Modify: `src/topo_shadow_box/core/models.py`
- Modify: `tests/test_core_models.py`

**Step 1: Update `TestOsmFeatureSet` in `tests/test_core_models.py`**

Replace the existing `TestOsmFeatureSet` class with:
```python
class TestOsmFeatureSet:
    def test_empty_feature_set(self):
        from topo_shadow_box.core.models import OsmFeatureSet
        fs = OsmFeatureSet()
        assert fs.roads == []
        assert fs.water == []
        assert fs.buildings == []

    def test_feature_set_with_typed_data(self):
        from topo_shadow_box.core.models import OsmFeatureSet
        from topo_shadow_box.models import RoadFeature, WaterFeature, Coordinate
        road = RoadFeature(id=1, coordinates=[
            Coordinate(lat=47.6, lon=-122.3),
            Coordinate(lat=47.7, lon=-122.2),
        ])
        water = WaterFeature(id=2, coordinates=[
            Coordinate(lat=47.6, lon=-122.3),
            Coordinate(lat=47.7, lon=-122.2),
            Coordinate(lat=47.65, lon=-122.1),
        ])
        fs = OsmFeatureSet(roads=[road], water=[water])
        assert len(fs.roads) == 1
        assert len(fs.water) == 1

    def test_rejects_wrong_feature_type_in_roads(self):
        from topo_shadow_box.core.models import OsmFeatureSet
        from topo_shadow_box.models import WaterFeature, Coordinate
        water = WaterFeature(id=2, coordinates=[
            Coordinate(lat=47.6, lon=-122.3),
            Coordinate(lat=47.7, lon=-122.2),
            Coordinate(lat=47.65, lon=-122.1),
        ])
        with pytest.raises(ValidationError):
            OsmFeatureSet(roads=[water])
```

**Step 2: Run tests to verify new test fails**

```bash
.venv/bin/pytest tests/test_core_models.py::TestOsmFeatureSet -v
```

Expected: FAIL on typed tests.

**Step 3: Update `OsmFeatureSet` in `core/models.py`**

Replace the placeholder with:
```python
from topo_shadow_box.models import RoadFeature, WaterFeature, BuildingFeature

class OsmFeatureSet(BaseModel):
    """Return type for fetch_osm_features."""
    roads: list[RoadFeature] = []
    water: list[WaterFeature] = []
    buildings: list[BuildingFeature] = []
```

**Step 4: Run all core model tests**

```bash
.venv/bin/pytest tests/test_core_models.py -v
```

Expected: all tests PASS.

**Step 5: Commit**

```bash
git add tests/test_core_models.py src/topo_shadow_box/core/models.py
git commit -m "feat: upgrade OsmFeatureSet to typed feature lists"
```

---

## Task 8: Convert `Bounds` to Pydantic (Layer 3, TDD)

**Files:**
- Create: `tests/test_state_models.py`
- Modify: `src/topo_shadow_box/state.py`

**Step 1: Write failing tests**

Create `tests/test_state_models.py`:
```python
"""Tests for state Pydantic models."""
import pytest
from pydantic import ValidationError


class TestBounds:
    def test_default_bounds_not_set(self):
        from topo_shadow_box.state import Bounds
        b = Bounds()
        assert b.is_set is False

    def test_valid_bounds(self):
        from topo_shadow_box.state import Bounds
        b = Bounds(north=48.0, south=47.0, east=-121.0, west=-122.0, is_set=True)
        assert b.north == 48.0
        assert b.lat_range == pytest.approx(1.0)
        assert b.lon_range == pytest.approx(1.0)
        assert b.center_lat == pytest.approx(47.5)
        assert b.center_lon == pytest.approx(-121.5)

    def test_north_must_be_gt_south_when_set(self):
        from topo_shadow_box.state import Bounds
        with pytest.raises(ValidationError):
            Bounds(north=47.0, south=48.0, east=-121.0, west=-122.0, is_set=True)

    def test_east_must_be_gt_west_when_set(self):
        from topo_shadow_box.state import Bounds
        with pytest.raises(ValidationError):
            Bounds(north=48.0, south=47.0, east=-122.0, west=-121.0, is_set=True)

    def test_lat_out_of_range(self):
        from topo_shadow_box.state import Bounds
        with pytest.raises(ValidationError):
            Bounds(north=91.0, south=47.0, east=-121.0, west=-122.0, is_set=True)

    def test_lon_out_of_range(self):
        from topo_shadow_box.state import Bounds
        with pytest.raises(ValidationError):
            Bounds(north=48.0, south=47.0, east=181.0, west=-122.0, is_set=True)

    def test_cross_field_validation_skipped_when_not_set(self):
        from topo_shadow_box.state import Bounds
        # is_set=False: north/south/east/west are all 0, cross-field check skipped
        b = Bounds()
        assert b.north == 0.0
```

**Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_state_models.py::TestBounds -v
```

Expected: FAIL — `Bounds` is still a dataclass without `is_set` field or validators.

**Step 3: Replace `Bounds` in `src/topo_shadow_box/state.py`**

Replace the `Bounds` dataclass entirely. New `state.py` starts with:
```python
"""Session state for the topo-shadow-box MCP server."""

from typing import Optional
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class Bounds(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    north: float = Field(default=0.0, ge=-90, le=90)
    south: float = Field(default=0.0, ge=-90, le=90)
    east: float = Field(default=0.0, ge=-180, le=180)
    west: float = Field(default=0.0, ge=-180, le=180)
    is_set: bool = False

    @model_validator(mode="after")
    def check_north_gt_south(self) -> "Bounds":
        if self.is_set and self.north <= self.south:
            raise ValueError(f"north ({self.north}) must be greater than south ({self.south})")
        return self

    @model_validator(mode="after")
    def check_east_gt_west(self) -> "Bounds":
        if self.is_set and self.east <= self.west:
            raise ValueError(f"east ({self.east}) must be greater than west ({self.west})")
        return self

    @property
    def lat_range(self) -> float:
        return self.north - self.south

    @property
    def lon_range(self) -> float:
        return self.east - self.west

    @property
    def center_lat(self) -> float:
        return (self.north + self.south) / 2

    @property
    def center_lon(self) -> float:
        return (self.east + self.west) / 2
```

Keep all other dataclasses unchanged for now.

**Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_state_models.py::TestBounds -v
```

Expected: all tests PASS.

**Step 5: Run full test suite to check no regressions**

```bash
.venv/bin/pytest -v
```

Expected: all existing tests PASS (Bounds is used in coords.py — verify no breakage).

**Step 6: Commit**

```bash
git add tests/test_state_models.py src/topo_shadow_box/state.py
git commit -m "feat: convert Bounds to pydantic BaseModel with cross-field validation"
```

---

## Task 9: Convert `ModelParams` to Pydantic (TDD)

**Files:**
- Modify: `tests/test_state_models.py`
- Modify: `src/topo_shadow_box/state.py`

**Step 1: Write failing tests — append to `tests/test_state_models.py`**

```python
class TestModelParams:
    def test_defaults(self):
        from topo_shadow_box.state import ModelParams
        mp = ModelParams()
        assert mp.width_mm == 200.0
        assert mp.vertical_scale == 1.5
        assert mp.base_height_mm == 10.0
        assert mp.shape == "square"

    def test_width_must_be_positive(self):
        from topo_shadow_box.state import ModelParams
        with pytest.raises(ValidationError):
            ModelParams(width_mm=0.0)

    def test_vertical_scale_must_be_positive(self):
        from topo_shadow_box.state import ModelParams
        with pytest.raises(ValidationError):
            ModelParams(vertical_scale=-1.0)

    def test_base_height_must_be_positive(self):
        from topo_shadow_box.state import ModelParams
        with pytest.raises(ValidationError):
            ModelParams(base_height_mm=0.0)

    def test_valid_shapes(self):
        from topo_shadow_box.state import ModelParams
        for shape in ("square", "circle", "hexagon", "rectangle"):
            mp = ModelParams(shape=shape)
            assert mp.shape == shape

    def test_invalid_shape(self):
        from topo_shadow_box.state import ModelParams
        with pytest.raises(ValidationError):
            ModelParams(shape="triangle")
```

**Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_state_models.py::TestModelParams -v
```

Expected: FAIL — `ModelParams` is still a dataclass.

**Step 3: Replace `ModelParams` in `state.py`**

```python
from typing import Literal

class ModelParams(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    width_mm: float = Field(default=200.0, gt=0)
    vertical_scale: float = Field(default=1.5, gt=0)
    base_height_mm: float = Field(default=10.0, gt=0)
    shape: Literal["square", "circle", "hexagon", "rectangle"] = "square"
```

**Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_state_models.py::TestModelParams -v && .venv/bin/pytest -v
```

Expected: all PASS.

**Step 5: Commit**

```bash
git add tests/test_state_models.py src/topo_shadow_box/state.py
git commit -m "feat: convert ModelParams to pydantic BaseModel"
```

---

## Task 10: Convert `Colors` to Pydantic (TDD)

**Files:**
- Modify: `tests/test_state_models.py`
- Modify: `src/topo_shadow_box/state.py`

**Step 1: Write failing tests — append to `tests/test_state_models.py`**

```python
class TestColors:
    def test_defaults(self):
        from topo_shadow_box.state import Colors
        c = Colors()
        assert c.terrain == "#C8A882"

    def test_valid_hex_color(self):
        from topo_shadow_box.state import Colors
        c = Colors(terrain="#FF0000")
        assert c.terrain == "#FF0000"

    def test_lowercase_hex_accepted(self):
        from topo_shadow_box.state import Colors
        c = Colors(terrain="#ff0000")
        assert c.terrain == "#FF0000"  # normalized to uppercase

    def test_invalid_hex_missing_hash(self):
        from topo_shadow_box.state import Colors
        with pytest.raises(ValidationError):
            Colors(terrain="FF0000")

    def test_invalid_hex_wrong_length(self):
        from topo_shadow_box.state import Colors
        with pytest.raises(ValidationError):
            Colors(terrain="#FFF")

    def test_invalid_hex_non_hex_chars(self):
        from topo_shadow_box.state import Colors
        with pytest.raises(ValidationError):
            Colors(terrain="#GGGGGG")

    def test_hex_to_rgb(self):
        from topo_shadow_box.state import Colors
        c = Colors(terrain="#FF8040")
        assert c.hex_to_rgb(c.terrain) == (255, 128, 64)

    def test_as_dict(self):
        from topo_shadow_box.state import Colors
        c = Colors()
        d = c.as_dict()
        assert "terrain" in d
        assert "water" in d
        assert len(d) == 6
```

**Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_state_models.py::TestColors -v
```

Expected: FAIL.

**Step 3: Replace `Colors` in `state.py`**

```python
import re

class Colors(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    terrain: str = "#C8A882"
    water: str = "#4682B4"
    roads: str = "#D4C5A9"
    buildings: str = "#E8D5B7"
    gpx_track: str = "#FF0000"
    map_insert: str = "#FFFFFF"

    @field_validator("terrain", "water", "roads", "buildings", "gpx_track", "map_insert", mode="before")
    @classmethod
    def validate_and_normalize_hex(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Color must be a string")
        v = v.strip()
        if not re.match(r'^#[0-9A-Fa-f]{6}$', v):
            raise ValueError(f"Invalid hex color '{v}'. Must be #RRGGBB format.")
        return f"#{v[1:].upper()}"

    def hex_to_rgb(self, hex_color: str) -> tuple[int, int, int]:
        h = hex_color.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    def as_dict(self) -> dict[str, str]:
        return {
            "terrain": self.terrain,
            "water": self.water,
            "roads": self.roads,
            "buildings": self.buildings,
            "gpx_track": self.gpx_track,
            "map_insert": self.map_insert,
        }
```

Also add `import re` at the top of `state.py`.

**Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_state_models.py::TestColors -v && .venv/bin/pytest -v
```

Expected: all PASS.

**Step 5: Commit**

```bash
git add tests/test_state_models.py src/topo_shadow_box/state.py
git commit -m "feat: convert Colors to pydantic BaseModel with hex validation"
```

---

## Task 11: Convert `ElevationData` and `MeshData` to Pydantic (TDD)

**Files:**
- Modify: `tests/test_state_models.py`
- Modify: `src/topo_shadow_box/state.py`

**Step 1: Write failing tests — append to `tests/test_state_models.py`**

```python
class TestElevationData:
    def test_defaults(self):
        from topo_shadow_box.state import ElevationData
        e = ElevationData()
        assert e.is_set is False
        assert e.grid is None

    def test_resolution_must_be_positive(self):
        from topo_shadow_box.state import ElevationData
        with pytest.raises(ValidationError):
            ElevationData(resolution=0)

    def test_resolution_must_not_exceed_1000(self):
        from topo_shadow_box.state import ElevationData
        with pytest.raises(ValidationError):
            ElevationData(resolution=1001)


class TestMeshData:
    def test_defaults(self):
        from topo_shadow_box.state import MeshData
        m = MeshData()
        assert m.vertices == []
        assert m.faces == []
        assert m.name == ""

    def test_with_data(self):
        from topo_shadow_box.state import MeshData
        m = MeshData(
            vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            faces=[[0, 1, 2]],
            name="Terrain",
            feature_type="terrain",
        )
        assert len(m.vertices) == 3
```

**Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_state_models.py::TestElevationData tests/test_state_models.py::TestMeshData -v
```

Expected: FAIL.

**Step 3: Replace `ElevationData` and `MeshData` in `state.py`**

```python
class ElevationData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    grid: Optional[np.ndarray] = None
    lats: Optional[np.ndarray] = None
    lons: Optional[np.ndarray] = None
    resolution: int = Field(default=200, gt=0, le=1000)
    min_elevation: float = 0.0
    max_elevation: float = 0.0
    is_set: bool = False


class MeshData(BaseModel):
    vertices: list[list[float]] = Field(default_factory=list)
    faces: list[list[int]] = Field(default_factory=list)
    name: str = ""
    feature_type: str = ""
```

**Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_state_models.py -v && .venv/bin/pytest -v
```

Expected: all PASS.

**Step 5: Commit**

```bash
git add tests/test_state_models.py src/topo_shadow_box/state.py
git commit -m "feat: convert ElevationData and MeshData to pydantic BaseModel"
```

---

## Task 12: Convert `SessionState` to Pydantic (TDD)

**Files:**
- Modify: `tests/test_state_models.py`
- Modify: `src/topo_shadow_box/state.py`

**Step 1: Write failing tests — append to `tests/test_state_models.py`**

```python
class TestSessionState:
    def test_defaults(self):
        from topo_shadow_box.state import SessionState
        s = SessionState()
        assert s.bounds.is_set is False
        assert s.elevation.is_set is False
        assert s.gpx_tracks == []
        assert s.gpx_waypoints == []
        assert s.terrain_mesh is None
        assert s.preview_port == 3333

    def test_preview_port_range(self):
        from topo_shadow_box.state import SessionState
        with pytest.raises(ValidationError):
            SessionState(preview_port=0)
        with pytest.raises(ValidationError):
            SessionState(preview_port=65536)

    def test_summary_returns_dict(self):
        from topo_shadow_box.state import SessionState
        s = SessionState()
        d = s.summary()
        assert "area" in d
        assert "data" in d
        assert "model" in d
```

**Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_state_models.py::TestSessionState -v
```

Expected: FAIL.

**Step 3: Replace `SessionState` in `state.py`**

Remove the `@dataclass` import. Add imports for `OsmFeatureSet`, `GpxTrack`, `GpxWaypoint` from their new modules. Full `SessionState`:

```python
from topo_shadow_box.core.models import OsmFeatureSet
from topo_shadow_box.models import GpxTrack, GpxWaypoint

class SessionState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    bounds: Bounds = Field(default_factory=Bounds)
    elevation: ElevationData = Field(default_factory=ElevationData)
    features: OsmFeatureSet = Field(default_factory=OsmFeatureSet)
    gpx_tracks: list[GpxTrack] = []
    gpx_waypoints: list[GpxWaypoint] = []
    model_params: ModelParams = Field(default_factory=ModelParams)
    colors: Colors = Field(default_factory=Colors)
    terrain_mesh: Optional[MeshData] = None
    feature_meshes: list[MeshData] = []
    gpx_mesh: Optional[MeshData] = None
    map_insert_mesh: Optional[MeshData] = None
    preview_port: int = Field(default=3333, gt=0, le=65535)
    preview_running: bool = False

    def summary(self) -> dict:
        return {
            "area": {
                "bounds_set": self.bounds.is_set,
                "north": self.bounds.north,
                "south": self.bounds.south,
                "east": self.bounds.east,
                "west": self.bounds.west,
            } if self.bounds.is_set else {"bounds_set": False},
            "data": {
                "elevation_loaded": self.elevation.is_set,
                "elevation_resolution": self.elevation.resolution if self.elevation.is_set else None,
                "elevation_range": (
                    f"{self.elevation.min_elevation:.0f}m - {self.elevation.max_elevation:.0f}m"
                    if self.elevation.is_set else None
                ),
                "features_loaded": bool(
                    self.features.roads or self.features.water or self.features.buildings
                ),
                "feature_counts": {
                    "roads": len(self.features.roads),
                    "water": len(self.features.water),
                    "buildings": len(self.features.buildings),
                },
                "gpx_tracks": len(self.gpx_tracks),
            },
            "model": {
                "width_mm": self.model_params.width_mm,
                "vertical_scale": self.model_params.vertical_scale,
                "base_height_mm": self.model_params.base_height_mm,
                "shape": self.model_params.shape,
            },
            "colors": self.colors.as_dict(),
            "meshes": {
                "terrain_generated": self.terrain_mesh is not None,
                "feature_meshes": len(self.feature_meshes),
                "gpx_mesh_generated": self.gpx_mesh is not None,
            },
            "preview": {
                "running": self.preview_running,
                "port": self.preview_port,
            },
        }


# Global session state — one per MCP server process
state = SessionState()
```

Remove all `from dataclasses import dataclass, field` imports from `state.py`.

**Step 4: Run all tests**

```bash
.venv/bin/pytest -v
```

Expected: all PASS.

**Step 5: Commit**

```bash
git add tests/test_state_models.py src/topo_shadow_box/state.py
git commit -m "feat: convert SessionState to pydantic BaseModel"
```

---

## Task 13: Update `core/elevation.py` to return `ElevationResult`

**Files:**
- Modify: `src/topo_shadow_box/core/elevation.py`
- Modify: `src/topo_shadow_box/tools/data.py`

**Step 1: Update `fetch_terrain_elevation` signature and return**

In `core/elevation.py`, change the return type and final return statement:

```python
from .models import ElevationResult

async def fetch_terrain_elevation(
    north: float, south: float, east: float, west: float,
    resolution: int = 200,
) -> ElevationResult:
    # ... existing body unchanged ...

    # At the end, replace `return { ... }` with:
    return ElevationResult(
        grid=grid_interp,
        lats=lats_new,
        lons=lons_new,
        resolution=resolution,
        min_elevation=float(p2),
        max_elevation=float(p98),
    )
```

(Read the end of `core/elevation.py` to find the exact `return` dict and replace it.)

**Step 2: Update `tools/data.py` to use `ElevationResult`**

In `tools/data.py`, find where `fetch_terrain_elevation` result is unpacked into `ElevationData`. Change from:
```python
result = await fetch_terrain_elevation(...)
state.elevation = ElevationData(
    grid=result["grid"],
    lats=result["lats"],
    ...
)
```
To:
```python
result = await fetch_terrain_elevation(...)
state.elevation = ElevationData(
    grid=result.grid,
    lats=result.lats,
    lons=result.lons,
    resolution=result.resolution,
    min_elevation=result.min_elevation,
    max_elevation=result.max_elevation,
    is_set=True,
)
```

**Step 3: Run full test suite**

```bash
.venv/bin/pytest -v
```

Expected: all PASS.

**Step 4: Commit**

```bash
git add src/topo_shadow_box/core/elevation.py src/topo_shadow_box/tools/data.py
git commit -m "feat: update fetch_terrain_elevation to return ElevationResult"
```

---

## Task 14: Update `core/mesh.py` mesh functions to return `MeshResult`

**Files:**
- Modify: `src/topo_shadow_box/core/mesh.py`
- Modify: `src/topo_shadow_box/tools/generate.py`

There are four functions returning dicts: `generate_terrain_mesh`, `generate_feature_meshes`, `generate_gpx_track_mesh`, and `generate_map_insert_plate` (in `core/map_insert.py`).

**Step 1: Add import to `core/mesh.py`**

```python
from .models import MeshResult
```

**Step 2: Update `generate_terrain_mesh` return type**

Find the final `return {"vertices": ..., "faces": ...}` in `generate_terrain_mesh` and replace with:
```python
return MeshResult(vertices=vertices, faces=faces, name="Terrain", feature_type="terrain")
```

Update the function signature annotation: `-> MeshResult`

**Step 3: Update `generate_feature_meshes` return type**

This function returns `list[dict]`. Change to `list[MeshResult]`. Find all `return` / `append` calls and update. Each feature mesh dict like:
```python
{"vertices": v, "faces": f, "name": name, "type": ftype}
```
becomes:
```python
MeshResult(vertices=v, faces=f, name=name, feature_type=ftype)
```

Update signature: `-> list[MeshResult]`

**Step 4: Update `generate_gpx_track_mesh` return type**

Change `-> Optional[dict]` to `-> Optional[MeshResult]`. Replace final return:
```python
return MeshResult(vertices=vertices, faces=faces, name="GPX Track", feature_type="gpx_track")
```

**Step 5: Update `tools/generate.py` to use `.` access instead of `[]`**

```python
terrain = generate_terrain_mesh(...)
state.terrain_mesh = MeshData(
    vertices=terrain.vertices,
    faces=terrain.faces,
    name=terrain.name,
    feature_type=terrain.feature_type,
)

# Feature meshes loop:
for fm in fmeshes:
    state.feature_meshes.append(MeshData(
        vertices=fm.vertices,
        faces=fm.faces,
        name=fm.name,
        feature_type=fm.feature_type,
    ))

# GPX mesh:
state.gpx_mesh = MeshData(
    vertices=gpx.vertices,
    faces=gpx.faces,
    name=gpx.name,
    feature_type=gpx.feature_type,
)
```

Also update the `if state.features:` check to:
```python
if state.features.roads or state.features.water or state.features.buildings:
```

**Step 6: Run full test suite**

```bash
.venv/bin/pytest -v
```

Expected: all PASS.

**Step 7: Commit**

```bash
git add src/topo_shadow_box/core/mesh.py src/topo_shadow_box/tools/generate.py
git commit -m "feat: update mesh generation functions to return MeshResult"
```

---

## Task 15: Update `core/map_insert.py` to return `MeshResult`

**Files:**
- Modify: `src/topo_shadow_box/core/map_insert.py`
- Modify: `src/topo_shadow_box/tools/generate.py`

**Step 1: Add import to `core/map_insert.py`**

```python
from .models import MeshResult
```

**Step 2: Update `generate_map_insert_plate` return**

Change `-> dict` to `-> MeshResult`. Replace final return dict with:
```python
return MeshResult(vertices=vertices, faces=faces, name="Map Insert", feature_type="map_insert")
```

**Step 3: Update `tools/generate.py` map insert section**

```python
plate = generate_map_insert_plate(...)
state.map_insert_mesh = MeshData(
    vertices=plate.vertices,
    faces=plate.faces,
    name=plate.name,
    feature_type=plate.feature_type,
)
```

**Step 4: Run full test suite**

```bash
.venv/bin/pytest -v
```

**Step 5: Commit**

```bash
git add src/topo_shadow_box/core/map_insert.py src/topo_shadow_box/tools/generate.py
git commit -m "feat: update generate_map_insert_plate to return MeshResult"
```

---

## Task 16: Update `core/osm.py` to return `OsmFeatureSet` with typed features

**Files:**
- Modify: `src/topo_shadow_box/core/osm.py`
- Modify: `src/topo_shadow_box/tools/data.py`

**Step 1: Add imports to `core/osm.py`**

```python
from .models import OsmFeatureSet
from topo_shadow_box.models import RoadFeature, WaterFeature, BuildingFeature, Coordinate
```

**Step 2: Update `_parse_features` to return typed models**

Currently `_parse_features` returns `list[dict]`. It now returns typed feature lists. Change its internal logic:

```python
def _parse_features(elements: list[dict], feature_type: str) -> list:
    features = []
    for elem in elements:
        raw_coords = _parse_way_coords(elem)
        if len(raw_coords) < 2:
            continue
        coords = [Coordinate(lat=c["lat"], lon=c["lon"]) for c in raw_coords]
        tags = elem.get("tags", {})
        name = tags.get("name", f"{feature_type}_{elem.get('id')}")

        if feature_type == "road":
            if len(coords) < 2:
                continue
            features.append(RoadFeature(
                id=elem.get("id", 0),
                coordinates=coords,
                tags=tags,
                name=name,
                road_type=tags.get("highway", "road"),
            ))
        elif feature_type == "water":
            if len(coords) < 3:
                continue
            features.append(WaterFeature(
                id=elem.get("id", 0),
                coordinates=coords,
                tags=tags,
                name=name,
            ))
        elif feature_type == "building":
            if len(coords) < 3:
                continue
            height = 10.0
            if "height" in tags:
                try:
                    height = float(tags["height"].replace("m", "").strip())
                except ValueError:
                    pass
            elif "building:levels" in tags:
                try:
                    height = float(tags["building:levels"]) * 3.0
                except ValueError:
                    pass
            height = max(height, 0.1)  # ensure gt=0 for BuildingFeature
            features.append(BuildingFeature(
                id=elem.get("id", 0),
                coordinates=coords,
                tags=tags,
                name=name,
                height=height,
            ))
    return features
```

**Step 3: Update `fetch_osm_features` return type**

Change return type to `-> OsmFeatureSet` and build the result:
```python
async def fetch_osm_features(...) -> OsmFeatureSet:
    # ... existing fetch logic ...
    roads = _parse_features(road_elements, "road")[:200]
    water = _parse_features(water_elements, "water")[:50]
    buildings = _parse_features(building_elements, "building")[:150]
    return OsmFeatureSet(roads=roads, water=water, buildings=buildings)
```

**Step 4: Update `tools/data.py` to store `OsmFeatureSet` directly**

Change from:
```python
result = await fetch_osm_features(...)
state.features = result  # was a dict
```
To:
```python
result = await fetch_osm_features(...)
state.features = result  # now OsmFeatureSet
```

Also update any string formatting that used `state.features.items()` — now use:
```python
f"roads={len(state.features.roads)}, water={len(state.features.water)}, buildings={len(state.features.buildings)}"
```

**Step 5: Update `tools/area.py` reset line**

Change `state.features = {}` to `state.features = OsmFeatureSet()`.

Import at top of `tools/area.py`:
```python
from ..core.models import OsmFeatureSet
```

**Step 6: Run full test suite**

```bash
.venv/bin/pytest -v
```

Expected: all PASS.

**Step 7: Commit**

```bash
git add src/topo_shadow_box/core/osm.py src/topo_shadow_box/tools/data.py src/topo_shadow_box/tools/area.py
git commit -m "feat: update fetch_osm_features to return typed OsmFeatureSet"
```

---

## Task 17: Update `core/gpx.py` to return typed models

**Files:**
- Modify: `src/topo_shadow_box/core/gpx.py`
- Modify: `src/topo_shadow_box/tools/area.py`

**Step 1: Update `parse_gpx_file` return type**

In `core/gpx.py`:
```python
from topo_shadow_box.models import GpxTrack, GpxPoint, GpxWaypoint

def parse_gpx_file(filepath: str) -> dict:
    # ... existing parsing ...

    # Change track construction:
    tracks = []
    for track in gpx.tracks:
        points = []
        for segment in track.segments:
            for point in segment.points:
                points.append(GpxPoint(
                    lat=point.latitude,
                    lon=point.longitude,
                    elevation=point.elevation if point.elevation else 0.0,
                ))
        if len(points) >= 2:
            tracks.append(GpxTrack(
                name=track.name or "Unnamed Track",
                points=points,
            ))

    # Change waypoint construction:
    waypoints = []
    for wp in gpx.waypoints:
        waypoints.append(GpxWaypoint(
            name=wp.name or "",
            lat=wp.latitude,
            lon=wp.longitude,
            elevation=wp.elevation if wp.elevation else 0.0,
            description=wp.description or "",
        ))

    # Return dict structure unchanged (bounds still a plain dict)
    return {
        "tracks": tracks,        # now list[GpxTrack]
        "waypoints": waypoints,  # now list[GpxWaypoint]
        "bounds": bounds_dict,
        "metadata": {...},
    }
```

**Step 2: Update `tools/area.py` — no changes needed for GPX storage**

`state.gpx_tracks = gpx_data["tracks"]` now stores `list[GpxTrack]` which matches `SessionState.gpx_tracks: list[GpxTrack]`. Same for waypoints. Only update the point-count calculation:

```python
total_points = sum(len(t.points) for t in state.gpx_tracks)
```

**Step 3: Update `tools/area.py` reset of elevation**

Change `state.elevation = state.elevation.__class__()` to `state.elevation = ElevationData()`.

Add import: `from ..state import state, Bounds, ElevationData`

**Step 4: Run full test suite**

```bash
.venv/bin/pytest -v
```

Expected: all PASS.

**Step 5: Commit**

```bash
git add src/topo_shadow_box/core/gpx.py src/topo_shadow_box/tools/area.py
git commit -m "feat: update parse_gpx_file to return typed GpxTrack/GpxWaypoint models"
```

---

## Task 18: Update `core/mesh.py` to accept typed feature/GPX models

The mesh functions that take `features` and `gpx_tracks` as arguments currently expect plain dicts. After the OSM and GPX updates, callers pass typed models.

**Files:**
- Modify: `src/topo_shadow_box/core/mesh.py`

**Step 1: Read the signatures of `generate_feature_meshes` and `generate_gpx_track_mesh`**

Find lines where `features["roads"]`, `features["water"]`, `features["buildings"]` are accessed — replace with `features.roads`, `features.water`, `features.buildings`.

Find lines where track dict keys like `track["points"]`, `track["name"]`, `point["lat"]`, `point["lon"]`, `point["elevation"]` are accessed — replace with attribute access: `track.points`, `track.name`, `point.lat`, `point.lon`, `point.elevation`.

Find lines where feature coordinate dicts like `coord["lat"]`, `coord["lon"]` are accessed — replace with `coord.lat`, `coord.lon`.

**Step 2: Update function signatures**

```python
from topo_shadow_box.core.models import OsmFeatureSet, MeshResult
from topo_shadow_box.models import GpxTrack

def generate_feature_meshes(
    features: OsmFeatureSet,
    ...
) -> list[MeshResult]:

def generate_gpx_track_mesh(
    tracks: list[GpxTrack],
    ...
) -> Optional[MeshResult]:
```

**Step 3: Run full test suite**

```bash
.venv/bin/pytest -v
```

Expected: all PASS. The existing `test_mesh.py` tests call core functions directly with their own fixtures — verify they still work.

**Step 4: Commit**

```bash
git add src/topo_shadow_box/core/mesh.py
git commit -m "feat: update mesh functions to accept typed OsmFeatureSet and GpxTrack models"
```

---

## Task 19: Simplify `tools/model.py` — validation moves to models

**Files:**
- Modify: `src/topo_shadow_box/tools/model.py`

**Step 1: Remove manual shape validation from `set_model_params`**

The `shape` validation (`if shape not in ("square", "circle", "rectangle", "hexagon")`) is now handled by `ModelParams`'s `Literal` type. Remove the manual check:

```python
# Remove this block:
if shape is not None:
    if shape not in ("square", "circle", "rectangle", "hexagon"):
        return f"Error: Invalid shape '{shape}'. Use: square, circle, rectangle, hexagon."
    p.shape = shape

# Replace with:
if shape is not None:
    try:
        p.shape = shape
    except Exception as e:
        return f"Error: {e}"
```

**Step 2: Simplify `set_colors` — validation moves to `Colors`**

Remove the manual hex validation loop and replace with direct attribute assignment (since `Colors` has `validate_assignment=True`):

```python
@mcp.tool()
def set_colors(terrain=None, water=None, roads=None, buildings=None, gpx_track=None, map_insert=None) -> str:
    c = state.colors
    for name, value in [
        ("terrain", terrain), ("water", water), ("roads", roads),
        ("buildings", buildings), ("gpx_track", gpx_track), ("map_insert", map_insert),
    ]:
        if value is not None:
            try:
                setattr(c, name, value)
            except Exception as e:
                return f"Error: {e}"
    return f"Colors: {c.as_dict()}"
```

**Step 3: Run full test suite**

```bash
.venv/bin/pytest -v
```

Expected: all PASS.

**Step 4: Commit**

```bash
git add src/topo_shadow_box/tools/model.py
git commit -m "refactor: remove manual validation from set_model_params and set_colors (moved to models)"
```

---

## Task 20: Update `core/map_insert.py` and `exporters/` for typed features

**Files:**
- Modify: `src/topo_shadow_box/core/map_insert.py`
- Modify: `src/topo_shadow_box/exporters/svg.py`

**Step 1: Update `map_insert.py` feature access**

Find any `features["roads"]`, `features["water"]`, `features["buildings"]` — replace with `.roads`, `.water`, `.buildings`.

Find `for coord in feature["coordinates"]:` — replace with `for coord in feature.coordinates:`.

Find `coord["lat"]`, `coord["lon"]` — replace with `coord.lat`, `coord.lon`.

Find `track["points"]`, `point["lat"]`, `point["lon"]` — replace with `track.points`, `point.lat`, `point.lon`.

**Step 2: Update `exporters/svg.py` the same way**

Same pattern: dict-style access → attribute access for features and GPX tracks.

**Step 3: Run full test suite**

```bash
.venv/bin/pytest -v
```

Expected: all PASS.

**Step 4: Commit**

```bash
git add src/topo_shadow_box/core/map_insert.py src/topo_shadow_box/exporters/svg.py
git commit -m "feat: update map_insert and svg exporter to use typed feature models"
```

---

## Task 21: Final cleanup — remove dataclass imports, run full suite

**Files:**
- Modify: `src/topo_shadow_box/state.py`

**Step 1: Remove unused imports from `state.py`**

Remove `from dataclasses import dataclass, field` (no longer used).

**Step 2: Run full test suite**

```bash
.venv/bin/pytest -v
```

Expected: all tests PASS.

**Step 3: Verify the server starts**

```bash
.venv/bin/python -c "from topo_shadow_box.state import state; print('OK:', state)"
```

Expected: prints `OK:` and the SessionState repr without errors.

**Step 4: Commit**

```bash
git add src/topo_shadow_box/state.py
git commit -m "chore: remove unused dataclass imports after pydantic migration"
```

---

## Task 22: Final verification

**Step 1: Run full test suite with verbose output**

```bash
.venv/bin/pytest -v
```

Expected: all tests pass, including the three new test files and all existing geometry tests.

**Step 2: Check test count grew appropriately**

```bash
.venv/bin/pytest --collect-only | tail -5
```

Expected: significantly more tests than before (was ~80, now should be 120+).

**Step 3: Verify no bare dict access remains**

```bash
grep -rn 'features\["roads"\]\|features\["water"\]\|features\["buildings"\]' src/
grep -rn 'track\["points"\]\|point\["lat"\]\|point\["lon"\]' src/
```

Expected: no matches.

**Step 4: Commit if anything was missed**

If any cleanup was found, commit it. Otherwise done.
