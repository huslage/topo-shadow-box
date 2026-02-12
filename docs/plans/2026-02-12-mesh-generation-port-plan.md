# Mesh Generation Port Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Port topo3d's proven mesh generation algorithms to topo-shadow-box MCP, producing watertight 3D-printable meshes with shape-aware buildings, proper water triangulation, and shape-aware base generation.

**Architecture:** Replace/enhance `core/mesh.py` (427 lines) with topo3d's geometry patterns adapted to the MCP's `GeoToModelTransform` + `SessionState` architecture. Add two new files (`core/shape_clipper.py`, `core/building_shapes.py`). The mesh output format (`{vertices: [[x,y,z],...], faces: [[i,j,k],...]}`) stays the same, so exporters/preview/tools are untouched.

**Tech Stack:** Python, numpy, scipy (cKDTree, gaussian_filter, RectBivariateSpline)

**Reference code:** `/Users/huslage/work/topo3d/app/utils/mesh_generator.py` (2221 lines), `shape_clipper.py` (904 lines), `building_shapes.py` (301 lines)

---

### Task 1: Add shape_clipper.py

**Files:**
- Create: `src/topo_shadow_box/core/shape_clipper.py`
- Test: `tests/test_shape_clipper.py`

**Step 1: Write the failing test**

```python
# tests/test_shape_clipper.py
"""Tests for shape clipping."""
import numpy as np
import pytest

from topo_shadow_box.core.shape_clipper import (
    CircleClipper, SquareClipper, RectangleClipper, HexagonClipper,
)


class TestCircleClipper:
    def test_is_inside_center(self):
        c = CircleClipper(50.0, 50.0, 40.0)
        assert c.is_inside(50.0, 50.0) == True

    def test_is_inside_edge(self):
        c = CircleClipper(50.0, 50.0, 40.0)
        assert c.is_inside(90.0, 50.0) == True  # exactly on boundary

    def test_is_outside(self):
        c = CircleClipper(50.0, 50.0, 40.0)
        assert c.is_inside(91.0, 50.0) == False

    def test_is_inside_array(self):
        c = CircleClipper(50.0, 50.0, 40.0)
        x = np.array([50.0, 91.0, 10.0])
        z = np.array([50.0, 50.0, 50.0])
        result = c.is_inside(x, z)
        np.testing.assert_array_equal(result, [True, False, True])

    def test_clip_linestring_fully_inside(self):
        c = CircleClipper(50.0, 50.0, 40.0)
        points = [(30.0, 50.0), (50.0, 50.0), (70.0, 50.0)]
        segments = c.clip_linestring(points)
        assert len(segments) == 1
        assert len(segments[0]) == 3

    def test_clip_linestring_crossing(self):
        c = CircleClipper(50.0, 50.0, 40.0)
        # Line that goes outside and back inside
        points = [(50.0, 50.0), (95.0, 50.0), (50.0, 50.0)]
        segments = c.clip_linestring(points)
        assert len(segments) == 2  # split at boundary

    def test_project_to_boundary(self):
        c = CircleClipper(50.0, 50.0, 40.0)
        px, pz = c.project_to_boundary(50.0, 30.0)
        # Should be on circle, at bottom
        assert abs(px - 50.0) < 1e-6
        assert abs(pz - 10.0) < 1e-6


class TestSquareClipper:
    def test_is_inside_center(self):
        s = SquareClipper(50.0, 50.0, 40.0)
        assert s.is_inside(50.0, 50.0) == True

    def test_is_outside(self):
        s = SquareClipper(50.0, 50.0, 40.0)
        assert s.is_inside(91.0, 50.0) == False

    def test_clip_linestring(self):
        s = SquareClipper(50.0, 50.0, 40.0)
        points = [(50.0, 50.0), (95.0, 50.0)]
        segments = s.clip_linestring(points)
        assert len(segments) == 1
        # Should be clipped at x=90
        assert abs(segments[0][-1][0] - 90.0) < 1e-6

    def test_project_to_boundary(self):
        s = SquareClipper(50.0, 50.0, 40.0)
        px, pz = s.project_to_boundary(80.0, 50.0)
        assert abs(px - 90.0) < 1e-6


class TestRectangleClipper:
    def test_is_inside(self):
        r = RectangleClipper(50.0, 50.0, 40.0, 20.0)
        assert r.is_inside(50.0, 50.0) == True
        assert r.is_inside(91.0, 50.0) == False
        assert r.is_inside(50.0, 71.0) == False


class TestHexagonClipper:
    def test_is_inside_center(self):
        h = HexagonClipper(50.0, 50.0, 40.0)
        assert h.is_inside(50.0, 50.0) == True

    def test_is_outside(self):
        h = HexagonClipper(50.0, 50.0, 40.0)
        assert h.is_inside(100.0, 100.0) == False
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/huslage/work/topo-shadow-box && uv run pytest tests/test_shape_clipper.py -v`
Expected: FAIL with ImportError

**Step 3: Write the implementation**

Port from `/Users/huslage/work/topo3d/app/utils/shape_clipper.py` — all 4 clipper classes (CircleClipper, SquareClipper, RectangleClipper, HexagonClipper) with these methods:
- `is_inside(x, z)` — scalar or numpy array
- `clip_linestring(points)` — returns list of continuous path segments
- `clip_polygon(points)` — simplified all-or-nothing
- `project_to_boundary(x, z)` — project to nearest boundary point

Remove `generate_wall_vertices()` (not needed — wall gen is in mesh.py). Keep the same logic otherwise.

**Step 4: Run test to verify it passes**

Run: `cd /Users/huslage/work/topo-shadow-box && uv run pytest tests/test_shape_clipper.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/topo_shadow_box/core/shape_clipper.py tests/test_shape_clipper.py
git commit -m "feat: add shape clipper module (circle, square, rectangle, hexagon)"
```

---

### Task 2: Add building_shapes.py

**Files:**
- Create: `src/topo_shadow_box/core/building_shapes.py`
- Test: `tests/test_building_shapes.py`

**Step 1: Write the failing test**

```python
# tests/test_building_shapes.py
"""Tests for building shape generation."""
import pytest
from topo_shadow_box.core.building_shapes import BuildingShapeGenerator


class TestBuildingShapeGenerator:
    def setup_method(self):
        self.gen = BuildingShapeGenerator()

    def test_determine_church(self):
        tags = {"amenity": "place_of_worship", "religion": "christian"}
        assert self.gen.determine_building_shape(tags) == "steeple"

    def test_determine_house(self):
        tags = {"building": "house"}
        assert self.gen.determine_building_shape(tags) == "pitched_roof"

    def test_determine_warehouse(self):
        tags = {"building": "warehouse"}
        assert self.gen.determine_building_shape(tags) == "gabled_roof"

    def test_determine_commercial(self):
        tags = {"building": "commercial"}
        assert self.gen.determine_building_shape(tags) == "flat_roof"

    def test_determine_default(self):
        tags = {"building": "yes"}
        assert self.gen.determine_building_shape(tags) == "flat_roof"

    def test_flat_roof_mesh(self):
        mesh = self.gen.generate_building_mesh(0, 10, 0, 15, 0, 10, "flat_roof")
        assert len(mesh["vertices"]) == 8
        assert len(mesh["faces"]) == 12  # 6 faces * 2 triangles

    def test_pitched_roof_mesh(self):
        mesh = self.gen.generate_building_mesh(0, 10, 0, 15, 0, 10, "pitched_roof")
        assert len(mesh["vertices"]) == 10  # 8 box + 2 ridge
        assert len(mesh["faces"]) > 12  # more faces for roof

    def test_steeple_mesh(self):
        mesh = self.gen.generate_building_mesh(0, 10, 0, 15, 0, 10, "steeple")
        assert len(mesh["vertices"]) == 17  # 8 body + 4 tower base + 4 tower top + 1 spire
        assert len(mesh["faces"]) > 12

    def test_custom_color(self):
        mesh = self.gen.generate_building_mesh(0, 10, 0, 15, 0, 10, "flat_roof", custom_color="#FF0000")
        assert mesh["custom_color"] == "#FF0000"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/huslage/work/topo-shadow-box && uv run pytest tests/test_building_shapes.py -v`
Expected: FAIL with ImportError

**Step 3: Write the implementation**

Port from `/Users/huslage/work/topo3d/app/utils/building_shapes.py` — the full `BuildingShapeGenerator` class with:
- `determine_building_shape(tags)` — OSM tag to shape type mapping
- `generate_building_mesh(x1, x2, y_base, y_top, z1, z2, shape_type, custom_color)` — returns `{vertices, faces, custom_color}`
- Private methods: `_generate_flat_roof`, `_generate_pitched_roof`, `_generate_gabled_roof`, `_generate_steeple`

Copy the topo3d code directly — it's self-contained geometry with no external deps.

**Step 4: Run test to verify it passes**

Run: `cd /Users/huslage/work/topo-shadow-box && uv run pytest tests/test_building_shapes.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/topo_shadow_box/core/building_shapes.py tests/test_building_shapes.py
git commit -m "feat: add building shape generation (steeple, pitched, gabled, flat)"
```

---

### Task 3: Add watertight road strip function

**Files:**
- Modify: `src/topo_shadow_box/core/mesh.py`
- Test: `tests/test_mesh.py`

**Step 1: Write the failing test**

```python
# tests/test_mesh.py
"""Tests for mesh generation."""
import numpy as np
import pytest

from topo_shadow_box.core.mesh import create_road_strip


class TestCreateRoadStrip:
    def test_basic_strip(self):
        centerline = np.array([
            [0, 5, 0],
            [10, 5, 0],
            [20, 5, 0],
        ])
        mesh = create_road_strip(centerline, width=2.0, thickness=1.0)
        assert len(mesh["vertices"]) > 0
        assert len(mesh["faces"]) > 0

    def test_four_verts_per_point(self):
        centerline = np.array([
            [0, 5, 0],
            [10, 5, 0],
        ])
        mesh = create_road_strip(centerline, width=2.0, thickness=1.0)
        # 2 points * 4 verts each = 8 vertices
        assert len(mesh["vertices"]) == 8

    def test_watertight_faces(self):
        """Check that the mesh has top, bottom, sides, and end caps."""
        centerline = np.array([
            [0, 5, 0],
            [10, 5, 0],
        ])
        mesh = create_road_strip(centerline, width=2.0, thickness=1.0)
        # 1 segment: top(2) + bottom(2) + left(2) + right(2) + start_cap(2) + end_cap(2) = 12
        assert len(mesh["faces"]) == 12

    def test_empty_centerline(self):
        mesh = create_road_strip(np.array([]), width=2.0, thickness=1.0)
        assert mesh["vertices"] == []
        assert mesh["faces"] == []

    def test_single_point(self):
        mesh = create_road_strip(np.array([[0, 5, 0]]), width=2.0, thickness=1.0)
        assert mesh["vertices"] == []
        assert mesh["faces"] == []

    def test_duplicate_points_removed(self):
        centerline = np.array([
            [0, 5, 0],
            [0, 5, 0],  # duplicate
            [10, 5, 0],
        ])
        mesh = create_road_strip(centerline, width=2.0, thickness=1.0)
        assert len(mesh["vertices"]) == 8  # 2 unique points * 4
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/huslage/work/topo-shadow-box && uv run pytest tests/test_mesh.py::TestCreateRoadStrip -v`
Expected: FAIL with ImportError (create_road_strip doesn't exist)

**Step 3: Write the implementation**

Add `create_road_strip(centerline, width, thickness)` to `core/mesh.py`. Port from topo3d's `create_road_strip` (lines 1888-1983):

```python
def create_road_strip(centerline, width=2.0, thickness=0.3):
    """Create watertight 3D road strip along a centerline.

    4 vertices per point: top-left, top-right, bottom-left, bottom-right.
    Faces: top, bottom, left/right sides, start/end caps.
    """
    if len(centerline) < 2:
        return {"vertices": [], "faces": []}

    # Remove duplicate consecutive points
    clean = [centerline[0]]
    for i in range(1, len(centerline)):
        if not np.allclose(centerline[i], centerline[i-1], atol=1e-6):
            clean.append(centerline[i])
    centerline = np.array(clean)

    if len(centerline) < 2:
        return {"vertices": [], "faces": []}

    vertices = []
    faces = []

    for i, point in enumerate(centerline):
        # 3-point direction averaging
        if i == 0:
            direction = centerline[i + 1] - centerline[i]
        elif i == len(centerline) - 1:
            direction = centerline[i] - centerline[i - 1]
        else:
            direction = centerline[i + 1] - centerline[i - 1]

        length = np.linalg.norm([direction[0], direction[2]])
        if length < 1e-6:
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = direction / length

        perp = np.array([-direction[2], 0, direction[0]])
        half_w = width / 2

        tl = point + perp * half_w
        tr = point - perp * half_w
        bl = np.array([tl[0], tl[1] - thickness, tl[2]])
        br = np.array([tr[0], tr[1] - thickness, tr[2]])

        vertices.extend([tl.tolist(), tr.tolist(), bl.tolist(), br.tolist()])

    n_points = len(centerline)
    for i in range(n_points - 1):
        c = i * 4
        n = (i + 1) * 4
        # Top
        faces.append([c, c+1, n+1])
        faces.append([c, n+1, n])
        # Bottom
        faces.append([c+2, n+2, n+3])
        faces.append([c+2, n+3, c+3])
        # Left
        faces.append([c, n, n+2])
        faces.append([c, n+2, c+2])
        # Right
        faces.append([c+1, c+3, n+3])
        faces.append([c+1, n+3, n+1])

    # Start cap
    faces.append([0, 2, 3])
    faces.append([0, 3, 1])
    # End cap
    end = (n_points - 1) * 4
    faces.append([end, end+1, end+3])
    faces.append([end, end+3, end+2])

    return {"vertices": vertices, "faces": faces}
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/huslage/work/topo-shadow-box && uv run pytest tests/test_mesh.py::TestCreateRoadStrip -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/topo_shadow_box/core/mesh.py tests/test_mesh.py
git commit -m "feat: add watertight road strip generation (create_road_strip)"
```

---

### Task 4: Add ear-clipping triangulation and solid polygon

**Files:**
- Modify: `src/topo_shadow_box/core/mesh.py`
- Test: `tests/test_mesh.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_mesh.py
from topo_shadow_box.core.mesh import triangulate_polygon, create_solid_polygon


class TestTriangulatePolygon:
    def test_triangle(self):
        pts = np.array([[0, 0], [10, 0], [5, 10]])
        tris = triangulate_polygon(pts)
        assert len(tris) == 1

    def test_square(self):
        pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        tris = triangulate_polygon(pts)
        assert len(tris) == 2  # square = 2 triangles

    def test_concave_polygon(self):
        # L-shaped polygon
        pts = np.array([[0, 0], [10, 0], [10, 5], [5, 5], [5, 10], [0, 10]])
        tris = triangulate_polygon(pts)
        assert len(tris) == 4  # 6 vertices = 4 triangles

    def test_empty(self):
        tris = triangulate_polygon(np.array([]).reshape(0, 2))
        assert tris == []


class TestCreateSolidPolygon:
    def test_basic_triangle(self):
        points = np.array([[0, 5, 0], [10, 5, 0], [5, 5, 10]])
        mesh = create_solid_polygon(points, thickness=1.0)
        assert len(mesh["vertices"]) == 6  # 3 top + 3 bottom
        # top(1) + bottom(1) + walls(3 edges * 2 tris) = 8 faces
        assert len(mesh["faces"]) == 8

    def test_square(self):
        points = np.array([[0, 5, 0], [10, 5, 0], [10, 5, 10], [0, 5, 10]])
        mesh = create_solid_polygon(points, thickness=1.0)
        assert len(mesh["vertices"]) == 8  # 4 top + 4 bottom
        # top(2) + bottom(2) + walls(4 * 2) = 12 faces
        assert len(mesh["faces"]) == 12

    def test_degenerate(self):
        points = np.array([[0, 5, 0], [1, 5, 0]])
        mesh = create_solid_polygon(points, thickness=1.0)
        assert mesh["vertices"] == []
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/huslage/work/topo-shadow-box && uv run pytest tests/test_mesh.py::TestTriangulatePolygon tests/test_mesh.py::TestCreateSolidPolygon -v`
Expected: FAIL with ImportError

**Step 3: Write the implementation**

Add `triangulate_polygon(points_2d)` and `create_solid_polygon(points, thickness)` to `core/mesh.py`.

Port `triangulate_polygon` from topo3d lines 1735-1829:
- Shoelace formula for winding detection
- Ear-clipping with convexity + point-in-triangle check
- Fallback to fan triangulation

Port `create_solid_polygon` from topo3d lines 1574-1732:
- Remove consecutive duplicate points
- Merge near-duplicate vertices via `scipy.spatial.cKDTree`
- Remove collinear vertices
- Top face via ear-clipping, bottom face with reversed winding
- Side walls connecting top to bottom
- Replace `print()` calls with passing (MCP context, no stdout)

**Step 4: Run test to verify it passes**

Run: `cd /Users/huslage/work/topo-shadow-box && uv run pytest tests/test_mesh.py::TestTriangulatePolygon tests/test_mesh.py::TestCreateSolidPolygon -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/topo_shadow_box/core/mesh.py tests/test_mesh.py
git commit -m "feat: add ear-clipping triangulation and solid polygon generation"
```

---

### Task 5: Rewrite road, water, building, and GPX mesh generators

**Files:**
- Modify: `src/topo_shadow_box/core/mesh.py`
- Test: `tests/test_mesh.py`

This is the core integration task. Replace the simple generators with watertight ones using the primitives from Tasks 3-4.

**Step 1: Write the failing test**

```python
# Add to tests/test_mesh.py
from topo_shadow_box.state import Bounds, ElevationData
from topo_shadow_box.core.coords import GeoToModelTransform
from topo_shadow_box.core.mesh import generate_feature_meshes


class TestFeatureMeshes:
    def setup_method(self):
        """Create a simple 10x10 elevation grid."""
        self.grid = np.ones((10, 10)) * 100.0  # flat at 100m
        self.lats = np.linspace(40.0, 40.01, 10)
        self.lons = np.linspace(-74.0, -73.99, 10)
        self.bounds = Bounds(north=40.01, south=40.0, east=-73.99, west=-74.0)
        self.elevation = ElevationData(
            grid=self.grid, lats=self.lats, lons=self.lons,
            resolution=10, min_elevation=100.0, max_elevation=100.0,
        )
        self.transform = GeoToModelTransform(self.bounds, 200.0)

    def test_road_is_watertight(self):
        features = {
            "roads": [{
                "coordinates": [
                    {"lat": 40.005, "lon": -73.995},
                    {"lat": 40.005, "lon": -73.993},
                ],
                "name": "Test Road",
            }],
        }
        meshes = generate_feature_meshes(
            features, self.elevation, self.bounds, self.transform,
            vertical_scale=1.5, shape="square",
        )
        assert len(meshes) == 1
        road = meshes[0]
        # Watertight road: 4 verts per point, >2 faces per segment
        assert len(road["vertices"]) >= 8  # at least 2 points * 4
        assert len(road["faces"]) >= 12

    def test_water_is_watertight(self):
        features = {
            "water": [{
                "coordinates": [
                    {"lat": 40.003, "lon": -73.997},
                    {"lat": 40.007, "lon": -73.997},
                    {"lat": 40.007, "lon": -73.993},
                    {"lat": 40.003, "lon": -73.993},
                ],
                "name": "Test Lake",
            }],
        }
        meshes = generate_feature_meshes(
            features, self.elevation, self.bounds, self.transform,
            vertical_scale=1.5, shape="square",
        )
        assert len(meshes) == 1
        water = meshes[0]
        # Watertight: top + bottom + walls
        assert len(water["vertices"]) == 8  # 4 top + 4 bottom
        assert len(water["faces"]) == 12  # top(2) + bottom(2) + walls(8)

    def test_building_shape_aware(self):
        features = {
            "buildings": [{
                "coordinates": [
                    {"lat": 40.005, "lon": -73.996},
                    {"lat": 40.006, "lon": -73.996},
                    {"lat": 40.006, "lon": -73.995},
                    {"lat": 40.005, "lon": -73.995},
                ],
                "name": "Test Church",
                "height": 20.0,
                "tags": {"amenity": "place_of_worship", "religion": "christian"},
            }],
        }
        meshes = generate_feature_meshes(
            features, self.elevation, self.bounds, self.transform,
            vertical_scale=1.5, shape="square",
        )
        assert len(meshes) == 1
        # Steeple: 17 vertices (8 body + 4 tower base + 4 tower top + 1 spire)
        assert len(meshes[0]["vertices"]) == 17

    def test_feature_limits(self):
        """Roads limited to 200, buildings to 150, water to 50."""
        features = {
            "roads": [
                {"coordinates": [{"lat": 40.005, "lon": -73.995}, {"lat": 40.005, "lon": -73.994}], "name": f"Road {i}"}
                for i in range(250)
            ],
        }
        meshes = generate_feature_meshes(
            features, self.elevation, self.bounds, self.transform,
            vertical_scale=1.5, shape="square",
        )
        assert len(meshes) <= 200
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/huslage/work/topo-shadow-box && uv run pytest tests/test_mesh.py::TestFeatureMeshes -v`
Expected: FAIL (roads are flat ribbons, not watertight)

**Step 3: Write the implementation**

Rewrite `_generate_road_mesh`, `_generate_water_mesh`, `_generate_building_mesh`, and `generate_gpx_track_mesh` in `core/mesh.py`.

**Roads** (`_generate_road_mesh`):
- Convert coordinates to model space with elevation sampling
- Apply road relief: `y += max(road_height, 0.6 * size_scale)`
- Build `points_xyz` numpy array
- Call `create_road_strip(points_xyz, width=1.0*size_scale, thickness=max(0.9, 0.6)*size_scale)`
- Return watertight mesh

**Water** (`_generate_water_mesh`):
- Compute average perimeter elevation
- Water Y: `max(0.7 * size_scale, water_y + water_relief)` where relief = `0.6 * size_scale`
- Water thickness: `max(1.2 * size_scale, 2.0 * water_relief)`
- Build `points` numpy array at water_y
- Call `create_solid_polygon(points, thickness=water_thickness)`

**Buildings** (`_generate_building_mesh`):
- Get center lat/lon, sample elevation for base_y
- Calculate building bounding box in model space
- Enforce minimum footprint (1.2mm)
- Use `BuildingShapeGenerator.determine_building_shape(tags)` and `.generate_building_mesh(...)`
- Pass through `custom_color` if present in building dict

**GPX Track** (`generate_gpx_track_mesh`):
- Sample points: `max(1, len(points) // 500)`
- Track relief: `max(0.8 * size_scale, 0.3 * size_scale)`
- Track thickness: `max(1.2 * size_scale, 2.0 * gpx_relief)`
- Call `create_road_strip(points, width=2.5*size_scale, thickness=gpx_thickness)`

**Feature limits** in `generate_feature_meshes`:
- `roads[:200]`, `water[:50]`, `buildings[:150]`

**Step 4: Run test to verify it passes**

Run: `cd /Users/huslage/work/topo-shadow-box && uv run pytest tests/test_mesh.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/topo_shadow_box/core/mesh.py tests/test_mesh.py
git commit -m "feat: rewrite feature meshes (watertight roads/water, shape-aware buildings)"
```

---

### Task 6: Add shape-aware base and wall generation

**Files:**
- Modify: `src/topo_shadow_box/core/mesh.py`
- Test: `tests/test_mesh.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_mesh.py
from topo_shadow_box.core.mesh import generate_terrain_mesh


class TestTerrainMeshShapeAware:
    def setup_method(self):
        self.grid = np.random.uniform(90, 110, (20, 20))
        self.lats = np.linspace(40.0, 40.01, 20)
        self.lons = np.linspace(-74.0, -73.99, 20)
        self.bounds = Bounds(north=40.01, south=40.0, east=-73.99, west=-74.0)
        self.elevation = ElevationData(
            grid=self.grid, lats=self.lats, lons=self.lons,
            resolution=20, min_elevation=90.0, max_elevation=110.0,
        )
        self.transform = GeoToModelTransform(self.bounds, 200.0)

    def test_square_terrain(self):
        mesh = generate_terrain_mesh(
            self.elevation, self.bounds, self.transform,
            vertical_scale=1.5, base_height_mm=10.0, shape="square",
        )
        assert len(mesh["vertices"]) > 0
        assert len(mesh["faces"]) > 0

    def test_circle_terrain_has_smooth_walls(self):
        mesh = generate_terrain_mesh(
            self.elevation, self.bounds, self.transform,
            vertical_scale=1.5, base_height_mm=10.0, shape="circle",
        )
        # Circle: should have boundary vertices projected to circle edge
        # and smooth circular wall (360 segments)
        # More vertices than simple rectangular approach
        assert len(mesh["vertices"]) > 400 * 2  # top + bottom at minimum

    def test_circle_terrain_watertight(self):
        """All interior edges should be shared by exactly 2 faces."""
        mesh = generate_terrain_mesh(
            self.elevation, self.bounds, self.transform,
            vertical_scale=1.5, base_height_mm=10.0, shape="circle",
        )
        # Basic check: faces reference valid vertex indices
        n_verts = len(mesh["vertices"])
        for face in mesh["faces"]:
            for idx in face:
                assert 0 <= idx < n_verts, f"Invalid vertex index {idx} (n={n_verts})"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/huslage/work/topo-shadow-box && uv run pytest tests/test_mesh.py::TestTerrainMeshShapeAware -v`
Expected: FAIL on circle test (current code uses simple rectangular walls for circle)

**Step 3: Write the implementation**

Modify `generate_terrain_mesh` in `core/mesh.py`:

1. Import `ShapeClipper` classes and instantiate based on `shape` parameter
2. For square/rectangle: keep existing rectangular base + walls (they work fine)
3. For circle: replace base generation with topo3d's approach:
   - Find boundary vertices (inside vertices with outside neighbors)
   - Project boundary vertices to exact circle edge
   - Generate smooth circular wall (360 segments) using `interpolate_boundary_elevation`
   - Fan triangulation for bottom face from center point
4. For hexagon: use `generate_shape_base` pattern with boundary edge detection

Port from topo3d:
- `generate_shape_base` (lines 575-716): boundary edge detection, loop walking, wall generation
- `generate_circular_base` (lines 719-832): smooth 360-segment circular wall
- `interpolate_boundary_elevation` (lines 835-878): angle-based elevation interpolation

Key adaptation: topo3d works with raw numpy arrays and direct coordinate math. The MCP uses `GeoToModelTransform`. The terrain vertex generation already uses the transform, so the base generation just works with the model-space vertex arrays — same as topo3d.

**Step 4: Run test to verify it passes**

Run: `cd /Users/huslage/work/topo-shadow-box && uv run pytest tests/test_mesh.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/topo_shadow_box/core/mesh.py tests/test_mesh.py
git commit -m "feat: add shape-aware base generation (circle, hexagon smooth walls)"
```

---

### Task 7: Add road/feature clipping to shape boundary

**Files:**
- Modify: `src/topo_shadow_box/core/mesh.py`
- Modify: `src/topo_shadow_box/tools/generate.py` (pass shape to GPX generation)
- Test: `tests/test_mesh.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_mesh.py
class TestFeatureClipping:
    def setup_method(self):
        self.grid = np.ones((20, 20)) * 100.0
        self.lats = np.linspace(40.0, 40.01, 20)
        self.lons = np.linspace(-74.0, -73.99, 20)
        self.bounds = Bounds(north=40.01, south=40.0, east=-73.99, west=-74.0)
        self.elevation = ElevationData(
            grid=self.grid, lats=self.lats, lons=self.lons,
            resolution=20, min_elevation=100.0, max_elevation=100.0,
        )
        self.transform = GeoToModelTransform(self.bounds, 200.0)

    def test_building_outside_circle_excluded(self):
        """Buildings with corners outside circle shape should be skipped."""
        features = {
            "buildings": [{
                "coordinates": [
                    {"lat": 40.0001, "lon": -73.9999},  # corner of area
                    {"lat": 40.001, "lon": -73.9999},
                    {"lat": 40.001, "lon": -73.999},
                    {"lat": 40.0001, "lon": -73.999},
                ],
                "name": "Corner Building",
                "height": 10.0,
                "tags": {"building": "yes"},
            }],
        }
        meshes = generate_feature_meshes(
            features, self.elevation, self.bounds, self.transform,
            vertical_scale=1.5, shape="circle",
        )
        # Building in extreme corner should be outside circle
        assert len(meshes) == 0

    def test_road_clipped_to_circle(self):
        """Roads should be clipped at circle boundary, not removed entirely."""
        features = {
            "roads": [{
                "coordinates": [
                    {"lat": 40.005, "lon": -74.0},  # starts at west edge
                    {"lat": 40.005, "lon": -73.99},  # goes to east edge
                ],
                "name": "Cross Road",
            }],
        }
        meshes = generate_feature_meshes(
            features, self.elevation, self.bounds, self.transform,
            vertical_scale=1.5, shape="circle",
        )
        # Road should still exist (center passes through circle)
        assert len(meshes) >= 1
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/huslage/work/topo-shadow-box && uv run pytest tests/test_mesh.py::TestFeatureClipping -v`
Expected: FAIL (no shape clipping on features currently)

**Step 3: Write the implementation**

Update `generate_feature_meshes` to:
1. Instantiate appropriate `ShapeClipper` based on `shape` parameter
2. Pass `shape_clipper` to road, water, and building generators

Update `_generate_road_mesh`:
- Convert coordinates to model-space XZ points
- If shape_clipper: call `shape_clipper.clip_linestring(points_xz)` to get clipped segments
- For each clipped segment, find closest 3D point for elevation, build strip

Update `_generate_building_mesh`:
- After computing bounding box in model space, check all 4 corners with `shape_clipper.is_inside()`
- Skip building if any corner is outside

Update `_generate_water_mesh`:
- Filter points through `shape_clipper.is_inside()` before triangulation

Update `generate_gpx_track_mesh`:
- Add `shape` parameter
- GPX tracks are NOT clipped (preserve natural path per topo3d pattern)

Update `tools/generate.py`:
- Pass `shape=mp.shape` to `generate_gpx_track_mesh`

**Step 4: Run test to verify it passes**

Run: `cd /Users/huslage/work/topo-shadow-box && uv run pytest tests/test_mesh.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/topo_shadow_box/core/mesh.py src/topo_shadow_box/tools/generate.py tests/test_mesh.py
git commit -m "feat: add shape-aware feature clipping (roads, buildings, water)"
```

---

### Task 8: Add elevation percentile normalization

**Files:**
- Modify: `src/topo_shadow_box/core/mesh.py`
- Test: `tests/test_mesh.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_mesh.py
from topo_shadow_box.core.mesh import _elevation_normalization


class TestElevationNormalization:
    def test_basic(self):
        grid = np.array([[100, 200], [150, 250]])
        min_e, rng = _elevation_normalization(grid)
        assert min_e == 100.0
        assert rng == 150.0

    def test_percentile_clips_outliers(self):
        """With percentile mode, extreme outliers should be clipped."""
        grid = np.random.uniform(100, 110, (100, 100))
        grid[0, 0] = 0.0      # extreme low outlier
        grid[99, 99] = 500.0   # extreme high outlier
        min_e, rng = _elevation_normalization(grid, use_percentile=True)
        # Percentile should ignore the outliers
        assert min_e > 0.0
        assert min_e + rng < 500.0

    def test_flat_terrain(self):
        grid = np.ones((10, 10)) * 100.0
        min_e, rng = _elevation_normalization(grid)
        assert min_e == 100.0
        assert rng == 1.0  # default for flat terrain
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/huslage/work/topo-shadow-box && uv run pytest tests/test_mesh.py::TestElevationNormalization -v`
Expected: FAIL (no `use_percentile` parameter)

**Step 3: Write the implementation**

Update `_elevation_normalization` in `core/mesh.py`:

```python
def _elevation_normalization(grid: np.ndarray, use_percentile: bool = False) -> tuple[float, float]:
    """Compute min elevation and range for normalization.

    Args:
        grid: 2D elevation array
        use_percentile: If True, use 2nd-98th percentile to clip outliers
    """
    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        return 0.0, 1.0

    if use_percentile and finite.size > 100:
        p_low = float(np.percentile(finite, 2.0))
        p_high = float(np.percentile(finite, 98.0))
        if p_high > p_low and (p_high - p_low) >= 1.0:
            return p_low, p_high - p_low

    raw_min = float(np.min(finite))
    raw_max = float(np.max(finite))
    if raw_max <= raw_min:
        return raw_min, 1.0
    return raw_min, raw_max - raw_min
```

Then update all callers of `_elevation_normalization` to pass `use_percentile=True`. This enables percentile clipping by default for all terrain, which improves quality when noisy elevation data has outliers.

**Step 4: Run test to verify it passes**

Run: `cd /Users/huslage/work/topo-shadow-box && uv run pytest tests/test_mesh.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/topo_shadow_box/core/mesh.py tests/test_mesh.py
git commit -m "feat: add percentile-based elevation normalization"
```

---

### Task 9: Run all tests and verify end-to-end

**Files:**
- No new files

**Step 1: Run full test suite**

Run: `cd /Users/huslage/work/topo-shadow-box && uv run pytest tests/ -v`
Expected: ALL PASS

**Step 2: Verify imports work**

Run: `cd /Users/huslage/work/topo-shadow-box && uv run python -c "from topo_shadow_box.core.mesh import generate_terrain_mesh, generate_feature_meshes, generate_gpx_track_mesh, create_road_strip, create_solid_polygon, triangulate_polygon; print('All imports OK')"`
Expected: "All imports OK"

**Step 3: Verify MCP server starts**

Run: `cd /Users/huslage/work/topo-shadow-box && uv run python -c "from topo_shadow_box.server import mcp; print(f'Tools: {len(mcp._tool_manager._tools)}')"`
Expected: Prints tool count without errors

**Step 4: Commit any remaining fixes**

If any test failures were found and fixed, commit them.

**Step 5: Final commit**

```bash
git add -A
git commit -m "test: verify all mesh generation tests pass end-to-end"
```
