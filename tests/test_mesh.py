"""Tests for mesh generation utilities."""

import numpy as np

from topo_shadow_box.core.mesh import (
    _elevation_normalization,
    create_gpx_cylinder_track,
    create_road_strip,
    create_solid_polygon,
    generate_feature_meshes,
    generate_gpx_track_mesh,
    generate_terrain_mesh,
    triangulate_polygon,
)
from topo_shadow_box.state import Bounds, ElevationData
from topo_shadow_box.core.coords import GeoToModelTransform


class TestCreateRoadStrip:
    """Tests for the watertight road strip generator."""

    def test_basic_strip(self):
        """3-point centerline produces vertices and faces."""
        centerline = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
        result = create_road_strip(centerline, width=2.0, thickness=0.3)
        assert len(result["vertices"]) > 0
        assert len(result["faces"]) > 0

    def test_four_verts_per_point(self):
        """2-point centerline produces 4 vertices per point = 8 total."""
        centerline = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
        result = create_road_strip(centerline, width=2.0, thickness=0.3)
        assert len(result["vertices"]) == 8

    def test_watertight_faces(self):
        """1 segment (2 points) produces 12 faces: 8 per segment + 2 start cap + 2 end cap."""
        centerline = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
        result = create_road_strip(centerline, width=2.0, thickness=0.3)
        # 8 faces for the segment (top 2 + bottom 2 + left 2 + right 2)
        # + 2 start cap + 2 end cap = 12
        assert len(result["faces"]) == 12

    def test_empty_centerline(self):
        """Empty centerline returns empty result."""
        result = create_road_strip([], width=2.0, thickness=0.3)
        assert result["vertices"] == []
        assert result["faces"] == []

    def test_single_point(self):
        """Single point centerline returns empty result (need at least 2 points)."""
        centerline = [[0.0, 0.0, 0.0]]
        result = create_road_strip(centerline, width=2.0, thickness=0.3)
        assert result["vertices"] == []
        assert result["faces"] == []

    def test_duplicate_points_removed(self):
        """Duplicate consecutive points are removed; 2 unique points produce 8 vertices."""
        centerline = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],  # duplicate
            [0.0, 0.0, 0.0],  # duplicate
            [1.0, 0.0, 0.0],
        ]
        result = create_road_strip(centerline, width=2.0, thickness=0.3)
        assert len(result["vertices"]) == 8


class TestTriangulatePolygon:
    """Tests for the ear-clipping polygon triangulator."""

    def test_triangle(self):
        """A triangle produces exactly 1 triangle."""
        pts = np.array([[0, 0], [10, 0], [5, 10]])
        tris = triangulate_polygon(pts)
        assert len(tris) == 1

    def test_square(self):
        """A square produces exactly 2 triangles."""
        pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        tris = triangulate_polygon(pts)
        assert len(tris) == 2

    def test_concave_polygon(self):
        """A concave L-shape with 6 vertices produces 4 triangles."""
        pts = np.array([[0, 0], [10, 0], [10, 5], [5, 5], [5, 10], [0, 10]])
        tris = triangulate_polygon(pts)
        assert len(tris) == 4  # 6 vertices = 4 triangles

    def test_empty(self):
        """Empty input returns empty list."""
        tris = triangulate_polygon(np.array([]).reshape(0, 2))
        assert tris == []


class TestCreateSolidPolygon:
    """Tests for the watertight solid polygon generator."""

    def test_basic_triangle(self):
        """Triangle produces 6 vertices (3 top + 3 bottom) and 8 faces."""
        points = np.array([[0, 5, 0], [10, 5, 0], [5, 5, 10]])
        mesh = create_solid_polygon(points, thickness=1.0)
        assert len(mesh["vertices"]) == 6  # 3 top + 3 bottom
        assert len(mesh["faces"]) == 8  # top(1) + bottom(1) + walls(3*2)

    def test_square(self):
        """Square produces 8 vertices and 12 faces."""
        points = np.array([[0, 5, 0], [10, 5, 0], [10, 5, 10], [0, 5, 10]])
        mesh = create_solid_polygon(points, thickness=1.0)
        assert len(mesh["vertices"]) == 8
        assert len(mesh["faces"]) == 12  # top(2) + bottom(2) + walls(4*2)

    def test_degenerate(self):
        """Fewer than 3 points returns empty mesh."""
        points = np.array([[0, 5, 0], [1, 5, 0]])
        mesh = create_solid_polygon(points, thickness=1.0)
        assert mesh["vertices"] == []


class TestFeatureMeshes:
    def setup_method(self):
        self.grid = np.ones((10, 10)) * 100.0
        self.lats = np.linspace(40.0, 40.01, 10)
        self.lons = np.linspace(-74.0, -73.99, 10)
        self.bounds = Bounds(north=40.01, south=40.0, east=-73.99, west=-74.0)
        self.elevation = ElevationData(
            grid=self.grid, lats=self.lats, lons=self.lons,
            resolution=10, min_elevation=100.0, max_elevation=100.0,
        )
        self.transform = GeoToModelTransform(self.bounds, 200.0)

    def test_road_is_watertight(self):
        features = {"roads": [{"coordinates": [
            {"lat": 40.005, "lon": -73.995}, {"lat": 40.005, "lon": -73.993}
        ], "name": "Test Road"}]}
        meshes = generate_feature_meshes(
            features, self.elevation, self.bounds, self.transform, 1.5, "square")
        assert len(meshes) == 1
        assert len(meshes[0].vertices) >= 8
        assert len(meshes[0].faces) >= 12

    def test_water_is_watertight(self):
        features = {"water": [{"coordinates": [
            {"lat": 40.003, "lon": -73.997}, {"lat": 40.007, "lon": -73.997},
            {"lat": 40.007, "lon": -73.993}, {"lat": 40.003, "lon": -73.993}
        ], "name": "Test Lake"}]}
        meshes = generate_feature_meshes(
            features, self.elevation, self.bounds, self.transform, 1.5, "square")
        assert len(meshes) == 1
        assert len(meshes[0].vertices) == 8
        assert len(meshes[0].faces) == 12

    def test_building_shape_aware(self):
        features = {"buildings": [{"coordinates": [
            {"lat": 40.005, "lon": -73.996}, {"lat": 40.006, "lon": -73.996},
            {"lat": 40.006, "lon": -73.995}, {"lat": 40.005, "lon": -73.995}
        ], "name": "Church", "height": 20.0,
           "tags": {"amenity": "place_of_worship", "religion": "christian"}}]}
        meshes = generate_feature_meshes(
            features, self.elevation, self.bounds, self.transform, 1.5, "square")
        assert len(meshes) == 1
        assert len(meshes[0].vertices) == 17  # steeple

    def test_feature_limits(self):
        features = {"roads": [
            {"coordinates": [{"lat": 40.005, "lon": -73.995}, {"lat": 40.005, "lon": -73.994}], "name": f"Road {i}"}
            for i in range(250)
        ]}
        meshes = generate_feature_meshes(
            features, self.elevation, self.bounds, self.transform, 1.5, "square")
        assert len(meshes) <= 200


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
            self.elevation, self.bounds, self.transform, 1.5, 10.0, "square")
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_circle_terrain_valid_indices(self):
        mesh = generate_terrain_mesh(
            self.elevation, self.bounds, self.transform, 1.5, 10.0, "circle")
        n_verts = len(mesh.vertices)
        for face in mesh.faces:
            for idx in face:
                assert 0 <= idx < n_verts

    def test_circle_terrain_has_faces(self):
        ci = generate_terrain_mesh(
            self.elevation, self.bounds, self.transform, 1.5, 10.0, "circle")
        # Circle clips to boundary, resulting in a valid non-empty mesh
        assert len(ci.vertices) > 0
        assert len(ci.faces) > 0

    def test_hexagon_terrain_valid_indices(self):
        mesh = generate_terrain_mesh(
            self.elevation, self.bounds, self.transform, 1.5, 10.0, "hexagon")
        n_verts = len(mesh.vertices)
        for face in mesh.faces:
            for idx in face:
                assert 0 <= idx < n_verts

    def test_hexagon_terrain_has_faces(self):
        mesh = generate_terrain_mesh(
            self.elevation, self.bounds, self.transform, 1.5, 10.0, "hexagon")
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0


class TestFeatureClipping:
    """Tests for clipping features to shape boundaries."""

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
        """A building in the far corner should be excluded by a circle clipper."""
        features = {"buildings": [{"coordinates": [
            {"lat": 40.0001, "lon": -73.9999},
            {"lat": 40.001, "lon": -73.9999},
            {"lat": 40.001, "lon": -73.999},
            {"lat": 40.0001, "lon": -73.999},
        ], "name": "Corner", "height": 10.0, "tags": {"building": "yes"}}]}
        meshes = generate_feature_meshes(
            features, self.elevation, self.bounds, self.transform, 1.5, "circle")
        assert len(meshes) == 0

    def test_building_inside_circle_kept(self):
        """A building near the center should be kept by a circle clipper."""
        features = {"buildings": [{"coordinates": [
            {"lat": 40.004, "lon": -73.996},
            {"lat": 40.006, "lon": -73.996},
            {"lat": 40.006, "lon": -73.994},
            {"lat": 40.004, "lon": -73.994},
        ], "name": "Center", "height": 10.0, "tags": {"building": "yes"}}]}
        meshes = generate_feature_meshes(
            features, self.elevation, self.bounds, self.transform, 1.5, "circle")
        assert len(meshes) == 1

    def test_road_clipped_to_circle(self):
        """A road crossing the full model width should be clipped to the circle."""
        features = {"roads": [{"coordinates": [
            {"lat": 40.005, "lon": -74.0},
            {"lat": 40.005, "lon": -73.99},
        ], "name": "Cross Road"}]}
        meshes = generate_feature_meshes(
            features, self.elevation, self.bounds, self.transform, 1.5, "circle")
        assert len(meshes) >= 1

    def test_road_fully_outside_circle_excluded(self):
        """A road entirely in the corner should produce no meshes in a circle."""
        features = {"roads": [{"coordinates": [
            {"lat": 40.0001, "lon": -73.9999},
            {"lat": 40.0005, "lon": -73.9995},
        ], "name": "Corner Road"}]}
        meshes = generate_feature_meshes(
            features, self.elevation, self.bounds, self.transform, 1.5, "circle")
        assert len(meshes) == 0

    def test_water_outside_circle_excluded(self):
        """Water body in the corner should be excluded by circle clipping."""
        features = {"water": [{"coordinates": [
            {"lat": 40.0001, "lon": -73.9999},
            {"lat": 40.001, "lon": -73.9999},
            {"lat": 40.001, "lon": -73.999},
            {"lat": 40.0001, "lon": -73.999},
        ], "name": "Corner Pond"}]}
        meshes = generate_feature_meshes(
            features, self.elevation, self.bounds, self.transform, 1.5, "circle")
        assert len(meshes) == 0

    def test_water_inside_circle_kept(self):
        """Water body near center should be kept by circle clipping."""
        features = {"water": [{"coordinates": [
            {"lat": 40.004, "lon": -73.996},
            {"lat": 40.006, "lon": -73.996},
            {"lat": 40.006, "lon": -73.994},
            {"lat": 40.004, "lon": -73.994},
        ], "name": "Center Lake"}]}
        meshes = generate_feature_meshes(
            features, self.elevation, self.bounds, self.transform, 1.5, "circle")
        assert len(meshes) == 1

    def test_square_shape_no_clipping(self):
        """Square shape should not clip anything (all features kept)."""
        features = {"buildings": [{"coordinates": [
            {"lat": 40.0001, "lon": -73.9999},
            {"lat": 40.001, "lon": -73.9999},
            {"lat": 40.001, "lon": -73.999},
            {"lat": 40.0001, "lon": -73.999},
        ], "name": "Corner", "height": 10.0, "tags": {"building": "yes"}}]}
        meshes = generate_feature_meshes(
            features, self.elevation, self.bounds, self.transform, 1.5, "square")
        assert len(meshes) == 1

    def test_building_outside_hexagon_excluded(self):
        """A building in the corner should be excluded by hexagon clipper."""
        features = {"buildings": [{"coordinates": [
            {"lat": 40.0001, "lon": -73.9999},
            {"lat": 40.001, "lon": -73.9999},
            {"lat": 40.001, "lon": -73.999},
            {"lat": 40.0001, "lon": -73.999},
        ], "name": "Corner", "height": 10.0, "tags": {"building": "yes"}}]}
        meshes = generate_feature_meshes(
            features, self.elevation, self.bounds, self.transform, 1.5, "hexagon")
        assert len(meshes) == 0

    def test_gpx_track_not_clipped(self):
        """GPX tracks should not be clipped, even with a circle shape."""
        tracks = [{"points": [
            {"lat": 40.005, "lon": -74.0, "elevation": 100.0},
            {"lat": 40.005, "lon": -73.99, "elevation": 100.0},
        ]}]
        result = generate_gpx_track_mesh(
            tracks, self.elevation, self.bounds, self.transform, 1.5, shape="circle")
        assert result is not None
        assert len(result.vertices) > 0


class TestElevationNormalization:
    """Tests for percentile-based elevation normalization."""

    def test_basic(self):
        grid = np.array([[100, 200], [150, 250]])
        min_e, rng = _elevation_normalization(grid)
        assert min_e == 100.0
        assert rng == 150.0

    def test_percentile_clips_outliers(self):
        grid = np.random.uniform(100, 110, (100, 100))
        grid[0, 0] = 0.0
        grid[99, 99] = 500.0
        min_e, rng = _elevation_normalization(grid, use_percentile=True)
        assert min_e > 0.0
        assert min_e + rng < 500.0

    def test_flat_terrain(self):
        grid = np.ones((10, 10)) * 100.0
        min_e, rng = _elevation_normalization(grid)
        assert min_e == 100.0
        assert rng == 1.0

    def test_percentile_disabled(self):
        grid = np.random.uniform(100, 110, (100, 100))
        grid[0, 0] = 0.0
        grid[99, 99] = 500.0
        min_e, rng = _elevation_normalization(grid, use_percentile=False)
        assert min_e == 0.0
        assert abs(rng - 500.0) < 0.1


def test_elevation_normalization_is_deterministic():
    """Same grid always produces the same normalization values."""
    from topo_shadow_box.core.mesh import _elevation_normalization
    import numpy as np

    rng = np.random.default_rng(42)
    grid = rng.random((100, 100)) * 1000
    result1 = _elevation_normalization(grid)
    result2 = _elevation_normalization(grid)
    assert result1 == result2


def test_elevation_normalization_nonzero_range_for_varied_grid():
    """A grid with varied elevations should have positive normalization range."""
    from topo_shadow_box.core.mesh import _elevation_normalization
    import numpy as np

    grid = np.linspace(0, 500, 200 * 200).reshape(200, 200)
    min_e, range_e = _elevation_normalization(grid)
    assert range_e > 0, "Non-flat grid should have positive range"


class TestCreateGpxCylinderTrack:
    """Tests for the watertight GPX cylinder tube generator."""

    def _directed_edge_counts(self, faces):
        """Return dict of directed edge -> count from face list."""
        counts = {}
        for face in faces:
            a, b, c = face[0], face[1], face[2]
            for e in [(a, b), (b, c), (c, a)]:
                counts[e] = counts.get(e, 0) + 1
        return counts

    def test_manifold_edges(self):
        """2-point centerline: every directed edge appears exactly once (manifold)."""
        centerline = np.array([
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ])
        result = create_gpx_cylinder_track(centerline, radius=1.0, n_sides=8)
        counts = self._directed_edge_counts(result["faces"])
        for e, count in counts.items():
            assert count == 1, f"Directed edge {e} appears {count} times (expected 1)"
        for e in counts:
            reverse = (e[1], e[0])
            assert reverse in counts, f"Reverse of directed edge {e} is missing"

    def test_longer_track_manifold(self):
        """5-point centerline: all rings including intermediates are manifold."""
        centerline = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 1.0, 0.0],
            [6.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
        ])
        result = create_gpx_cylinder_track(centerline, radius=1.0, n_sides=8)
        counts = self._directed_edge_counts(result["faces"])
        for e, count in counts.items():
            assert count == 1, f"Directed edge {e} appears {count} times (expected 1)"
        for e in counts:
            reverse = (e[1], e[0])
            assert reverse in counts, f"Reverse of directed edge {e} is missing"

    def test_vertex_and_face_count(self):
        """Vertex and face counts match analytical formula for n_points and n_sides."""
        n_points = 4
        n_sides = 8
        centerline = np.linspace([0, 0, 0], [10, 0, 0], n_points)
        result = create_gpx_cylinder_track(centerline, radius=1.0, n_sides=n_sides)
        # n_points rings * n_sides vertices + 2 cap centers
        assert len(result["vertices"]) == n_points * n_sides + 2
        # 2 * n_sides * (n_points - 1) wall quads + 2 * n_sides cap tris = 2 * n_sides * n_points
        assert len(result["faces"]) == 2 * n_sides * n_points


class TestEdgeCases:
    """Edge cases for mesh generation."""

    def _make_elevation(self, rows=10, cols=10, value=0.0):
        """Create an elevation grid with a constant value."""
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

    def test_all_zero_elevation_grid_produces_valid_mesh(self):
        """An all-zero elevation grid should produce a valid (flat) terrain mesh."""
        from topo_shadow_box.core.mesh import generate_terrain_mesh
        from topo_shadow_box.core.coords import GeoToModelTransform

        bounds = self._make_bounds()
        elev = self._make_elevation(value=0.0)
        transform = GeoToModelTransform(bounds=bounds, model_width_mm=200.0)

        result = generate_terrain_mesh(
            elevation=elev, bounds=bounds, transform=transform,
            vertical_scale=1.5, base_height_mm=10.0, shape="square",
        )
        assert len(result.vertices) > 0, "Should produce vertices even for flat terrain"
        assert len(result.faces) > 0, "Should produce faces even for flat terrain"

    def test_uniform_elevation_grid_produces_flat_top_surface(self):
        """A constant (non-zero) elevation grid should produce a flat top surface."""
        from topo_shadow_box.core.mesh import generate_terrain_mesh
        from topo_shadow_box.core.coords import GeoToModelTransform
        import numpy as np

        bounds = self._make_bounds()
        elev = self._make_elevation(value=500.0)
        transform = GeoToModelTransform(bounds=bounds, model_width_mm=200.0)

        result = generate_terrain_mesh(
            elevation=elev, bounds=bounds, transform=transform,
            vertical_scale=1.5, base_height_mm=10.0, shape="square",
        )
        verts = np.array(result.vertices)
        # Top surface vertices (Y >= 0) should all have the same Y value (flat)
        top_ys = verts[verts[:, 1] >= 0, 1]
        assert len(top_ys) > 0, "Should have top-surface vertices"
        assert np.allclose(top_ys, top_ys[0], atol=0.5), (
            f"Uniform elevation should produce flat top. Y range: {top_ys.min():.3f} to {top_ys.max():.3f}"
        )

    def test_empty_feature_set_produces_no_feature_meshes(self):
        """generate_feature_meshes with empty OsmFeatureSet should return empty list."""
        from topo_shadow_box.core.mesh import generate_feature_meshes
        from topo_shadow_box.core.models import OsmFeatureSet
        from topo_shadow_box.core.coords import GeoToModelTransform

        bounds = self._make_bounds()
        elev = self._make_elevation()
        transform = GeoToModelTransform(bounds=bounds, model_width_mm=200.0)

        result = generate_feature_meshes(
            features=OsmFeatureSet(roads=[], water=[], buildings=[]),
            elevation=elev,
            bounds=bounds,
            transform=transform,
            vertical_scale=1.5,
            shape="square",
        )
        assert len(result) == 0, f"Empty feature set should produce no meshes, got {len(result)}"
