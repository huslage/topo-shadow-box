"""Tests for mesh generation utilities."""

import numpy as np
import pytest

from topo_shadow_box.core.mesh import (
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
        assert len(meshes[0]["vertices"]) >= 8
        assert len(meshes[0]["faces"]) >= 12

    def test_water_is_watertight(self):
        features = {"water": [{"coordinates": [
            {"lat": 40.003, "lon": -73.997}, {"lat": 40.007, "lon": -73.997},
            {"lat": 40.007, "lon": -73.993}, {"lat": 40.003, "lon": -73.993}
        ], "name": "Test Lake"}]}
        meshes = generate_feature_meshes(
            features, self.elevation, self.bounds, self.transform, 1.5, "square")
        assert len(meshes) == 1
        assert len(meshes[0]["vertices"]) == 8
        assert len(meshes[0]["faces"]) == 12

    def test_building_shape_aware(self):
        features = {"buildings": [{"coordinates": [
            {"lat": 40.005, "lon": -73.996}, {"lat": 40.006, "lon": -73.996},
            {"lat": 40.006, "lon": -73.995}, {"lat": 40.005, "lon": -73.995}
        ], "name": "Church", "height": 20.0,
           "tags": {"amenity": "place_of_worship", "religion": "christian"}}]}
        meshes = generate_feature_meshes(
            features, self.elevation, self.bounds, self.transform, 1.5, "square")
        assert len(meshes) == 1
        assert len(meshes[0]["vertices"]) == 17  # steeple

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
        assert len(mesh["vertices"]) > 0
        assert len(mesh["faces"]) > 0

    def test_circle_terrain_valid_indices(self):
        mesh = generate_terrain_mesh(
            self.elevation, self.bounds, self.transform, 1.5, 10.0, "circle")
        n_verts = len(mesh["vertices"])
        for face in mesh["faces"]:
            for idx in face:
                assert 0 <= idx < n_verts

    def test_circle_has_more_verts_than_square(self):
        sq = generate_terrain_mesh(
            self.elevation, self.bounds, self.transform, 1.5, 10.0, "square")
        ci = generate_terrain_mesh(
            self.elevation, self.bounds, self.transform, 1.5, 10.0, "circle")
        # Circle adds 360 wall segments (720 verts) + center
        assert len(ci["vertices"]) > len(sq["vertices"])

    def test_hexagon_terrain_valid_indices(self):
        mesh = generate_terrain_mesh(
            self.elevation, self.bounds, self.transform, 1.5, 10.0, "hexagon")
        n_verts = len(mesh["vertices"])
        for face in mesh["faces"]:
            for idx in face:
                assert 0 <= idx < n_verts

    def test_hexagon_terrain_has_faces(self):
        mesh = generate_terrain_mesh(
            self.elevation, self.bounds, self.transform, 1.5, 10.0, "hexagon")
        assert len(mesh["vertices"]) > 0
        assert len(mesh["faces"]) > 0


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
        assert len(result["vertices"]) > 0
