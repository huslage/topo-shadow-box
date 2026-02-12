"""Tests for mesh generation utilities."""

import numpy as np
import pytest

from topo_shadow_box.core.mesh import (
    create_road_strip,
    create_solid_polygon,
    triangulate_polygon,
)


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
