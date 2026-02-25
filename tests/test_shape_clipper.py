"""Tests for shape clipping utilities."""

import numpy as np

from topo_shadow_box.core.shape_clipper import (
    CircleClipper,
    HexagonClipper,
    RectangleClipper,
    SquareClipper,
)


# --- CircleClipper ---


class TestCircleClipper:
    def setup_method(self):
        self.clipper = CircleClipper(center_x=0.0, center_z=0.0, radius=10.0)

    def test_is_inside_center(self):
        assert self.clipper.is_inside(0.0, 0.0)

    def test_is_inside_edge(self):
        # Point exactly on the boundary should be inside (<=)
        assert self.clipper.is_inside(10.0, 0.0)
        assert self.clipper.is_inside(0.0, 10.0)

    def test_is_inside_outside(self):
        assert not self.clipper.is_inside(11.0, 0.0)
        assert not self.clipper.is_inside(8.0, 8.0)  # sqrt(128) > 10

    def test_is_inside_array(self):
        x = np.array([0.0, 10.0, 11.0, 5.0])
        z = np.array([0.0, 0.0, 0.0, 5.0])
        result = self.clipper.is_inside(x, z)
        expected = np.array([True, True, False, True])
        np.testing.assert_array_equal(result, expected)

    def test_clip_linestring_fully_inside(self):
        points = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
        segments = self.clipper.clip_linestring(points)
        assert len(segments) == 1
        np.testing.assert_allclose(segments[0], np.array(points))

    def test_clip_linestring_crossing(self):
        # Line goes from inside to outside
        points = [(0.0, 0.0), (20.0, 0.0)]
        segments = self.clipper.clip_linestring(points)
        assert len(segments) == 1
        # Should end at the circle boundary (10, 0)
        np.testing.assert_allclose(segments[0][-1], [10.0, 0.0], atol=1e-6)

    def test_clip_linestring_enters_and_exits(self):
        # Line: outside -> inside -> outside
        points = [(-20.0, 0.0), (0.0, 0.0), (20.0, 0.0)]
        segments = self.clipper.clip_linestring(points)
        assert len(segments) == 1
        # Should start near (-10, 0) and end near (10, 0)
        np.testing.assert_allclose(segments[0][0], [-10.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(segments[0][-1], [10.0, 0.0], atol=1e-6)

    def test_clip_linestring_empty(self):
        segments = self.clipper.clip_linestring([(0.0, 0.0)])
        assert segments == []

    def test_clip_linestring_fully_outside(self):
        points = [(20.0, 0.0), (30.0, 0.0)]
        segments = self.clipper.clip_linestring(points)
        assert segments == []

    def test_clip_polygon_all_inside(self):
        polygon = [(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)]
        result = self.clipper.clip_polygon(polygon)
        assert result is not None
        np.testing.assert_allclose(result, np.array(polygon))

    def test_clip_polygon_partially_outside(self):
        polygon = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0)]
        result = self.clipper.clip_polygon(polygon)
        assert result is None

    def test_clip_polygon_empty(self):
        result = self.clipper.clip_polygon([])
        assert result is None

    def test_project_to_boundary(self):
        # Point outside, should project to boundary
        px, pz = self.clipper.project_to_boundary(20.0, 0.0)
        np.testing.assert_allclose((px, pz), (10.0, 0.0), atol=1e-6)

    def test_project_to_boundary_inside(self):
        # Point inside, still projects to boundary
        px, pz = self.clipper.project_to_boundary(5.0, 0.0)
        np.testing.assert_allclose((px, pz), (10.0, 0.0), atol=1e-6)

    def test_project_to_boundary_center(self):
        # Point at center projects to arbitrary boundary point
        px, pz = self.clipper.project_to_boundary(0.0, 0.0)
        assert np.isclose(px, 10.0)
        assert np.isclose(pz, 0.0)

    def test_project_to_boundary_diagonal(self):
        px, pz = self.clipper.project_to_boundary(5.0, 5.0)
        dist = np.sqrt(px**2 + pz**2)
        np.testing.assert_allclose(dist, 10.0, atol=1e-6)


# --- SquareClipper ---


class TestSquareClipper:
    def setup_method(self):
        self.clipper = SquareClipper(center_x=0.0, center_z=0.0, half_width=10.0)

    def test_is_inside_center(self):
        assert self.clipper.is_inside(0.0, 0.0)

    def test_is_inside_outside(self):
        assert not self.clipper.is_inside(11.0, 0.0)
        assert not self.clipper.is_inside(0.0, 11.0)

    def test_is_inside_edge(self):
        assert self.clipper.is_inside(10.0, 10.0)

    def test_is_inside_corner(self):
        # Corners should be inside for a square (unlike circle)
        assert self.clipper.is_inside(10.0, 10.0)
        assert self.clipper.is_inside(-10.0, -10.0)

    def test_clip_linestring_fully_inside(self):
        points = [(0.0, 0.0), (5.0, 5.0)]
        segments = self.clipper.clip_linestring(points)
        assert len(segments) == 1

    def test_clip_linestring_crossing(self):
        # Line crosses right boundary
        points = [(0.0, 0.0), (20.0, 0.0)]
        segments = self.clipper.clip_linestring(points)
        assert len(segments) == 1
        np.testing.assert_allclose(segments[0][-1], [10.0, 0.0], atol=1e-6)

    def test_clip_linestring_fully_outside(self):
        points = [(20.0, 0.0), (30.0, 0.0)]
        segments = self.clipper.clip_linestring(points)
        assert segments == []

    def test_project_to_boundary(self):
        # Point on the +x axis: closer to right edge
        px, pz = self.clipper.project_to_boundary(5.0, 0.0)
        assert px == 10.0
        assert pz == 0.0

    def test_project_to_boundary_closer_to_top(self):
        # Point closer to the +z edge
        px, pz = self.clipper.project_to_boundary(0.0, 5.0)
        assert px == 0.0
        assert pz == 10.0


# --- RectangleClipper ---


class TestRectangleClipper:
    def setup_method(self):
        self.clipper = RectangleClipper(
            center_x=0.0, center_z=0.0, half_width=10.0, half_height=5.0
        )

    def test_is_inside_center(self):
        assert self.clipper.is_inside(0.0, 0.0)

    def test_is_inside_within_width_outside_height(self):
        # Inside width but outside height
        assert not self.clipper.is_inside(5.0, 6.0)

    def test_is_inside_within_height_outside_width(self):
        # Inside height but outside width
        assert not self.clipper.is_inside(11.0, 3.0)

    def test_is_inside_corner(self):
        assert self.clipper.is_inside(10.0, 5.0)
        assert self.clipper.is_inside(-10.0, -5.0)

    def test_is_inside_array(self):
        x = np.array([0.0, 10.0, 11.0, 5.0])
        z = np.array([0.0, 5.0, 0.0, 6.0])
        result = self.clipper.is_inside(x, z)
        expected = np.array([True, True, False, False])
        np.testing.assert_array_equal(result, expected)

    def test_clip_linestring_crossing(self):
        points = [(0.0, 0.0), (20.0, 0.0)]
        segments = self.clipper.clip_linestring(points)
        assert len(segments) == 1
        np.testing.assert_allclose(segments[0][-1], [10.0, 0.0], atol=1e-6)

    def test_clip_polygon_all_inside(self):
        polygon = [(1.0, 1.0), (5.0, 1.0), (5.0, 3.0)]
        result = self.clipper.clip_polygon(polygon)
        assert result is not None

    def test_clip_polygon_partially_outside(self):
        polygon = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0)]
        result = self.clipper.clip_polygon(polygon)
        assert result is None

    def test_project_to_boundary(self):
        # With half_width=10, half_height=5, point at (8,0):
        # ratio_x = 8/10 = 0.8, ratio_z = 0/5 = 0.0
        # ratio_x >= ratio_z, dx >= 0, so project to right edge
        px, pz = self.clipper.project_to_boundary(8.0, 0.0)
        assert px == 10.0
        assert pz == 0.0


# --- HexagonClipper ---


class TestHexagonClipper:
    def setup_method(self):
        self.clipper = HexagonClipper(center_x=0.0, center_z=0.0, radius=10.0)

    def test_is_inside_center(self):
        assert self.clipper.is_inside(0.0, 0.0)

    def test_is_inside_outside(self):
        assert not self.clipper.is_inside(20.0, 0.0)
        assert not self.clipper.is_inside(0.0, 20.0)

    def test_is_inside_near_vertex(self):
        # Just inside a vertex at (radius, 0) = (10, 0)
        assert self.clipper.is_inside(9.0, 0.0)

    def test_is_inside_array(self):
        x = np.array([0.0, 20.0, 5.0])
        z = np.array([0.0, 0.0, 0.0])
        result = self.clipper.is_inside(x, z)
        expected = np.array([True, False, True])
        np.testing.assert_array_equal(result, expected)

    def test_clip_linestring_fully_inside(self):
        points = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]
        segments = self.clipper.clip_linestring(points)
        assert len(segments) == 1

    def test_clip_polygon_all_inside(self):
        polygon = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        result = self.clipper.clip_polygon(polygon)
        assert result is not None

    def test_clip_polygon_outside(self):
        polygon = [(20.0, 20.0), (25.0, 20.0), (25.0, 25.0)]
        result = self.clipper.clip_polygon(polygon)
        assert result is None

    def test_project_to_boundary(self):
        # Project center outward; should land on nearest edge
        px, pz = self.clipper.project_to_boundary(0.0, 0.0)
        assert px is not None
        assert pz is not None
        # Result should be on the hexagon boundary
