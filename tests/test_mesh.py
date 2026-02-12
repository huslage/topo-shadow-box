"""Tests for create_road_strip mesh generation."""

import numpy as np
import pytest

from topo_shadow_box.core.mesh import create_road_strip


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
