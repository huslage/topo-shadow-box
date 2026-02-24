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
