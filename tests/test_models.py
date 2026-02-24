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
