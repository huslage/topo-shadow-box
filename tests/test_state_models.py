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
