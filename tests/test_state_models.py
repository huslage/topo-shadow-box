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
