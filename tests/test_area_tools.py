"""Tests for validate_area tool."""
from unittest.mock import MagicMock
from topo_shadow_box.state import state, Bounds, ElevationData
import numpy as np


def _register_and_get(tool_name: str):
    """Register area tools against a mock MCP and extract the named tool."""
    from topo_shadow_box.tools.area import register_area_tools
    tools = {}
    mock_mcp = MagicMock()
    def capture(**kwargs):
        def decorator(fn):
            tools[fn.__name__] = fn
            return fn
        return decorator
    mock_mcp.tool = capture
    register_area_tools(mock_mcp)
    return tools[tool_name]


def test_validate_area_small_area_returns_error():
    validate_area = _register_and_get("validate_area")
    # Set an area that is ~50m x 50m (way too small)
    state.bounds = Bounds(
        north=37.7501, south=37.7500, east=-122.4499, west=-122.4500, is_set=True
    )
    result = validate_area()
    assert "too small" in result.lower() or "error" in result.lower()


def test_validate_area_large_area_returns_warning():
    validate_area = _register_and_get("validate_area")
    # Set an area ~600km x 600km
    state.bounds = Bounds(
        north=47.75, south=37.75, east=-112.45, west=-122.45, is_set=True
    )
    result = validate_area()
    assert "large" in result.lower() or "warning" in result.lower()


def test_validate_area_flat_relief_warns_when_elevation_set():
    validate_area = _register_and_get("validate_area")
    state.bounds = Bounds(
        north=37.80, south=37.75, east=-122.40, west=-122.45, is_set=True
    )
    # Flat elevation (5m relief)
    state.elevation = ElevationData(
        grid=np.full((10, 10), 100.0),
        lats=np.linspace(37.75, 37.80, 10),
        lons=np.linspace(-122.45, -122.40, 10),
        resolution=10,
        min_elevation=98.0,
        max_elevation=103.0,
        is_set=True,
    )
    result = validate_area()
    assert "flat" in result.lower() or "relief" in result.lower()


def test_validate_area_good_area_no_elevation_passes():
    validate_area = _register_and_get("validate_area")
    from topo_shadow_box.state import ElevationData
    state.bounds = Bounds(
        north=37.80, south=37.75, east=-122.40, west=-122.45, is_set=True
    )
    state.elevation = ElevationData()  # not set
    result = validate_area()
    assert "good" in result.lower() or "ok" in result.lower()


def test_validate_area_good_area_good_relief_passes():
    validate_area = _register_and_get("validate_area")
    state.bounds = Bounds(
        north=37.80, south=37.75, east=-122.40, west=-122.45, is_set=True
    )
    state.elevation = ElevationData(
        grid=np.linspace(0, 500, 100).reshape(10, 10),
        lats=np.linspace(37.75, 37.80, 10),
        lons=np.linspace(-122.45, -122.40, 10),
        resolution=10,
        min_elevation=0.0,
        max_elevation=500.0,
        is_set=True,
    )
    result = validate_area()
    assert "good" in result.lower() or "ok" in result.lower()


def test_validate_area_requires_bounds_set():
    validate_area = _register_and_get("validate_area")
    from topo_shadow_box.state import Bounds
    state.bounds = Bounds()  # not set
    result = validate_area()
    assert "error" in result.lower() or "area" in result.lower()


def test_geocode_candidate_model():
    from topo_shadow_box.models import GeocodeCandidate
    c = GeocodeCandidate(
        display_name="Mount Hood, Hood River County, Oregon, United States",
        lat=45.3736,
        lon=-121.6959,
        place_type="peak",
        bbox_north=45.3936,
        bbox_south=45.3536,
        bbox_east=-121.6759,
        bbox_west=-121.7159,
    )
    assert c.lat == 45.3736
    assert c.bbox_north == 45.3936
