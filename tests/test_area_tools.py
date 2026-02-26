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


def test_geocode_place_returns_candidates(monkeypatch):
    import httpx
    from unittest.mock import MagicMock

    geocode_place = _register_and_get("geocode_place")

    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = [
        {
            "display_name": "Mount Hood, Hood River County, Oregon, United States",
            "lat": "45.3736",
            "lon": "-121.6959",
            "type": "peak",
            "boundingbox": ["45.3536", "45.3936", "-121.7159", "-121.6759"],
        },
        {
            "display_name": "Mount Hood Meadows, Clackamas County, Oregon, United States",
            "lat": "45.3300",
            "lon": "-121.6660",
            "type": "resort",
            "boundingbox": ["45.3200", "45.3400", "-121.6760", "-121.6560"],
        },
    ]

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: fake_response)

    result = geocode_place(query="Mount Hood")
    assert "1." in result
    assert "2." in result
    assert "45.3736" in result
    assert "peak" in result


def test_geocode_place_no_results(monkeypatch):
    import httpx
    from unittest.mock import MagicMock

    geocode_place = _register_and_get("geocode_place")

    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = []

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: fake_response)

    result = geocode_place(query="xyzzy_nowhere_12345")
    assert "no locations found" in result.lower()


def test_geocode_place_network_error(monkeypatch):
    import httpx

    geocode_place = _register_and_get("geocode_place")

    def raise_error(*a, **kw):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", raise_error)

    result = geocode_place(query="Mount Hood")
    assert "error" in result.lower()


def test_geocode_place_limit_clamped(monkeypatch):
    import httpx
    from unittest.mock import MagicMock

    geocode_place = _register_and_get("geocode_place")

    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = []

    captured = {}

    def fake_get(url, **kwargs):
        captured["params"] = kwargs.get("params", {})
        return fake_response

    monkeypatch.setattr(httpx, "get", fake_get)

    geocode_place(query="test", limit=99)
    assert captured["params"]["limit"] <= 10

    geocode_place(query="test", limit=0)
    assert captured["params"]["limit"] >= 1


FAKE_GEOCODE_RESULTS = [
    {
        "display_name": "Mount Hood, Hood River County, Oregon, United States",
        "lat": "45.3736",
        "lon": "-121.6959",
        "type": "peak",
        "boundingbox": ["45.3536", "45.3936", "-121.7159", "-121.6759"],
    },
    {
        "display_name": "Mount Hood Meadows, Clackamas County, Oregon, United States",
        "lat": "45.3300",
        "lon": "-121.6660",
        "type": "resort",
        "boundingbox": ["45.3200", "45.3400", "-121.6760", "-121.6560"],
    },
]


def _make_fake_geocode_response(results=None):
    from unittest.mock import MagicMock
    fake = MagicMock()
    fake.raise_for_status = MagicMock()
    fake.json.return_value = results if results is not None else FAKE_GEOCODE_RESULTS
    return fake


def test_geocode_place_stores_candidates_in_state(monkeypatch):
    import httpx

    geocode_place = _register_and_get("geocode_place")
    state.pending_geocode_candidates = []

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _make_fake_geocode_response())

    geocode_place(query="Mount Hood")

    assert len(state.pending_geocode_candidates) == 2
    assert state.pending_geocode_candidates[0].lat == 45.3736
    assert state.pending_geocode_candidates[1].place_type == "resort"


def test_select_geocode_result_sets_area_from_bbox(monkeypatch):
    import httpx
    from topo_shadow_box.models import GeocodeCandidate

    geocode_place = _register_and_get("geocode_place")
    select_geocode_result = _register_and_get("select_geocode_result")

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _make_fake_geocode_response())
    geocode_place(query="Mount Hood")

    result = select_geocode_result(number=1)

    assert "area set" in result.lower()
    assert state.bounds.is_set
    assert abs(state.bounds.north - 45.3936) < 0.001
    assert abs(state.bounds.south - 45.3536) < 0.001
    assert abs(state.bounds.east - (-121.6759)) < 0.001
    assert abs(state.bounds.west - (-121.7159)) < 0.001


def test_select_geocode_result_clears_pending_candidates(monkeypatch):
    import httpx

    geocode_place = _register_and_get("geocode_place")
    select_geocode_result = _register_and_get("select_geocode_result")

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _make_fake_geocode_response())
    geocode_place(query="Mount Hood")

    select_geocode_result(number=2)

    assert state.pending_geocode_candidates == []


def test_select_geocode_result_number_too_high_returns_error(monkeypatch):
    import httpx
    from topo_shadow_box.state import Bounds

    geocode_place = _register_and_get("geocode_place")
    select_geocode_result = _register_and_get("select_geocode_result")

    state.bounds = Bounds()  # reset

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _make_fake_geocode_response())
    geocode_place(query="Mount Hood")

    result = select_geocode_result(number=99)

    assert "error" in result.lower()
    assert state.bounds.is_set is False


def test_select_geocode_result_number_zero_returns_error(monkeypatch):
    import httpx

    geocode_place = _register_and_get("geocode_place")
    select_geocode_result = _register_and_get("select_geocode_result")

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _make_fake_geocode_response())
    geocode_place(query="Mount Hood")

    result = select_geocode_result(number=0)

    assert "error" in result.lower()


def test_select_geocode_result_no_pending_candidates_returns_error():
    select_geocode_result = _register_and_get("select_geocode_result")
    state.pending_geocode_candidates = []

    result = select_geocode_result(number=1)

    assert "error" in result.lower()
    assert "geocode" in result.lower() or "search" in result.lower()
