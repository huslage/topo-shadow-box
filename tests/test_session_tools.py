"""Tests for save_session and load_session tools."""
import json
import os
import pytest
import numpy as np
from unittest.mock import MagicMock
from pathlib import Path


def _get_session_tools():
    from topo_shadow_box.tools.session import register_session_tools
    tools = {}
    mock_mcp = MagicMock()
    def capture(**kwargs):
        def decorator(fn):
            tools[fn.__name__] = fn
            return fn
        return decorator
    mock_mcp.tool = capture
    register_session_tools(mock_mcp)
    return tools


def _configure_state():
    """Put known data into state for round-trip tests."""
    from topo_shadow_box.state import state, Bounds, ElevationData
    state.bounds = Bounds(north=37.80, south=37.75, east=-122.40, west=-122.45, is_set=True)
    state.model_params.width_mm = 150.0
    state.model_params.vertical_scale = 2.0
    state.colors.terrain = "#AABBCC"
    state.elevation = ElevationData(
        grid=np.zeros((10, 10)),
        lats=np.zeros(10),
        lons=np.zeros(10),
        resolution=10,
        min_elevation=0.0,
        max_elevation=100.0,
        is_set=True,
    )


def test_save_session_creates_file(tmp_path):
    tools = _get_session_tools()
    _configure_state()
    path = str(tmp_path / "session.json")
    result = tools["save_session"](path=path)
    assert os.path.exists(path), f"File not created. Result: {result}"
    assert "saved" in result.lower()


def test_save_session_file_is_valid_json(tmp_path):
    tools = _get_session_tools()
    _configure_state()
    path = str(tmp_path / "session.json")
    tools["save_session"](path=path)
    with open(path) as f:
        data = json.load(f)
    assert "bounds" in data
    assert "model_params" in data
    assert "colors" in data


def test_save_session_does_not_include_elevation_grid(tmp_path):
    tools = _get_session_tools()
    _configure_state()
    path = str(tmp_path / "session.json")
    tools["save_session"](path=path)
    with open(path) as f:
        data = json.load(f)
    # Elevation metadata (min/max) is fine, but not the full grid array
    if "elevation" in data:
        assert "grid" not in data["elevation"], "Should not serialize numpy grid"


def test_load_session_restores_bounds_and_params(tmp_path):
    tools = _get_session_tools()
    _configure_state()
    path = str(tmp_path / "session.json")
    tools["save_session"](path=path)

    # Reset state
    from topo_shadow_box.state import state, Bounds, ElevationData
    state.bounds = Bounds()
    state.elevation = ElevationData()
    state.model_params.width_mm = 200.0

    result = tools["load_session"](path=path)
    assert state.bounds.north == pytest.approx(37.80)
    assert state.bounds.south == pytest.approx(37.75)
    assert state.model_params.width_mm == pytest.approx(150.0)
    assert "restored" in result.lower() or "loaded" in result.lower()


def test_load_session_clears_elevation_and_meshes(tmp_path):
    tools = _get_session_tools()
    _configure_state()
    path = str(tmp_path / "session.json")
    tools["save_session"](path=path)

    from topo_shadow_box.state import state, MeshData
    # Pretend meshes exist
    state.terrain_mesh = MeshData(vertices=[[0,0,0]], faces=[[0,0,0]], name="T", feature_type="terrain")

    tools["load_session"](path=path)
    assert state.elevation.is_set is False, "Elevation should need re-fetching after load"
    assert state.terrain_mesh is None, "Meshes should be cleared after load"


def test_load_session_missing_file_returns_error(tmp_path):
    tools = _get_session_tools()
    result = tools["load_session"](path=str(tmp_path / "nonexistent.json"))
    assert "error" in result.lower() or "not found" in result.lower()


def test_save_session_default_path_creates_directory(tmp_path, monkeypatch):
    """Default path (~/.cache/topo-shadow-box/session.json) directory is created."""
    tools = _get_session_tools()
    _configure_state()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    result = tools["save_session"]()
    expected = tmp_path / ".cache" / "topo-shadow-box" / "session.json"
    assert expected.exists(), f"Default file not created. Result: {result}"
