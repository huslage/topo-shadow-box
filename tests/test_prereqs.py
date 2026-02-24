"""Tests for tool prerequisite helpers."""
import pytest


def test_require_state_raises_when_bounds_not_set():
    from topo_shadow_box.tools._prereqs import require_state
    from topo_shadow_box.state import SessionState

    mock_state = SessionState()  # bounds.is_set defaults to False
    with pytest.raises(ValueError, match="area"):
        require_state(mock_state, bounds=True)


def test_require_state_passes_when_bounds_set():
    from topo_shadow_box.tools._prereqs import require_state
    from topo_shadow_box.state import SessionState, Bounds

    mock_state = SessionState()
    mock_state.bounds = Bounds(north=37.8, south=37.75, east=-122.4, west=-122.45, is_set=True)
    # Should not raise
    require_state(mock_state, bounds=True)


def test_require_state_raises_when_elevation_not_set():
    from topo_shadow_box.tools._prereqs import require_state
    from topo_shadow_box.state import SessionState, Bounds

    mock_state = SessionState()
    mock_state.bounds = Bounds(north=37.8, south=37.75, east=-122.4, west=-122.45, is_set=True)
    with pytest.raises(ValueError, match="elevation"):
        require_state(mock_state, bounds=True, elevation=True)


def test_require_state_raises_when_mesh_not_generated():
    from topo_shadow_box.tools._prereqs import require_state
    from topo_shadow_box.state import SessionState, Bounds, ElevationData
    import numpy as np

    mock_state = SessionState()
    mock_state.bounds = Bounds(north=37.8, south=37.75, east=-122.4, west=-122.45, is_set=True)
    mock_state.elevation = ElevationData(
        grid=np.zeros((10, 10)),
        lats=np.zeros(10),
        lons=np.zeros(10),
        resolution=10,
        is_set=True,
    )
    with pytest.raises(ValueError, match="model"):
        require_state(mock_state, bounds=True, elevation=True, mesh=True)


def test_require_state_no_flags_does_not_raise():
    from topo_shadow_box.tools._prereqs import require_state
    from topo_shadow_box.state import SessionState

    mock_state = SessionState()
    # No flags — should never raise
    require_state(mock_state)
