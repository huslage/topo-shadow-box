"""Tests for generate_model progress notifications."""
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock


def _make_state_with_features(n_roads=2, n_water=1, n_buildings=1, has_gpx=False):
    """Configure state with known feature counts."""
    from topo_shadow_box.state import state, Bounds, ElevationData
    from topo_shadow_box.core.models import OsmFeatureSet
    from topo_shadow_box.models import RoadFeature, WaterFeature, BuildingFeature, Coordinate, GpxTrack, GpxPoint

    state.bounds = Bounds(north=37.80, south=37.75, east=-122.40, west=-122.45, is_set=True)
    grid = np.linspace(0, 200, 20 * 20).reshape(20, 20)
    state.elevation = ElevationData(
        grid=grid,
        lats=np.linspace(37.75, 37.80, 20),
        lons=np.linspace(-122.45, -122.40, 20),
        resolution=20,
        min_elevation=0.0,
        max_elevation=200.0,
        is_set=True,
    )

    coord_pair = [
        Coordinate(lat=37.76, lon=-122.44),
        Coordinate(lat=37.77, lon=-122.43),
    ]
    coord_triple = [
        Coordinate(lat=37.76, lon=-122.44),
        Coordinate(lat=37.77, lon=-122.43),
        Coordinate(lat=37.78, lon=-122.42),
    ]

    roads = [RoadFeature(id=i, coordinates=coord_pair, name=f"Road{i}", road_type="primary")
             for i in range(n_roads)]
    waters = [WaterFeature(id=i+100, coordinates=coord_triple, name=f"Water{i}")
              for i in range(n_water)]
    buildings = [BuildingFeature(id=i+200, coordinates=coord_triple, name=f"Bldg{i}", height=10.0)
                 for i in range(n_buildings)]
    state.features = OsmFeatureSet(roads=roads, water=waters, buildings=buildings)

    if has_gpx:
        state.gpx_tracks = [GpxTrack(name="Track", points=[
            GpxPoint(lat=37.76, lon=-122.44, elevation=100),
            GpxPoint(lat=37.77, lon=-122.43, elevation=110),
        ])]
    else:
        state.gpx_tracks = []

    return state


def _get_generate_model_fn():
    from topo_shadow_box.tools.generate import register_generate_tools
    tools = {}
    mock_mcp = MagicMock()
    def capture():
        def decorator(fn):
            tools[fn.__name__] = fn
            return fn
        return decorator
    mock_mcp.tool = capture
    register_generate_tools(mock_mcp)
    return tools["generate_model"]


@pytest.mark.anyio
async def test_generate_model_reports_progress():
    """generate_model should call ctx.report_progress for each work unit."""
    _make_state_with_features(n_roads=2, n_water=1, n_buildings=1, has_gpx=False)
    generate_model = _get_generate_model_fn()

    mock_ctx = AsyncMock()
    mock_ctx.report_progress = AsyncMock()

    result = await generate_model(ctx=mock_ctx)

    assert mock_ctx.report_progress.called, "report_progress should have been called"
    calls = mock_ctx.report_progress.call_args_list
    # 1 terrain + 2 roads + 1 water + 1 building = 5 total
    assert len(calls) >= 4, f"Expected at least 4 progress calls, got {len(calls)}"


@pytest.mark.anyio
async def test_generate_model_progress_is_nondecreasing():
    """Progress values should never decrease."""
    _make_state_with_features(n_roads=3, n_water=0, n_buildings=0, has_gpx=False)
    generate_model = _get_generate_model_fn()

    mock_ctx = AsyncMock()
    progress_values = []

    async def capture_progress(current, total):
        progress_values.append((current, total))

    mock_ctx.report_progress = capture_progress

    await generate_model(ctx=mock_ctx)

    assert len(progress_values) >= 2
    currents = [p[0] for p in progress_values]
    assert currents == sorted(currents), f"Progress not non-decreasing: {currents}"


@pytest.mark.anyio
async def test_generate_model_final_progress_equals_total():
    """Final progress call should have current == total."""
    _make_state_with_features(n_roads=1, n_water=1, n_buildings=0, has_gpx=False)
    generate_model = _get_generate_model_fn()

    mock_ctx = AsyncMock()
    progress_values = []

    async def capture_progress(current, total):
        progress_values.append((current, total))

    mock_ctx.report_progress = capture_progress

    await generate_model(ctx=mock_ctx)

    last_current, last_total = progress_values[-1]
    assert last_current == last_total, (
        f"Final progress {last_current} should equal total {last_total}"
    )


@pytest.mark.anyio
async def test_generate_model_with_gpx_includes_gpx_in_total():
    """GPX track counts as one work unit."""
    _make_state_with_features(n_roads=0, n_water=0, n_buildings=0, has_gpx=True)
    generate_model = _get_generate_model_fn()

    mock_ctx = AsyncMock()
    progress_values = []

    async def capture_progress(current, total):
        progress_values.append((current, total))

    mock_ctx.report_progress = capture_progress

    await generate_model(ctx=mock_ctx)

    # 1 terrain + 1 gpx = 2 total
    assert len(progress_values) >= 2
    totals = [p[1] for p in progress_values]
    assert all(t == totals[0] for t in totals), "Total should be consistent across calls"
    assert totals[0] == 2, f"Expected total=2 (terrain+gpx), got {totals[0]}"
