"""End-to-end integration test for the full pipeline with mocked HTTP."""

import os
import pytest
from unittest.mock import patch, AsyncMock
from PIL import Image
from io import BytesIO


def _make_tile_response():
    """Create a mock HTTP response with a valid 256x256 RGB elevation tile."""
    # Encode ~100m elevation in Terrarium format: R*256 + G + B/256 - 32768 = elevation
    # For ~100m: 128*256 + 100 + 0/256 - 32768 = 32768 + 100 - 32768 = 100m
    r, g, b = 128, 100, 0
    img = Image.new("RGB", (256, 256), color=(r, g, b))
    buf = BytesIO()
    img.save(buf, format="PNG")
    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.content = buf.getvalue()
    return mock_resp


@pytest.mark.anyio
async def test_full_pipeline_set_area_fetch_elevation_generate_export(tmp_path):
    """Full pipeline: area → elevation (mocked) → terrain mesh → 3MF export."""
    from topo_shadow_box.state import state, Bounds, ElevationData, MeshData
    from topo_shadow_box.core.elevation import fetch_terrain_elevation
    from topo_shadow_box.core.mesh import generate_terrain_mesh
    from topo_shadow_box.core.coords import GeoToModelTransform
    from topo_shadow_box.exporters.threemf import export_3mf

    # Reset state
    state.bounds = Bounds(north=37.8, south=37.75, east=-122.4, west=-122.45, is_set=True)

    # Step 1: Fetch elevation (mocked)
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=_make_tile_response())
        mock_client_cls.return_value = mock_client

        elevation = await fetch_terrain_elevation(
            north=37.8, south=37.75, east=-122.4, west=-122.45, resolution=20
        )

    state.elevation = ElevationData(
        grid=elevation.grid,
        lats=elevation.lats,
        lons=elevation.lons,
        resolution=elevation.resolution,
        min_elevation=elevation.min_elevation,
        max_elevation=elevation.max_elevation,
        is_set=True,
    )

    # Step 2: Generate terrain mesh
    b = state.bounds
    mp = state.model_params
    transform = GeoToModelTransform(bounds=b, model_width_mm=mp.width_mm)
    terrain = generate_terrain_mesh(
        elevation=state.elevation,
        bounds=b,
        transform=transform,
        vertical_scale=mp.vertical_scale,
        base_height_mm=mp.base_height_mm,
        shape="square",
    )
    state.terrain_mesh = MeshData(
        vertices=terrain.vertices,
        faces=terrain.faces,
        name=terrain.name,
        feature_type=terrain.feature_type,
    )

    # Verify terrain mesh is valid
    assert len(state.terrain_mesh.vertices) > 0, "Terrain mesh should have vertices"
    assert len(state.terrain_mesh.faces) > 0, "Terrain mesh should have faces"

    # Step 3: Export to 3MF
    out = str(tmp_path / "output.3mf")
    meshes = [{
        "name": state.terrain_mesh.name,
        "type": state.terrain_mesh.feature_type,
        "vertices": state.terrain_mesh.vertices,
        "faces": state.terrain_mesh.faces,
        "color": "#C8A882",
    }]
    result = export_3mf(meshes, out)

    # Verify output
    assert os.path.exists(out), "3MF file should be created"
    assert os.path.getsize(out) > 0, "3MF file should not be empty"
    assert result["success"] is True
    assert result["objects"] == 1
