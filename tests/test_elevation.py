import logging
import pytest
from unittest.mock import patch, AsyncMock


def test_elevation_module_has_logger():
    import topo_shadow_box.core.elevation as elev_mod
    assert hasattr(elev_mod, 'logger'), "elevation module should have a module-level logger"


@pytest.mark.anyio
async def test_failed_tile_logs_warning(caplog):
    """A failed tile fetch should log a warning, not silently pass."""
    from topo_shadow_box.core.elevation import fetch_terrain_elevation

    with caplog.at_level(logging.WARNING, logger="topo_shadow_box.core.elevation"):
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(side_effect=Exception("tile fetch failed"))
            mock_client_cls.return_value = mock_client

            try:
                await fetch_terrain_elevation(37.8, 37.75, -122.4, -122.45, resolution=10)
            except Exception:
                pass

    assert any(
        r.name == "topo_shadow_box.core.elevation" and r.levelno == logging.WARNING
        for r in caplog.records
    ), f"Should log a WARNING from topo_shadow_box.core.elevation. Got: {[(r.name, r.levelno, r.message) for r in caplog.records]}"


@pytest.mark.anyio
async def test_all_tiles_fail_does_not_crash():
    """When all tile fetches raise exceptions, function should handle gracefully."""
    from topo_shadow_box.core.elevation import fetch_terrain_elevation

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=Exception("network error"))
        mock_client_cls.return_value = mock_client

        # Should either raise ValueError (no data to interpolate) or return normally
        # Either behavior is acceptable — we just verify it doesn't crash with an unhandled exception
        try:
            result = await fetch_terrain_elevation(37.8, 37.75, -122.4, -122.45, resolution=10)
            # If it returns, the grid should exist (may be all zeros)
            assert result.grid is not None
        except ValueError:
            pass  # "Bounding box doesn't overlap" or "Cropped region too small" — acceptable


@pytest.mark.anyio
async def test_partial_tile_failure_logs_warning_for_failed_tile(caplog):
    """When some tiles fail and some succeed, warnings should be logged for failures."""
    import logging
    from topo_shadow_box.core.elevation import fetch_terrain_elevation
    from PIL import Image
    from io import BytesIO

    call_count = 0

    async def mock_get(url, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("first tile network failure")
        # Return a valid 256x256 RGB tile for all other calls
        img = Image.new("RGB", (256, 256), color=(128, 100, 0))
        buf = BytesIO()
        img.save(buf, format="PNG")
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.content = buf.getvalue()
        return mock_resp

    with caplog.at_level(logging.WARNING, logger="topo_shadow_box.core.elevation"):
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = mock_get
            mock_client_cls.return_value = mock_client

            try:
                await fetch_terrain_elevation(37.8, 37.75, -122.4, -122.45, resolution=10)
            except Exception:
                pass

    # At least one warning should have been logged for the failed tile
    assert any(
        r.name == "topo_shadow_box.core.elevation" and r.levelno == logging.WARNING
        for r in caplog.records
    ), f"Expected WARNING from elevation logger. Got: {[(r.name, r.levelno, r.message) for r in caplog.records]}"


@pytest.mark.anyio
async def test_all_tile_urls_are_fetched_without_duplicates():
    """All tile URLs for the bounding box should be fetched, with no duplicates."""
    from topo_shadow_box.core.elevation import fetch_terrain_elevation
    from PIL import Image
    from io import BytesIO

    fetched_urls = []

    async def mock_get(url, **kwargs):
        fetched_urls.append(url)
        img = Image.new("RGB", (256, 256), color=(128, 100, 0))
        buf = BytesIO()
        img.save(buf, format="PNG")
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.content = buf.getvalue()
        return mock_resp

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = mock_get
        mock_client_cls.return_value = mock_client

        result = await fetch_terrain_elevation(37.8, 37.75, -122.4, -122.45, resolution=10)

    assert len(fetched_urls) >= 1, "Should have fetched at least one tile"
    assert result is not None
    assert len(fetched_urls) == len(set(fetched_urls)), "Should not fetch same tile twice"
