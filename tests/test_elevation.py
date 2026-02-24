import logging
import pytest
import anyio
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
