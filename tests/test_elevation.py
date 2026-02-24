import asyncio
import logging
import pytest
from unittest.mock import patch, AsyncMock


def test_elevation_module_has_logger():
    import topo_shadow_box.core.elevation as elev_mod
    assert hasattr(elev_mod, 'logger'), "elevation module should have a module-level logger"


def test_failed_tile_logs_warning(caplog):
    """A failed tile fetch should log a warning, not silently pass."""
    from topo_shadow_box.core.elevation import fetch_terrain_elevation

    async def run():
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

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(run())
    finally:
        loop.close()

    assert any(
        "tile" in r.message.lower() or "failed" in r.message.lower()
        for r in caplog.records
    ), f"Should log a warning for failed tile. Got records: {[r.message for r in caplog.records]}"
