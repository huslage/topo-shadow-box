import logging
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import httpx


def test_osm_module_has_logger():
    import topo_shadow_box.core.osm as osm_mod
    assert hasattr(osm_mod, 'logger')


@pytest.mark.anyio
async def test_timeout_logs_warning_and_tries_next_server(caplog):
    """TimeoutException on one server should warn and try the next."""
    from topo_shadow_box.core.osm import _query_overpass

    call_count = 0

    async def mock_post(url, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise httpx.TimeoutException("timeout")
        # Return a synchronous mock since raise_for_status() and json() are sync
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value={"elements": [{"id": 1}]})
        return mock_resp

    with caplog.at_level(logging.WARNING, logger="topo_shadow_box.core.osm"):
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = mock_post
            mock_client_cls.return_value = mock_client

            result = await _query_overpass("test query")

    assert result == [{"id": 1}]
    assert any(
        r.name == "topo_shadow_box.core.osm" and r.levelno == logging.WARNING
        and "timed out" in r.message.lower()
        for r in caplog.records
    )


@pytest.mark.anyio
async def test_all_servers_fail_logs_warning(caplog):
    """When all servers fail, return empty list and log warning about all servers."""
    from topo_shadow_box.core.osm import _query_overpass

    with caplog.at_level(logging.WARNING, logger="topo_shadow_box.core.osm"):
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_client_cls.return_value = mock_client

            result = await _query_overpass("test query")

    assert result == []
    assert any(
        r.name == "topo_shadow_box.core.osm" and r.levelno == logging.WARNING
        and ("all" in r.message.lower() or "server" in r.message.lower())
        for r in caplog.records
    )


@pytest.mark.anyio
async def test_http_status_error_logs_status(caplog):
    """HTTP 429 response should log status code and try next server."""
    from topo_shadow_box.core.osm import _query_overpass

    mock_request = MagicMock()
    mock_response_obj = MagicMock()
    mock_response_obj.status_code = 429

    async def mock_post(url, **kwargs):
        # Return a synchronous mock since raise_for_status() is sync in httpx
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "429 Too Many Requests",
                request=mock_request,
                response=mock_response_obj,
            )
        )
        return mock_resp

    with caplog.at_level(logging.WARNING, logger="topo_shadow_box.core.osm"):
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = mock_post
            mock_client_cls.return_value = mock_client

            result = await _query_overpass("test query")

    assert result == []
    assert any(
        r.name == "topo_shadow_box.core.osm" and r.levelno == logging.WARNING
        and "429" in r.message
        for r in caplog.records
    )


@pytest.mark.anyio
async def test_fetch_features_returns_explicit_message_when_empty():
    """fetch_features should clearly report zero results, not silently return empty dict."""
    from topo_shadow_box.state import state, Bounds
    from topo_shadow_box.core.models import OsmFeatureSet
    from topo_shadow_box.tools.data import register_data_tools

    state.bounds = Bounds(north=37.8, south=37.75, east=-122.4, west=-122.45, is_set=True)

    # Capture the registered tool function by intercepting mcp.tool() decoration
    registered_tools = {}
    mock_mcp = MagicMock()
    def capture_tool():
        def decorator(fn):
            registered_tools[fn.__name__] = fn
            return fn
        return decorator
    mock_mcp.tool = capture_tool
    register_data_tools(mock_mcp)

    fetch_features_fn = registered_tools["fetch_features"]

    with patch("topo_shadow_box.tools.data.fetch_osm_features", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = OsmFeatureSet(roads=[], water=[], buildings=[])
        result = await fetch_features_fn(include=["roads", "water", "buildings"])

    assert "none found" in result, f"Expected 'none found' in result, got: {result!r}"
    assert "check server logs" in result or "unexpected" in result, f"Expected hint in result, got: {result!r}"
