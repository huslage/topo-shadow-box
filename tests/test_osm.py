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
    def capture_tool(**kwargs):
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


@pytest.mark.anyio
async def test_osm_client_sends_user_agent():
    """OSM HTTP client must include User-Agent header."""
    from topo_shadow_box.core.osm import _query_overpass

    captured_init_kwargs = {}

    class CapturingClient:
        def __init__(self, **kwargs):
            captured_init_kwargs.update(kwargs)
            self._inner = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json = MagicMock(return_value={"elements": []})
            self._inner.post = AsyncMock(return_value=mock_resp)

        async def __aenter__(self):
            return self._inner

        async def __aexit__(self, *args):
            pass

    with patch("httpx.AsyncClient", CapturingClient):
        await _query_overpass("test query")

    assert "headers" in captured_init_kwargs, "httpx.AsyncClient should be initialized with headers"
    assert "User-Agent" in captured_init_kwargs["headers"], "headers should include User-Agent"
    assert "topo-shadow-box" in captured_init_kwargs["headers"]["User-Agent"]


@pytest.mark.anyio
async def test_fetch_osm_features_sleeps_between_queries():
    """fetch_osm_features should sleep 1s between Overpass queries (OSM rate limit)."""
    from topo_shadow_box.core.osm import fetch_osm_features

    sleep_calls = []

    async def mock_sleep(seconds):
        sleep_calls.append(seconds)

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value={"elements": []})

    with patch("asyncio.sleep", side_effect=mock_sleep):
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            await fetch_osm_features(
                north=37.8, south=37.75, east=-122.4, west=-122.45,
                feature_types=["roads", "water", "buildings"],
            )

    # 3 feature types → 2 inter-query sleeps minimum
    assert len(sleep_calls) >= 1, f"Should sleep between OSM queries. Got sleep_calls={sleep_calls}"
    assert all(s >= 1.0 for s in sleep_calls), f"Each sleep should be >= 1.0s. Got: {sleep_calls}"
