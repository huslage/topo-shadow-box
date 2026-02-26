"""MCP server smoke test: verifies the server starts and responds to initialize."""

import sys
import pytest
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession


@pytest.mark.anyio
async def test_mcp_server_initializes():
    """Server starts and responds to MCP initialize with correct name and tools."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "topo_shadow_box"],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            result = await session.initialize()

            assert result.serverInfo.name == "topo-shadow-box"


@pytest.mark.anyio
async def test_mcp_server_exposes_expected_tools():
    """Server exposes the expected MCP tools."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "topo_shadow_box"],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()

        tool_names = {t.name for t in tools.tools}
        assert "set_area_from_coordinates" in tool_names
        assert "set_area_from_gpx" in tool_names
        assert "fetch_elevation" in tool_names
        assert "generate_model" in tool_names
        assert "export_3mf" in tool_names
        assert "geocode_place" in tool_names
