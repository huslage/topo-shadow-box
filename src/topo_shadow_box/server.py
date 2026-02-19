"""MCP server for topo-shadow-box.

Registers all tools and runs via stdio transport.
"""

from mcp.server.fastmcp import FastMCP

from .tools.area import register_area_tools
from .tools.data import register_data_tools
from .tools.model import register_model_tools
from .tools.generate import register_generate_tools
from .tools.preview import register_preview_tools
from .tools.export import register_export_tools
from .tools.status import register_status_tools

mcp = FastMCP(
    "topo-shadow-box",
    instructions="Generate 3D-printed shadow boxes with topographical terrain, map features, and GPX tracks",
)

# Register all tool groups
register_area_tools(mcp)
register_data_tools(mcp)
register_model_tools(mcp)
register_generate_tools(mcp)
register_preview_tools(mcp)
register_export_tools(mcp)
register_status_tools(mcp)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
