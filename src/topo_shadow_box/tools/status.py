"""Status tool: get_status."""

import json
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from ..state import state


def register_status_tools(mcp: FastMCP):

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def get_status() -> str:
        """Return a summary of the current model state.

        Shows what data has been loaded, current parameters, and which meshes
        have been generated.
        """
        return json.dumps(state.summary(), indent=2)
