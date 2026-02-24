"""Preview tool: launch/refresh Three.js viewer."""

import webbrowser
from mcp.server.fastmcp import FastMCP

from ..state import state
from ..preview.server import start_preview_server, update_preview


def register_preview_tools(mcp: FastMCP):

    @mcp.tool()
    async def preview() -> str:
        """Open or refresh the Three.js preview in the browser.

        Starts a local HTTP server with WebSocket updates on localhost:3333.
        If the preview is already running, it sends updated mesh data via WebSocket.
        """
        from ._prereqs import require_state
        try:
            require_state(state, mesh=True)
        except ValueError as e:
            return f"Error: {e}"

        if not state.preview_running:
            await start_preview_server(state)
            state.preview_running = True
            webbrowser.open(f"http://localhost:{state.preview_port}")
            return f"Preview opened at http://localhost:{state.preview_port}"
        else:
            await update_preview(state)
            return f"Preview updated at http://localhost:{state.preview_port}"
