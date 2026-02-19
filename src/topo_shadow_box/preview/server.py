"""HTTP + WebSocket preview server for Three.js viewer."""

import asyncio
import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from threading import Thread

import websockets

_ws_clients: set = set()
_http_server: HTTPServer | None = None
_ws_server = None

VIEWER_HTML = os.path.join(os.path.dirname(__file__), "viewer.html")


class PreviewHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            with open(VIEWER_HTML, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress HTTP logs


def _state_to_json(state) -> str:
    """Convert session state meshes to JSON for the viewer."""
    data = {"meshes": []}

    if state.terrain_mesh:
        data["meshes"].append({
            "name": state.terrain_mesh.name,
            "type": state.terrain_mesh.feature_type,
            "vertices": state.terrain_mesh.vertices,
            "faces": state.terrain_mesh.faces,
            "color": state.colors.terrain,
        })

    for fm in state.feature_meshes:
        color = getattr(state.colors, fm.feature_type, "#808080")
        data["meshes"].append({
            "name": fm.name,
            "type": fm.feature_type,
            "vertices": fm.vertices,
            "faces": fm.faces,
            "color": color,
        })

    if state.gpx_mesh:
        data["meshes"].append({
            "name": state.gpx_mesh.name,
            "type": state.gpx_mesh.feature_type,
            "vertices": state.gpx_mesh.vertices,
            "faces": state.gpx_mesh.faces,
            "color": state.colors.gpx_track,
        })

    return json.dumps(data)


async def _ws_handler(websocket):
    _ws_clients.add(websocket)
    try:
        async for _ in websocket:
            pass  # We only send to clients, not receive
    finally:
        _ws_clients.discard(websocket)


async def start_preview_server(state, http_port: int = 3333, ws_port: int = 3334):
    """Start the HTTP and WebSocket servers."""
    global _http_server, _ws_server

    # Start HTTP server in a thread
    _http_server = HTTPServer(("localhost", http_port), PreviewHandler)
    http_thread = Thread(target=_http_server.serve_forever, daemon=True)
    http_thread.start()

    # Start WebSocket server
    _ws_server = await websockets.serve(_ws_handler, "localhost", ws_port)

    # Send initial data after a brief delay for client connection
    asyncio.get_event_loop().call_later(1.0, lambda: asyncio.ensure_future(update_preview(state)))


async def update_preview(state):
    """Send updated mesh data to all connected WebSocket clients."""
    if not _ws_clients:
        return

    data = _state_to_json(state)
    await asyncio.gather(
        *[client.send(data) for client in _ws_clients],
        return_exceptions=True,
    )
