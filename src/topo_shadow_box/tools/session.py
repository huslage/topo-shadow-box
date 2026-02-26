"""Session persistence tools: save_session, load_session."""

import json
import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from ..state import state, Bounds, ElevationData, ModelParams, Colors
from ..core.models import OsmFeatureSet

logger = logging.getLogger(__name__)


def _default_path() -> Path:
    return Path.home() / ".cache" / "topo-shadow-box" / "session.json"


def register_session_tools(mcp: FastMCP):

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True))
    def save_session(path: str | None = None) -> str:
        """Save the current session to a JSON file for later resumption.

        Saves bounds, model params, colors, and GPX tracks.
        Does NOT save the elevation grid or meshes (regenerate after loading).
        **Next:** load_session in a future session to restore this configuration.

        Args:
            path: Where to save. Default: ~/.cache/topo-shadow-box/session.json
        """
        save_path = Path(path) if path else _default_path()
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data: dict = {
            "bounds": None,
            "model_params": state.model_params.model_dump(),
            "colors": state.colors.model_dump(),
            "elevation_metadata": None,
            "gpx_tracks": [],
        }

        if state.bounds.is_set:
            data["bounds"] = {
                "north": state.bounds.north,
                "south": state.bounds.south,
                "east": state.bounds.east,
                "west": state.bounds.west,
            }

        if state.elevation.is_set:
            data["elevation_metadata"] = {
                "resolution": state.elevation.resolution,
                "min_elevation": state.elevation.min_elevation,
                "max_elevation": state.elevation.max_elevation,
            }

        if state.gpx_tracks:
            data["gpx_tracks"] = [
                {
                    "name": t.name,
                    "points": [{"lat": p.lat, "lon": p.lon, "elevation": p.elevation}
                               for p in t.points],
                }
                for t in state.gpx_tracks
            ]

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Session saved to %s", save_path)
        return f"Session saved to {save_path}"

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False))
    def load_session(path: str | None = None) -> str:
        """Load a previously saved session from a JSON file.

        Restores bounds, model params, colors, and GPX tracks.
        Clears elevation and meshes — you will need to re-run fetch_elevation
        and generate_model after loading.
        **Next:** fetch_elevation, then optionally fetch_features, then generate_model.

        Args:
            path: Path to load from. Default: ~/.cache/topo-shadow-box/session.json
        """
        from ..models import GpxTrack, GpxPoint

        load_path = Path(path) if path else _default_path()

        if not load_path.exists():
            return f"Error: Session file not found at {load_path}"

        try:
            with open(load_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return f"Error: Invalid session file — {e}"

        # Restore bounds
        if data.get("bounds"):
            b = data["bounds"]
            state.bounds = Bounds(
                north=b["north"], south=b["south"],
                east=b["east"], west=b["west"],
                is_set=True,
            )

        # Restore model params
        if data.get("model_params"):
            state.model_params = ModelParams(**data["model_params"])

        # Restore colors
        if data.get("colors"):
            state.colors = Colors(**data["colors"])

        # Restore GPX tracks
        if data.get("gpx_tracks"):
            state.gpx_tracks = [
                GpxTrack(
                    name=t["name"],
                    points=[GpxPoint(**p) for p in t["points"]],
                )
                for t in data["gpx_tracks"]
            ]
        else:
            state.gpx_tracks = []

        # Clear things that need regeneration
        state.elevation = ElevationData()
        state.features = OsmFeatureSet()
        state.terrain_mesh = None
        state.feature_meshes = []
        state.gpx_mesh = None

        restored = []
        if state.bounds.is_set:
            restored.append("bounds")
        restored.append("model_params")
        restored.append("colors")
        if state.gpx_tracks:
            restored.append(f"{len(state.gpx_tracks)} GPX track(s)")

        return (
            f"Session restored from {load_path}. "
            f"Restored: {', '.join(restored)}. "
            "Still needed: fetch_elevation, then generate_model."
        )
