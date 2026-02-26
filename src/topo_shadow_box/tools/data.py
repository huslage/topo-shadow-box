"""Data acquisition tools: fetch_elevation, fetch_features."""

import logging

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from ..state import state, ElevationData
from ..core.elevation import fetch_terrain_elevation
from ..core.osm import fetch_osm_features
from ._prereqs import require_state

logger = logging.getLogger(__name__)


def register_data_tools(mcp: FastMCP):

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True))
    async def fetch_elevation(resolution: int = 200) -> str:
        """Fetch terrain elevation data for the current area of interest.

        Uses AWS Terrain-RGB tiles (free, globally available).
        **Requires:** set_area_from_coordinates or set_area_from_gpx first.
        **Next:** Optionally fetch_features, then generate_model.

        Args:
            resolution: Grid points per axis (default 200). Higher = more detail
                but slower generate_model. Use 100 for quick previews.
        """
        try:
            require_state(state, bounds=True)
        except ValueError as e:
            return f"Error: {e}"

        b = state.bounds
        result = await fetch_terrain_elevation(
            north=b.north, south=b.south, east=b.east, west=b.west,
            resolution=resolution,
        )

        state.elevation = ElevationData(
            grid=result.grid,
            lats=result.lats,
            lons=result.lons,
            resolution=result.resolution,
            min_elevation=result.min_elevation,
            max_elevation=result.max_elevation,
            is_set=True,
        )

        # Clear meshes since elevation changed
        state.terrain_mesh = None
        state.feature_meshes = []
        state.gpx_mesh = None

        return (
            f"Elevation fetched: {resolution}x{resolution} grid, "
            f"range {result.min_elevation:.0f}m to {result.max_elevation:.0f}m "
            f"({result.max_elevation - result.min_elevation:.0f}m relief)"
        )

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True))
    async def fetch_features(include: list[str] | None = None) -> str:
        """Fetch OpenStreetMap roads, water, and buildings for the current area.

        **Requires:** set_area_from_coordinates or set_area_from_gpx first.
        **Next:** generate_model (features are optional — skip if not needed).

        Args:
            include: Feature types to fetch. Options: 'roads', 'water', 'buildings'.
                     Default: all three. Omit types you don't want in the model.
        """
        try:
            require_state(state, bounds=True)
        except ValueError as e:
            return f"Error: {e}"

        if include is None:
            include = ["roads", "water", "buildings"]

        b = state.bounds
        features = await fetch_osm_features(
            north=b.north, south=b.south, east=b.east, west=b.west,
            feature_types=include,
        )

        state.features = features

        # Clear feature meshes
        state.feature_meshes = []

        counts = {
            k: v for k, v in {
                "roads": len(features.roads),
                "water": len(features.water),
                "buildings": len(features.buildings),
            }.items() if k in include
        }
        if all(v == 0 for v in counts.values()):
            logger.debug("fetch_features returned zero results for types: %s", include)
            return f"Features fetched: none found (check server logs if unexpected) — {counts}"
        return f"Features fetched: {counts}"
