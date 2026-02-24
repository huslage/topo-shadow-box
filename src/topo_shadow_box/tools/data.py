"""Data acquisition tools: fetch_elevation, fetch_features."""

from mcp.server.fastmcp import FastMCP

from ..state import state, ElevationData
from ..core.elevation import fetch_terrain_elevation
from ..core.osm import fetch_osm_features


def register_data_tools(mcp: FastMCP):

    @mcp.tool()
    async def fetch_elevation(resolution: int = 200) -> str:
        """Fetch terrain elevation data for the current area of interest.

        Uses AWS Terrain-RGB tiles (Terrarium format) for global elevation data.
        The elevation grid is interpolated to the requested resolution.

        Args:
            resolution: Grid size (points per axis). Default 200. Higher = more detail but slower.
        """
        if not state.bounds.is_set:
            return "Error: Set an area first with set_area_from_coordinates or set_area_from_gpx."

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

    @mcp.tool()
    async def fetch_features(include: list[str] | None = None) -> str:
        """Fetch OpenStreetMap features for the current area.

        Args:
            include: Feature types to fetch. Options: roads, water, buildings.
                     Default: all three.
        """
        if not state.bounds.is_set:
            return "Error: Set an area first with set_area_from_coordinates or set_area_from_gpx."

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

        counts = {k: len(v) for k, v in features.items() if v}
        return f"Features fetched: {counts}"
