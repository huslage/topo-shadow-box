"""Area definition tools: set_area_from_coordinates, set_area_from_gpx."""

from mcp.server.fastmcp import FastMCP

from ..state import state, Bounds
from ..core.gpx import parse_gpx_file
from ..core.coords import add_padding_to_bounds


def register_area_tools(mcp: FastMCP):

    @mcp.tool()
    def set_area_from_coordinates(
        lat: float | None = None,
        lon: float | None = None,
        radius_m: float | None = None,
        north: float | None = None,
        south: float | None = None,
        east: float | None = None,
        west: float | None = None,
    ) -> str:
        """Define the area of interest by center+radius or explicit bounding box.

        Either provide (lat, lon, radius_m) for a circular area,
        or (north, south, east, west) for a rectangular bounding box.
        """
        if lat is not None and lon is not None and radius_m is not None:
            bounds = add_padding_to_bounds(
                Bounds(north=lat, south=lat, east=lon, west=lon),
                padding_m=radius_m,
            )
        elif all(v is not None for v in [north, south, east, west]):
            bounds = Bounds(north=north, south=south, east=east, west=west)
        else:
            return "Error: Provide either (lat, lon, radius_m) or (north, south, east, west)."

        state.bounds = bounds
        # Clear downstream data when area changes
        state.elevation = state.elevation.__class__()
        state.features = {}
        state.terrain_mesh = None
        state.feature_meshes = []
        state.gpx_mesh = None

        return (
            f"Area set: N={bounds.north:.6f}, S={bounds.south:.6f}, "
            f"E={bounds.east:.6f}, W={bounds.west:.6f} "
            f"(~{bounds.lat_range * 111_000:.0f}m x {bounds.lon_range * 111_000:.0f}m)"
        )

    @mcp.tool()
    def set_area_from_gpx(file_path: str, padding_m: float = 500.0) -> str:
        """Load a GPX file and use its bounds (plus padding) as the area of interest.

        Also stores the GPX tracks for later rendering.

        Args:
            file_path: Absolute path to a .gpx file
            padding_m: Padding in meters around the GPX bounds (default 500m)
        """
        gpx_data = parse_gpx_file(file_path)

        if not gpx_data["bounds"]:
            return "Error: GPX file has no track data with bounds."

        b = gpx_data["bounds"]
        raw_bounds = Bounds(north=b["north"], south=b["south"], east=b["east"], west=b["west"])
        padded = add_padding_to_bounds(raw_bounds, padding_m=padding_m)

        state.bounds = padded
        state.gpx_tracks = gpx_data["tracks"]
        state.gpx_waypoints = gpx_data.get("waypoints", [])

        # Clear downstream data
        state.elevation = state.elevation.__class__()
        state.features = {}
        state.terrain_mesh = None
        state.feature_meshes = []
        state.gpx_mesh = None

        total_points = sum(len(t["points"]) for t in state.gpx_tracks)
        return (
            f"GPX loaded: {len(state.gpx_tracks)} track(s), {total_points} points, "
            f"{len(state.gpx_waypoints)} waypoint(s). "
            f"Area set: N={padded.north:.6f}, S={padded.south:.6f}, "
            f"E={padded.east:.6f}, W={padded.west:.6f} "
            f"(padding: {padding_m}m)"
        )
