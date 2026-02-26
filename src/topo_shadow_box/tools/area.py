"""Area definition tools: set_area_from_coordinates, set_area_from_gpx."""

import math

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from ..state import state, Bounds, ElevationData
from ..core.gpx import parse_gpx_file
from ..core.coords import add_padding_to_bounds
from ..core.models import OsmFeatureSet


def register_area_tools(mcp: FastMCP):

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False))
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

        **Next:** Optionally call validate_area to check for problems,
        then fetch_elevation, then fetch_features (optional), then generate_model.

        Args:
            lat: Center latitude (degrees). Use with lon and radius_m.
            lon: Center longitude (degrees). Use with lat and radius_m.
            radius_m: Radius in meters around the center point.
            north/south/east/west: Explicit bounding box (degrees).
        """
        if lat is not None and lon is not None and radius_m is not None:
            bounds = add_padding_to_bounds(
                Bounds(north=lat, south=lat, east=lon, west=lon),
                padding_m=radius_m,
                is_set=True,
            )
        elif all(v is not None for v in [north, south, east, west]):
            bounds = Bounds(north=north, south=south, east=east, west=west, is_set=True)
        else:
            return "Error: Provide either (lat, lon, radius_m) or (north, south, east, west)."

        state.bounds = bounds
        # Clear downstream data when area changes
        state.elevation = ElevationData()
        state.features = OsmFeatureSet()
        state.terrain_mesh = None
        state.feature_meshes = []
        state.gpx_mesh = None

        return (
            f"Area set: N={bounds.north:.6f}, S={bounds.south:.6f}, "
            f"E={bounds.east:.6f}, W={bounds.west:.6f} "
            f"(~{bounds.lat_range * 111_000:.0f}m x {bounds.lon_range * 111_000:.0f}m)"
        )

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False))
    def set_area_from_gpx(file_path: str, padding_m: float = 500.0) -> str:
        """Load a GPX file and use its bounds (plus padding) as the area of interest.

        Also stores the GPX tracks for rendering as a raised strip on the terrain.
        **Next:** fetch_elevation, then fetch_features (optional), then generate_model.

        Args:
            file_path: Absolute path to a .gpx file.
            padding_m: Padding in meters around the GPX bounds (default 500m).
        """
        gpx_data = parse_gpx_file(file_path)

        if not gpx_data["bounds"]:
            return "Error: GPX file has no track data with bounds."

        b = gpx_data["bounds"]
        raw_bounds = Bounds(north=b["north"], south=b["south"], east=b["east"], west=b["west"])
        padded = add_padding_to_bounds(raw_bounds, padding_m=padding_m, is_set=True)

        state.bounds = padded
        state.gpx_tracks = gpx_data["tracks"]
        state.gpx_waypoints = gpx_data.get("waypoints", [])

        # Clear downstream data
        state.elevation = ElevationData()
        state.features = OsmFeatureSet()
        state.terrain_mesh = None
        state.feature_meshes = []
        state.gpx_mesh = None

        total_points = sum(len(t.points) for t in state.gpx_tracks)
        return (
            f"GPX loaded: {len(state.gpx_tracks)} track(s), {total_points} points, "
            f"{len(state.gpx_waypoints)} waypoint(s). "
            f"Area set: N={padded.north:.6f}, S={padded.south:.6f}, "
            f"E={padded.east:.6f}, W={padded.west:.6f} "
            f"(padding: {padding_m}m)"
        )

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def validate_area() -> str:
        """Check the current area for potential problems before fetching data.

        **Requires:** set_area_from_coordinates or set_area_from_gpx first.
        **Next:** fetch_elevation, then fetch_features (optional), then generate_model.

        Checks area size, and elevation relief if elevation has been fetched.
        Returns warnings but does not block the pipeline.
        """
        if not state.bounds.is_set:
            return "Error: Set an area first with set_area_from_coordinates or set_area_from_gpx."

        b = state.bounds
        # Approximate span in meters (1 degree lat ≈ 111km)
        lat_m = b.lat_range * 111_000
        lon_m = b.lon_range * 111_000 * abs(math.cos(math.radians(b.center_lat)))
        min_span_m = min(lat_m, lon_m)
        max_span_m = max(lat_m, lon_m)

        warnings = []

        if min_span_m < 100:
            return (
                f"Error: Area too small ({min_span_m:.0f}m minimum span). "
                "Use a larger area for meaningful terrain detail."
            )
        if max_span_m > 500_000:
            warnings.append(
                f"Very large area ({max_span_m / 1000:.0f}km span) — "
                "fetching will be slow and terrain detail will be low."
            )

        if state.elevation.is_set:
            relief = state.elevation.max_elevation - state.elevation.min_elevation
            if relief < 20:
                warnings.append(
                    f"Low elevation relief ({relief:.0f}m) — "
                    "model will print nearly flat. Consider a more mountainous area."
                )

        if warnings:
            return "Warnings: " + " | ".join(warnings)
        return "Area looks good."
