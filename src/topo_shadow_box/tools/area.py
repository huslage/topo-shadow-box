"""Area definition tools: set_area_from_coordinates, set_area_from_gpx."""

import math

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from ..state import state, Bounds, ElevationData
from ..core.gpx import parse_gpx_file
from ..core.coords import add_padding_to_bounds
from ..core.models import OsmFeatureSet
from ..models import GeocodeCandidate


def _set_area_from_candidate(candidate: GeocodeCandidate) -> Bounds:
    """Set session area bounds from a geocode candidate and clear downstream state."""
    bounds = Bounds(
        north=candidate.bbox_north,
        south=candidate.bbox_south,
        east=candidate.bbox_east,
        west=candidate.bbox_west,
        is_set=True,
    )
    state.bounds = bounds
    state.pending_geocode_candidates = []
    state.elevation = ElevationData()
    state.features = OsmFeatureSet()
    state.terrain_mesh = None
    state.feature_meshes = []
    state.gpx_mesh = None
    return bounds


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

        **Prior:** If you only have a place name, call geocode_place first to get coordinates.

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

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False))
    def geocode_place(query: str, limit: int = 5) -> str:
        """Search for a place by name and set the area of interest.

        Use this when the user provides a place name but no coordinates or GPX file.
        If the user provides a GPX file, use set_area_from_gpx instead — no geocoding needed.

        - 1 result: area is set automatically, no further action needed.
        - 2+ results: raises an error with a numbered list. Present it to the user,
          wait for them to reply with a number, then call select_geocode_result.

        **Next (1 result):** fetch_elevation directly.
        **Next (2+ results):** select_geocode_result with the user's chosen number.

        Args:
            query: Place name to search for (e.g., "Mount Hood", "Grand Canyon", "Portland, Oregon").
            limit: Maximum number of candidates to return (1–10, default 5).
        """
        limit = max(1, min(10, limit))

        try:
            response = httpx.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": query, "format": "json", "limit": limit},
                headers={"User-Agent": "topo-shadow-box/1.0"},
                timeout=10.0,
            )
            response.raise_for_status()
            results = response.json()
        except httpx.HTTPStatusError as exc:
            return f"Error: Nominatim returned HTTP {exc.response.status_code}."
        except Exception as exc:
            return f"Error contacting geocoding service: {exc}"

        if not results:
            return f"No locations found for '{query}'. Try a more specific name or add a region (e.g., 'Portland, Oregon')."

        candidates = []
        for item in results:
            bbox = item.get("boundingbox", [])
            # Nominatim boundingbox order: [south, north, west, east]
            candidates.append(
                GeocodeCandidate(
                    display_name=item["display_name"],
                    lat=float(item["lat"]),
                    lon=float(item["lon"]),
                    place_type=item.get("type", "unknown"),
                    bbox_south=float(bbox[0]) if len(bbox) >= 4 else float(item["lat"]),
                    bbox_north=float(bbox[1]) if len(bbox) >= 4 else float(item["lat"]),
                    bbox_west=float(bbox[2]) if len(bbox) >= 4 else float(item["lon"]),
                    bbox_east=float(bbox[3]) if len(bbox) >= 4 else float(item["lon"]),
                )
            )

        # Single result: auto-select without requiring user input
        if len(candidates) == 1:
            c = candidates[0]
            bounds = _set_area_from_candidate(c)
            return (
                f"Found 1 result: '{c.display_name}' (auto-selected). "
                f"Area set: N={bounds.north:.6f}, S={bounds.south:.6f}, "
                f"E={bounds.east:.6f}, W={bounds.west:.6f} "
                f"(~{bounds.lat_range * 111_000:.0f}m x {bounds.lon_range * 111_000:.0f}m)"
            )

        state.pending_geocode_candidates = candidates

        lines = [f"Found {len(candidates)} location(s) for '{query}':\n"]
        for i, c in enumerate(candidates, 1):
            lines.append(
                f"{i}. {c.display_name}\n"
                f"   Type: {c.place_type} | Center: {c.lat:.5f}, {c.lon:.5f}\n"
                f"   Bbox: N={c.bbox_north:.5f}, S={c.bbox_south:.5f}, "
                f"E={c.bbox_east:.5f}, W={c.bbox_west:.5f}"
            )
        lines.append(
            f"\nUser input required: ask the user which number (1–{len(candidates)}) "
            "they want, then call select_geocode_result with that number."
        )
        raise ValueError("\n".join(lines))

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False))
    def select_geocode_result(number: int) -> str:
        """Select a geocode candidate by number and set it as the area of interest.

        Only call this after geocode_place returned multiple candidates AND the user
        has replied with their chosen number. Single results are auto-selected by
        geocode_place — do not call this in that case.

        **Requires:** geocode_place called first with multiple results (candidates stored in session).
        **Next:** fetch_elevation, then fetch_features (optional), then generate_model.

        Args:
            number: 1-based index of the candidate the user selected.
        """
        if not state.pending_geocode_candidates:
            return "Error: No geocode search results pending. Call geocode_place first."

        n = len(state.pending_geocode_candidates)
        if number < 1 or number > n:
            return f"Error: Invalid selection {number}. Choose a number between 1 and {n}."

        candidate = state.pending_geocode_candidates[number - 1]
        bounds = _set_area_from_candidate(candidate)

        return (
            f"Area set from '{candidate.display_name}': "
            f"N={bounds.north:.6f}, S={bounds.south:.6f}, "
            f"E={bounds.east:.6f}, W={bounds.west:.6f} "
            f"(~{bounds.lat_range * 111_000:.0f}m x {bounds.lon_range * 111_000:.0f}m)"
        )
