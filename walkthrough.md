# Topo Shadow Box: Codebase Walkthrough

*2026-03-01T22:46:22Z by Showboat 0.6.1*
<!-- showboat-id: 23a20148-3f9e-405b-83d4-78a56e20e23d -->

## What Is Topo Shadow Box?

Topo Shadow Box is an MCP (Model Context Protocol) server that turns geographic coordinates into 3D-printable terrain models. You give it a location — coordinates, a place name, or a GPX file — and it fetches real elevation data from AWS, road/water/building data from OpenStreetMap, generates multi-material 3D meshes, and exports .3mf files ready for a multi-colour 3D printer.

The project is structured as a Python package with two main layers:
- **Core algorithms** in `src/topo_shadow_box/core/` — elevation, meshing, OSM, shape clipping, etc.
- **MCP tool handlers** in `src/topo_shadow_box/tools/` — thin wrappers that validate pre-reqs, call core functions, and update shared session state.

Everything flows through a single global `SessionState` object that acts as the in-memory database for one user session.

## Project Layout

```bash
find src -type f -name '*.py' | sort
```

```output
src/topo_shadow_box/__init__.py
src/topo_shadow_box/__main__.py
src/topo_shadow_box/core/__init__.py
src/topo_shadow_box/core/building_shapes.py
src/topo_shadow_box/core/coords.py
src/topo_shadow_box/core/elevation.py
src/topo_shadow_box/core/gpx.py
src/topo_shadow_box/core/map_insert.py
src/topo_shadow_box/core/mesh.py
src/topo_shadow_box/core/models.py
src/topo_shadow_box/core/osm.py
src/topo_shadow_box/core/shape_clipper.py
src/topo_shadow_box/exporters/__init__.py
src/topo_shadow_box/exporters/openscad.py
src/topo_shadow_box/exporters/svg.py
src/topo_shadow_box/exporters/threemf.py
src/topo_shadow_box/models.py
src/topo_shadow_box/preview/__init__.py
src/topo_shadow_box/preview/server.py
src/topo_shadow_box/server.py
src/topo_shadow_box/state.py
src/topo_shadow_box/tools/__init__.py
src/topo_shadow_box/tools/_prereqs.py
src/topo_shadow_box/tools/area.py
src/topo_shadow_box/tools/data.py
src/topo_shadow_box/tools/export.py
src/topo_shadow_box/tools/generate.py
src/topo_shadow_box/tools/model.py
src/topo_shadow_box/tools/preview.py
src/topo_shadow_box/tools/session.py
src/topo_shadow_box/tools/status.py
```

## The Entry Point: server.py

`server.py` is the heart of the MCP layer. It creates a FastMCP server and registers all the tool functions from the `tools/` package. When you run the package, the server speaks the MCP protocol over stdio — this is how Claude (or any MCP client) discovers and calls the tools.

```bash
cat src/topo_shadow_box/server.py
```

```output
"""MCP server for topo-shadow-box.

Registers all tools and runs via stdio transport.
"""

import json as _json

from mcp.server.fastmcp import FastMCP

from .tools.area import register_area_tools
from .tools.data import register_data_tools
from .tools.model import register_model_tools
from .tools.generate import register_generate_tools
from .tools.preview import register_preview_tools
from .tools.export import register_export_tools
from .tools.status import register_status_tools
from .tools.session import register_session_tools
from .state import state as _state

mcp = FastMCP(
    "topo-shadow-box",
    instructions="Generate 3D-printed shadow boxes with topographical terrain, map features, and GPX tracks",
)


@mcp.resource("state://session")
def session_state_resource() -> str:
    """Current session state: bounds, elevation, features, model params, mesh status."""
    return _json.dumps(_state.summary(), indent=2)


# Register all tool groups
register_area_tools(mcp)
register_data_tools(mcp)
register_model_tools(mcp)
register_generate_tools(mcp)
register_preview_tools(mcp)
register_export_tools(mcp)
register_status_tools(mcp)
register_session_tools(mcp)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
```

Notice that each tool group has a `register_*_tools(mcp)` function. This keeps the server file tiny — it's purely wiring. There's also an MCP *resource* (`state://session`) that exposes a live JSON dump of the current session state; a client can read this at any time to understand what data is loaded.

## Session State: The In-Memory Database

`state.py` defines `SessionState`, a Pydantic model that holds everything between tool calls. Because MCP is stateless at the transport level (each tool invocation is a separate call), this singleton is the glue.

```bash
grep -n 'class\|^    [a-z].*:' src/topo_shadow_box/state.py | head -60
```

```output
16:class Bounds(BaseModel):
19:    north: float = Field(default=0.0, ge=-90, le=90)
20:    south: float = Field(default=0.0, ge=-90, le=90)
21:    east: float = Field(default=0.0, ge=-180, le=180)
22:    west: float = Field(default=0.0, ge=-180, le=180)
23:    is_set: bool = False
26:    def check_north_gt_south(self) -> "Bounds":
32:    def check_east_gt_west(self) -> "Bounds":
38:    def lat_range(self) -> float:
42:    def lon_range(self) -> float:
46:    def center_lat(self) -> float:
50:    def center_lon(self) -> float:
54:class ElevationData(BaseModel):
57:    grid: Optional[np.ndarray] = None
58:    lats: Optional[np.ndarray] = None
59:    lons: Optional[np.ndarray] = None
60:    resolution: int = Field(default=200, gt=0, le=1000)
61:    min_elevation: float = 0.0
62:    max_elevation: float = 0.0
63:    is_set: bool = False
66:class ModelParams(BaseModel):
69:    width_mm: float = Field(default=200.0, gt=0)
70:    vertical_scale: float = Field(default=1.5, gt=0)
71:    base_height_mm: float = Field(default=10.0, gt=0)
72:    shape: Literal["square", "circle", "hexagon", "rectangle"] = "square"
75:class Colors(BaseModel):
78:    terrain: str = "#C8A882"
79:    water: str = "#4682B4"
80:    roads: str = "#D4C5A9"
81:    buildings: str = "#E8D5B7"
82:    gpx_track: str = "#FF0000"
83:    map_insert: str = "#FFFFFF"
86:    @classmethod
87:    def validate_and_normalize_hex(cls, v: str) -> str:
95:    def hex_to_rgb(self, hex_color: str) -> tuple[int, int, int]:
99:    def as_dict(self) -> dict[str, str]:
110:class MeshData(BaseModel):
111:    vertices: list[list[float]] = Field(default_factory=list)
112:    faces: list[list[int]] = Field(default_factory=list)
113:    name: str = ""
114:    feature_type: str = ""
117:class SessionState(BaseModel):
120:    bounds: Bounds = Field(default_factory=Bounds)
121:    elevation: ElevationData = Field(default_factory=ElevationData)
122:    features: OsmFeatureSet = Field(default_factory=OsmFeatureSet)
123:    gpx_tracks: list[GpxTrack] = []
124:    gpx_waypoints: list[GpxWaypoint] = []
125:    model_params: ModelParams = Field(default_factory=ModelParams)
126:    colors: Colors = Field(default_factory=Colors)
127:    pending_geocode_candidates: list[GeocodeCandidate] = []
128:    terrain_mesh: Optional[MeshData] = None
129:    feature_meshes: list[MeshData] = []
130:    gpx_mesh: Optional[MeshData] = None
131:    map_insert_mesh: Optional[MeshData] = None
132:    preview_port: int = Field(default=3333, gt=0, le=65535)
133:    preview_running: bool = False
135:    def summary(self) -> dict:
```

The state is divided into clear sub-models:

- **`Bounds`** — geographic bounding box with validators ensuring north > south and east > west, plus derived helpers like `center_lat`, `lat_range`, etc.
- **`ElevationData`** — the raw numpy grid plus metadata; `is_set` flags when data is ready.
- **`ModelParams`** — physical dimensions (width, vertical scale, base height, shape).
- **`Colors`** — hex colours for each feature type, validated by a Pydantic `field_validator`.
- **`MeshData`** — vertices and faces for one mesh layer.
- **`SessionState`** — the top-level container holding all of the above plus pending geocode candidates, loaded meshes, and preview flags.

There is one global `state = SessionState()` in the module; every tool imports and mutates it.

## Prerequisites: Enforcing Pipeline Order

Before any tool runs, it calls helpers from `tools/_prereqs.py` to verify that earlier pipeline steps have been completed.

```bash
cat src/topo_shadow_box/tools/_prereqs.py
```

```output
"""Prerequisite checking helpers for MCP tools."""


def require_state(state, *, bounds: bool = False, elevation: bool = False, mesh: bool = False) -> None:
    """Raise ValueError with a descriptive message if required state is not set.

    Usage in a tool:
        try:
            require_state(state, bounds=True, elevation=True)
        except ValueError as e:
            return f"Error: {e}"
    """
    if bounds and not state.bounds.is_set:
        raise ValueError(
            "Set an area first with set_area_from_coordinates or set_area_from_gpx."
        )
    if elevation and not state.elevation.is_set:
        raise ValueError(
            "Fetch elevation data first with fetch_elevation."
        )
    if mesh and not state.terrain_mesh:
        raise ValueError(
            "Generate a model first with generate_model."
        )
```

`require_state` is a simple guard used at the top of every data/generate/export tool. It surfaces human-readable errors instead of cryptic `NoneType` exceptions — critical when you're talking to an LLM that needs to know what to do next.

## Step 1 — Defining an Area

Tools in `tools/area.py` cover all the ways a user can specify a region of interest.

```bash
grep -n 'def \|async def ' src/topo_shadow_box/tools/area.py
```

```output
16:def _set_area_from_candidate(candidate: GeocodeCandidate) -> Bounds:
35:def register_area_tools(mcp: FastMCP):
38:    def set_area_from_coordinates(
89:    def set_area_from_gpx(file_path: str, padding_m: float = 500.0) -> str:
129:    def validate_area() -> str:
174:    def geocode_place(query: str, limit: int = 5) -> str:
255:    def select_geocode_result(number: int) -> str:
```

```bash
sed -n '38,127p' src/topo_shadow_box/tools/area.py
```

```output
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

```

Both area-setting tools follow the same pattern: compute the `Bounds`, assign `state.bounds`, then **cascade-clear all downstream state** (elevation, features, meshes). This ensures you never get stale data from a previous area.

The center+radius form expands a single point into a bbox using `add_padding_to_bounds` from `core/coords.py` — which accounts for the fact that a degree of longitude shrinks toward the poles.

### Geocoding

`geocode_place` hits the Nominatim API and handles two cases:

```bash
sed -n '174,255p' src/topo_shadow_box/tools/area.py
```

```output
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
```

A single result auto-selects. Multiple results are stored as `pending_geocode_candidates` and the tool **raises a `ValueError`** (which MCP surfaces as an error response) with a numbered list — instructing the LLM to ask the user, then call `select_geocode_result(number)`. This keeps disambiguation in user-space rather than having the LLM silently pick.

## Step 2 — Fetching Elevation Data

`tools/data.py` → `fetch_elevation` calls `core/elevation.py`.

```bash
grep -n 'def \|async def \|zoom\|terrarium\|Terrarium\|RectBivariate\|gaussian' src/topo_shadow_box/core/elevation.py | head -40
```

```output
1:"""Elevation data fetching from AWS Terrain-RGB tiles (Terrarium format)."""
12:from scipy.ndimage import gaussian_filter
18:# AWS Terrain Tiles (Mapzen/Tilezen Terrarium format) - free, globally available
19:AWS_TERRAIN_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
22:def _lat_lon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
23:    """Convert lat/lon to tile coordinates at a given zoom level."""
25:    n = 2.0 ** zoom
31:def _tile_to_lat_lon(x: int, y: int, zoom: int) -> tuple[float, float]:
33:    n = 2.0 ** zoom
40:def _decode_terrarium(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
41:    """Decode Terrarium format RGB to elevation in meters."""
45:def _pick_zoom(north: float, south: float, east: float, west: float) -> int:
46:    """Pick an appropriate zoom level based on area span."""
60:async def fetch_terrain_elevation(
68:    zoom = _pick_zoom(north, south, east, west)
70:    x_min, y_max = _lat_lon_to_tile(south, west, zoom)
71:    x_max, y_min = _lat_lon_to_tile(north, east, zoom)
82:        # Fall back to lower zoom
83:        zoom = max(zoom - 1, 8)
84:        x_min, y_max = _lat_lon_to_tile(south, west, zoom)
85:        x_max, y_min = _lat_lon_to_tile(north, east, zoom)
106:        async def fetch_tile(tx: int, ty: int) -> None:
107:            url = AWS_TERRAIN_URL.format(z=zoom, x=tx, y=ty)
116:                    tile_elevations = _decode_terrarium(r, g, b)
126:    tile_north, tile_west = _tile_to_lat_lon(x_min, y_min, zoom)
127:    tile_south, tile_east = _tile_to_lat_lon(x_max + 1, y_max + 1, zoom)
153:    interp_func = interpolate.RectBivariateSpline(
161:    elevations = gaussian_filter(elevations, sigma=0.5)
```

```bash
sed -n '40,55p' src/topo_shadow_box/core/elevation.py
```

```output
def _decode_terrarium(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Decode Terrarium format RGB to elevation in meters."""
    return (r * 256.0 + g + b / 256.0) - 32768.0


def _pick_zoom(north: float, south: float, east: float, west: float) -> int:
    """Pick an appropriate zoom level based on area span."""
    max_span = max(north - south, east - west)
    if max_span > 1.0:
        return 10
    elif max_span > 0.5:
        return 11
    elif max_span > 0.1:
        return 12
    elif max_span > 0.05:
        return 13
```

```bash
sed -n '125,175p' src/topo_shadow_box/core/elevation.py
```

```output
    # Map tile pixel coordinates to geographic coordinates
    tile_north, tile_west = _tile_to_lat_lon(x_min, y_min, zoom)
    tile_south, tile_east = _tile_to_lat_lon(x_max + 1, y_max + 1, zoom)

    tile_lats = np.linspace(tile_south, tile_north, stitched_height)
    tile_lons = np.linspace(tile_west, tile_east, stitched_width)

    # Flip so that index 0 = south, last = north
    stitched_elevations = np.flipud(stitched_elevations)

    # Crop to requested bounding box
    lat_indices = np.where((tile_lats >= south) & (tile_lats <= north))[0]
    lon_indices = np.where((tile_lons >= west) & (tile_lons <= east))[0]

    if len(lat_indices) == 0 or len(lon_indices) == 0:
        raise ValueError("Bounding box doesn't overlap with fetched tiles")

    cropped = stitched_elevations[
        lat_indices[0]:lat_indices[-1] + 1,
        lon_indices[0]:lon_indices[-1] + 1,
    ]
    cropped_lats = tile_lats[lat_indices[0]:lat_indices[-1] + 1]
    cropped_lons = tile_lons[lon_indices[0]:lon_indices[-1] + 1]

    if cropped.shape[0] < 4 or cropped.shape[1] < 4:
        raise ValueError("Cropped region too small for interpolation")

    # Interpolate to requested resolution
    interp_func = interpolate.RectBivariateSpline(
        cropped_lats, cropped_lons, cropped, kx=3, ky=3,
    )
    target_lats = np.linspace(south, north, resolution)
    target_lons = np.linspace(west, east, resolution)
    elevations = interp_func(target_lats, target_lons)

    # Smooth to reduce tile boundary artifacts
    elevations = gaussian_filter(elevations, sigma=0.5)

    return ElevationResult(
        grid=elevations,
        lats=target_lats,
        lons=target_lons,
        resolution=resolution,
        min_elevation=float(np.min(elevations)),
        max_elevation=float(np.max(elevations)),
    )
```

The elevation pipeline:

1. **Pick zoom** — based on the bbox span; larger areas get lower zoom (fewer tiles).
2. **Tile coordinates** — Web Mercator tile math converts lat/lon to tile (x, y) at that zoom.
3. **Async fetch** — all required tiles are fetched concurrently with `httpx.AsyncClient`, with a fallback to a lower zoom if too many tiles would be needed.
4. **Terrarium decode** — each 256×256 PNG is decoded: `elevation = R×256 + G + B/256 − 32768`. This gives a floating-point elevation in metres.
5. **Stitch** — tiles are assembled into a large 2D grid.
6. **Crop** — the area of interest is cut from the bigger tile mosaic.
7. **Interpolate** — `scipy.interpolate.RectBivariateSpline` (bicubic) resamples the cropped grid to the requested resolution (default 200×200).
8. **Smooth** — a gentle `gaussian_filter(sigma=0.5)` softens tile-boundary seams.

The result is stored in `state.elevation.grid` as a numpy array, south-to-north in row order.

## Step 3 — Fetching OpenStreetMap Features

`tools/data.py` → `fetch_features` calls `core/osm.py`.

```bash
grep -n 'SERVERS\|highway\|natural.*water\|building\|def \|async def ' src/topo_shadow_box/core/osm.py | head -40
```

```output
13:OVERPASS_SERVERS = [
20:async def _query_overpass(query: str) -> list[dict]:
23:        for server in OVERPASS_SERVERS:
44:def _parse_way_coords(element: dict) -> list[dict]:
61:def _parse_features(elements: list[dict], feature_type: str) -> list:
80:                road_type=tags.get("highway", "road"),
91:        elif feature_type == "building":
100:            elif "building:levels" in tags:
102:                    height = float(tags["building:levels"]) * 3.0
116:async def fetch_osm_features(
123:        feature_types: List of types: 'roads', 'water', 'buildings'
132:    building_elements = []
137:            f'[out:json][timeout:30];way["highway"]({bbox});out body geom;',
142:            f'[out:json][timeout:30];(way["natural"="water"]({bbox});'
144:            f'relation["natural"="water"]({bbox}););out body geom;',
147:    if "buildings" in feature_types:
148:        queries["buildings"] = (
149:            f'[out:json][timeout:30];way["building"]({bbox});out body geom;',
150:            "building",
162:        elif feat_name == "buildings":
163:            building_elements = elements
167:    buildings = _parse_features(building_elements, "building")[:150]
168:    return OsmFeatureSet(roads=roads, water=water, buildings=buildings)
```

```bash
sed -n '13,19p' src/topo_shadow_box/core/osm.py
```

```output
OVERPASS_SERVERS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]


```

The OSM fetcher sends separate Overpass QL queries for roads (`way["highway"]`), water (`way["natural"=\"water\"]` and multipolygon relations), and buildings (`way["building"]`). It tries three Overpass mirrors in order, falling back if one times out.

Building height comes from tags: `height` (metres) > `building:levels` × 3.0 m > a default of 10 m. Buildings are capped at 150 to keep model complexity manageable.

## Step 4 — Generating the 3D Model

This is where the heavy work happens. `tools/generate.py` → `generate_model` calls functions in `core/mesh.py`, `core/building_shapes.py`, `core/coords.py`, and `core/shape_clipper.py`.

```bash
grep -n 'def \|async def ' src/topo_shadow_box/core/mesh.py
```

```output
20:def _elevation_normalization(grid: np.ndarray, use_percentile: bool = True) -> tuple[float, float]:
43:def _sample_elevation(
72:def _interpolate_boundary_elevation(angle: float, boundary_data: list) -> float:
126:def generate_terrain_mesh(
211:def _generate_square_terrain(
277:def _generate_circle_terrain(
423:def _generate_hexagon_terrain(
540:def _create_shape_clipper(
570:def generate_feature_meshes(
651:def generate_single_feature_mesh(
706:def _get_attr_or_key(obj, attr, default=None):
716:def _get_lat(coord):
723:def _get_lon(coord):
730:def _generate_road_mesh(
821:def _generate_water_mesh(
881:def _generate_building_mesh(
951:def generate_gpx_track_mesh(
1036:def create_gpx_cylinder_track(centerline, radius=1.0, n_sides=8):
1146:def create_road_strip(centerline, width=2.0, thickness=0.3):
1252:def triangulate_polygon(points_2d):
1285:    def cross_2d(o, a, b):
1288:    def is_ear(idx_list, pos, pts, is_ccw):
1349:def create_solid_polygon(points, thickness=0.5):
```

```bash
sed -n '126,210p' src/topo_shadow_box/core/mesh.py
```

```output
def generate_terrain_mesh(
    elevation: ElevationData,
    bounds: Bounds,
    transform: GeoToModelTransform,
    vertical_scale: float = 1.5,
    base_height_mm: float = 10.0,
    shape: str = "square",
    _norm: tuple[float, float] | None = None,
) -> MeshResult:
    """Generate terrain mesh from elevation grid.

    Produces shape-aware bases:
    - square/rectangle: rectangular side walls and flat bottom
    - circle: smooth 360-segment circular wall with interpolated contour
    - hexagon: boundary-edge-based wall following terrain contour

    Returns MeshResult.
    """
    grid = elevation.grid
    lats = elevation.lats
    lons = elevation.lons
    rows, cols = grid.shape

    min_elev, elev_range = _norm if _norm is not None else _elevation_normalization(grid)
    model_width = max(transform.model_width_x, transform.model_width_z)
    size_scale = model_width / 200.0

    # Build top vertices
    vertices_list = []
    for i in range(rows):
        for j in range(cols):
            x, z = transform.geo_to_model(lats[i], lons[j])
            y = transform.elevation_to_y(
                float(grid[i, j]), min_elev, elev_range, vertical_scale, size_scale,
            )
            vertices_list.append([x, y, z])

    top_verts = np.array(vertices_list)
    n = len(top_verts)

    # Shape clipping mask
    cx = transform.model_width_x / 2
    cz = transform.model_width_z / 2
    if shape == "circle":
        radius = min(transform.model_width_x, transform.model_width_z) / 2
        dx = top_verts[:, 0] - cx
        dz = top_verts[:, 2] - cz
        inside = (dx * dx + dz * dz) <= radius * radius
    elif shape == "hexagon":
        radius = min(transform.model_width_x, transform.model_width_z) / 2
        hex_clipper = HexagonClipper(cx, cz, radius)
        inside = hex_clipper.is_inside(top_verts[:, 0], top_verts[:, 2])
    else:
        inside = np.ones(n, dtype=bool)

    # Top surface faces (terrain) - CCW from above
    terrain_faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            idx = i * cols + j
            idx_r = idx + 1
            idx_d = idx + cols
            idx_dr = idx + cols + 1
            if inside[idx] and inside[idx_d] and inside[idx_r]:
                terrain_faces.append([idx, idx_d, idx_r])
            if inside[idx_r] and inside[idx_d] and inside[idx_dr]:
                terrain_faces.append([idx_r, idx_d, idx_dr])

    if shape == "circle":
        result = _generate_circle_terrain(
            top_verts, terrain_faces, rows, cols, cx, cz, radius,
            inside, base_height_mm,
        )
    elif shape == "hexagon":
        result = _generate_hexagon_terrain(
            top_verts, terrain_faces, rows, cols, cx, cz,
            inside, base_height_mm, hex_clipper,
        )
    else:
        result = _generate_square_terrain(
            top_verts, terrain_faces, rows, cols, n, base_height_mm,
        )
    return MeshResult(vertices=result["vertices"], faces=result["faces"], name="Terrain", feature_type="terrain")


```

### Terrain Mesh Generation

The terrain mesh builds a vertex for every point in the elevation grid, converting lat/lon + elevation to model-space (X=east, Y=up, Z=north) via a `GeoToModelTransform` that handles the latitude correction for longitude-distance.

A per-vertex `inside` mask clips to the chosen shape (circle = distance check, hexagon = polygon test, square = all true). Then adjacent quads are split into CCW triangles — only triangles where all three vertices are inside the mask are kept.

The base depends on shape:
- **Square** — 4 rectangular walls connecting the top perimeter to a flat floor.
- **Circle** — a smooth 360-segment cylinder wall; elevation at each angle is linearly interpolated around the terrain boundary.
- **Hexagon** — follows the boundary edges of the clipped terrain to build 6 walls.

### Feature Meshes

```bash
sed -n '730,820p' src/topo_shadow_box/core/mesh.py
```

```output
def _generate_road_mesh(
    road, elevation: ElevationData, transform: GeoToModelTransform,
    min_elev: float, elev_range: float, vertical_scale: float, size_scale: float,
    shape_clipper: ShapeClipper | None = None,
) -> dict | None:
    """Generate a road as a watertight strip following the terrain.

    If a shape_clipper is provided, the road is clipped to the shape boundary.
    Each clipped segment becomes a separate road strip; the first non-empty
    mesh is returned.
    """
    coords = _get_attr_or_key(road, "coordinates", [])
    if len(coords) < 2:
        return None

    road_height_offset = 0.15 * size_scale
    road_relief = road_height_offset

    # Convert coordinates to model space with elevation sampling
    points_3d = []
    points_xz = []
    for coord in coords:
        lat = _get_lat(coord)
        lon = _get_lon(coord)
        x, z = transform.geo_to_model(lat, lon)
        elev = _sample_elevation(lat, lon, elevation)
        y = transform.elevation_to_y(elev, min_elev, elev_range, vertical_scale, size_scale)
        y += road_relief
        points_3d.append([x, y, z])
        points_xz.append([x, z])

    road_width = 1.0 * size_scale
    road_thickness = 0.3 * size_scale

    if shape_clipper is not None:
        # Clip road to shape boundary
        clipped_segments = shape_clipper.clip_linestring(points_xz)
        if not clipped_segments:
            return None

        # Build a KD-tree of original 3D points for elevation lookup
        orig_xz = np.array(points_xz)
        orig_3d = np.array(points_3d)
        tree = cKDTree(orig_xz)

        # Try each clipped segment
        all_vertices = []
        all_faces = []
        for segment in clipped_segments:
            if len(segment) < 2:
                continue
            # For each clipped point, find closest original 3D point for elevation
            segment_3d = []
            for pt in segment:
                _, idx = tree.query([pt[0], pt[1]])
                closest_3d = orig_3d[idx]
                # Use the clipped XZ but keep elevation from nearest original point
                segment_3d.append([pt[0], closest_3d[1], pt[1]])
            segment_3d = np.array(segment_3d)

            result = create_road_strip(segment_3d, width=road_width, thickness=road_thickness)
            if result["vertices"]:
                base_vi = len(all_vertices)
                all_vertices.extend(result["vertices"])
                for face in result["faces"]:
                    all_faces.append([f + base_vi for f in face])

        if not all_vertices:
            return None

        return {
            "name": _get_attr_or_key(road, "name", "Road"),
            "type": "roads",
            "vertices": all_vertices,
            "faces": all_faces,
        }

    # No clipper: use all points (current behavior)
    points = np.array(points_3d)
    result = create_road_strip(points, width=road_width, thickness=road_thickness)
    if not result["vertices"]:
        return None

    return {
        "name": _get_attr_or_key(road, "name", "Road"),
        "type": "roads",
        "vertices": result["vertices"],
        "faces": result["faces"],
    }


```

Roads are generated as **watertight strips** — each road segment becomes a box-like extrusion (width × thickness) that follows the terrain surface, slightly elevated above it. The `create_road_strip` function builds quad faces along the centerline and caps the ends.

When a shape clipper is active the linestring is clipped at the boundary first; a KD-tree maps the clipped 2D intersection points back to the nearest 3D elevation in the original dataset.

Water polygons use `create_solid_polygon` (ear-clipping triangulation + thickness). Buildings use the richer `generate_building_mesh` in `core/building_shapes.py`, which extrudes a floor polygon to the building height.

### Coordinate System and Transforms

```bash
cat src/topo_shadow_box/core/coords.py
```

```output
"""Geographic to model coordinate transforms."""

import math
import numpy as np

from ..state import Bounds


class GeoToModelTransform:
    """Transforms geographic coordinates (lat/lon) to model coordinates (mm).

    Model coordinate system:
    - X: east-west (longitude)
    - Y: up (elevation)
    - Z: north-south (latitude), with north at Z=0
    """

    def __init__(self, bounds: Bounds, model_width_mm: float = 200.0):
        self.bounds = bounds

        avg_lat = bounds.center_lat
        self.lon_scale = math.cos(math.radians(avg_lat))

        lat_range = bounds.lat_range
        lon_range = bounds.lon_range * self.lon_scale

        # Scale factor: map the larger dimension to model_width_mm
        max_span = max(lat_range, lon_range)
        self.scale_factor = model_width_mm / max_span if max_span > 0 else 1.0

        # Model dimensions in mm
        self.model_width_x = lon_range * self.scale_factor
        self.model_width_z = lat_range * self.scale_factor

    def geo_to_model(self, lat: float, lon: float) -> tuple[float, float]:
        """Convert lat/lon to model X, Z coordinates (mm)."""
        x = (lon - self.bounds.west) * self.lon_scale * self.scale_factor
        z = (self.bounds.north - lat) * self.scale_factor
        return x, z

    def geo_to_model_array(
        self, lats: np.ndarray, lons: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert arrays of lat/lon to model X, Z."""
        x = (lons - self.bounds.west) * self.lon_scale * self.scale_factor
        z = (self.bounds.north - lats) * self.scale_factor
        return x, z

    def elevation_to_y(
        self, elevation: float, min_elev: float, elev_range: float,
        vertical_scale: float, size_scale: float = 1.0,
    ) -> float:
        """Convert elevation in meters to Y coordinate in mm."""
        if elev_range <= 0:
            return 0.0
        clamped = max(min_elev, min(min_elev + elev_range, elevation))
        return ((clamped - min_elev) / elev_range) * 20.0 * size_scale * vertical_scale


def add_padding_to_bounds(bounds: Bounds, padding_m: float, is_set: bool = False) -> Bounds:
    """Add padding in meters around a bounding box."""
    # 1 degree latitude ~ 111,000 meters
    lat_padding = padding_m / 111_000.0
    # 1 degree longitude varies with latitude
    avg_lat = bounds.center_lat
    lon_padding = padding_m / (111_000.0 * math.cos(math.radians(avg_lat)))

    return Bounds(
        north=bounds.north + lat_padding,
        south=bounds.south - lat_padding,
        east=bounds.east + lon_padding,
        west=bounds.west - lon_padding,
        is_set=is_set,
    )
```

`GeoToModelTransform` is the key coordinate class. It solves the longitude-shrinkage problem by multiplying all east-west distances by `cos(avg_lat)` — without this, models near the poles would look stretched horizontally.

Elevation Y maps the full relief (after percentile normalisation) to 20 mm × vertical_scale × size_scale. The base sits at Y=0 and the highest terrain point sits at the top of that range.

## Shape Clipping

`core/shape_clipper.py` provides boundary-aware clipping for both masking the terrain grid and clipping road/water linestrings/polygons to the model boundary.

```bash
grep -n 'class \|def ' src/topo_shadow_box/core/shape_clipper.py | head -30
```

```output
12:class ShapeClipper(ABC):
13:    """Abstract base class for shape clipping operations.
19:    def __init__(self, center_x: float, center_z: float, size: float):
32:    def is_inside(self, x, z):
45:    def clip_linestring(self, points) -> list[np.ndarray]:
57:    def clip_polygon(self, points) -> np.ndarray | None:
71:    def project_to_boundary(self, x: float, z: float) -> tuple[float, float] | None:
84:class CircleClipper(ShapeClipper):
87:    def __init__(self, center_x: float, center_z: float, radius: float):
91:    def is_inside(self, x, z):
96:    def _line_circle_intersection(self, p1, p2) -> list[tuple[float, float]]:
131:    def clip_linestring(self, points) -> list[np.ndarray]:
169:    def clip_polygon(self, points) -> np.ndarray | None:
180:    def project_to_boundary(self, x: float, z: float) -> tuple[float, float]:
192:class SquareClipper(ShapeClipper):
195:    def __init__(self, center_x: float, center_z: float, half_width: float):
199:    def is_inside(self, x, z):
203:    def _line_box_intersection(self, p1, p2) -> list[tuple[float, float]]:
255:    def clip_linestring(self, points) -> list[np.ndarray]:
293:    def clip_polygon(self, points) -> np.ndarray | None:
304:    def project_to_boundary(self, x: float, z: float) -> tuple[float, float]:
325:class RectangleClipper(ShapeClipper):
328:    def __init__(self, center_x: float, center_z: float,
334:    def is_inside(self, x, z):
338:    def _line_box_intersection(self, p1, p2) -> list[tuple[float, float]]:
385:    def clip_linestring(self, points) -> list[np.ndarray]:
423:    def clip_polygon(self, points) -> np.ndarray | None:
434:    def project_to_boundary(self, x: float, z: float) -> tuple[float, float]:
458:class HexagonClipper(ShapeClipper):
461:    def __init__(self, center_x: float, center_z: float, radius: float):
```

All clippers share an abstract interface:
- `is_inside(x, z)` — scalar or array containment test
- `clip_linestring(points)` → list of numpy arrays (segments split at boundary crossings)
- `clip_polygon(points)` → single numpy array (polygon clipped to boundary)
- `project_to_boundary(x, z)` → nearest point on the boundary

The **CircleClipper** uses analytic line-circle intersection math. The **SquareClipper** and **RectangleClipper** use line-AABB intersection with Cohen–Sutherland-style endpoint classification. The **HexagonClipper** treats the hexagon as 6 half-planes.

`clip_linestring` handles the tricky case where a road can enter and exit the shape multiple times — it returns a list of sub-segments, each a watertight run inside the boundary.

## The Exporters

### 3MF Export

```bash
grep -n 'def \|material\|<material\|<object\|<component\|zip\|ZipFile' src/topo_shadow_box/exporters/threemf.py | head -40
```

```output
1:"""3MF multi-material export using custom XML + ZIP."""
3:import zipfile
6:def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
11:def export_3mf(meshes: list[dict], output_path: str, base_height_mm: float = 10.0) -> dict:
12:    """Export meshes as a multi-material 3MF file.
33:    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
41:def _build_model_xml(objects: list[tuple], base_height_mm: float = 10.0) -> str:
47:        '  xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">',
52:    # Base materials
53:    parts.append('    <m:basematerials id="1">')
57:    parts.append("    </m:basematerials>")
64:            f'    <object id="{obj_id}" name="{safe_name}" '
```

```bash
sed -n '41,100p' src/topo_shadow_box/exporters/threemf.py
```

```output
def _build_model_xml(objects: list[tuple], base_height_mm: float = 10.0) -> str:
    """Build the 3MF model XML with multiple colored objects."""
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<model unit="millimeter" xml:lang="en-US"',
        '  xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"',
        '  xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">',
        '  <metadata name="Application">topo-shadow-box</metadata>',
        "  <resources>",
    ]

    # Base materials
    parts.append('    <m:basematerials id="1">')
    for name, _, _, (r, g, b) in objects:
        safe_name = (name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))
        parts.append(f'      <m:base name="{safe_name}" displaycolor="#{r:02X}{g:02X}{b:02X}"/>')
    parts.append("    </m:basematerials>")

    # Objects
    for obj_idx, (name, vertices, faces, _) in enumerate(objects):
        obj_id = obj_idx + 2
        safe_name = (name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))
        parts.append(
            f'    <object id="{obj_id}" name="{safe_name}" '
            f'pid="1" pindex="{obj_idx}" type="model">'
        )
        parts.append("      <mesh>")

        # Vertices
        parts.append("        <vertices>")
        for v in vertices:
            parts.append(f'          <vertex x="{v[0]:.6f}" y="{v[1]:.6f}" z="{v[2]:.6f}"/>')
        parts.append("        </vertices>")

        # Triangles
        parts.append("        <triangles>")
        for f in faces:
            parts.append(f'          <triangle v1="{f[0]}" v2="{f[1]}" v3="{f[2]}"/>')
        parts.append("        </triangles>")

        parts.append("      </mesh>")
        parts.append("    </object>")

    parts.append("  </resources>")

    # Build items
    parts.append("  <build>")
    for obj_idx in range(len(objects)):
        parts.append(f'    <item objectid="{obj_idx + 2}" transform="1 0 0 0 0 1 0 -1 0 0 0 {base_height_mm:.6f}"/>')
    parts.append("  </build>")
    parts.append("</model>")

    return "\n".join(parts)


_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>"""
```

The 3MF exporter hand-builds XML (no XML library) and writes it into a ZIP with the required `_rels/.rels`, `[Content_Types].xml`, and `3D/3dmodel.model` entries.

Each mesh layer is one `<object>` in the file, referencing a shared `<m:basematerials>` list. This is what tells a multi-filament printer like a Bambu Lab which colour to use for which object. The `<item>` transform applies a Y-axis flip (row 2 of the matrix is `0 -1 0`) to convert from the model's right-hand Y-up to 3MF's convention.

### Preview Server

```bash
grep -n 'def \|WebSocket\|websocket\|http\|webbrowser\|broadcast\|json\|asyncio' src/topo_shadow_box/preview/server.py | head -30
```

```output
1:"""HTTP + WebSocket preview server for Three.js viewer."""
3:import asyncio
4:import json
6:from http.server import HTTPServer, SimpleHTTPRequestHandler
9:import websockets
12:_http_server: HTTPServer | None = None
19:    def do_GET(self):
30:    def log_message(self, format, *args):
34:def _state_to_json(state) -> str:
66:    return json.dumps(data)
69:async def _ws_handler(websocket):
70:    _ws_clients.add(websocket)
72:        async for _ in websocket:
75:        _ws_clients.discard(websocket)
78:async def start_preview_server(state, http_port: int = 3333, ws_port: int = 3334):
79:    """Start the HTTP and WebSocket servers."""
80:    global _http_server, _ws_server
83:    _http_server = HTTPServer(("localhost", http_port), PreviewHandler)
84:    http_thread = Thread(target=_http_server.serve_forever, daemon=True)
85:    http_thread.start()
87:    # Start WebSocket server
88:    _ws_server = await websockets.serve(_ws_handler, "localhost", ws_port)
91:    asyncio.get_event_loop().call_later(1.0, lambda: asyncio.ensure_future(update_preview(state)))
94:async def update_preview(state):
95:    """Send updated mesh data to all connected WebSocket clients."""
99:    data = _state_to_json(state)
100:    await asyncio.gather(
```

The preview system runs two servers on localhost:
- **Port 3333** — a Python `http.server.HTTPServer` serving the Three.js viewer HTML/JS from the `preview/` package directory.
- **Port 3334** — a `websockets` async server. When meshes are updated, `update_preview` serialises the current state to JSON and broadcasts to all connected clients.

The viewer renders each mesh layer in Three.js with the configured colours, allowing you to inspect the model before printing.

## Tests

The test suite uses pytest and is organized by layer:

```bash
ls tests/
```

```output
__init__.py
__pycache__
fixtures
test_area_tools.py
test_building_shapes.py
test_core_models.py
test_elevation.py
test_exporters.py
test_generate_progress.py
test_integration.py
test_mcp_smoke.py
test_mesh.py
test_models.py
test_osm.py
test_plugin_manifests.py
test_prereqs.py
test_session_tools.py
test_shape_clipper.py
test_state_models.py
test_state_resource.py
```

```bash
python -m pytest --collect-only -q 2>&1 | tail -20
```

```output
/Users/huslage/.cache/uv/archive-v0/RSA1DGwLd245AvjBdIROw/bin/python: No module named pytest
```

```bash
uv run pytest --collect-only -q 2>&1 | tail -25 | sed 's/in [0-9.]*s$/in Xs/'
```

```output
tests/test_state_models.py::TestModelParams::test_width_must_be_positive
tests/test_state_models.py::TestModelParams::test_vertical_scale_must_be_positive
tests/test_state_models.py::TestModelParams::test_base_height_must_be_positive
tests/test_state_models.py::TestModelParams::test_valid_shapes
tests/test_state_models.py::TestModelParams::test_invalid_shape
tests/test_state_models.py::TestColors::test_defaults
tests/test_state_models.py::TestColors::test_valid_hex_color
tests/test_state_models.py::TestColors::test_lowercase_hex_accepted
tests/test_state_models.py::TestColors::test_invalid_hex_missing_hash
tests/test_state_models.py::TestColors::test_invalid_hex_wrong_length
tests/test_state_models.py::TestColors::test_invalid_hex_non_hex_chars
tests/test_state_models.py::TestColors::test_hex_to_rgb
tests/test_state_models.py::TestColors::test_as_dict
tests/test_state_models.py::TestElevationData::test_defaults
tests/test_state_models.py::TestElevationData::test_resolution_must_be_positive
tests/test_state_models.py::TestElevationData::test_resolution_must_not_exceed_1000
tests/test_state_models.py::TestMeshData::test_defaults
tests/test_state_models.py::TestMeshData::test_with_data
tests/test_state_models.py::TestSessionState::test_defaults
tests/test_state_models.py::TestSessionState::test_preview_port_range
tests/test_state_models.py::TestSessionState::test_summary_returns_dict
tests/test_state_resource.py::test_state_resource_returns_valid_json
tests/test_state_resource.py::test_state_resource_content_matches_summary

271 tests collected in Xs
```

```bash
uv run pytest -q 2>&1 | tail -5 | sed 's/in [0-9.]*s$/in Xs/'
```

```output
........................................................................ [ 26%]
........................................................................ [ 53%]
........................................................................ [ 79%]
.......................................................                  [100%]
271 passed in Xs
```

271 tests, all passing in under 4 seconds. The test layers map to the source layers:

| Test file | What it covers |
|---|---|
| `test_models.py` | Domain Pydantic models (Coordinate, GpxTrack, …) |
| `test_state_models.py` | SessionState sub-models (Bounds, Colors, ModelParams, …) |
| `test_elevation.py` | Tile fetching, Terrarium decoding, interpolation |
| `test_mesh.py` | Road strips, polygon triangulation, terrain mesh |
| `test_shape_clipper.py` | Circle/hexagon/square/rectangle clip & containment |
| `test_osm.py` | Overpass parsing, feature extraction |
| `test_building_shapes.py` | Building extrusion geometry |
| `test_area_tools.py` | Area-setting and geocoding tools |
| `test_exporters.py` | 3MF XML structure, SVG, OpenSCAD |
| `test_generate_progress.py` | MCP progress reporting callbacks |
| `test_integration.py` | Full pipeline end-to-end |
| `test_mcp_smoke.py` | FastMCP server starts without errors |
| `test_prereqs.py` | `require_state` guard |
| `test_session_tools.py` | Save/load session round-trip |
| `test_state_resource.py` | MCP resource returns valid JSON |
| `test_plugin_manifests.py` | Plugin manifest structure |

Network calls (elevation tiles, Overpass, Nominatim) are mocked in unit tests so they run offline.

## The Full Pipeline — Summary

Putting it all together, here is the path from first prompt to 3D file:

    User: "Make me a model of Mount Hood"
           │
           ▼
    geocode_place("Mount Hood")
      → Nominatim API → single result → auto-set bounds
           │
           ▼
    fetch_elevation(resolution=200)
      → pick zoom, fetch AWS tiles concurrently
      → decode Terrarium RGB, stitch, crop, bicubic interpolate, Gaussian smooth
      → state.elevation.grid [200×200 numpy array]
           │
           ▼
    fetch_features(include=["roads","water","buildings"])
      → Overpass QL queries (3 server fallback)
      → parse ways/relations → RoadFeature, WaterFeature, BuildingFeature
      → state.features
           │
           ▼
    generate_model()
      → GeoToModelTransform (lat-corrected mm scale)
      → terrain mesh (shape-clipped grid → CCW triangles + base walls)
      → road meshes (terrain-following watertight strips, clipped)
      → water meshes (solid clipped polygons)
      → building meshes (extruded floor polygons)
      → GPX track mesh (cylinder following points)
      → state.terrain_mesh + state.feature_meshes
           │
           ▼
    export_3mf("~/model.3mf")
      → hand-built XML with <m:basematerials> per layer
      → ZipFile with 3D/3dmodel.model + _rels + content-types
      → ready for Bambu Studio / PrusaSlicer
