# topo-shadow-box

An MCP (Model Context Protocol) server that generates 3D-printable topographic shadow boxes. Given a geographic area, it fetches real elevation data and OpenStreetMap features, then produces multi-material 3MF files ready for slicing and printing.

## Features

- Terrain mesh from AWS Terrain-RGB elevation tiles (free, global coverage)
- OpenStreetMap overlay: roads, water bodies, and buildings
- GPX track visualization as raised strips on the terrain
- Multiple output shapes: square, circle, hexagon, rectangle
- Export to 3MF (multi-material), OpenSCAD `.scad`, or SVG map insert
- Live 3D preview via a local Three.js viewer (port 3333)
- Session save/load to resume work across server restarts

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

```bash
git clone https://github.com/your-org/topo-shadow-box
cd topo-shadow-box
uv sync
```

## Example Agent Conversations

These show how to prompt Claude (or any MCP-capable agent) to generate a shadow box. The agent calls the tools automatically — you just describe what you want.

---

### Example 1: Shadow box from coordinates

**User:**
> Make me a 3D-printable shadow box of downtown Asheville, NC. Hexagon shape, 150mm wide. Save it to my Desktop.

**Agent:**

```
set_area_from_coordinates(lat=35.5951, lon=-82.5515, radius_m=2000)
→ Area set: 35.545°N–35.645°N, 82.596°W–82.507°W (≈4.0×4.0 km)

validate_area()
→ OK — area 16.0 km², elevation relief 312 m

fetch_elevation(resolution=200)
→ Elevation grid fetched: 200×200 (40 000 points), range 606–918 m

fetch_features()
→ Fetched 87 roads, 12 water bodies, 43 buildings from OpenStreetMap

set_model_params(width_mm=150, shape="hexagon", vertical_scale=1.5)
→ Model params updated

generate_model()
→ Terrain mesh … roads … water … buildings … frame — done (6 meshes)

export_3mf(output_path="/Users/you/Desktop/asheville.3mf")
→ Exported 6-material 3MF to /Users/you/Desktop/asheville.3mf (4.2 MB)
```

**Agent reply:**
> Done! `asheville.3mf` is on your Desktop. It has six material slots: terrain, roads, water, buildings, frame, and base. Open it in PrusaSlicer or Bambu Studio and assign a filament color to each slot.

---

### Example 2: Shadow box from a GPX track

**User:**
> I have a GPX file of my morning ride at `/Users/you/rides/morning_ride.gpx`. Generate a shadow box showing the route with the terrain underneath. Circle shape. Export to `/tmp/ride.3mf`.

**Agent:**

```
set_area_from_gpx(file_path="/Users/you/rides/morning_ride.gpx", padding_m=500)
→ GPX loaded: 1 track, 2 043 points. Area set with 500 m padding.

validate_area()
→ OK — area 28.4 km², elevation relief 487 m

fetch_elevation(resolution=200)
→ Elevation grid fetched: 200×200, range 312–799 m

fetch_features(include=["roads", "water"])
→ Fetched 134 roads, 8 water bodies

set_model_params(shape="circle", vertical_scale=2.0)
→ Model params updated

generate_model()
→ Terrain mesh … roads … water … GPX track strip … frame — done (5 meshes)

export_3mf(output_path="/tmp/ride.3mf")
→ Exported 5-material 3MF to /tmp/ride.3mf (5.8 MB)
```

**Agent reply:**
> Your ride shadow box is at `/tmp/ride.3mf`. The GPX track is its own material object so you can print it in a contrasting color. The circle footprint clips the terrain and roads cleanly to the boundary.

---

## Running as an MCP Server in Claude Code

### Quick setup with `claude mcp add`

```bash
claude mcp add topo-shadow-box -- /path/to/topo-shadow-box/.venv/bin/topo-shadow-box
```

Replace `/path/to/topo-shadow-box` with the actual path where you cloned the repository.

### Manual setup via `claude_desktop_config.json`

Add the server to your Claude Code MCP configuration. The config file is located at:

- **macOS**: `~/.claude/claude_desktop_config.json`
- **Linux**: `~/.config/claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "topo-shadow-box": {
      "command": "/path/to/topo-shadow-box/.venv/bin/topo-shadow-box",
      "args": []
    }
  }
}
```

Or using `uv run` if you prefer not to reference the venv directly:

```json
{
  "mcpServers": {
    "topo-shadow-box": {
      "command": "uv",
      "args": ["--directory", "/path/to/topo-shadow-box", "run", "topo-shadow-box"]
    }
  }
}
```

After editing the config, restart Claude Code. The server's tools will appear in your tool list.

## Running from the Command Line

The server communicates over stdio (standard MCP transport). You can start it directly:

```bash
# Using the installed script
.venv/bin/topo-shadow-box

# Using Python module
.venv/bin/python -m topo_shadow_box

# Using uv run (no venv activation needed)
uv run topo-shadow-box
```

The server starts and waits for MCP JSON-RPC messages on stdin. In normal use you won't interact with it directly — an MCP client (Claude Code, Claude Desktop, or any MCP-compatible client) handles the protocol.

## Tool Reference

Tools must be called in the order shown. Each tool validates that its prerequisites are satisfied and returns a plain-text status message.

### 1. Define the Area

**`set_area_from_coordinates`** — Set area by center point or bounding box.

```
set_area_from_coordinates(lat=47.6062, lon=-122.3321, radius_m=3000)
# or
set_area_from_coordinates(north=47.65, south=47.55, east=-122.28, west=-122.38)
```

**`set_area_from_gpx`** — Load a GPX file; use its extent as the area and render the track.

```
set_area_from_gpx(file_path="/Users/you/rides/mountain_loop.gpx", padding_m=500)
```

**`validate_area`** — Check for problems (area too small/large, low elevation relief).

```
validate_area()
```

### 2. Fetch Data

**`fetch_elevation`** — Download AWS Terrain-RGB tiles and build an elevation grid.

```
fetch_elevation(resolution=200)   # 200x200 grid; use 100 for quick previews
```

**`fetch_features`** *(optional)* — Download roads, water, and buildings from OpenStreetMap.

```
fetch_features()                             # all feature types
fetch_features(include=["roads", "water"])   # subset only
```

### 3. Configure the Model *(optional)*

**`set_model_params`** — Adjust dimensions and shape before generating.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `width_mm` | 200 | Model width in mm (larger geographic span maps to this) |
| `vertical_scale` | 1.5 | Elevation exaggeration. Use 2–3 for flat terrain, 1 for mountains |
| `base_height_mm` | 10 | Thickness of the solid base |
| `shape` | `"square"` | Outline shape: `square`, `circle`, `hexagon`, or `rectangle` |

```
set_model_params(width_mm=150, vertical_scale=2.0, shape="hexagon")
```

**`set_colors`** — Set hex colors for each material layer (applied at export time).

```
set_colors(terrain="#8B7355", water="#4A90D9", roads="#CCCCCC", buildings="#E8E0D0")
```

### 4. Generate the Model

**`generate_model`** — Compute all meshes. Reports fine-grained progress.

```
generate_model()
```

Re-run this after changing `set_model_params` or if you want to regenerate after adjusting features.

**`generate_map_insert`** *(optional)* — Generate a flat map plate (for 3MF) and/or SVG for paper printing.

```
generate_map_insert(format="both")   # "svg", "plate", or "both"
```

### 5. Preview *(optional)*

**`preview`** — Open a Three.js viewer at `http://localhost:3333` showing all meshes.

```
preview()
```

### 6. Export

**`export_3mf`** — Multi-material 3MF file for 3D printing. Each feature type is a separate material object.

```
export_3mf(output_path="/Users/you/Desktop/seattle.3mf")
```

**`export_openscad`** — Parametric `.scad` file. Open in OpenSCAD to render or modify.

```
export_openscad(output_path="/Users/you/Desktop/seattle.scad")
```

**`export_svg`** — SVG map insert for paper printing behind the terrain model. Does not require `generate_model`.

```
export_svg(output_path="/Users/you/Desktop/seattle_map.svg")
```

### Session Persistence

**`save_session`** — Save bounds, model params, colors, and GPX tracks to JSON.

```
save_session()                                          # saves to ~/.cache/topo-shadow-box/session.json
save_session(path="/Users/you/projects/seattle.json")   # custom path
```

**`load_session`** — Restore a saved session. Elevation and meshes must be regenerated after loading.

```
load_session()
load_session(path="/Users/you/projects/seattle.json")
```

After loading, run `fetch_elevation` → `generate_model` to rebuild the meshes.

## Typical Workflow

```
1. set_area_from_coordinates(lat=47.6062, lon=-122.3321, radius_m=2000)
2. validate_area()
3. fetch_elevation(resolution=200)
4. fetch_features()
5. set_model_params(shape="hexagon", vertical_scale=2.0)
6. set_colors(terrain="#8B7355", water="#4A90D9", roads="#FFFFFF")
7. generate_model()
8. preview()
9. export_3mf(output_path="/Users/you/Desktop/seattle.3mf")
```

For a GPX-based model:

```
1. set_area_from_gpx(file_path="/Users/you/rides/loop.gpx", padding_m=500)
2. fetch_elevation(resolution=200)
3. fetch_features(include=["roads", "water"])
4. generate_model()
5. export_3mf(output_path="/Users/you/Desktop/loop.3mf")
```

## Output Formats

| Format | Use |
|--------|-----|
| `.3mf` | Multi-material 3D print file. Import into PrusaSlicer, Bambu Studio, or Orca Slicer. Each feature type is a separate object assigned its own filament. |
| `.scad` | OpenSCAD source. Use for parametric customization or rendering to other formats. |
| `.svg` | Flat map insert. Print on paper or cardstock to place behind the terrain in a frame. |

## Data Sources

- **Elevation**: [AWS Terrain Tiles](https://registry.opendata.aws/terrain-tiles/) — Terrarium format, globally available at no cost.
- **Map features**: [OpenStreetMap](https://www.openstreetmap.org/) via the [Overpass API](https://overpass-api.de/) (three mirror servers with automatic fallback).

## Development

```bash
# Install dev dependencies
uv sync

# Run all tests
.venv/bin/pytest

# Run a specific test file
.venv/bin/pytest tests/test_mesh.py

# Run a specific test
.venv/bin/pytest tests/test_mesh.py::TestCreateRoadStrip::test_basic_strip
```

## License

See [LICENSE](LICENSE).
