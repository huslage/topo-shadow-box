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

## Installation

Download a prebuilt binary from GitHub Releases, or build from source with Go:

```bash
cd go
go build -o /usr/local/bin/topo-shadow-box ./cmd/topo-shadow-box
```

### Adding to Claude Code

```
/plugin marketplace add huslage/topo-shadow-box
/plugin install topo-shadow-box@topo-shadow-box
```

The MCP server starts automatically when you install the plugin.

### Adding to Codex

Add this to `~/.codex/config.toml`:

```toml
[mcp_servers.topo-shadow-box]
command = "topo-shadow-box"
args = ["serve"]
```

Then restart Codex.

## Permissions

The first time Claude calls each tool, it will ask for permission. Click **"Allow always"** (not "Allow") so the approval is saved and you won't be asked again.

If you'd rather pre-approve everything upfront, add this to your `~/.claude/settings.json`:

```json
{
  "permissions": {
    "allow": ["mcp__plugin_topo-shadow-box_topo-shadow-box__*"]
  }
}
```

## CLI Usage

The Go binary supports two modes:

- `topo-shadow-box serve` starts the MCP server (Claude Code plugin mode).
- `topo-shadow-box [flags]` runs one-shot CLI generation and writes an output file.

### Build the Binary

```bash
cd go
go build -o /usr/local/bin/topo-shadow-box ./cmd/topo-shadow-box
```

### Quickstart Examples

```bash
# Circle area -> 3MF
topo-shadow-box --lat 35.99 --lon -78.90 --radius 5000 --output durham.3mf

# Bounding box -> OpenSCAD
topo-shadow-box --north 36.1 --south 35.9 --east -78.8 --west -79.0 --output area.scad

# GPX track area -> SVG
topo-shadow-box --gpx /path/to/ride.gpx --output ride.svg

# Custom model options + subset of map features
topo-shadow-box \
  --lat 35.99 --lon -78.90 --radius 5000 \
  --output model.3mf \
  --width 150 \
  --vertical-scale 2.0 \
  --base-height 8 \
  --shape circle \
  --resolution 250 \
  --features roads,water \
  --color-terrain "#8B7355" \
  --color-water "#4A90D9"
```

### CLI Flag Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--lat`, `--lon`, `--radius` | — | Circular area input (`radius` in meters) |
| `--north`, `--south`, `--east`, `--west` | — | Explicit bounding box input |
| `--gpx` | — | GPX file path (uses GPX bounds + default padding) |
| `--output`, `-o` | — | Output path; extension selects format (`.3mf`, `.scad`, `.svg`) |
| `--width` | `200` | Model width in mm |
| `--vertical-scale` | `1.5` | Elevation exaggeration multiplier |
| `--base-height` | `10` | Solid base thickness in mm |
| `--shape` | `square` | `square`, `circle`, `hexagon`, `rectangle` |
| `--resolution` | `200` | Elevation grid resolution per axis |
| `--features` | `roads,water,buildings` | CSV feature list; use empty string to skip features |
| `--color-terrain` | `#C8A882` | Terrain color |
| `--color-roads` | `#D4C5A9` | Roads color |
| `--color-water` | `#4682B4` | Water color |
| `--color-buildings` | `#E8D5B7` | Buildings color |
| `--color-gpx-track` | `#FF0000` | GPX track color |

### Validation Rules

- Exactly one input mode is required:
  - `--lat` + `--lon` + `--radius`
  - `--north` + `--south` + `--east` + `--west`
  - `--gpx`
- `--output` is required.
- `--shape` must be one of `square`, `circle`, `hexagon`, `rectangle`.
- Colors must be `#RRGGBB`.
- Output format is inferred from file extension.

### Run as MCP Server

```bash
topo-shadow-box serve
```

### CLI Troubleshooting

- `error: --output is required`
  - Add `--output <path>` (or `-o <path>`). The extension must be `.3mf`, `.scad`, or `.svg`.

- `one of --lat/--lon/--radius, --north/--south/--east/--west, or --gpx is required`
  - Provide exactly one input mode. Do not mix coordinate/bbox/GPX inputs.

- `--lat/--lon/--radius ... are mutually exclusive` (or bbox/GPX equivalent)
  - Remove extra input flags so only one area-definition method is used.

- `cannot infer output format from extension ...`
  - Use a supported output extension: `.3mf`, `.scad`, or `.svg`.

- `invalid --shape ...`
  - Use one of: `square`, `circle`, `hexagon`, `rectangle`.

- Color format errors (`must be in #RRGGBB format`)
  - Use 6-digit hex with leading `#`, e.g. `#C8A882`.

- Network/data fetch failures during elevation or feature steps
  - Retry command (transient network issues are common).
  - Try smaller area or lower `--resolution` if requests are timing out.
  - For feature-only failures, you can still generate terrain by setting `--features ''`.

- GPX errors (`open gpx...`, parse failures, or no bounds)
  - Verify the file exists and is valid GPX XML with track points.
  - Try exporting the GPX again from your source app/device.

- Preview tool reports no mesh generated (MCP mode)
  - Run `generate_model` first, then call `preview`.

## Example Agent Conversations

These show how to prompt Claude (or any MCP-capable agent) to generate a shadow box. The agent calls the tools automatically — you just describe what you want.

---

### Example 1: Shadow box from coordinates

**User:**
> Make me a 3D-printable shadow box of downtown Asheville, NC. Hexagon shape, 150mm wide. Save it to my Desktop.

**Agent:**

```
⏺ set_area_from_coordinates(lat=35.5951, lon=-82.5515, radius_m=2000)
  ↳ Area set: 35.545°N–35.645°N, 82.596°W–82.507°W (≈4.0×4.0 km)

⏺ validate_area()
  ↳ OK — area 16.0 km², elevation relief 312 m

⏺ fetch_elevation(resolution=200)
  ↳ Elevation grid fetched: 200×200 (40 000 points), range 606–918 m

⏺ fetch_features()
  ↳ Fetched 87 roads, 12 water bodies, 43 buildings from OpenStreetMap

⏺ set_model_params(width_mm=150, shape="hexagon", vertical_scale=1.5)
  ↳ Model params updated

⏺ generate_model()
  ↳ Terrain mesh … roads … water … buildings … frame — done (6 meshes)

⏺ export_3mf(output_path="/Users/you/Desktop/asheville.3mf")
  ↳ Exported 6-material 3MF to /Users/you/Desktop/asheville.3mf (4.2 MB)
```

**Agent reply:**
> Done! `asheville.3mf` is on your Desktop. It has six material slots: terrain, roads, water, buildings, frame, and base. Open it in PrusaSlicer or Bambu Studio and assign a filament color to each slot.

---

### Example 2: Shadow box from a GPX track

**User:**
> I have a GPX file of my morning ride at `/Users/you/rides/morning_ride.gpx`. Generate a shadow box showing the route with the terrain underneath. Circle shape. Export to `/tmp/ride.3mf`.

**Agent:**

```
⏺ set_area_from_gpx(file_path="/Users/you/rides/morning_ride.gpx", padding_m=500)
  ↳ GPX loaded: 1 track, 2 043 points. Area set with 500 m padding.

⏺ validate_area()
  ↳ OK — area 28.4 km², elevation relief 487 m

⏺ fetch_elevation(resolution=200)
  ↳ Elevation grid fetched: 200×200, range 312–799 m

⏺ fetch_features(include=["roads", "water"])
  ↳ Fetched 134 roads, 8 water bodies

⏺ set_model_params(shape="circle", vertical_scale=2.0)
  ↳ Model params updated

⏺ generate_model()
  ↳ Terrain mesh … roads … water … GPX track strip … frame — done (5 meshes)

⏺ export_3mf(output_path="/tmp/ride.3mf")
  ↳ Exported 5-material 3MF to /tmp/ride.3mf (5.8 MB)
```

**Agent reply:**
> Your ride shadow box is at `/tmp/ride.3mf`. The GPX track is its own material object so you can print it in a contrasting color. The circle footprint clips the terrain and roads cleanly to the boundary.

---

## Tool Reference

See [docs/tool-reference.md](docs/tool-reference.md) for the full tool reference, typical workflows, output formats, and data sources.

## Development

See [docs/development.md](docs/development.md).

## License

See [LICENSE](LICENSE).
