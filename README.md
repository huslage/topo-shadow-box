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

_Note_: Requires [uv](https://docs.astral.sh/uv/) to install Python dependencies 

### Adding to Claude Code:

```
/plugin marketplace add huslage/topo-shadow-box
/plugin install topo-shadow-box@topo-shadow-box
```

The MCP server starts automatically when you install the plugin.

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
