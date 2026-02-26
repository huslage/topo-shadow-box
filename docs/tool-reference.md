# Tool Reference

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
