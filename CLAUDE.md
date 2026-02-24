# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run tests
.venv/bin/pytest

# Run a single test file
.venv/bin/pytest tests/test_mesh.py

# Run a specific test
.venv/bin/pytest tests/test_mesh.py::TestCreateRoadStrip::test_basic_strip

# Run the MCP server
.venv/bin/topo-shadow-box
# or
.venv/bin/python -m topo_shadow_box
```

## Architecture Overview

This is an **MCP (Model Context Protocol) server** that generates 3D-printable topographic shadow boxes. It exposes tools that an LLM client calls sequentially to set up an area, fetch data, generate meshes, and export files.

### Entry Points

- `src/topo_shadow_box/server.py` — FastMCP server, registers all tool groups, exposes `main()` which runs via stdio transport.
- `src/topo_shadow_box/state.py` — **Global singleton `state`** (`SessionState`) that holds all session data. Every tool reads/writes this object. There is no persistence between server restarts.

### Typical Tool Call Flow

```
set_area_from_coordinates / set_area_from_gpx
  → fetch_elevation
  → fetch_features (optional)
  → set_model_params / set_frame_params / set_colors (optional)
  → generate_model
  → preview (optional)
  → export_3mf / export_openscad / export_svg
```

### Layers

**`src/topo_shadow_box/tools/`** — MCP tool handlers (thin layer). Each `register_*_tools(mcp)` function registers tools against the FastMCP instance. They validate prerequisites, call core/exporter functions, and update `state`.

**`src/topo_shadow_box/core/`** — Pure computation (no MCP/state dependencies except function arguments):
- `coords.py` — `GeoToModelTransform`: maps lat/lon → mm model coordinates. X=east, Y=up, Z=north-to-south.
- `elevation.py` — Fetches AWS Terrain-RGB tiles (Terrarium format), stitches them, interpolates to requested resolution.
- `osm.py` — Fetches roads/water/buildings from Overpass API (with server fallback).
- `mesh.py` — All mesh generation: terrain surface, feature meshes (roads/water/buildings), GPX track strips. Uses ear-clipping triangulation. All meshes target watertight geometry for 3D printing.
- `frame.py` — Generates the decorative outer frame mesh.
- `shape_clipper.py` — `ShapeClipper` hierarchy (Circle, Hexagon, Rectangle) for clipping features to model shape.
- `building_shapes.py` — `BuildingShapeGenerator` for shape-aware building extrusion (pitched roof, steeple, flat roof, etc.).
- `map_insert.py` — Generates SVG map and 3D flat plate for placing behind the terrain.
- `gpx.py` — GPX file parsing.

**`src/topo_shadow_box/exporters/`** — File format writers:
- `threemf.py` — Multi-material 3MF (ZIP of XML). Each mesh type gets its own material color.
- `openscad.py` — Parametric `.scad` file with `polyhedron()` calls.
- `svg.py` — SVG map insert for paper printing.

**`src/topo_shadow_box/preview/`** — HTTP + WebSocket server (port 3333) serving a Three.js viewer (`viewer.html`). WebSocket pushes updated mesh data when `preview` tool is called.

### Model Coordinate System

- **X**: east–west (longitude)
- **Y**: up (elevation in mm)
- **Z**: north–south (north = Z=0)
- Model width defaults to 200mm; larger geographic dimension is mapped to `width_mm`.
- Elevation uses 2nd–98th percentile normalization to prevent outlier flattening.

### Shapes

Model shape (`square`, `circle`, `hexagon`, `rectangle`) affects:
1. Terrain boundary clipping (via `ShapeClipper`)
2. Base/wall generation (circle: smooth 360-segment wall; hexagon: boundary-edge walk; square/rectangle: 4-sided walls)
3. Feature clipping: roads are clipped via `clip_linestring`, water/buildings by point-in-shape checks

### Key Design Constraints

- **Watertight meshes** are required for 3D printing. All mesh generators produce closed solids (terrain surface + side walls + bottom cap).
- **Feature limits**: max 200 roads, 50 water bodies, 150 buildings are processed.
- **GPX tracks** are NOT clipped to shape boundaries (preserve natural path).
- Elevation data uses the AWS Terrarium tile service (`s3.amazonaws.com/elevation-tiles-prod/terrarium/`).
- OSM data uses Overpass API with three mirror servers as fallback.
