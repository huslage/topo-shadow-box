# Design: CLI for topo-shadow-box

**Date:** 2026-03-04
**Status:** Approved

## Goal

Add a streamlined CLI to the Go binary so users can generate 3D models without running an MCP session. The binary continues to serve as an MCP server via the `serve` subcommand.

## Interface

```
# Circular area
topo-shadow-box --lat 35.99 --lon -78.90 --radius 5000 --output durham.3mf

# Bounding box
topo-shadow-box --north 36.1 --south 35.9 --east -78.8 --west -79.0 --output area.scad

# GPX file
topo-shadow-box --gpx morning_ride.gpx --output ride.3mf

# MCP server
topo-shadow-box serve
```

### Flags

| Flag | Default | Notes |
|------|---------|-------|
| `--lat` | — | Center latitude (use with `--lon`, `--radius`) |
| `--lon` | — | Center longitude |
| `--radius` | — | Radius in meters |
| `--north/--south/--east/--west` | — | Bounding box |
| `--gpx` | — | Path to GPX file |
| `--output` / `-o` | — | Required; format inferred from extension (.3mf/.scad/.svg) |
| `--width` | 200 | Model width in mm |
| `--vertical-scale` | 1.5 | Elevation exaggeration |
| `--base-height` | 10 | Base thickness in mm |
| `--shape` | square | square / circle / hexagon / rectangle |
| `--resolution` | 200 | Grid points per axis |
| `--features` | roads,water,buildings | Comma-separated; empty string skips OSM fetch |
| `--color-terrain` | (default) | Hex #RRGGBB |
| `--color-roads` | (default) | Hex #RRGGBB |
| `--color-water` | (default) | Hex #RRGGBB |
| `--color-buildings` | (default) | Hex #RRGGBB |
| `--color-gpx-track` | (default) | Hex #RRGGBB |

Input methods are mutually exclusive (`--lat/--lon/--radius`, `--north/--south/--east/--west`, `--gpx`). Exactly one must be provided. `--output` is always required.

## Architecture

```
cmd/topo-shadow-box/
  main.go      ← routes to `serve` subcommand or CLI pipeline
  cli.go       ← flag parsing + pipeline execution
  serve.go     ← MCP server startup

internal/
  session/     ← shared Session struct
  core/        ← elevation, osm, gpx, mesh (shared with MCP tools)
  exporters/   ← 3mf, openscad, svg (shared with MCP tools)
  tools/       ← MCP tool handlers
```

The CLI calls `internal/core` and `internal/exporters` directly — no MCP layer. This means zero code duplication; both CLI and MCP server share the same pipeline functions.

### CLI Pipeline

1. Parse flags → populate `session.Config`
2. Validate input (exactly one input method, `--output` required, valid values)
3. Fetch elevation
4. Fetch OSM features (if `--features` non-empty)
5. Generate model
6. Export to output file

Progress messages → stderr. Exit 0 on success, non-zero on error.

## Testing

`cmd/topo-shadow-box/cli_test.go`:
- Flag parsing: valid combinations, mutual exclusion errors, missing `--output`
- Format inference: `.3mf` → 3MF, `.scad` → OpenSCAD, `.svg` → SVG
- `--features` parsing: comma-split, invalid value errors

Integration coverage is inherited from `internal/core` and `internal/exporters` tests (mock HTTP client).
