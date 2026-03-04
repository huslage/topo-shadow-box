# Design: Port topo-shadow-box to Go

**Date:** 2026-03-02
**Status:** Approved

## Goal

Produce a single downloadable binary (`topo-shadow-box`) that users can drop in their PATH. No Python, no `uv`, no runtime dependencies required.

## Language & Toolchain

**Go**, using `mark3labs/mcp-go` as the MCP SDK. Binaries built with `go build` and cross-compiled for macOS (arm64/amd64), Linux (amd64), and Windows (amd64) via GitHub Actions.

## Project Layout

```
topo-shadow-box/
├── cmd/topo-shadow-box/main.go   ← entry point, wires up MCP server
├── internal/
│   ├── session/                  ← Session struct (Config/FetchedData/Results)
│   ├── core/                     ← elevation, osm, gpx, mesh, shape_clipper
│   ├── exporters/                ← threemf, openscad, svg
│   └── tools/                   ← MCP tool handlers
└── go.mod
```

The Python package structure maps 1:1 to Go packages under `internal/`.

## Dependency Mapping

| Python | Go |
|--------|-----|
| `mcp[cli]` | `mark3labs/mcp-go` |
| `numpy` + `scipy` | `gonum.org/v1/gonum` (mat, interp) |
| `Pillow` (PNG decode) | stdlib `image/png` |
| `httpx` | stdlib `net/http` |
| `gpxpy` | `tkrajina/gpx-go` |
| `pydantic` | Go struct tags + manual validators |
| shape clipping | `paulmach/orb/clip` |
| 3MF export | stdlib `archive/zip` + `encoding/xml` |
| SVG / OpenSCAD export | stdlib `text/template` |

## Session State Redesign

The Python `SessionState` is a flat global singleton mixing config, fetched data, and computed results. The Go port splits this into three clearly named sub-structs passed as a pointer:

```go
type Session struct {
    Config      Config      // serialized by save/load
    FetchedData FetchedData // in-memory only, not persisted
    Results     Results     // computed, always regenerable
    mu          sync.Mutex  // single-client concurrency guard
}

type Config struct {
    Bounds      Bounds
    ModelParams ModelParams
    Colors      Colors
    GpxTracks   []GpxTrack
}

type FetchedData struct {
    Elevation *ElevationData
    Features  *OsmFeatureSet
}

type Results struct {
    TerrainMesh   *Mesh
    FeatureMeshes []Mesh
    GpxMesh       *Mesh
    MapInsert     *Mesh
}
```

`Session` is created once in `main.go` and injected into each tool handler. No global. Save/load only marshals `Config` to JSON.

## Features Included / Excluded

**Included (full parity):**
- All MCP tools: set_area_*, validate_area, fetch_elevation, fetch_features, generate_model, generate_map_insert, export_3mf, export_openscad, export_svg, get_status, save_session, load_session, set_model_params, set_colors, geocode_place, select_geocode_result

**Excluded:**
- Live Three.js preview (`preview` tool / WebSocket server on port 3333) — users open the 3MF directly in their slicer

## Error Handling

Go `error` return convention throughout. Tool handlers wrap errors with context (`fmt.Errorf("fetch elevation: %w", err)`). MCP tool responses surface error strings to the agent on failure, matching current Python behavior.

## Testing

Unit tests per package (`core/elevation_test.go`, etc.) using stdlib `testing`. HTTP calls are wrapped behind an interface so tests inject a mock client. Mirrors the existing Python test structure.

## Distribution

GitHub Actions builds and attaches binaries to releases for:
- `darwin/arm64`, `darwin/amd64`
- `linux/amd64`
- `windows/amd64`

The Claude Code plugin manifest updated to reference the binary instead of launching via `uv`.
