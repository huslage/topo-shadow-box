# Port topo3d Mesh Generation to topo-shadow-box MCP

**Date:** 2026-02-12
**Status:** Approved

## Problem

topo-shadow-box produces lower quality meshes than topo3d. Key gaps:
- Flat ribbon roads/GPX tracks (not watertight, not printable)
- Fan-triangulated water bodies (no bottom/walls, not watertight)
- Simple box buildings (no shape variety)
- Simple min/max elevation normalization (outliers flatten terrain)
- Rectangular base/walls only (even for circular/hex shapes)

## Approach

Direct port of topo3d's proven geometry algorithms into the MCP's existing architecture. The mesh output format (vertices + faces dicts) stays the same, so tools, exporters, and preview are unaffected.

## Changes

### 1. Elevation normalization (`core/elevation.py`)

Add percentile-based normalization (2nd-98th percentile) to clip outliers and preserve local terrain relief. Store percentile bounds in ElevationData for use during mesh generation.

### 2. Watertight road and GPX strip generation (`core/mesh.py`)

Replace flat ribbons with topo3d's `create_road_strip` pattern:
- 4 vertices per centerline point (top-left, top-right, bottom-left, bottom-right)
- Top faces, bottom faces, left/right side walls, start/end caps
- 3-point direction averaging for smoother ribbons
- Road width: 1.0mm * size_scale, thickness: max(0.9, 0.6) * size_scale
- GPX width: 2.5mm, height offset: 0.8mm above terrain
- Road height offset: 0.2mm above terrain

### 3. Ear-clipping water body triangulation (`core/mesh.py`)

Replace fan triangulation with ear-clipping algorithm:
- Shoelace formula for winding detection
- Ear vertex identification (convex + no interior vertices)
- Fallback to fan triangulation if degenerate
- Collinear vertex removal before triangulation
- Duplicate vertex deduplication via scipy cKDTree (r=1e-4)
- Watertight output: top face at average perimeter elevation, bottom face reversed, side walls

### 4. Shape-aware building generation (`core/building_shapes.py`)

New file porting topo3d's BuildingShapeGenerator:
- OSM tag mapping: church/cathedral -> steeple, house/residential -> pitched roof, warehouse/barn -> gabled roof, commercial/office -> flat roof
- Steeple: main body (60%), tower (25%), spire (15%)
- Pitched roof: 70% walls, 30% roof with ridge along X
- Gabled roof: 80% walls, 20% roof
- Minimum footprint enforcement (1.2mm) for printability

### 5. Shape-aware base and wall generation (`core/shape_clipper.py`)

New file with shape clippers (circle, square, rectangle, hexagon):
- `is_inside(x, z)` for point testing
- `clip_linestring()` for road/GPX segment clipping
- Terrain: mask-based vertex filtering (existing)
- Buildings: skip if any corner outside shape
- Roads: segment clipping preserving continuity

Enhanced base generation in `core/mesh.py`:
- Boundary edge detection from terrain faces
- Smooth circular walls (360 segments)
- Manifold topology enforcement
- Shape-projected boundary vertices

### 6. Feature limits

- 150 buildings, 200 roads, 50 water features
- GPX point sampling: max ~500 per track

## Files

| File | Change |
|------|--------|
| `core/mesh.py` | Major rewrite: watertight strips, ear-clipping, shape-aware base, feature limits, elevation caching |
| `core/elevation.py` | Add percentile normalization |
| `core/building_shapes.py` | New: building shape generation |
| `core/shape_clipper.py` | New: shape clipping for features |

## Not changed

tools/, exporters/, preview/, state.py, server.py, coords.py — mesh output format (vertices + faces dicts) is unchanged.
