"""Mesh generation for terrain, features, and GPX tracks."""

from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from ..state import Bounds, ElevationData
from .coords import GeoToModelTransform
from .building_shapes import BuildingShapeGenerator
from .models import MeshResult
from .shape_clipper import (
    CircleClipper,
    HexagonClipper,
    RectangleClipper,
    ShapeClipper,
)


def _elevation_normalization(grid: np.ndarray, use_percentile: bool = True) -> tuple[float, float]:
    """Compute min elevation and range for normalization.

    When use_percentile is True and grid has >100 elements, uses 2nd-98th
    percentile to clip outliers, preserving local terrain relief.
    """
    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        return 0.0, 1.0

    if use_percentile and finite.size > 100:
        p_low = float(np.percentile(finite, 2.0))
        p_high = float(np.percentile(finite, 98.0))
        if p_high > p_low and (p_high - p_low) >= 1.0:
            return p_low, p_high - p_low

    raw_min = float(np.min(finite))
    raw_max = float(np.max(finite))
    if raw_max <= raw_min:
        return raw_min, 1.0
    return raw_min, raw_max - raw_min


def _sample_elevation(
    lat: float, lon: float, elevation: ElevationData
) -> float:
    """Sample elevation at a geographic point using bilinear interpolation."""
    if not elevation.is_set:
        return 0.0
    grid = elevation.grid
    lats = elevation.lats
    lons = elevation.lons

    # Find position in grid
    lat_frac = (lat - lats[0]) / (lats[-1] - lats[0]) * (len(lats) - 1)
    lon_frac = (lon - lons[0]) / (lons[-1] - lons[0]) * (len(lons) - 1)

    i = int(np.clip(lat_frac, 0, len(lats) - 2))
    j = int(np.clip(lon_frac, 0, len(lons) - 2))

    di = lat_frac - i
    dj = lon_frac - j

    # Bilinear interpolation
    return float(
        grid[i, j] * (1 - di) * (1 - dj)
        + grid[i + 1, j] * di * (1 - dj)
        + grid[i, j + 1] * (1 - di) * dj
        + grid[i + 1, j + 1] * di * dj
    )


def _interpolate_boundary_elevation(angle: float, boundary_data: list) -> float:
    """Interpolate elevation at a given angle from sorted boundary vertices.

    Args:
        angle: Target angle in radians (-pi to pi).
        boundary_data: List of tuples (angle, elevation, ...) sorted by angle.

    Returns:
        Interpolated elevation value.
    """
    n = len(boundary_data)
    if n == 0:
        return 0.0

    # Find bracketing boundary vertices
    upper_idx = 0
    lower_idx = n - 1
    for i in range(n):
        if boundary_data[i][0] >= angle:
            upper_idx = i
            lower_idx = (i - 1) % n
            break
    else:
        # Angle is past all boundary angles, wrap around
        upper_idx = 0
        lower_idx = n - 1

    lower_angle = boundary_data[lower_idx][0]
    lower_elev = boundary_data[lower_idx][1]
    upper_angle = boundary_data[upper_idx][0]
    upper_elev = boundary_data[upper_idx][1]

    # Handle wraparound
    if upper_angle < lower_angle:
        if angle < 0:
            upper_angle_adj = upper_angle
            lower_angle_adj = lower_angle - 2 * np.pi
        else:
            upper_angle_adj = upper_angle + 2 * np.pi
            lower_angle_adj = lower_angle
    else:
        upper_angle_adj = upper_angle
        lower_angle_adj = lower_angle

    # Linear interpolation
    span = upper_angle_adj - lower_angle_adj
    if abs(span) < 1e-6:
        return lower_elev

    t = (angle - lower_angle_adj) / span
    t = max(0.0, min(1.0, t))
    return lower_elev + t * (upper_elev - lower_elev)


def generate_terrain_mesh(
    elevation: ElevationData,
    bounds: Bounds,
    transform: GeoToModelTransform,
    vertical_scale: float = 1.5,
    base_height_mm: float = 10.0,
    shape: str = "square",
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

    min_elev, elev_range = _elevation_normalization(grid)
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


def _generate_square_terrain(
    top_verts: np.ndarray,
    terrain_faces: list,
    rows: int,
    cols: int,
    n: int,
    base_height_mm: float,
) -> dict:
    """Generate rectangular base with side walls for square/rectangle shapes."""
    # Bottom vertices (same X,Z but Y = -base_height)
    bottom_verts = top_verts.copy()
    bottom_verts[:, 1] = -base_height_mm
    all_verts = np.vstack([top_verts, bottom_verts])

    faces = list(terrain_faces)

    # Side walls
    # West wall (j=0)
    for i in range(rows - 1):
        t0 = i * cols
        t1 = (i + 1) * cols
        b0 = n + t0
        b1 = n + t1
        faces.append([t1, t0, b0])
        faces.append([t1, b0, b1])

    # East wall (j=cols-1)
    for i in range(rows - 1):
        t0 = i * cols + (cols - 1)
        t1 = (i + 1) * cols + (cols - 1)
        b0 = n + t0
        b1 = n + t1
        faces.append([t0, t1, b1])
        faces.append([t0, b1, b0])

    # North wall (i=0)
    for j in range(cols - 1):
        t0 = j
        t1 = j + 1
        b0 = n + t0
        b1 = n + t1
        faces.append([t0, t1, b1])
        faces.append([t0, b1, b0])

    # South wall (i=rows-1)
    for j in range(cols - 1):
        t0 = (rows - 1) * cols + j
        t1 = (rows - 1) * cols + j + 1
        b0 = n + t0
        b1 = n + t1
        faces.append([t1, t0, b0])
        faces.append([t1, b0, b1])

    # Bottom face (CCW from below = CW from above)
    for i in range(rows - 1):
        for j in range(cols - 1):
            idx = n + i * cols + j
            faces.append([idx, idx + 1, idx + cols])
            faces.append([idx + 1, idx + cols + 1, idx + cols])

    return {
        "vertices": all_verts.tolist(),
        "faces": faces,
    }


def _generate_circle_terrain(
    top_verts: np.ndarray,
    terrain_faces: list,
    rows: int,
    cols: int,
    cx: float,
    cz: float,
    radius: float,
    inside: np.ndarray,
    base_height_mm: float,
) -> dict:
    """Generate watertight circular terrain by stitching wall to boundary edges.

    1. Stretches boundary vertices outward to the circle radius
    2. Finds boundary edges (edges used by exactly 1 terrain face)
    3. Orders them into a loop around the circle
    4. Builds wall faces from boundary edges down to the base
    5. Closes the bottom with a fan

    This guarantees manifold geometry since wall faces share edges
    directly with terrain faces.
    """
    top_verts = top_verts.copy()
    n = len(top_verts)

    # Find boundary vertices (inside with an outside neighbor)
    boundary_set = set()
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if not inside[idx]:
                continue
            neighbors = []
            if i > 0:
                neighbors.append((i - 1) * cols + j)
            if i < rows - 1:
                neighbors.append((i + 1) * cols + j)
            if j > 0:
                neighbors.append(i * cols + j - 1)
            if j < cols - 1:
                neighbors.append(i * cols + j + 1)
            for nb_idx in neighbors:
                if not inside[nb_idx]:
                    boundary_set.add(idx)
                    break

    faces = list(terrain_faces)

    # Use directed edges from terrain faces to find boundary loop.
    # For each terrain face [a,b,c], directed edges are (a,b), (b,c), (c,a).
    # A boundary directed edge is one whose reverse does NOT exist.
    # This gives exactly one successor per boundary vertex = clean loop walk.
    all_directed = set()
    for face in terrain_faces:
        all_directed.add((face[0], face[1]))
        all_directed.add((face[1], face[2]))
        all_directed.add((face[2], face[0]))

    # Boundary directed edges: those without a reverse
    boundary_next: dict[int, int] = {}
    for a, b in all_directed:
        if (b, a) not in all_directed:
            boundary_next[a] = b

    if len(boundary_next) < 3:
        return {"vertices": top_verts.tolist(), "faces": faces}

    # Walk the directed boundary loop - each vertex has exactly one successor
    start = next(iter(boundary_next))
    loop = [start]
    current = boundary_next[start]
    while current != start:
        loop.append(current)
        current = boundary_next[current]

    if len(loop) < 3:
        return {"vertices": top_verts.tolist(), "faces": faces}

    # Determine loop winding direction using signed area (shoelace in XZ)
    signed_area = 0.0
    for i in range(len(loop)):
        next_i = (i + 1) % len(loop)
        v0 = top_verts[loop[i]]
        v1 = top_verts[loop[next_i]]
        signed_area += (v0[0] - cx) * (v1[2] - cz) - (v1[0] - cx) * (v0[2] - cz)

    # Make loop CW for outward wall normals
    if signed_area > 0:
        loop = loop[::-1]

    # Redistribute boundary vertices evenly around the circle for a smooth edge.
    # Calculate each vertex's current angle, then assign evenly-spaced angles
    # while preserving the loop order.
    n_boundary = len(loop)
    # Find the starting angle (vertex with smallest angle) to anchor distribution
    angles = []
    for vi in loop:
        v = top_verts[vi]
        angles.append(np.arctan2(v[2] - cz, v[0] - cx))

    # Evenly distribute angles around the full circle, starting from the
    # angle of the first vertex in the CW loop
    start_angle = angles[0]
    for i, vi in enumerate(loop):
        # CW loop: subtract increasing angles
        even_angle = start_angle - (2.0 * np.pi * i / n_boundary)
        top_verts[vi][0] = cx + radius * np.cos(even_angle)
        top_verts[vi][2] = cz + radius * np.sin(even_angle)

    # Create bottom vertices for each boundary loop vertex
    # Layout: n .. n+len(loop)-1 = bottom vertices, n+len(loop) = center bottom
    bottom_start = n
    center_bottom_idx = n + len(loop)

    bottom_verts = []
    for vi in loop:
        v = top_verts[vi]
        bottom_verts.append([v[0], -base_height_mm, v[2]])
    bottom_verts.append([cx, -base_height_mm, cz])

    all_verts = np.vstack([top_verts, np.array(bottom_verts)])

    # Wall faces: connect each boundary edge to its bottom counterpart
    # Loop is CW, so wall normals point outward with this winding
    for i in range(len(loop)):
        next_i = (i + 1) % len(loop)
        top0 = loop[i]
        top1 = loop[next_i]
        bot0 = bottom_start + i
        bot1 = bottom_start + next_i
        faces.append([top0, top1, bot0])
        faces.append([top1, bot1, bot0])

    # Bottom face: fan from center (CCW from below = CW from above)
    for i in range(len(loop)):
        next_i = (i + 1) % len(loop)
        bot0 = bottom_start + i
        bot1 = bottom_start + next_i
        faces.append([center_bottom_idx, bot0, bot1])

    return {
        "vertices": all_verts.tolist(),
        "faces": faces,
    }


def _generate_hexagon_terrain(
    top_verts: np.ndarray,
    terrain_faces: list,
    rows: int,
    cols: int,
    cx: float,
    cz: float,
    inside: np.ndarray,
    base_height_mm: float,
    hex_clipper: HexagonClipper,
) -> dict:
    """Generate hexagonal base using boundary edge detection from terrain faces.

    Finds edges referenced exactly once (boundary edges), walks the boundary
    loop, projects vertices to the hexagon edge, and generates wall + bottom.
    """
    top_verts = top_verts.copy()
    n = len(top_verts)

    # Find boundary edges from actual terrain faces
    # An edge that appears in exactly one face is a boundary edge
    edge_count: dict[tuple[int, int], int] = defaultdict(int)
    for face in terrain_faces:
        v0, v1, v2 = int(face[0]), int(face[1]), int(face[2])
        for a, b in [(v0, v1), (v1, v2), (v2, v0)]:
            edge = (min(a, b), max(a, b))
            edge_count[edge] += 1

    boundary_edge_list = [e for e, c in edge_count.items() if c == 1]

    if not boundary_edge_list:
        # Fallback: no boundary edges, return terrain-only mesh
        return {
            "vertices": top_verts.tolist(),
            "faces": terrain_faces,
        }

    # Build adjacency from boundary edges to walk the loop
    adjacency: dict[int, list[int]] = defaultdict(list)
    for v1, v2 in boundary_edge_list:
        adjacency[v1].append(v2)
        adjacency[v2].append(v1)

    # Walk the boundary loop starting from any vertex
    start = boundary_edge_list[0][0]
    boundary_indices = [start]
    visited = {start}

    current = start
    while True:
        neighbors = adjacency[current]
        next_v = None
        for nb in neighbors:
            if nb not in visited:
                next_v = nb
                break
        if next_v is None:
            break
        boundary_indices.append(next_v)
        visited.add(next_v)
        current = next_v

    if len(boundary_indices) < 3:
        return {
            "vertices": top_verts.tolist(),
            "faces": terrain_faces,
        }

    # Project boundary vertices to the hexagon edge
    for idx in boundary_indices:
        v = top_verts[idx]
        projected = hex_clipper.project_to_boundary(v[0], v[2])
        if projected:
            top_verts[idx][0] = projected[0]
            top_verts[idx][2] = projected[1]

    num_boundary = len(boundary_indices)

    # Create base vertices (one below each boundary vertex at base height)
    base_verts = []
    for idx in boundary_indices:
        v = top_verts[idx]
        base_verts.append([v[0], -base_height_mm, v[2]])

    # Center bottom vertex for fan triangulation
    base_verts.append([cx, -base_height_mm, cz])
    center_bottom_idx = n + num_boundary

    all_verts = np.vstack([top_verts, np.array(base_verts)])

    faces = list(terrain_faces)

    # Wall faces connecting terrain boundary to base vertices (CCW from outside)
    for i in range(num_boundary):
        next_i = (i + 1) % num_boundary
        terrain_top = boundary_indices[i]
        terrain_top_next = boundary_indices[next_i]
        base_bottom = n + i
        base_bottom_next = n + next_i

        faces.append([terrain_top, base_bottom, terrain_top_next])
        faces.append([terrain_top_next, base_bottom, base_bottom_next])

    # Bottom face: fan from center (CCW from below)
    for i in range(num_boundary):
        next_i = (i + 1) % num_boundary
        base_bottom = n + i
        base_bottom_next = n + next_i
        faces.append([center_bottom_idx, base_bottom_next, base_bottom])

    return {
        "vertices": all_verts.tolist(),
        "faces": faces,
    }



def _create_shape_clipper(
    shape: str, transform: GeoToModelTransform,
) -> ShapeClipper | None:
    """Create the appropriate shape clipper based on model shape.

    Args:
        shape: Model shape ('square', 'circle', 'hexagon', 'rectangle').
        transform: Geographic-to-model coordinate transform.

    Returns:
        ShapeClipper instance, or None for square (no clipping needed).
    """
    model_width_x = transform.model_width_x
    model_width_z = transform.model_width_z
    center_x = model_width_x / 2
    center_z = model_width_z / 2

    if shape == "circle":
        radius = min(model_width_x, model_width_z) / 2
        return CircleClipper(center_x, center_z, radius)
    elif shape == "hexagon":
        radius = min(model_width_x, model_width_z) / 2
        return HexagonClipper(center_x, center_z, radius)
    elif shape == "rectangle":
        return RectangleClipper(center_x, center_z, model_width_x / 2, model_width_z / 2)
    else:
        # Square/default: clip features to terrain boundary
        return RectangleClipper(center_x, center_z, model_width_x / 2, model_width_z / 2)


def generate_feature_meshes(
    features,
    elevation: ElevationData,
    bounds: Bounds,
    transform: GeoToModelTransform,
    vertical_scale: float = 1.5,
    shape: str = "square",
) -> list[MeshResult]:
    """Generate meshes for OSM features (roads, water, buildings).

    Features are clipped to the model shape boundary for non-square shapes.

    Returns list of MeshResult.
    """
    min_elev, elev_range = _elevation_normalization(elevation.grid)
    model_width = max(transform.model_width_x, transform.model_width_z)
    size_scale = model_width / 200.0

    # Create shape clipper for boundary clipping
    clipper = _create_shape_clipper(shape, transform)

    meshes = []

    # Determine if features is an OsmFeatureSet or a plain dict
    if hasattr(features, "roads"):
        roads_list = features.roads[:200]
        water_list = features.water[:50]
        buildings_list = features.buildings[:150]
    else:
        roads_list = features.get("roads", [])[:200]
        water_list = features.get("water", [])[:50]
        buildings_list = features.get("buildings", [])[:150]

    # Roads: extruded strips following terrain
    for road in roads_list:
        road_mesh = _generate_road_mesh(
            road, elevation, transform, min_elev, elev_range,
            vertical_scale, size_scale, shape_clipper=clipper,
        )
        if road_mesh:
            meshes.append(MeshResult(
                vertices=road_mesh["vertices"],
                faces=road_mesh["faces"],
                name=road_mesh.get("name", "Road"),
                feature_type="road",
            ))

    # Water: solid polygons at water level
    for water in water_list:
        water_mesh = _generate_water_mesh(
            water, elevation, transform, min_elev, elev_range,
            vertical_scale, size_scale, shape_clipper=clipper,
        )
        if water_mesh:
            meshes.append(MeshResult(
                vertices=water_mesh["vertices"],
                faces=water_mesh["faces"],
                name=water_mesh.get("name", "Water"),
                feature_type="water",
            ))

    # Buildings: shape-aware extruded footprints
    building_gen = BuildingShapeGenerator()
    for building in buildings_list:
        building_mesh = _generate_building_mesh(
            building, elevation, transform, min_elev, elev_range,
            vertical_scale, size_scale, shape_clipper=clipper,
            building_shape_gen=building_gen,
        )
        if building_mesh:
            meshes.append(MeshResult(
                vertices=building_mesh["vertices"],
                faces=building_mesh["faces"],
                name=building_mesh.get("name", "Building"),
                feature_type="building",
            ))

    return meshes


def _get_attr_or_key(obj, attr, default=None):
    """Get attribute from typed model or key from dict."""
    if hasattr(obj, attr):
        val = getattr(obj, attr)
        return val if val is not None else default
    elif isinstance(obj, dict):
        return obj.get(attr, default)
    return default


def _get_lat(coord):
    """Get lat from Coordinate model or dict."""
    if hasattr(coord, "lat"):
        return coord.lat
    return coord["lat"]


def _get_lon(coord):
    """Get lon from Coordinate model or dict."""
    if hasattr(coord, "lon"):
        return coord.lon
    return coord["lon"]


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


def _generate_water_mesh(
    water, elevation: ElevationData, transform: GeoToModelTransform,
    min_elev: float, elev_range: float, vertical_scale: float, size_scale: float,
    shape_clipper: ShapeClipper | None = None,
) -> dict | None:
    """Generate a water body as a watertight solid polygon.

    If a shape_clipper is provided, only points inside the shape are kept.
    If fewer than 3 points remain inside, the water body is excluded.
    """
    coords = _get_attr_or_key(water, "coordinates", [])
    if len(coords) < 3:
        return None

    # Compute average perimeter elevation
    elevs = []
    xz_points = []
    for coord in coords:
        lat = _get_lat(coord)
        lon = _get_lon(coord)
        x, z = transform.geo_to_model(lat, lon)
        elev = _sample_elevation(lat, lon, elevation)
        elevs.append(elev)
        xz_points.append((x, z))

    # Filter points through shape clipper
    if shape_clipper is not None:
        filtered_xz = []
        filtered_elevs = []
        for i, (x, z) in enumerate(xz_points):
            if shape_clipper.is_inside(x, z):
                filtered_xz.append((x, z))
                filtered_elevs.append(elevs[i])
        if len(filtered_xz) < 3:
            return None
        xz_points = filtered_xz
        elevs = filtered_elevs

    avg_elev = sum(elevs) / len(elevs)
    water_y_base = transform.elevation_to_y(avg_elev, min_elev, elev_range, vertical_scale, size_scale)

    water_relief = 0.6 * size_scale
    water_y = max(0.7 * size_scale, water_y_base + water_relief)
    water_thickness = max(1.2 * size_scale, 2.0 * water_relief)

    # Build numpy array of [x, water_y, z] points
    points = np.array([[p[0], water_y, p[1]] for p in xz_points])

    result = create_solid_polygon(points, thickness=water_thickness)
    if not result["vertices"]:
        return None

    return {
        "name": _get_attr_or_key(water, "name", "Water"),
        "type": "water",
        "vertices": result["vertices"],
        "faces": result["faces"],
    }


def _generate_building_mesh(
    building, elevation: ElevationData, transform: GeoToModelTransform,
    min_elev: float, elev_range: float, vertical_scale: float, size_scale: float,
    shape_clipper: ShapeClipper | None = None,
    building_shape_gen: BuildingShapeGenerator | None = None,
) -> dict | None:
    """Generate a building using BuildingShapeGenerator for shape variety.

    If a shape_clipper is provided, the building's four corners are checked.
    If any corner is outside the shape boundary, the building is excluded.
    """
    coords = _get_attr_or_key(building, "coordinates", [])
    if len(coords) < 3:
        return None

    # Get bounding box from coordinates
    lats = [_get_lat(c) for c in coords]
    lons = [_get_lon(c) for c in coords]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    # Convert to model space
    x1, z1 = transform.geo_to_model(max_lat, min_lon)  # north-west corner
    x2, z2 = transform.geo_to_model(min_lat, max_lon)  # south-east corner

    # Enforce minimum footprint (1.2mm)
    min_size = 1.2
    if abs(x2 - x1) < min_size:
        cx = (x1 + x2) / 2
        x1 = cx - min_size / 2
        x2 = cx + min_size / 2
    if abs(z2 - z1) < min_size:
        cz = (z1 + z2) / 2
        z1 = cz - min_size / 2
        z2 = cz + min_size / 2

    # Check all 4 corners against shape clipper
    if shape_clipper is not None:
        corners = [(x1, z1), (x1, z2), (x2, z1), (x2, z2)]
        for cx, cz in corners:
            if not shape_clipper.is_inside(cx, cz):
                return None

    # Sample elevation at center for base_y
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    center_elev = _sample_elevation(center_lat, center_lon, elevation)
    base_y = transform.elevation_to_y(center_elev, min_elev, elev_range, vertical_scale, size_scale)

    # Compute height
    height_scale = 1.0
    height_mm = _get_attr_or_key(building, "height", 8.0) * height_scale * 0.15 * size_scale
    height_mm = max(height_mm, min_size)  # Minimum 1.2mm

    # Determine building shape from tags
    tags = _get_attr_or_key(building, "tags", {})
    gen = building_shape_gen or BuildingShapeGenerator()
    shape_type = gen.determine_building_shape(tags)

    # Generate the mesh
    result = gen.generate_building_mesh(x1, x2, base_y, base_y + height_mm, z1, z2, shape_type)

    return {
        "name": _get_attr_or_key(building, "name", "Building"),
        "type": "buildings",
        "vertices": result["vertices"],
        "faces": result["faces"],
    }


def generate_gpx_track_mesh(
    tracks: list,
    elevation: ElevationData,
    bounds: Bounds,
    transform: GeoToModelTransform,
    vertical_scale: float = 1.5,
    shape: str = "square",
) -> MeshResult | None:
    """Generate a GPX track as a cylindrical tube above the terrain.

    GPX tracks are NOT clipped to shape boundaries per convention
    (preserve the natural path). The shape parameter is accepted
    for future use but does not affect output.

    The tube has a circular cross-section with radius = 1mm * size_scale,
    centered at terrain_y + radius so the bottom touches terrain and
    the top protrudes 2 * radius above terrain.
    """
    min_elev, elev_range = _elevation_normalization(elevation.grid)
    model_width = max(transform.model_width_x, transform.model_width_z)
    size_scale = model_width / 200.0

    cylinder_radius = 1.0 * size_scale

    all_vertices = []
    all_faces = []

    for track in tracks:
        points = _get_attr_or_key(track, "points", [])
        if len(points) < 2:
            continue

        # Sample points: only take every Nth point
        sample_rate = max(1, len(points) // 500)
        sampled = points[::sample_rate]
        # Always include the last point
        if sampled[-1] is not points[-1]:
            sampled.append(points[-1])

        if len(sampled) < 2:
            continue

        # Build numpy array of sampled [x, y, z] points
        # Y is set to terrain_y + radius so bottom of cylinder rests on terrain
        track_points = []
        for pt in sampled:
            lat = _get_lat(pt)
            lon = _get_lon(pt)
            x, z = transform.geo_to_model(lat, lon)

            # Use GPX elevation if available, otherwise sample from DEM
            pt_elev = _get_attr_or_key(pt, "elevation", 0)
            if pt_elev and pt_elev > 0:
                elev = pt_elev
            else:
                elev = _sample_elevation(lat, lon, elevation)

            y = transform.elevation_to_y(elev, min_elev, elev_range, vertical_scale, size_scale)
            y += cylinder_radius  # Center of cylinder at terrain + radius
            track_points.append([x, y, z])

        track_points = np.array(track_points)

        result = create_gpx_cylinder_track(track_points, radius=cylinder_radius, n_sides=8)
        if not result["vertices"]:
            continue

        # Offset face indices for merged mesh
        base_vi = len(all_vertices)
        all_vertices.extend(result["vertices"])
        for face in result["faces"]:
            all_faces.append([f + base_vi for f in face])

    if not all_vertices:
        return None

    return MeshResult(
        vertices=all_vertices,
        faces=all_faces,
        name="GPX Track",
        feature_type="gpx_track",
    )


def create_gpx_cylinder_track(centerline, radius=1.0, n_sides=8):
    """Create a cylindrical tube along a centerline path.

    Generates a circular cross-section tube (n_sides polygon) following
    the centerline. The center of the tube is at each centerline point;
    setting centerline Y to terrain_y + radius makes the bottom touch the
    terrain and top protrude 2*radius above it.

    Args:
        centerline: Numpy array of [x, y, z] points defining the tube center.
        radius: Radius of the circular cross-section in mm.
        n_sides: Number of sides for the polygon approximating the circle.

    Returns:
        Dict with 'vertices' (list of [x,y,z]) and 'faces' (list of [i,j,k]).
    """
    if len(centerline) < 2:
        return {"vertices": [], "faces": []}

    # Remove duplicate consecutive points
    clean = [centerline[0]]
    for i in range(1, len(centerline)):
        if not np.allclose(centerline[i], centerline[i - 1], atol=1e-6):
            clean.append(centerline[i])
    centerline = np.array(clean)

    if len(centerline) < 2:
        return {"vertices": [], "faces": []}

    n_points = len(centerline)
    up = np.array([0.0, 1.0, 0.0])

    # Build per-point tangent and perpendicular frame (u, v)
    frames = []
    for i in range(n_points):
        if i == 0:
            tangent = centerline[1] - centerline[0]
        elif i == n_points - 1:
            tangent = centerline[-1] - centerline[-2]
        else:
            tangent = centerline[i + 1] - centerline[i - 1]

        t_len = np.linalg.norm(tangent)
        if t_len < 1e-8:
            tangent = np.array([1.0, 0.0, 0.0])
        else:
            tangent = tangent / t_len

        # Build orthonormal frame perpendicular to tangent
        ref = up if abs(np.dot(tangent, up)) < 0.99 else np.array([1.0, 0.0, 0.0])
        u = np.cross(tangent, ref)
        u /= np.linalg.norm(u)
        v = np.cross(tangent, u)
        v /= np.linalg.norm(v)
        frames.append((u, v))

    # Build rings of vertices
    # Ring i has n_sides vertices at angles 0..2pi
    vertices = []
    for i in range(n_points):
        u, v = frames[i]
        center = centerline[i]
        for j in range(n_sides):
            angle = 2.0 * np.pi * j / n_sides
            pt = center + radius * (np.cos(angle) * u + np.sin(angle) * v)
            vertices.append(pt.tolist())

    faces = []

    # Tube walls: connect adjacent rings
    for i in range(n_points - 1):
        ring_curr = i * n_sides
        ring_next = (i + 1) * n_sides
        for j in range(n_sides):
            j_next = (j + 1) % n_sides
            a = ring_curr + j
            b = ring_curr + j_next
            c = ring_next + j
            d = ring_next + j_next
            # Two triangles per quad (CCW from outside)
            faces.append([a, c, b])
            faces.append([b, c, d])

    # Start cap: fan from center of first ring
    start_center_idx = len(vertices)
    vertices.append(centerline[0].tolist())
    for j in range(n_sides):
        j_next = (j + 1) % n_sides
        a = j
        b = j_next
        # CCW winding facing the start direction
        faces.append([start_center_idx, a, b])

    # End cap: fan from center of last ring
    end_center_idx = len(vertices)
    vertices.append(centerline[-1].tolist())
    end_ring = (n_points - 1) * n_sides
    for j in range(n_sides):
        j_next = (j + 1) % n_sides
        a = end_ring + j
        b = end_ring + j_next
        # CCW winding facing the end direction
        faces.append([end_center_idx, b, a])

    return {
        "vertices": vertices,
        "faces": faces,
    }


def create_road_strip(centerline, width=2.0, thickness=0.3):
    """Create a 3D-printable road strip with thickness along a centerline.

    Creates a box-like extrusion that's watertight for 3D printing.
    Vertex layout per point: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right

    Args:
        centerline: List of [x, y, z] points defining the road center.
        width: Width of the road strip in mm.
        thickness: Vertical thickness of the strip in mm.

    Returns:
        Dict with 'vertices' (list of [x,y,z]) and 'faces' (list of [i,j,k]).
    """
    if len(centerline) < 2:
        return {"vertices": [], "faces": []}

    # Remove duplicate consecutive points
    clean_centerline = [centerline[0]]
    for i in range(1, len(centerline)):
        if not np.allclose(centerline[i], centerline[i - 1], atol=1e-6):
            clean_centerline.append(centerline[i])
    centerline = np.array(clean_centerline)

    if len(centerline) < 2:
        return {"vertices": [], "faces": []}

    vertices = []
    faces = []

    # Generate vertices for top and bottom surfaces
    for i, point in enumerate(centerline):
        if i == 0:
            direction = centerline[i + 1] - centerline[i]
        elif i == len(centerline) - 1:
            direction = centerline[i] - centerline[i - 1]
        else:
            direction = centerline[i + 1] - centerline[i - 1]

        # Normalize in XZ plane (Y is up)
        length = np.linalg.norm([direction[0], direction[2]])
        if length < 1e-6:
            # Degenerate direction, use previous or default
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = direction / length

        # Perpendicular in XZ plane (horizontal) - pointing left when facing direction
        perpendicular = np.array([-direction[2], 0, direction[0]])

        # Create 4 vertices per point: top-left, top-right, bottom-left, bottom-right
        half_width = width / 2
        top_left = point + perpendicular * half_width
        top_right = point - perpendicular * half_width
        bottom_left = np.array([top_left[0], top_left[1] - thickness, top_left[2]])
        bottom_right = np.array([top_right[0], top_right[1] - thickness, top_right[2]])

        vertices.extend([
            top_left.tolist(),
            top_right.tolist(),
            bottom_left.tolist(),
            bottom_right.tolist(),
        ])

    n_points = len(centerline)

    # Create faces for each segment with consistent outward-facing winding
    # Indices: TL=0, TR=1, BL=2, BR=3 (per point)
    for i in range(n_points - 1):
        curr = i * 4          # Current point base index
        next_pt = (i + 1) * 4  # Next point base index

        # Current: TL=curr+0, TR=curr+1, BL=curr+2, BR=curr+3
        # Next:    TL=next+0, TR=next+1, BL=next+2, BR=next+3

        # Top face (normal pointing up +Y) - CCW when viewed from above
        faces.append([curr + 0, curr + 1, next_pt + 1])
        faces.append([curr + 0, next_pt + 1, next_pt + 0])

        # Bottom face (normal pointing down -Y) - CCW when viewed from below
        faces.append([curr + 2, next_pt + 2, next_pt + 3])
        faces.append([curr + 2, next_pt + 3, curr + 3])

        # Left side (normal pointing left) - CCW when viewed from left
        faces.append([curr + 0, next_pt + 0, next_pt + 2])
        faces.append([curr + 0, next_pt + 2, curr + 2])

        # Right side (normal pointing right) - CCW when viewed from right
        faces.append([curr + 1, curr + 3, next_pt + 3])
        faces.append([curr + 1, next_pt + 3, next_pt + 1])

    # Start cap (normal pointing backward along road) - CCW when viewed from start
    faces.append([0, 2, 3])
    faces.append([0, 3, 1])

    # End cap (normal pointing forward along road) - CCW when viewed from end
    end = (n_points - 1) * 4
    faces.append([end + 0, end + 1, end + 3])
    faces.append([end + 0, end + 3, end + 2])

    return {
        "vertices": vertices,
        "faces": faces,
    }


def triangulate_polygon(points_2d):
    """Triangulate a 2D polygon using ear-clipping.

    Ear-clipping preserves all polygon boundary edges, which is critical
    for manifold mesh generation (walls reference boundary edges).

    Args:
        points_2d: Nx2 numpy array of 2D points forming polygon boundary.

    Returns:
        List of triangle index triplets (indices into original points_2d).
    """
    n = len(points_2d)
    if n < 3:
        return []
    if n == 3:
        return [[0, 1, 2]]

    points_2d = np.array(points_2d, dtype=np.float64)

    # Determine polygon winding (need CCW for ear-clipping)
    # Shoelace formula for signed area
    signed_area = 0.0
    for i in range(n):
        j = (i + 1) % n
        signed_area += points_2d[i, 0] * points_2d[j, 1]
        signed_area -= points_2d[j, 0] * points_2d[i, 1]
    ccw = signed_area > 0

    # Build index list (we remove vertices as we clip ears)
    indices = list(range(n))
    triangles = []

    def cross_2d(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def is_ear(idx_list, pos, pts, is_ccw):
        """Check if vertex at position pos in idx_list is an ear."""
        m = len(idx_list)
        prev_pos = (pos - 1) % m
        next_pos = (pos + 1) % m

        a = pts[idx_list[prev_pos]]
        b = pts[idx_list[pos]]
        c = pts[idx_list[next_pos]]

        # Check if this vertex is convex (cross product sign matches winding)
        cross = cross_2d(a, b, c)
        if is_ccw and cross <= 1e-12:
            return False
        if not is_ccw and cross >= -1e-12:
            return False

        # Check that no other polygon vertex is inside this triangle
        for j in range(m):
            if j == prev_pos or j == pos or j == next_pos:
                continue
            p = pts[idx_list[j]]
            # Point-in-triangle test using barycentric coordinates
            d1 = cross_2d(a, b, p)
            d2 = cross_2d(b, c, p)
            d3 = cross_2d(c, a, p)
            has_neg = (d1 < -1e-12) or (d2 < -1e-12) or (d3 < -1e-12)
            has_pos = (d1 > 1e-12) or (d2 > 1e-12) or (d3 > 1e-12)
            if not (has_neg and has_pos):
                # Point is inside or on edge of triangle
                return False
        return True

    max_iterations = n * n  # Safety limit
    iteration = 0
    while len(indices) > 3 and iteration < max_iterations:
        ear_found = False
        m = len(indices)
        for i in range(m):
            if is_ear(indices, i, points_2d, ccw):
                prev_pos = (i - 1) % m
                next_pos = (i + 1) % m
                triangles.append([indices[prev_pos], indices[i], indices[next_pos]])
                indices.pop(i)
                ear_found = True
                break
        if not ear_found:
            # No ear found - polygon may be degenerate
            # Use remaining vertices as fan triangulation fallback
            for i in range(1, len(indices) - 1):
                triangles.append([indices[0], indices[i], indices[i + 1]])
            break
        iteration += 1

    # Add last triangle
    if len(indices) == 3:
        triangles.append([indices[0], indices[1], indices[2]])

    return triangles


def create_solid_polygon(points, thickness=0.5):
    """Create a 3D-printable solid polygon with thickness.

    Creates a watertight mesh with top surface, bottom surface, and side walls.
    Uses ear-clipping triangulation for proper polygon fill.

    Args:
        points: Numpy array of [x, y, z] coordinates forming the polygon outline.
        thickness: Height of the extrusion (in model units).

    Returns:
        dict: Mesh with 'vertices' and 'faces' lists.
    """
    n = len(points)
    if n < 3:
        return {"vertices": [], "faces": []}

    points = np.array(points)

    # Remove duplicate consecutive points
    unique_mask = np.ones(n, dtype=bool)
    for i in range(n):
        next_i = (i + 1) % n
        if np.allclose(points[i], points[next_i], atol=1e-6):
            unique_mask[next_i] = False
    points = points[unique_mask]
    n = len(points)

    if n < 3:
        return {"vertices": [], "faces": []}

    # Merge non-consecutive near-duplicate vertices (common in OSM water polygons
    # where the boundary visits the same point twice, e.g. at pinch points)
    tree = cKDTree(points[:, [0, 2]])  # Match in XZ plane
    pairs = tree.query_pairs(r=1e-4)  # Find near-duplicate pairs
    if pairs:
        # Build mapping: for each duplicate, map to the lowest index
        remap = list(range(n))
        for i, j in pairs:
            lo, hi = min(i, j), max(i, j)
            remap[hi] = lo
        # Resolve chains (if a->b->c, make a->c and b->c)
        for i in range(n):
            while remap[i] != remap[remap[i]]:
                remap[i] = remap[remap[i]]
        # Rebuild points list removing duplicates but preserving order
        keep = [i for i in range(n) if remap[i] == i]
        old_to_new = {}
        for new_idx, old_idx in enumerate(keep):
            old_to_new[old_idx] = new_idx
        # Map all remapped indices
        index_map = [old_to_new[remap[i]] for i in range(n)]
        # Rebuild polygon: walk original order but use remapped indices,
        # skip consecutive dupes
        new_indices = []
        for i in range(n):
            mapped = index_map[i]
            if len(new_indices) == 0 or mapped != new_indices[-1]:
                new_indices.append(mapped)
        # Remove wrap-around duplicate
        if len(new_indices) > 1 and new_indices[0] == new_indices[-1]:
            new_indices.pop()
        points = points[keep]
        n = len(points)

    if n < 3:
        return {"vertices": [], "faces": []}

    # Remove collinear vertices (common in OSM data where straight edges
    # have many nodes). Collinear vertices cause ear-clipping to fail
    # because the cross product is zero.
    non_collinear = []
    for i in range(n):
        prev_i = (i - 1) % n
        next_i = (i + 1) % n
        ax, az = points[prev_i, 0], points[prev_i, 2]
        bx, bz = points[i, 0], points[i, 2]
        cx, cz = points[next_i, 0], points[next_i, 2]
        cross = (bx - ax) * (cz - az) - (bz - az) * (cx - ax)
        if abs(cross) > 1e-10:
            non_collinear.append(i)
    if len(non_collinear) < n and len(non_collinear) >= 3:
        points = points[non_collinear]
        n = len(points)

    if n < 3:
        return {"vertices": [], "faces": []}

    vertices = []
    faces = []

    # Create top vertices (original points)
    for p in points:
        vertices.append([float(p[0]), float(p[1]), float(p[2])])

    # Create bottom vertices (offset down by thickness)
    for p in points:
        vertices.append([float(p[0]), float(p[1]) - thickness, float(p[2])])

    # Triangulate top/bottom faces using ear-clipping algorithm
    top_faces = triangulate_polygon(points[:, [0, 2]])  # Project to XZ plane

    if len(top_faces) == 0:
        # Fallback to simple fan triangulation
        for i in range(1, n - 1):
            top_faces.append([0, i, i + 1])

    # Top face
    for tri in top_faces:
        faces.append([int(tri[0]), int(tri[1]), int(tri[2])])

    # Bottom face (reversed winding)
    for tri in top_faces:
        faces.append([int(n + tri[0]), int(n + tri[2]), int(n + tri[1])])

    # Side walls - connect top and bottom edges
    for i in range(n):
        next_i = (i + 1) % n
        top_curr = i
        top_next = next_i
        bot_curr = n + i
        bot_next = n + next_i

        # Two triangles per side segment (consistent winding for outward normals)
        faces.append([top_curr, bot_curr, top_next])
        faces.append([top_next, bot_curr, bot_next])

    return {
        "vertices": vertices,
        "faces": faces,
    }
