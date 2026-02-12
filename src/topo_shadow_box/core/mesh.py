"""Mesh generation for terrain, features, and GPX tracks."""

from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from ..state import Bounds, ElevationData
from .coords import GeoToModelTransform
from .building_shapes import BuildingShapeGenerator
from .shape_clipper import HexagonClipper


def _elevation_normalization(grid: np.ndarray) -> tuple[float, float]:
    """Compute min elevation and range for normalization."""
    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        return 0.0, 1.0
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
) -> dict:
    """Generate terrain mesh from elevation grid.

    Produces shape-aware bases:
    - square/rectangle: rectangular side walls and flat bottom
    - circle: smooth 360-segment circular wall with interpolated contour
    - hexagon: boundary-edge-based wall following terrain contour

    Returns dict with 'vertices' and 'faces' (lists of lists).
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
        return _generate_circle_terrain(
            top_verts, terrain_faces, rows, cols, cx, cz, radius,
            inside, base_height_mm,
        )
    elif shape == "hexagon":
        return _generate_hexagon_terrain(
            top_verts, terrain_faces, rows, cols, cx, cz,
            inside, base_height_mm, hex_clipper,
        )
    else:
        return _generate_square_terrain(
            top_verts, terrain_faces, rows, cols, n, base_height_mm,
        )


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
    """Generate smooth circular base with 360-segment wall.

    Finds boundary vertices (inside with an outside neighbor), interpolates
    elevation around the circle, creates a smooth wall, and stretches boundary
    terrain vertices outward to close gaps.
    """
    top_verts = top_verts.copy()
    n = len(top_verts)
    num_segments = 360

    # Find boundary vertices and their angles for elevation interpolation
    # A boundary vertex is inside the circle AND has at least one grid neighbor outside
    boundary_data = []  # (angle, elevation, x, z, vertex_index)
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
                    v = top_verts[idx]
                    angle = np.arctan2(v[2] - cz, v[0] - cx)
                    boundary_data.append((angle, v[1], v[0], v[2], idx))
                    break

    # Sort by angle for interpolation
    boundary_data.sort(key=lambda x: x[0])

    # Move ALL boundary vertices outward to exactly match the wall circle
    # This stretches the terrain mesh to close the gap with the smooth wall
    for data in boundary_data:
        idx = data[4]
        v = top_verts[idx]
        dx = v[0] - cx
        dz = v[2] - cz
        dist = np.sqrt(dx * dx + dz * dz)
        if dist > 0:
            scale = radius / dist
            top_verts[idx][0] = cx + dx * scale
            top_verts[idx][2] = cz + dz * scale

    # Vertex layout for wall/base:
    # n .. n+359:          outer top circle (wall top edge on exact circle)
    # n+360 .. n+719:      outer bottom circle (wall base)
    # n+720:               center bottom vertex
    outer_top_start = n
    outer_bottom_start = n + num_segments
    center_bottom_idx = n + 2 * num_segments

    wall_verts = []
    for i in range(num_segments):
        angle = 2 * np.pi * i / num_segments - np.pi  # -pi to pi

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        outer_x = cx + radius * cos_a
        outer_z = cz + radius * sin_a

        # Interpolate elevation from boundary vertices
        elev = _interpolate_boundary_elevation(angle, boundary_data)

        # Top follows terrain contour
        wall_verts.append([outer_x, elev, outer_z])

    for i in range(num_segments):
        angle = 2 * np.pi * i / num_segments - np.pi
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        outer_x = cx + radius * cos_a
        outer_z = cz + radius * sin_a
        # Bottom at base
        wall_verts.append([outer_x, -base_height_mm, outer_z])

    # Center bottom vertex
    wall_verts.append([cx, -base_height_mm, cz])

    all_verts = np.vstack([top_verts, np.array(wall_verts)])

    faces = list(terrain_faces)

    # Outer wall faces (CCW when viewed from outside)
    for i in range(num_segments):
        next_i = (i + 1) % num_segments
        ot0 = outer_top_start + i
        ot1 = outer_top_start + next_i
        ob0 = outer_bottom_start + i
        ob1 = outer_bottom_start + next_i
        faces.append([ot0, ob0, ot1])
        faces.append([ot1, ob0, ob1])

    # Bottom face: fan triangulation from center (CCW from below)
    for i in range(num_segments):
        next_i = (i + 1) % num_segments
        ob0 = outer_bottom_start + i
        ob1 = outer_bottom_start + next_i
        faces.append([center_bottom_idx, ob1, ob0])

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



def generate_feature_meshes(
    features: dict,
    elevation: ElevationData,
    bounds: Bounds,
    transform: GeoToModelTransform,
    vertical_scale: float = 1.5,
    shape: str = "square",
) -> list[dict]:
    """Generate meshes for OSM features (roads, water, buildings).

    Returns list of dicts with: name, type, vertices, faces.
    """
    min_elev, elev_range = _elevation_normalization(elevation.grid)
    model_width = max(transform.model_width_x, transform.model_width_z)
    size_scale = model_width / 200.0

    meshes = []

    # Roads: extruded strips following terrain
    for road in features.get("roads", [])[:200]:
        road_mesh = _generate_road_mesh(
            road, elevation, transform, min_elev, elev_range, vertical_scale, size_scale,
        )
        if road_mesh:
            meshes.append(road_mesh)

    # Water: solid polygons at water level
    for water in features.get("water", [])[:50]:
        water_mesh = _generate_water_mesh(
            water, elevation, transform, min_elev, elev_range, vertical_scale, size_scale,
        )
        if water_mesh:
            meshes.append(water_mesh)

    # Buildings: shape-aware extruded footprints
    for building in features.get("buildings", [])[:150]:
        building_mesh = _generate_building_mesh(
            building, elevation, transform, min_elev, elev_range, vertical_scale, size_scale,
        )
        if building_mesh:
            meshes.append(building_mesh)

    return meshes


def _generate_road_mesh(
    road: dict, elevation: ElevationData, transform: GeoToModelTransform,
    min_elev: float, elev_range: float, vertical_scale: float, size_scale: float,
) -> dict | None:
    """Generate a road as a watertight strip following the terrain."""
    coords = road.get("coordinates", [])
    if len(coords) < 2:
        return None

    road_height_offset = 0.2 * size_scale
    road_relief = max(road_height_offset, 0.6 * size_scale)

    # Convert coordinates to model space with elevation sampling
    points = []
    for coord in coords:
        x, z = transform.geo_to_model(coord["lat"], coord["lon"])
        elev = _sample_elevation(coord["lat"], coord["lon"], elevation)
        y = transform.elevation_to_y(elev, min_elev, elev_range, vertical_scale, size_scale)
        y += road_relief
        points.append([x, y, z])

    points = np.array(points)

    road_width = 1.0 * size_scale
    road_thickness = max(0.9 * size_scale, road_relief)

    result = create_road_strip(points, width=road_width, thickness=road_thickness)
    if not result["vertices"]:
        return None

    return {
        "name": road.get("name", "Road"),
        "type": "roads",
        "vertices": result["vertices"],
        "faces": result["faces"],
    }


def _generate_water_mesh(
    water: dict, elevation: ElevationData, transform: GeoToModelTransform,
    min_elev: float, elev_range: float, vertical_scale: float, size_scale: float,
) -> dict | None:
    """Generate a water body as a watertight solid polygon."""
    coords = water.get("coordinates", [])
    if len(coords) < 3:
        return None

    # Compute average perimeter elevation
    elevs = []
    xz_points = []
    for coord in coords:
        x, z = transform.geo_to_model(coord["lat"], coord["lon"])
        elev = _sample_elevation(coord["lat"], coord["lon"], elevation)
        elevs.append(elev)
        xz_points.append((x, z))

    avg_elev = sum(elevs) / len(elevs)
    water_y_base = transform.elevation_to_y(avg_elev, min_elev, elev_range, vertical_scale, size_scale)

    water_relief = max(0.6 * size_scale, 0.5 * size_scale)  # = 0.6 * size_scale
    water_y = max(0.7 * size_scale, water_y_base + water_relief)
    water_thickness = max(1.2 * size_scale, 2.0 * water_relief)

    # Build numpy array of [x, water_y, z] points
    points = np.array([[p[0], water_y, p[1]] for p in xz_points])

    result = create_solid_polygon(points, thickness=water_thickness)
    if not result["vertices"]:
        return None

    return {
        "name": water.get("name", "Water"),
        "type": "water",
        "vertices": result["vertices"],
        "faces": result["faces"],
    }


def _generate_building_mesh(
    building: dict, elevation: ElevationData, transform: GeoToModelTransform,
    min_elev: float, elev_range: float, vertical_scale: float, size_scale: float,
) -> dict | None:
    """Generate a building using BuildingShapeGenerator for shape variety."""
    coords = building.get("coordinates", [])
    if len(coords) < 3:
        return None

    # Get bounding box from coordinates
    lats = [c["lat"] for c in coords]
    lons = [c["lon"] for c in coords]
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

    # Sample elevation at center for base_y
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    center_elev = _sample_elevation(center_lat, center_lon, elevation)
    base_y = transform.elevation_to_y(center_elev, min_elev, elev_range, vertical_scale, size_scale)

    # Compute height
    height_scale = 1.0
    height_mm = building.get("height", 8.0) * height_scale * 0.15 * size_scale
    height_mm = max(height_mm, min_size)  # Minimum 1.2mm

    # Determine building shape from tags
    tags = building.get("tags", {})
    gen = BuildingShapeGenerator()
    shape_type = gen.determine_building_shape(tags)

    # Generate the mesh
    result = gen.generate_building_mesh(x1, x2, base_y, base_y + height_mm, z1, z2, shape_type)

    return {
        "name": building.get("name", "Building"),
        "type": "buildings",
        "vertices": result["vertices"],
        "faces": result["faces"],
    }


def generate_gpx_track_mesh(
    tracks: list[dict],
    elevation: ElevationData,
    bounds: Bounds,
    transform: GeoToModelTransform,
    vertical_scale: float = 1.5,
) -> dict | None:
    """Generate a GPX track as a watertight strip above the terrain."""
    min_elev, elev_range = _elevation_normalization(elevation.grid)
    model_width = max(transform.model_width_x, transform.model_width_z)
    size_scale = model_width / 200.0

    gpx_relief = max(0.8 * size_scale, 0.3 * size_scale)  # = 0.8 * size_scale
    gpx_thickness = max(1.2 * size_scale, 2.0 * gpx_relief)
    gpx_width = 2.5 * size_scale

    all_vertices = []
    all_faces = []

    for track in tracks:
        points = track.get("points", [])
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
        track_points = []
        for pt in sampled:
            x, z = transform.geo_to_model(pt["lat"], pt["lon"])

            # Use GPX elevation if available, otherwise sample from DEM
            if pt.get("elevation", 0) > 0:
                elev = pt["elevation"]
            else:
                elev = _sample_elevation(pt["lat"], pt["lon"], elevation)

            y = transform.elevation_to_y(elev, min_elev, elev_range, vertical_scale, size_scale)
            y += gpx_relief
            track_points.append([x, y, z])

        track_points = np.array(track_points)

        result = create_road_strip(track_points, width=gpx_width, thickness=gpx_thickness)
        if not result["vertices"]:
            continue

        # Offset face indices for merged mesh
        base_vi = len(all_vertices)
        all_vertices.extend(result["vertices"])
        for face in result["faces"]:
            all_faces.append([f + base_vi for f in face])

    if not all_vertices:
        return None

    return {
        "vertices": all_vertices,
        "faces": all_faces,
        "name": "GPX Track",
        "type": "gpx_track",
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
