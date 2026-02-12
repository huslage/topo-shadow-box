"""Mesh generation for terrain, features, and GPX tracks."""

import numpy as np
from scipy import interpolate as sci_interpolate

from ..state import Bounds, ElevationData
from .coords import GeoToModelTransform


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


def generate_terrain_mesh(
    elevation: ElevationData,
    bounds: Bounds,
    transform: GeoToModelTransform,
    vertical_scale: float = 1.5,
    base_height_mm: float = 10.0,
    shape: str = "square",
) -> dict:
    """Generate terrain mesh from elevation grid.

    Returns dict with 'vertices' and 'faces' (lists of lists).
    """
    grid = elevation.grid
    lats = elevation.lats
    lons = elevation.lons
    rows, cols = grid.shape

    min_elev, elev_range = _elevation_normalization(grid)
    size_scale = transform.scale_factor * (bounds.lat_range if bounds.lat_range > 0 else 1.0) / 200.0
    # Simpler: model_width / 200 for consistent scaling
    model_width = max(transform.model_width_x, transform.model_width_z)
    size_scale = model_width / 200.0

    vertices = []
    for i in range(rows):
        for j in range(cols):
            x, z = transform.geo_to_model(lats[i], lons[j])
            y = transform.elevation_to_y(
                float(grid[i, j]), min_elev, elev_range, vertical_scale, size_scale,
            )
            vertices.append([x, y, z])

    vertices = np.array(vertices)

    # Shape clipping mask
    if shape == "circle":
        cx = transform.model_width_x / 2
        cz = transform.model_width_z / 2
        radius = min(transform.model_width_x, transform.model_width_z) / 2
        dx = vertices[:, 0] - cx
        dz = vertices[:, 2] - cz
        inside = (dx * dx + dz * dz) <= radius * radius
    else:
        inside = np.ones(len(vertices), dtype=bool)

    # Generate faces (two triangles per quad, CCW from above)
    faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            idx = i * cols + j
            idx_r = idx + 1
            idx_d = idx + cols
            idx_dr = idx + cols + 1

            if inside[idx] and inside[idx_d] and inside[idx_r]:
                faces.append([idx, idx_d, idx_r])
            if inside[idx_r] and inside[idx_d] and inside[idx_dr]:
                faces.append([idx_r, idx_d, idx_dr])

    # Add watertight base
    base_verts, base_faces = _generate_base(vertices, base_height_mm, rows, cols, inside)
    n_top = len(vertices)
    vertices = np.vstack([vertices, base_verts])
    offset_base_faces = base_faces + n_top
    # Base faces reference both top (original) and bottom (offset) vertices
    # We need to fix the base faces that reference top vertices
    all_faces = faces + _build_base_faces(n_top, rows, cols, inside, base_height_mm, vertices)

    # Simpler: use the proven pattern from topo3d
    # Rebuild with proper base
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

    # Bottom vertices (same X,Z but Y = -base_height)
    bottom_verts = top_verts.copy()
    bottom_verts[:, 1] = -base_height_mm

    all_verts = np.vstack([top_verts, bottom_verts])

    faces = []
    # Top surface (terrain) - CCW from above
    for i in range(rows - 1):
        for j in range(cols - 1):
            idx = i * cols + j
            idx_r = idx + 1
            idx_d = idx + cols
            idx_dr = idx + cols + 1
            if inside[idx] and inside[idx_d] and inside[idx_r]:
                faces.append([idx, idx_d, idx_r])
            if inside[idx_r] and inside[idx_d] and inside[idx_dr]:
                faces.append([idx_r, idx_d, idx_dr])

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


def _generate_base(vertices, base_height, rows, cols, inside):
    """Placeholder - actual base generation is inline above."""
    bottom = vertices.copy()
    bottom[:, 1] = -base_height
    return bottom, np.array([]).reshape(0, 3).astype(int)


def _build_base_faces(n_top, rows, cols, inside, base_height, vertices):
    """Placeholder - actual base faces are built inline above."""
    return []


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
    for road in features.get("roads", []):
        road_mesh = _generate_road_mesh(
            road, elevation, transform, min_elev, elev_range, vertical_scale, size_scale,
        )
        if road_mesh:
            meshes.append(road_mesh)

    # Water: flat polygons at water level
    for water in features.get("water", []):
        water_mesh = _generate_water_mesh(
            water, elevation, transform, min_elev, elev_range, vertical_scale, size_scale,
        )
        if water_mesh:
            meshes.append(water_mesh)

    # Buildings: extruded footprints
    for building in features.get("buildings", []):
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
    """Generate a road as a flat ribbon following the terrain."""
    coords = road.get("coordinates", [])
    if len(coords) < 2:
        return None

    road_width_mm = 1.0 * size_scale  # 1mm at 200mm model
    road_height_offset = 0.2 * size_scale

    vertices = []
    faces = []

    for i, coord in enumerate(coords):
        x, z = transform.geo_to_model(coord["lat"], coord["lon"])
        elev = _sample_elevation(coord["lat"], coord["lon"], elevation)
        y = transform.elevation_to_y(elev, min_elev, elev_range, vertical_scale, size_scale)
        y += road_height_offset

        # Compute perpendicular direction for road width
        if i < len(coords) - 1:
            next_c = coords[i + 1]
            nx, nz = transform.geo_to_model(next_c["lat"], next_c["lon"])
            dx, dz = nx - x, nz - z
        else:
            prev_c = coords[i - 1]
            px, pz = transform.geo_to_model(prev_c["lat"], prev_c["lon"])
            dx, dz = x - px, z - pz

        length = (dx * dx + dz * dz) ** 0.5
        if length < 1e-10:
            continue

        # Perpendicular vector
        px = -dz / length * road_width_mm / 2
        pz = dx / length * road_width_mm / 2

        vi = len(vertices)
        vertices.append([x + px, y, z + pz])
        vertices.append([x - px, y, z - pz])

        if i > 0 and vi >= 2:
            # Connect to previous segment
            faces.append([vi - 2, vi, vi + 1])
            faces.append([vi - 2, vi + 1, vi - 1])

    if len(vertices) < 4:
        return None

    return {
        "name": road.get("name", "Road"),
        "type": "roads",
        "vertices": vertices,
        "faces": faces,
    }


def _generate_water_mesh(
    water: dict, elevation: ElevationData, transform: GeoToModelTransform,
    min_elev: float, elev_range: float, vertical_scale: float, size_scale: float,
) -> dict | None:
    """Generate a water body as a flat polygon."""
    coords = water.get("coordinates", [])
    if len(coords) < 3:
        return None

    # Find average elevation for water surface
    elevs = []
    points_2d = []
    for coord in coords:
        x, z = transform.geo_to_model(coord["lat"], coord["lon"])
        elev = _sample_elevation(coord["lat"], coord["lon"], elevation)
        elevs.append(elev)
        points_2d.append((x, z))

    avg_elev = sum(elevs) / len(elevs)
    y = transform.elevation_to_y(avg_elev, min_elev, elev_range, vertical_scale, size_scale)
    y -= 0.1 * size_scale  # Slightly below terrain

    vertices = [[p[0], y, p[1]] for p in points_2d]

    # Simple fan triangulation from first vertex
    faces = []
    for i in range(1, len(vertices) - 1):
        faces.append([0, i, i + 1])

    return {
        "name": water.get("name", "Water"),
        "type": "water",
        "vertices": vertices,
        "faces": faces,
    }


def _generate_building_mesh(
    building: dict, elevation: ElevationData, transform: GeoToModelTransform,
    min_elev: float, elev_range: float, vertical_scale: float, size_scale: float,
) -> dict | None:
    """Generate a building as an extruded footprint."""
    coords = building.get("coordinates", [])
    if len(coords) < 3:
        return None

    height_m = building.get("height", 10.0)
    height_mm = height_m * 0.1 * size_scale  # Scale building height

    # Compute footprint vertices at ground level
    ground_verts = []
    for coord in coords:
        x, z = transform.geo_to_model(coord["lat"], coord["lon"])
        elev = _sample_elevation(coord["lat"], coord["lon"], elevation)
        y = transform.elevation_to_y(elev, min_elev, elev_range, vertical_scale, size_scale)
        ground_verts.append([x, y, z])

    n = len(ground_verts)
    # Roof vertices
    roof_verts = [[v[0], v[1] + height_mm, v[2]] for v in ground_verts]

    vertices = ground_verts + roof_verts
    faces = []

    # Roof (fan triangulation, CCW from above)
    for i in range(1, n - 1):
        faces.append([n + 0, n + i, n + i + 1])

    # Walls
    for i in range(n):
        j = (i + 1) % n
        gi, gj = i, j
        ri, rj = n + i, n + j
        # CCW from outside
        faces.append([gi, gj, rj])
        faces.append([gi, rj, ri])

    return {
        "name": building.get("name", "Building"),
        "type": "buildings",
        "vertices": vertices,
        "faces": faces,
    }


def generate_gpx_track_mesh(
    tracks: list[dict],
    elevation: ElevationData,
    bounds: Bounds,
    transform: GeoToModelTransform,
    vertical_scale: float = 1.5,
) -> dict | None:
    """Generate a GPX track as a ribbon mesh above the terrain."""
    min_elev, elev_range = _elevation_normalization(elevation.grid)
    model_width = max(transform.model_width_x, transform.model_width_z)
    size_scale = model_width / 200.0
    track_width = 1.5 * size_scale
    track_height_offset = 0.5 * size_scale

    all_vertices = []
    all_faces = []

    for track in tracks:
        points = track.get("points", [])
        if len(points) < 2:
            continue

        base_vi = len(all_vertices)
        for i, pt in enumerate(points):
            x, z = transform.geo_to_model(pt["lat"], pt["lon"])

            # Use GPX elevation if available, otherwise sample from DEM
            if pt.get("elevation", 0) > 0:
                elev = pt["elevation"]
            else:
                elev = _sample_elevation(pt["lat"], pt["lon"], elevation)

            y = transform.elevation_to_y(elev, min_elev, elev_range, vertical_scale, size_scale)
            y += track_height_offset

            # Perpendicular for ribbon width
            if i < len(points) - 1:
                npt = points[i + 1]
                nx, nz = transform.geo_to_model(npt["lat"], npt["lon"])
                dx, dz = nx - x, nz - z
            else:
                ppt = points[i - 1]
                px, pz = transform.geo_to_model(ppt["lat"], ppt["lon"])
                dx, dz = x - px, z - pz

            length = (dx * dx + dz * dz) ** 0.5
            if length < 1e-10:
                # Use previous direction or skip
                if i > 0:
                    all_vertices.append(all_vertices[-2][:])
                    all_vertices.append(all_vertices[-2][:])
                continue

            perp_x = -dz / length * track_width / 2
            perp_z = dx / length * track_width / 2

            all_vertices.append([x + perp_x, y, z + perp_z])
            all_vertices.append([x - perp_x, y, z - perp_z])

            vi = len(all_vertices) - 2
            if vi >= base_vi + 2:
                all_faces.append([vi - 2, vi, vi + 1])
                all_faces.append([vi - 2, vi + 1, vi - 1])

    if not all_vertices:
        return None

    return {
        "vertices": all_vertices,
        "faces": all_faces,
        "name": "GPX Track",
        "type": "gpx_track",
    }
