"""Map insert generation: SVG background map and 3D flat plate."""

from ..state import Bounds, Colors
from .coords import GeoToModelTransform
from .models import MeshResult


def generate_map_insert_svg(
    bounds: Bounds,
    features: dict,
    gpx_tracks: list,
    colors: Colors,
) -> str:
    """Generate an SVG map of features for paper printing.

    Returns SVG string. Stored in state for later export.
    """
    # Coordinate transform: geo -> SVG viewport
    width = 800  # SVG pixels
    lat_range = bounds.lat_range
    lon_range = bounds.lon_range
    import math
    lon_scale = math.cos(math.radians(bounds.center_lat))
    aspect = (lon_range * lon_scale) / lat_range if lat_range > 0 else 1.0
    height = int(width / aspect) if aspect > 0 else width

    def geo_to_svg(lat: float, lon: float) -> tuple[float, float]:
        x = (lon - bounds.west) / lon_range * width if lon_range > 0 else 0
        y = (bounds.north - lat) / lat_range * height if lat_range > 0 else 0
        return x, y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{colors.map_insert}"/>',
    ]

    # Water bodies
    for water in features.get("water", []):
        coords = water.get("coordinates", [])
        if len(coords) < 3:
            continue
        points = " ".join(f"{geo_to_svg(c['lat'], c['lon'])[0]:.1f},{geo_to_svg(c['lat'], c['lon'])[1]:.1f}" for c in coords)
        parts.append(f'<polygon points="{points}" fill="{colors.water}" opacity="0.6"/>')

    # Roads
    for road in features.get("roads", []):
        coords = road.get("coordinates", [])
        if len(coords) < 2:
            continue
        points = " ".join(f"{geo_to_svg(c['lat'], c['lon'])[0]:.1f},{geo_to_svg(c['lat'], c['lon'])[1]:.1f}" for c in coords)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colors.roads}" stroke-width="1" opacity="0.5"/>')

    # Buildings
    for bldg in features.get("buildings", []):
        coords = bldg.get("coordinates", [])
        if len(coords) < 3:
            continue
        points = " ".join(f"{geo_to_svg(c['lat'], c['lon'])[0]:.1f},{geo_to_svg(c['lat'], c['lon'])[1]:.1f}" for c in coords)
        parts.append(f'<polygon points="{points}" fill="{colors.buildings}" opacity="0.4"/>')

    # GPX tracks
    for track in gpx_tracks:
        pts = track.get("points", [])
        if len(pts) < 2:
            continue
        points = " ".join(f"{geo_to_svg(p['lat'], p['lon'])[0]:.1f},{geo_to_svg(p['lat'], p['lon'])[1]:.1f}" for p in pts)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colors.gpx_track}" stroke-width="2"/>')

    parts.append("</svg>")
    return "\n".join(parts)


def generate_map_insert_plate(
    bounds: Bounds,
    features: dict,
    gpx_tracks: list,
    transform: GeoToModelTransform,
    plate_thickness_mm: float = 1.0,
) -> MeshResult:
    """Generate a thin 3D plate for the map insert.

    The plate is a flat rectangle matching the model dimensions, with features
    as very slightly raised regions for visual/tactile effect.

    Returns dict with vertices and faces.
    """
    w = transform.model_width_x
    h = transform.model_width_z
    t = plate_thickness_mm

    # Place the plate below the terrain (at -base_height - some offset)
    y_top = 0.0
    y_bot = -t

    vertices = [
        [0, y_top, 0],      # 0: top NW
        [w, y_top, 0],      # 1: top NE
        [w, y_top, h],      # 2: top SE
        [0, y_top, h],      # 3: top SW
        [0, y_bot, 0],      # 4: bot NW
        [w, y_bot, 0],      # 5: bot NE
        [w, y_bot, h],      # 6: bot SE
        [0, y_bot, h],      # 7: bot SW
    ]

    faces = [
        # Top
        [0, 1, 2], [0, 2, 3],
        # Bottom
        [4, 6, 5], [4, 7, 6],
        # Front
        [3, 2, 6], [3, 6, 7],
        # Back
        [0, 5, 1], [0, 4, 5],
        # Left
        [0, 3, 7], [0, 7, 4],
        # Right
        [1, 6, 2], [1, 5, 6],
    ]

    return MeshResult(vertices=vertices, faces=faces, name="Map Insert", feature_type="map_insert")
