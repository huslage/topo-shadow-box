"""Elevation data fetching from AWS Terrain-RGB tiles (Terrarium format)."""

import math
import numpy as np
from io import BytesIO

import httpx
from PIL import Image
from scipy import interpolate
from scipy.ndimage import gaussian_filter

# AWS Terrain Tiles (Mapzen/Tilezen Terrarium format) - free, globally available
AWS_TERRAIN_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"


def _lat_lon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """Convert lat/lon to tile coordinates at a given zoom level."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def _tile_to_lat_lon(x: int, y: int, zoom: int) -> tuple[float, float]:
    """Convert tile coordinates to lat/lon (northwest corner)."""
    n = 2.0 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def _decode_terrarium(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Decode Terrarium format RGB to elevation in meters."""
    return (r * 256.0 + g + b / 256.0) - 32768.0


def _pick_zoom(north: float, south: float, east: float, west: float) -> int:
    """Pick an appropriate zoom level based on area span."""
    max_span = max(north - south, east - west)
    if max_span > 1.0:
        return 10
    elif max_span > 0.5:
        return 11
    elif max_span > 0.1:
        return 12
    elif max_span > 0.05:
        return 13
    else:
        return 14


async def fetch_terrain_elevation(
    north: float, south: float, east: float, west: float,
    resolution: int = 200,
) -> dict:
    """Fetch elevation data for a bounding box from AWS Terrain-RGB tiles.

    Returns dict with: grid (ndarray), lats (ndarray), lons (ndarray),
    min_elevation, max_elevation
    """
    zoom = _pick_zoom(north, south, east, west)

    x_min, y_max = _lat_lon_to_tile(south, west, zoom)
    x_max, y_min = _lat_lon_to_tile(north, east, zoom)

    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min

    num_tiles_x = x_max - x_min + 1
    num_tiles_y = y_max - y_min + 1

    if num_tiles_x * num_tiles_y > 25:
        # Fall back to lower zoom
        zoom = max(zoom - 1, 8)
        x_min, y_max = _lat_lon_to_tile(south, west, zoom)
        x_max, y_min = _lat_lon_to_tile(north, east, zoom)
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min
        num_tiles_x = x_max - x_min + 1
        num_tiles_y = y_max - y_min + 1

    tile_size = 256
    stitched_width = num_tiles_x * tile_size
    stitched_height = num_tiles_y * tile_size
    stitched_elevations = np.zeros((stitched_height, stitched_width))

    async with httpx.AsyncClient(timeout=30.0) as client:
        for ty in range(y_min, y_max + 1):
            for tx in range(x_min, x_max + 1):
                url = AWS_TERRAIN_URL.format(z=zoom, x=tx, y=ty)
                try:
                    response = await client.get(url)
                    if response.status_code == 200:
                        img = Image.open(BytesIO(response.content))
                        img_array = np.array(img)
                        r = img_array[:, :, 0].astype(np.float64)
                        g = img_array[:, :, 1].astype(np.float64)
                        b = img_array[:, :, 2].astype(np.float64)
                        tile_elevations = _decode_terrarium(r, g, b)

                        px = (tx - x_min) * tile_size
                        py = (ty - y_min) * tile_size
                        stitched_elevations[py:py + tile_size, px:px + tile_size] = tile_elevations
                except Exception:
                    pass  # Leave zeros for failed tiles

    # Map tile pixel coordinates to geographic coordinates
    tile_north, tile_west = _tile_to_lat_lon(x_min, y_min, zoom)
    tile_south, tile_east = _tile_to_lat_lon(x_max + 1, y_max + 1, zoom)

    tile_lats = np.linspace(tile_south, tile_north, stitched_height)
    tile_lons = np.linspace(tile_west, tile_east, stitched_width)

    # Flip so that index 0 = south, last = north
    stitched_elevations = np.flipud(stitched_elevations)

    # Crop to requested bounding box
    lat_indices = np.where((tile_lats >= south) & (tile_lats <= north))[0]
    lon_indices = np.where((tile_lons >= west) & (tile_lons <= east))[0]

    if len(lat_indices) == 0 or len(lon_indices) == 0:
        raise ValueError("Bounding box doesn't overlap with fetched tiles")

    cropped = stitched_elevations[
        lat_indices[0]:lat_indices[-1] + 1,
        lon_indices[0]:lon_indices[-1] + 1,
    ]
    cropped_lats = tile_lats[lat_indices[0]:lat_indices[-1] + 1]
    cropped_lons = tile_lons[lon_indices[0]:lon_indices[-1] + 1]

    if cropped.shape[0] < 4 or cropped.shape[1] < 4:
        raise ValueError("Cropped region too small for interpolation")

    # Interpolate to requested resolution
    interp_func = interpolate.RectBivariateSpline(
        cropped_lats, cropped_lons, cropped, kx=3, ky=3,
    )
    target_lats = np.linspace(south, north, resolution)
    target_lons = np.linspace(west, east, resolution)
    elevations = interp_func(target_lats, target_lons)

    # Smooth to reduce tile boundary artifacts
    elevations = gaussian_filter(elevations, sigma=0.5)

    return {
        "grid": elevations,
        "lats": target_lats,
        "lons": target_lons,
        "min_elevation": float(np.min(elevations)),
        "max_elevation": float(np.max(elevations)),
    }
