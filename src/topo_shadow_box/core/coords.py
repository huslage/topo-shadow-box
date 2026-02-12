"""Geographic to model coordinate transforms."""

import math
import numpy as np

from ..state import Bounds


class GeoToModelTransform:
    """Transforms geographic coordinates (lat/lon) to model coordinates (mm).

    Model coordinate system:
    - X: east-west (longitude)
    - Y: up (elevation)
    - Z: north-south (latitude), with north at Z=0
    """

    def __init__(self, bounds: Bounds, model_width_mm: float = 200.0):
        self.bounds = bounds

        avg_lat = bounds.center_lat
        self.lon_scale = math.cos(math.radians(avg_lat))

        lat_range = bounds.lat_range
        lon_range = bounds.lon_range * self.lon_scale

        # Scale factor: map the larger dimension to model_width_mm
        max_span = max(lat_range, lon_range)
        self.scale_factor = model_width_mm / max_span if max_span > 0 else 1.0

        # Model dimensions in mm
        self.model_width_x = lon_range * self.scale_factor
        self.model_width_z = lat_range * self.scale_factor

    def geo_to_model(self, lat: float, lon: float) -> tuple[float, float]:
        """Convert lat/lon to model X, Z coordinates (mm)."""
        x = (lon - self.bounds.west) * self.lon_scale * self.scale_factor
        z = (self.bounds.north - lat) * self.scale_factor
        return x, z

    def geo_to_model_array(
        self, lats: np.ndarray, lons: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert arrays of lat/lon to model X, Z."""
        x = (lons - self.bounds.west) * self.lon_scale * self.scale_factor
        z = (self.bounds.north - lats) * self.scale_factor
        return x, z

    def elevation_to_y(
        self, elevation: float, min_elev: float, elev_range: float,
        vertical_scale: float, size_scale: float = 1.0,
    ) -> float:
        """Convert elevation in meters to Y coordinate in mm."""
        if elev_range <= 0:
            return 0.0
        clamped = max(min_elev, min(min_elev + elev_range, elevation))
        return ((clamped - min_elev) / elev_range) * 20.0 * size_scale * vertical_scale


def add_padding_to_bounds(bounds: Bounds, padding_m: float) -> Bounds:
    """Add padding in meters around a bounding box."""
    # 1 degree latitude ~ 111,000 meters
    lat_padding = padding_m / 111_000.0
    # 1 degree longitude varies with latitude
    avg_lat = bounds.center_lat
    lon_padding = padding_m / (111_000.0 * math.cos(math.radians(avg_lat)))

    return Bounds(
        north=bounds.north + lat_padding,
        south=bounds.south - lat_padding,
        east=bounds.east + lon_padding,
        west=bounds.west - lon_padding,
    )
