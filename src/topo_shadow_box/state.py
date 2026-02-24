"""Session state for the topo-shadow-box MCP server.

Holds all data for the current shadow box: area bounds, elevation grid,
OSM features, GPX tracks, model parameters, generated meshes, etc.
"""

import re
from typing import Optional, Literal
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class Bounds(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    north: float = Field(default=0.0, ge=-90, le=90)
    south: float = Field(default=0.0, ge=-90, le=90)
    east: float = Field(default=0.0, ge=-180, le=180)
    west: float = Field(default=0.0, ge=-180, le=180)
    is_set: bool = False

    @model_validator(mode="after")
    def check_north_gt_south(self) -> "Bounds":
        if self.is_set and self.north <= self.south:
            raise ValueError(f"north ({self.north}) must be greater than south ({self.south})")
        return self

    @model_validator(mode="after")
    def check_east_gt_west(self) -> "Bounds":
        if self.is_set and self.east <= self.west:
            raise ValueError(f"east ({self.east}) must be greater than west ({self.west})")
        return self

    @property
    def lat_range(self) -> float:
        return self.north - self.south

    @property
    def lon_range(self) -> float:
        return self.east - self.west

    @property
    def center_lat(self) -> float:
        return (self.north + self.south) / 2

    @property
    def center_lon(self) -> float:
        return (self.east + self.west) / 2


class ElevationData:
    """Placeholder - will be replaced in Task 11."""
    def __init__(self, grid=None, lats=None, lons=None, resolution=200,
                 min_elevation=0.0, max_elevation=0.0):
        self.grid = grid
        self.lats = lats
        self.lons = lons
        self.resolution = resolution
        self.min_elevation = min_elevation
        self.max_elevation = max_elevation

    @property
    def is_set(self) -> bool:
        return self.grid is not None


class ModelParams(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    width_mm: float = Field(default=200.0, gt=0)
    vertical_scale: float = Field(default=1.5, gt=0)
    base_height_mm: float = Field(default=10.0, gt=0)
    shape: Literal["square", "circle", "hexagon", "rectangle"] = "square"


class Colors:
    """Placeholder - will be replaced in Task 10."""
    def __init__(self):
        self.terrain = "#C8A882"
        self.water = "#4682B4"
        self.roads = "#D4C5A9"
        self.buildings = "#E8D5B7"
        self.gpx_track = "#FF0000"
        self.map_insert = "#FFFFFF"

    def hex_to_rgb(self, hex_color: str) -> tuple[int, int, int]:
        h = hex_color.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    def as_dict(self) -> dict[str, str]:
        return {
            "terrain": self.terrain,
            "water": self.water,
            "roads": self.roads,
            "buildings": self.buildings,
            "gpx_track": self.gpx_track,
            "map_insert": self.map_insert,
        }


class MeshData:
    """Placeholder - will be replaced in Task 11."""
    def __init__(self, vertices=None, faces=None, name="", feature_type=""):
        self.vertices = vertices or []
        self.faces = faces or []
        self.name = name
        self.feature_type = feature_type


class SessionState:
    """Placeholder - will be replaced in Task 12."""
    def __init__(self):
        self.bounds = Bounds()
        self.elevation = ElevationData()
        self.features = {}
        self.gpx_tracks = []
        self.gpx_waypoints = []
        self.model_params = ModelParams()
        self.colors = Colors()
        self.terrain_mesh = None
        self.feature_meshes = []
        self.gpx_mesh = None
        self.map_insert_mesh = None
        self.preview_port = 3333
        self.preview_running = False

    def summary(self) -> dict:
        return {
            "area": {
                "bounds_set": self.bounds.is_set,
                "north": self.bounds.north,
                "south": self.bounds.south,
                "east": self.bounds.east,
                "west": self.bounds.west,
            } if self.bounds.is_set else {"bounds_set": False},
            "data": {
                "elevation_loaded": self.elevation.is_set,
                "elevation_resolution": self.elevation.resolution if self.elevation.is_set else None,
                "elevation_range": f"{self.elevation.min_elevation:.0f}m - {self.elevation.max_elevation:.0f}m" if self.elevation.is_set else None,
                "features_loaded": bool(self.features),
                "feature_counts": {k: len(v) for k, v in self.features.items()} if self.features else {},
                "gpx_tracks": len(self.gpx_tracks),
            },
            "model": {
                "width_mm": self.model_params.width_mm,
                "vertical_scale": self.model_params.vertical_scale,
                "base_height_mm": self.model_params.base_height_mm,
                "shape": self.model_params.shape,
            },
            "colors": self.colors.as_dict(),
            "meshes": {
                "terrain_generated": self.terrain_mesh is not None,
                "feature_meshes": len(self.feature_meshes),
                "gpx_mesh_generated": self.gpx_mesh is not None,
            },
            "preview": {
                "running": self.preview_running,
                "port": self.preview_port,
            },
        }


# Global session state — one per MCP server process
state = SessionState()
