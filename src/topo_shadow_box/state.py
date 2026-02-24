"""Session state for the topo-shadow-box MCP server.

Holds all data for the current shadow box: area bounds, elevation grid,
OSM features, GPX tracks, model parameters, generated meshes, etc.
"""

import re
from typing import Optional, Literal
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from topo_shadow_box.core.models import OsmFeatureSet
from topo_shadow_box.models import GpxTrack, GpxWaypoint


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


class ElevationData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    grid: Optional[np.ndarray] = None
    lats: Optional[np.ndarray] = None
    lons: Optional[np.ndarray] = None
    resolution: int = Field(default=200, gt=0, le=1000)
    min_elevation: float = 0.0
    max_elevation: float = 0.0
    is_set: bool = False


class ModelParams(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    width_mm: float = Field(default=200.0, gt=0)
    vertical_scale: float = Field(default=1.5, gt=0)
    base_height_mm: float = Field(default=10.0, gt=0)
    shape: Literal["square", "circle", "hexagon", "rectangle"] = "square"


class Colors(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    terrain: str = "#C8A882"
    water: str = "#4682B4"
    roads: str = "#D4C5A9"
    buildings: str = "#E8D5B7"
    gpx_track: str = "#FF0000"
    map_insert: str = "#FFFFFF"

    @field_validator("terrain", "water", "roads", "buildings", "gpx_track", "map_insert", mode="before")
    @classmethod
    def validate_and_normalize_hex(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Color must be a string")
        v = v.strip()
        if not re.match(r'^#[0-9A-Fa-f]{6}$', v):
            raise ValueError(f"Invalid hex color '{v}'. Must be #RRGGBB format.")
        return f"#{v[1:].upper()}"

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


class MeshData(BaseModel):
    vertices: list[list[float]] = Field(default_factory=list)
    faces: list[list[int]] = Field(default_factory=list)
    name: str = ""
    feature_type: str = ""


class SessionState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    bounds: Bounds = Field(default_factory=Bounds)
    elevation: ElevationData = Field(default_factory=ElevationData)
    features: OsmFeatureSet = Field(default_factory=OsmFeatureSet)
    gpx_tracks: list[GpxTrack] = []
    gpx_waypoints: list[GpxWaypoint] = []
    model_params: ModelParams = Field(default_factory=ModelParams)
    colors: Colors = Field(default_factory=Colors)
    terrain_mesh: Optional[MeshData] = None
    feature_meshes: list[MeshData] = []
    gpx_mesh: Optional[MeshData] = None
    map_insert_mesh: Optional[MeshData] = None
    preview_port: int = Field(default=3333, gt=0, le=65535)
    preview_running: bool = False

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
                "elevation_range": (
                    f"{self.elevation.min_elevation:.0f}m - {self.elevation.max_elevation:.0f}m"
                    if self.elevation.is_set else None
                ),
                "features_loaded": bool(
                    self.features.roads or self.features.water or self.features.buildings
                ),
                "feature_counts": {
                    "roads": len(self.features.roads),
                    "water": len(self.features.water),
                    "buildings": len(self.features.buildings),
                },
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
