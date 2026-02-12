"""Session state for the topo-shadow-box MCP server.

Holds all data for the current shadow box: area bounds, elevation grid,
OSM features, GPX tracks, model parameters, generated meshes, etc.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Bounds:
    north: float = 0.0
    south: float = 0.0
    east: float = 0.0
    west: float = 0.0

    @property
    def is_set(self) -> bool:
        return not (self.north == 0 and self.south == 0 and self.east == 0 and self.west == 0)

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


@dataclass
class ElevationData:
    grid: Optional[np.ndarray] = None
    lats: Optional[np.ndarray] = None
    lons: Optional[np.ndarray] = None
    resolution: int = 200
    min_elevation: float = 0.0
    max_elevation: float = 0.0

    @property
    def is_set(self) -> bool:
        return self.grid is not None


@dataclass
class ModelParams:
    width_mm: float = 200.0
    vertical_scale: float = 1.5
    base_height_mm: float = 10.0
    shape: str = "square"  # square, circle, rectangle, hexagon


@dataclass
class FrameParams:
    frame_width_mm: float = 10.0
    frame_depth_mm: float = 30.0
    wall_thickness_mm: float = 2.0


@dataclass
class Colors:
    terrain: str = "#228B22"       # ForestGreen
    water: str = "#4682B4"         # SteelBlue
    roads: str = "#696969"         # DimGray
    buildings: str = "#A9A9A9"     # DarkGray
    gpx_track: str = "#FF0000"     # Red
    frame: str = "#8B4513"         # SaddleBrown
    map_insert: str = "#FFFFFF"    # White

    def hex_to_rgb(self, hex_color: str) -> tuple[int, int, int]:
        hex_color = hex_color.lstrip("#")
        return (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "terrain": self.terrain,
            "water": self.water,
            "roads": self.roads,
            "buildings": self.buildings,
            "gpx_track": self.gpx_track,
            "frame": self.frame,
            "map_insert": self.map_insert,
        }


@dataclass
class MeshData:
    """Holds generated mesh vertices and faces."""
    vertices: list = field(default_factory=list)
    faces: list = field(default_factory=list)
    name: str = ""
    feature_type: str = ""


@dataclass
class SessionState:
    """Complete session state for a shadow box generation."""

    # Area of interest
    bounds: Bounds = field(default_factory=Bounds)

    # Data
    elevation: ElevationData = field(default_factory=ElevationData)
    features: dict = field(default_factory=dict)  # {roads: [...], water: [...], buildings: [...]}
    gpx_tracks: list = field(default_factory=list)  # [{name, points: [{lat, lon, elevation}]}]
    gpx_waypoints: list = field(default_factory=list)

    # Parameters
    model_params: ModelParams = field(default_factory=ModelParams)
    frame_params: FrameParams = field(default_factory=FrameParams)
    colors: Colors = field(default_factory=Colors)

    # Generated meshes
    terrain_mesh: Optional[MeshData] = None
    feature_meshes: list = field(default_factory=list)  # [MeshData, ...]
    gpx_mesh: Optional[MeshData] = None
    frame_mesh: Optional[MeshData] = None
    map_insert_mesh: Optional[MeshData] = None

    # Preview server state
    preview_port: int = 3333
    preview_running: bool = False

    def summary(self) -> dict:
        """Return a summary of the current state for get_status."""
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
            "frame": {
                "width_mm": self.frame_params.frame_width_mm,
                "depth_mm": self.frame_params.frame_depth_mm,
                "wall_thickness_mm": self.frame_params.wall_thickness_mm,
            },
            "colors": self.colors.as_dict(),
            "meshes": {
                "terrain_generated": self.terrain_mesh is not None,
                "feature_meshes": len(self.feature_meshes),
                "gpx_mesh_generated": self.gpx_mesh is not None,
                "frame_generated": self.frame_mesh is not None,
            },
            "preview": {
                "running": self.preview_running,
                "port": self.preview_port,
            },
        }


# Global session state — one per MCP server process
state = SessionState()
