"""Pydantic return models for core computation functions."""

from typing import Optional
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from topo_shadow_box.models import RoadFeature, WaterFeature, BuildingFeature


class MeshResult(BaseModel):
    """Return type for all mesh generation functions."""
    vertices: list[list[float]]
    faces: list[list[int]]
    name: str = ""
    feature_type: str = ""

    @field_validator("vertices")
    @classmethod
    def vertices_must_be_3d(cls, v: list[list[float]]) -> list[list[float]]:
        for i, vertex in enumerate(v):
            if len(vertex) != 3:
                raise ValueError(f"Vertex {i} must have exactly 3 components, got {len(vertex)}")
        return v

    @field_validator("faces")
    @classmethod
    def faces_must_be_triangles_with_non_negative_indices(
        cls, v: list[list[int]]
    ) -> list[list[int]]:
        for i, face in enumerate(v):
            if len(face) != 3:
                raise ValueError(f"Face {i} must have exactly 3 indices, got {len(face)}")
            for idx in face:
                if idx < 0:
                    raise ValueError(f"Face {i} has negative index {idx}")
        return v

    @model_validator(mode="after")
    def face_indices_must_be_valid(self) -> "MeshResult":
        n_verts = len(self.vertices)
        if n_verts == 0:
            return self
        for i, face in enumerate(self.faces):
            for idx in face:
                if idx >= n_verts:
                    raise ValueError(
                        f"Face {i} references vertex {idx} but only {n_verts} vertices exist"
                    )
        return self


class ElevationResult(BaseModel):
    """Return type for fetch_terrain_elevation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    grid: np.ndarray
    lats: np.ndarray
    lons: np.ndarray
    resolution: int = Field(gt=0, le=1000)
    min_elevation: float
    max_elevation: float


class OsmFeatureSet(BaseModel):
    """Return type for fetch_osm_features."""
    roads: list[RoadFeature] = []
    water: list[WaterFeature] = []
    buildings: list[BuildingFeature] = []
