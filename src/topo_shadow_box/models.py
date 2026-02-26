"""Pydantic domain models for geographic features and GPX data."""

from typing import Literal

from pydantic import BaseModel, Field


class Coordinate(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class GpxPoint(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    elevation: float


class GpxWaypoint(BaseModel):
    name: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    elevation: float
    description: str = ""


class GpxTrack(BaseModel):
    name: str
    points: list[GpxPoint] = Field(min_length=2)


class RoadFeature(BaseModel):
    id: int
    type: Literal["road"] = "road"
    coordinates: list[Coordinate] = Field(min_length=2)
    tags: dict = Field(default_factory=dict)
    name: str = ""
    road_type: str = ""


class WaterFeature(BaseModel):
    id: int
    type: Literal["water"] = "water"
    coordinates: list[Coordinate] = Field(min_length=3)
    tags: dict = Field(default_factory=dict)
    name: str = ""


class BuildingFeature(BaseModel):
    id: int
    type: Literal["building"] = "building"
    coordinates: list[Coordinate] = Field(min_length=3)
    tags: dict = Field(default_factory=dict)
    name: str = ""
    height: float = Field(default=10.0, gt=0)


class GeocodeCandidate(BaseModel):
    display_name: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    place_type: str
    bbox_north: float = Field(ge=-90, le=90)
    bbox_south: float = Field(ge=-90, le=90)
    bbox_east: float = Field(ge=-180, le=180)
    bbox_west: float = Field(ge=-180, le=180)
