"""Pydantic domain models for geographic features and GPX data."""

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
