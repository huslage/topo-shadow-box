"""OpenStreetMap feature fetching via Overpass API."""

import asyncio
import logging

import httpx

from .models import OsmFeatureSet
from topo_shadow_box.models import RoadFeature, WaterFeature, BuildingFeature, Coordinate

logger = logging.getLogger(__name__)

OVERPASS_SERVERS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]


async def _query_overpass(query: str) -> list[dict]:
    """Execute an Overpass API query with server fallback."""
    async with httpx.AsyncClient(timeout=45.0, headers={"User-Agent": "topo-shadow-box/1.0"}) as client:
        for server in OVERPASS_SERVERS:
            try:
                response = await client.post(server, data={"data": query})
                response.raise_for_status()
                data = response.json()
                return data.get("elements", [])
            except httpx.TimeoutException as exc:
                logger.warning("Overpass server %s timed out: %s", server, exc)
                continue
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "Overpass server %s returned HTTP %s", server, exc.response.status_code
                )
                continue
            except Exception as exc:
                logger.warning("Overpass server %s failed: %s", server, exc)
                continue
    logger.warning("All Overpass servers failed for query")
    return []


def _parse_way_coords(element: dict) -> list[dict]:
    """Extract coordinate list from a way or relation element."""
    coords = []
    elem_type = element.get("type")

    if elem_type == "way":
        for pt in element.get("geometry", []):
            coords.append({"lat": pt["lat"], "lon": pt["lon"]})
    elif elem_type == "relation":
        for member in element.get("members", []):
            if member.get("role") == "outer" and member.get("type") == "way":
                for pt in member.get("geometry", []):
                    coords.append({"lat": pt["lat"], "lon": pt["lon"]})
                break  # Use first outer ring
    return coords


def _parse_features(elements: list[dict], feature_type: str) -> list:
    """Parse Overpass elements into typed feature models."""
    features = []
    for elem in elements:
        raw_coords = _parse_way_coords(elem)
        if len(raw_coords) < 2:
            continue
        coords = [Coordinate(lat=c["lat"], lon=c["lon"]) for c in raw_coords]
        tags = elem.get("tags", {})
        name = tags.get("name", f"{feature_type}_{elem.get('id')}")

        if feature_type == "road":
            if len(coords) < 2:
                continue
            features.append(RoadFeature(
                id=elem.get("id", 0),
                coordinates=coords,
                tags=tags,
                name=name,
                road_type=tags.get("highway", "road"),
            ))
        elif feature_type == "water":
            if len(coords) < 3:
                continue
            features.append(WaterFeature(
                id=elem.get("id", 0),
                coordinates=coords,
                tags=tags,
                name=name,
            ))
        elif feature_type == "building":
            if len(coords) < 3:
                continue
            height = 10.0
            if "height" in tags:
                try:
                    height = float(tags["height"].replace("m", "").strip())
                except ValueError:
                    pass
            elif "building:levels" in tags:
                try:
                    height = float(tags["building:levels"]) * 3.0
                except ValueError:
                    pass
            height = max(height, 0.1)  # ensure gt=0 for BuildingFeature
            features.append(BuildingFeature(
                id=elem.get("id", 0),
                coordinates=coords,
                tags=tags,
                name=name,
                height=height,
            ))
    return features


async def fetch_osm_features(
    north: float, south: float, east: float, west: float,
    feature_types: list[str],
) -> OsmFeatureSet:
    """Fetch OSM features for a bounding box.

    Args:
        feature_types: List of types: 'roads', 'water', 'buildings'

    Returns:
        OsmFeatureSet with typed feature lists.
    """
    bbox = f"{south},{west},{north},{east}"

    road_elements = []
    water_elements = []
    building_elements = []

    queries = {}
    if "roads" in feature_types:
        queries["roads"] = (
            f'[out:json][timeout:30];way["highway"]({bbox});out body geom;',
            "road",
        )
    if "water" in feature_types:
        queries["water"] = (
            f'[out:json][timeout:30];(way["natural"="water"]({bbox});'
            f'way["waterway"]({bbox});'
            f'relation["natural"="water"]({bbox}););out body geom;',
            "water",
        )
    if "buildings" in feature_types:
        queries["buildings"] = (
            f'[out:json][timeout:30];way["building"]({bbox});out body geom;',
            "building",
        )

    queries_list = list(queries.items())
    for i, (feat_name, (query, parse_type)) in enumerate(queries_list):
        if i > 0:
            await asyncio.sleep(1.0)  # OSM rate limit: 1 req/sec
        try:
            elements = await _query_overpass(query)
        except Exception:
            elements = []
        if feat_name == "roads":
            road_elements = elements
        elif feat_name == "water":
            water_elements = elements
        elif feat_name == "buildings":
            building_elements = elements

    roads = _parse_features(road_elements, "road")[:200]
    water = _parse_features(water_elements, "water")[:50]
    buildings = _parse_features(building_elements, "building")[:150]
    return OsmFeatureSet(roads=roads, water=water, buildings=buildings)
