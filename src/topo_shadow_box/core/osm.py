"""OpenStreetMap feature fetching via Overpass API."""

import httpx

OVERPASS_SERVERS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]


async def _query_overpass(query: str) -> list[dict]:
    """Execute an Overpass API query with server fallback."""
    async with httpx.AsyncClient(timeout=45.0) as client:
        for server in OVERPASS_SERVERS:
            try:
                response = await client.post(server, data={"data": query})
                response.raise_for_status()
                data = response.json()
                return data.get("elements", [])
            except Exception:
                continue
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


def _parse_features(elements: list[dict], feature_type: str) -> list[dict]:
    """Parse Overpass elements into simplified feature dicts."""
    features = []
    for elem in elements:
        coords = _parse_way_coords(elem)
        if len(coords) < 2:
            continue

        tags = elem.get("tags", {})
        feature = {
            "id": elem.get("id"),
            "type": feature_type,
            "coordinates": coords,
            "tags": tags,
            "name": tags.get("name", f"{feature_type}_{elem.get('id')}"),
        }

        if feature_type == "road":
            feature["road_type"] = tags.get("highway", "road")
        elif feature_type == "building":
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
            feature["height"] = height

        features.append(feature)
    return features


async def fetch_osm_features(
    north: float, south: float, east: float, west: float,
    feature_types: list[str],
) -> dict:
    """Fetch OSM features for a bounding box.

    Args:
        feature_types: List of types: 'roads', 'water', 'buildings'

    Returns:
        Dict mapping type names to lists of feature dicts.
    """
    bbox = f"{south},{west},{north},{east}"
    results = {}

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

    for feat_name, (query, parse_type) in queries.items():
        try:
            elements = await _query_overpass(query)
            results[feat_name] = _parse_features(elements, parse_type)
        except Exception:
            results[feat_name] = []

    return results
