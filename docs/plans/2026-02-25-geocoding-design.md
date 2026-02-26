# Geocoding Feature Design

**Date:** 2026-02-25
**Status:** Approved

## Overview

Add a `geocode_place` MCP tool that resolves a place name to geographic coordinates via the Nominatim API. This removes the need for the model to guess or ask the user for raw coordinates when only a place name is known.

## Problem

When a user asks to generate a shadow box for a named location (e.g., "Mount Hood" or "the Grand Canyon"), the model currently has no tool to look up coordinates — it must either ask the user or guess. This creates friction and errors.

The GPX file workflow is unaffected: when a user provides a GPX file, `set_area_from_gpx` extracts bounds directly from the track data and geocoding is not needed.

## Design

### New Tool: `geocode_place`

**Location:** `src/topo_shadow_box/tools/area.py` (alongside existing area tools)

**Signature:**
```python
def geocode_place(query: str, limit: int = 5) -> str
```

**Parameters:**
- `query`: Place name to search for (e.g., "Mount Hood, Oregon")
- `limit`: Max number of candidates to return (1–10, default 5)

**Behavior:**
1. Calls Nominatim search API with the query
2. Returns a numbered list of candidates for the model to present to the user
3. User picks one; model calls `set_area_from_coordinates` with the chosen lat/lon or bounding box
4. Does **not** modify session state

**Returns:** Formatted string with up to `limit` candidates, each showing:
- Display name (city, region, country for disambiguation)
- Latitude / longitude (center point)
- Bounding box (north/south/east/west) for natural extent
- Type (city, peak, park, etc.)

### New Pydantic Model: `GeocodeCandidate`

**Location:** `src/topo_shadow_box/models.py`

```python
class GeocodeCandidate(BaseModel):
    display_name: str
    lat: float
    lon: float
    type: str
    bbox_north: float
    bbox_south: float
    bbox_east: float
    bbox_west: float
```

### API: Nominatim

- **Endpoint:** `https://nominatim.openstreetmap.org/search`
- **Params:** `q={query}&format=json&limit={limit}&addressdetails=0`
- **Headers:** `User-Agent: topo-shadow-box/1.x` (required by Nominatim policy)
- **Rate limit:** 1 request/second (same pattern as Overpass API)
- **No API key required**

### Workflow Guidance (Docstrings)

- `geocode_place`: "Use when the user provides a place name and no coordinates or GPX file are available. If the user provides a GPX file, use `set_area_from_gpx` instead. After the user selects a candidate, call `set_area_from_coordinates`."
- `set_area_from_coordinates`: Updated `Next:` hint to reference `geocode_place` as the prior step when only a name is known.

## Error Handling

| Condition | Response |
|-----------|----------|
| No results | Clear message: "No locations found for '{query}'. Try a more specific name." |
| Network error | Propagate with descriptive message (same pattern as `fetch_elevation`) |
| `limit` out of range | Clamp to 1–10 silently |

## Testing

- Mock `httpx` responses (no live API calls in tests)
- Happy path: multiple candidates returned and formatted
- Single result: still presented as a list for consistency
- No results: returns descriptive error string
- Network error: raises/returns error message
- `limit` clamping: values outside 1–10 are clamped

## Non-Goals

- Reverse geocoding (coordinates → name)
- Caching geocode results across sessions
- Structured address input (city, state, country as separate fields)
