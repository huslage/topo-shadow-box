# Pydantic Refactor Design

**Date:** 2026-02-24
**Scope:** Full refactor — core return types, domain models, state models
**Approach:** Bottom-up, layer by layer with red/green TDD

---

## Context

The codebase currently uses Python `@dataclass` for 6 state models (`Bounds`, `ElevationData`, `ModelParams`, `Colors`, `MeshData`, `SessionState`) and plain `dict` for all feature data (roads, water, buildings, GPX tracks/waypoints) and core function return values. There is no input validation, no cross-field constraints, and no structured typing on core return values.

---

## Goals

- Replace all `@dataclass` with Pydantic `BaseModel`
- Replace all plain `dict` return types from core functions with typed Pydantic models
- Replace all plain `dict` feature/GPX storage in state with typed Pydantic models
- Add thorough validation: lat/lon ranges, positive dimensions, hex color format, cross-field constraints, geometry invariants
- Maintain existing test coverage; add new tests for all validation behavior

---

## Architecture

Three layers built bottom-up, each independently testable before the next builds on it.

```
Layer 1 — Core return models      src/topo_shadow_box/core/models.py
Layer 2 — Domain models           src/topo_shadow_box/models.py
Layer 3 — State models            src/topo_shadow_box/state.py  (updated in-place)
```

New test files:
```
tests/test_core_models.py
tests/test_models.py
tests/test_state_models.py
```

Existing tests (`test_mesh.py`, `test_building_shapes.py`, `test_shape_clipper.py`) are untouched.

---

## Layer 1: Core Return Models (`core/models.py`)

```python
class ElevationResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    grid: np.ndarray
    lats: np.ndarray
    lons: np.ndarray
    resolution: int = Field(gt=0, le=1000)
    min_elevation: float
    max_elevation: float

class MeshResult(BaseModel):
    vertices: list[list[float]]   # each inner list: exactly 3 floats
    faces: list[list[int]]        # each inner list: exactly 3 non-negative ints
    name: str = ""
    feature_type: str = ""
    # cross-field validator: face indices must be valid vertex indices

class OsmFeatureSet(BaseModel):
    roads: list[RoadFeature] = []
    water: list[WaterFeature] = []
    buildings: list[BuildingFeature] = []
```

Note: `ElevationResult` uses `arbitrary_types_allowed=True` for numpy arrays. These fields are never serialized — they are internal computation state only.

Core function signatures change from returning `dict` to returning these models:
- `fetch_terrain_elevation(...) -> ElevationResult`
- `generate_terrain_mesh(...) -> MeshResult`
- `fetch_osm_features(...) -> OsmFeatureSet`
- All other mesh generators `-> MeshResult`

---

## Layer 2: Domain Models (`models.py`)

```python
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
    tags: dict = {}
    name: str = ""
    road_type: str = ""

class WaterFeature(BaseModel):
    id: int
    type: Literal["water"] = "water"
    coordinates: list[Coordinate] = Field(min_length=3)
    tags: dict = {}
    name: str = ""

class BuildingFeature(BaseModel):
    id: int
    type: Literal["building"] = "building"
    coordinates: list[Coordinate] = Field(min_length=3)
    tags: dict = {}
    name: str = ""
    height: float = Field(default=10.0, gt=0)
```

---

## Layer 3: State Models (`state.py`)

```python
class Bounds(BaseModel):
    north: float = Field(default=0.0, ge=-90, le=90)
    south: float = Field(default=0.0, ge=-90, le=90)
    east: float = Field(default=0.0, ge=-180, le=180)
    west: float = Field(default=0.0, ge=-180, le=180)
    is_set: bool = False
    # model_validator: north > south when is_set
    # model_validator: east > west when is_set

class ModelParams(BaseModel):
    width_mm: float = Field(default=200.0, gt=0)
    vertical_scale: float = Field(default=1.5, gt=0)
    base_height_mm: float = Field(default=3.0, gt=0)
    shape: Literal["square", "circle", "hexagon", "rectangle"] = "square"

class Colors(BaseModel):
    terrain: str = "#C8A882"
    water: str = "#4682B4"
    roads: str = "#D4C5A9"
    buildings: str = "#E8D5B7"
    gpx_track: str = "#FF0000"
    map_insert: str = "#FFFFFF"
    # field_validator("*"): must match r'^#[0-9A-Fa-f]{6}$'

class ElevationData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    grid: Optional[np.ndarray] = None
    lats: Optional[np.ndarray] = None
    lons: Optional[np.ndarray] = None
    resolution: int = Field(default=200, gt=0, le=1000)
    min_elevation: float = 0.0
    max_elevation: float = 0.0
    is_set: bool = False

class MeshData(BaseModel):
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    name: str = ""
    feature_type: str = ""

class SessionState(BaseModel):
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
```

Tool handlers are simplified: hex color validation and lat/lon range checks move out of tool code into model validators. Tools construct models; `ValidationError` surfaces automatically.

---

## Testing Strategy

Red/green TDD for every new model:
1. Write a failing test asserting a validator rejects invalid input
2. Write a failing test asserting valid input is accepted
3. Implement the model/validator to make both pass

Test files:
- `tests/test_core_models.py` — `ElevationResult`, `MeshResult`, `OsmFeatureSet`
- `tests/test_models.py` — all Layer 2 domain models
- `tests/test_state_models.py` — `Bounds`, `ModelParams`, `Colors`, `ElevationData`, `MeshData`, `SessionState`

---

## Dependencies

Add to `pyproject.toml`:
```
pydantic>=2.0
```
