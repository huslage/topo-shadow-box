"""Prerequisite checking helpers for MCP tools."""


def require_state(state, *, bounds: bool = False, elevation: bool = False, mesh: bool = False) -> None:
    """Raise ValueError with a descriptive message if required state is not set.

    Usage in a tool:
        try:
            require_state(state, bounds=True, elevation=True)
        except ValueError as e:
            return f"Error: {e}"
    """
    if bounds and not state.bounds.is_set:
        raise ValueError(
            "Set an area first with set_area_from_coordinates or set_area_from_gpx."
        )
    if elevation and not state.elevation.is_set:
        raise ValueError(
            "Fetch elevation data first with fetch_elevation."
        )
    if mesh and not state.terrain_mesh:
        raise ValueError(
            "Generate a model first with generate_model."
        )
