"""Model configuration tools: set_model_params, set_colors."""

from mcp.server.fastmcp import FastMCP

from ..state import state


def register_model_tools(mcp: FastMCP):

    @mcp.tool()
    def set_model_params(
        width_mm: float | None = None,
        vertical_scale: float | None = None,
        base_height_mm: float | None = None,
        shape: str | None = None,
    ) -> str:
        """Set model geometry parameters.

        Can be called any time before generate_model.
        **Next:** generate_model (re-run after changing params to update meshes).

        Args:
            width_mm: Model width in mm (default 200). The larger geographic
                dimension maps to this value.
            vertical_scale: Elevation exaggeration multiplier (default 1.5).
                Use 2-3 for flat terrain, 1 for mountains.
            base_height_mm: Thickness of the solid base (default 10mm).
            shape: Model outline shape — 'square', 'circle', 'hexagon', or 'rectangle'.
        """
        p = state.model_params
        if width_mm is not None:
            p.width_mm = width_mm
        if vertical_scale is not None:
            p.vertical_scale = vertical_scale
        if base_height_mm is not None:
            p.base_height_mm = base_height_mm
        if shape is not None:
            try:
                p.shape = shape
            except Exception as e:
                return f"Error: {e}"

        # Clear meshes since params changed
        state.terrain_mesh = None
        state.feature_meshes = []

        return (
            f"Model params: {p.width_mm}mm wide, vertical_scale={p.vertical_scale}, "
            f"base={p.base_height_mm}mm, shape={p.shape}"
        )

    @mcp.tool()
    def set_colors(
        terrain: str | None = None,
        water: str | None = None,
        roads: str | None = None,
        buildings: str | None = None,
        gpx_track: str | None = None,
        map_insert: str | None = None,
    ) -> str:
        """Set material colors for each feature type (hex #RRGGBB).

        Can be called any time before export.
        **Next:** generate_model or export (colors are applied at export time).

        Args:
            terrain/water/roads/buildings/gpx_track/map_insert: Hex color strings.
        """
        c = state.colors
        for name, value in [
            ("terrain", terrain), ("water", water), ("roads", roads),
            ("buildings", buildings), ("gpx_track", gpx_track),
            ("map_insert", map_insert),
        ]:
            if value is not None:
                try:
                    setattr(c, name, value)
                except Exception as e:
                    return f"Error: {e}"

        return f"Colors: {c.as_dict()}"
