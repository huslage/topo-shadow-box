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
        """Configure the shadow box model dimensions.

        Args:
            width_mm: Model width in millimeters (default 200)
            vertical_scale: Vertical exaggeration factor (default 1.5)
            base_height_mm: Base platform height in mm (default 10)
            shape: Model shape: square, circle, rectangle, hexagon (default square)
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
        """Set colors for multi-material export. All colors are hex strings (e.g. '#FF0000').

        Args:
            terrain: Terrain color (default #228B22 ForestGreen)
            water: Water color (default #4682B4 SteelBlue)
            roads: Road color (default #696969 DimGray)
            buildings: Building color (default #A9A9A9 DarkGray)
            gpx_track: GPX track color (default #FF0000 Red)
            map_insert: Map insert plate color (default #FFFFFF White)
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
