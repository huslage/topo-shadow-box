"""Export tools: export_3mf, export_openscad, export_svg."""

import logging
import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP

from ..state import state
from ..exporters.threemf import export_3mf as do_export_3mf
from ..exporters.openscad import export_openscad as do_export_openscad
from ..exporters.svg import export_svg as do_export_svg
from ._prereqs import require_state

logger = logging.getLogger(__name__)


def _validate_output_path(output_path: str) -> None:
    """Raise ValueError if output_path resolves outside the user's home directory."""
    resolved = Path(output_path).resolve()
    home = Path.home().resolve()
    try:
        resolved.relative_to(home)
    except ValueError:
        raise ValueError(
            f"Output path {output_path!r} is outside the home directory. "
            "Use a path within your home directory."
        )


def _collect_meshes() -> list[dict]:
    """Collect all generated meshes with their colors for export."""
    meshes = []
    colors = state.colors

    if state.terrain_mesh:
        meshes.append({
            "name": state.terrain_mesh.name,
            "type": state.terrain_mesh.feature_type,
            "vertices": state.terrain_mesh.vertices,
            "faces": state.terrain_mesh.faces,
            "color": colors.terrain,
        })

    # Merge features by type into single objects
    feature_groups: dict[str, dict] = {}
    for fm in state.feature_meshes:
        ftype = fm.feature_type
        if ftype not in feature_groups:
            color = getattr(colors, ftype, None)
            if color is None:
                logger.warning(
                    "No color defined for feature type %r — using default gray #808080", ftype
                )
                color = "#808080"
            feature_groups[ftype] = {
                "name": ftype.capitalize(),
                "type": ftype,
                "vertices": [],
                "faces": [],
                "color": color,
            }
        group = feature_groups[ftype]
        base_vi = len(group["vertices"])
        group["vertices"].extend(fm.vertices)
        group["faces"].extend([[f + base_vi for f in face] for face in fm.faces])

    for group in feature_groups.values():
        if group["vertices"]:
            meshes.append(group)

    if state.gpx_mesh:
        meshes.append({
            "name": state.gpx_mesh.name,
            "type": state.gpx_mesh.feature_type,
            "vertices": state.gpx_mesh.vertices,
            "faces": state.gpx_mesh.faces,
            "color": colors.gpx_track,
        })

    if state.map_insert_mesh:
        meshes.append({
            "name": state.map_insert_mesh.name,
            "type": state.map_insert_mesh.feature_type,
            "vertices": state.map_insert_mesh.vertices,
            "faces": state.map_insert_mesh.faces,
            "color": colors.map_insert,
        })

    return meshes


def register_export_tools(mcp: FastMCP):

    @mcp.tool()
    def export_3mf(output_path: str) -> str:
        """Export the model as a multi-material 3MF file.

        Each feature type (terrain, roads, water, buildings, GPX track)
        is a separate object with its own material color.

        Args:
            output_path: Where to save the .3mf file (absolute path)
        """
        try:
            require_state(state, mesh=True)
        except ValueError as e:
            return f"Error: {e}"

        meshes = _collect_meshes()
        if not meshes:
            return "Error: No mesh data to export."

        try:
            _validate_output_path(output_path)
        except ValueError as e:
            return f"Error: {e}"

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        result = do_export_3mf(meshes, output_path, base_height_mm=state.model_params.base_height_mm)
        return f"3MF exported to {output_path} ({result['objects']} objects)"

    @mcp.tool()
    def export_openscad(output_path: str) -> str:
        """Export the model as a parametric OpenSCAD file.

        The .scad file includes editable parameters at the top and polyhedron()
        calls for each mesh. Open in OpenSCAD for parametric customization.

        Args:
            output_path: Where to save the .scad file (absolute path)
        """
        try:
            require_state(state, mesh=True)
        except ValueError as e:
            return f"Error: {e}"

        meshes = _collect_meshes()
        if not meshes:
            return "Error: No mesh data to export."

        try:
            _validate_output_path(output_path)
        except ValueError as e:
            return f"Error: {e}"

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        do_export_openscad(
            meshes=meshes,
            output_path=output_path,
            model_params=state.model_params,
        )
        return f"OpenSCAD exported to {output_path}"

    @mcp.tool()
    def export_svg(output_path: str) -> str:
        """Export the map insert as an SVG file for paper printing.

        The SVG shows streets, water, parks, and GPX tracks styled for printing
        as a background insert behind the 3D terrain.

        Args:
            output_path: Where to save the .svg file (absolute path)
        """
        try:
            require_state(state, bounds=True)
        except ValueError as e:
            return f"Error: {e}"

        try:
            _validate_output_path(output_path)
        except ValueError as e:
            return f"Error: {e}"

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        do_export_svg(
            bounds=state.bounds,
            features=state.features,
            gpx_tracks=state.gpx_tracks,
            colors=state.colors,
            output_path=output_path,
            model_width_mm=state.model_params.width_mm,
        )
        return f"SVG map exported to {output_path}"
