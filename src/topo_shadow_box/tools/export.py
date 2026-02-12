"""Export tools: export_3mf, export_openscad, export_svg."""

import os
from mcp.server.fastmcp import FastMCP

from ..state import state
from ..exporters.threemf import export_3mf as do_export_3mf
from ..exporters.openscad import export_openscad as do_export_openscad
from ..exporters.svg import export_svg as do_export_svg


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

    for fm in state.feature_meshes:
        color = getattr(colors, fm.feature_type, "#808080")
        meshes.append({
            "name": fm.name,
            "type": fm.feature_type,
            "vertices": fm.vertices,
            "faces": fm.faces,
            "color": color,
        })

    if state.gpx_mesh:
        meshes.append({
            "name": state.gpx_mesh.name,
            "type": state.gpx_mesh.feature_type,
            "vertices": state.gpx_mesh.vertices,
            "faces": state.gpx_mesh.faces,
            "color": colors.gpx_track,
        })

    if state.frame_mesh:
        meshes.append({
            "name": state.frame_mesh.name,
            "type": state.frame_mesh.feature_type,
            "vertices": state.frame_mesh.vertices,
            "faces": state.frame_mesh.faces,
            "color": colors.frame,
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

        Each feature type (terrain, roads, water, buildings, GPX track, frame)
        is a separate object with its own material color.

        Args:
            output_path: Where to save the .3mf file (absolute path)
        """
        if not state.terrain_mesh:
            return "Error: Generate a model first."

        meshes = _collect_meshes()
        if not meshes:
            return "Error: No mesh data to export."

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        result = do_export_3mf(meshes, output_path)
        return f"3MF exported to {output_path} ({result['objects']} objects)"

    @mcp.tool()
    def export_openscad(output_path: str) -> str:
        """Export the model as a parametric OpenSCAD file.

        The .scad file includes editable parameters at the top and polyhedron()
        calls for each mesh. Open in OpenSCAD for parametric customization.

        Args:
            output_path: Where to save the .scad file (absolute path)
        """
        if not state.terrain_mesh:
            return "Error: Generate a model first."

        meshes = _collect_meshes()
        if not meshes:
            return "Error: No mesh data to export."

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        do_export_openscad(
            meshes=meshes,
            output_path=output_path,
            model_params=state.model_params,
            frame_params=state.frame_params,
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
        if not state.bounds.is_set:
            return "Error: Set an area first."

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
