"""Generation tools: generate_model, generate_map_insert."""

from mcp.server.fastmcp import FastMCP

from ..state import state, MeshData
from mcp.server.fastmcp import Context
from ..core.mesh import (
    generate_terrain_mesh, generate_gpx_track_mesh, generate_single_feature_mesh,
    _elevation_normalization,
)
from ..core.map_insert import generate_map_insert_svg, generate_map_insert_plate
from ..core.coords import GeoToModelTransform
from ._prereqs import require_state


def register_generate_tools(mcp: FastMCP):

    @mcp.tool()
    async def generate_model(ctx: Context) -> str:
        """Generate the full 3D model from current state.

        **Requires:** set_area_from_coordinates/gpx + fetch_elevation.
        Features and GPX tracks are optional but must be fetched before calling this.
        **Next:** preview (optional), then export_3mf / export_openscad / export_svg.

        Re-run this after changing model params or colors to update the meshes.
        Reports fine-grained progress as each feature is processed.
        """
        try:
            require_state(state, bounds=True, elevation=True)
        except ValueError as e:
            return f"Error: {e}"

        b = state.bounds
        mp = state.model_params

        transform = GeoToModelTransform(bounds=b, model_width_mm=mp.width_mm)
        norm = _elevation_normalization(state.elevation.grid)

        # Count total work units upfront
        features = state.features
        roads = features.roads[:200] if features else []
        waters = features.water[:50] if features else []
        buildings = features.buildings[:150] if features else []
        has_gpx = bool(state.gpx_tracks)
        total = 1 + len(roads) + len(waters) + len(buildings) + (1 if has_gpx else 0)
        current = 0

        # Generate terrain
        terrain = generate_terrain_mesh(
            elevation=state.elevation, bounds=b, transform=transform,
            vertical_scale=mp.vertical_scale, base_height_mm=mp.base_height_mm,
            shape=mp.shape, _norm=norm,
        )
        state.terrain_mesh = MeshData(
            vertices=terrain.vertices, faces=terrain.faces,
            name=terrain.name, feature_type=terrain.feature_type,
        )
        current += 1
        await ctx.report_progress(current, total)

        # Generate feature meshes one-by-one for progress
        state.feature_meshes = []
        for road in roads:
            fm = generate_single_feature_mesh(
                road, "road", state.elevation, b, transform,
                mp.vertical_scale, mp.shape, norm,
            )
            if fm:
                state.feature_meshes.append(MeshData(
                    vertices=fm.vertices, faces=fm.faces,
                    name=fm.name, feature_type=fm.feature_type,
                ))
            current += 1
            await ctx.report_progress(current, total)

        for water in waters:
            fm = generate_single_feature_mesh(
                water, "water", state.elevation, b, transform,
                mp.vertical_scale, mp.shape, norm,
            )
            if fm:
                state.feature_meshes.append(MeshData(
                    vertices=fm.vertices, faces=fm.faces,
                    name=fm.name, feature_type=fm.feature_type,
                ))
            current += 1
            await ctx.report_progress(current, total)

        for building in buildings:
            fm = generate_single_feature_mesh(
                building, "building", state.elevation, b, transform,
                mp.vertical_scale, mp.shape, norm,
            )
            if fm:
                state.feature_meshes.append(MeshData(
                    vertices=fm.vertices, faces=fm.faces,
                    name=fm.name, feature_type=fm.feature_type,
                ))
            current += 1
            await ctx.report_progress(current, total)

        # Generate GPX track mesh
        state.gpx_mesh = None
        if has_gpx:
            gpx = generate_gpx_track_mesh(
                tracks=state.gpx_tracks, elevation=state.elevation,
                bounds=b, transform=transform,
                vertical_scale=mp.vertical_scale, shape=mp.shape, _norm=norm,
            )
            if gpx:
                state.gpx_mesh = MeshData(
                    vertices=gpx.vertices, faces=gpx.faces,
                    name=gpx.name, feature_type=gpx.feature_type,
                )
            current += 1
            await ctx.report_progress(current, total)

        terrain_verts = len(state.terrain_mesh.vertices)
        terrain_faces = len(state.terrain_mesh.faces)
        feature_count = len(state.feature_meshes)
        total_verts = terrain_verts + sum(len(m.vertices) for m in state.feature_meshes)
        if state.gpx_mesh:
            total_verts += len(state.gpx_mesh.vertices)

        return (
            f"Model generated: {terrain_verts} terrain vertices, {terrain_faces} faces, "
            f"{feature_count} feature meshes, "
            f"GPX: {'yes' if state.gpx_mesh else 'no'}. "
            f"Total vertices: {total_verts}"
        )

    @mcp.tool()
    def generate_map_insert(format: str = "both") -> str:
        """Generate a background map insert (SVG for paper printing and/or 3D plate).

        **Requires:** set_area_from_coordinates or set_area_from_gpx first.
        **Next:** export_svg (for paper) or export_3mf (includes the plate).

        Args:
            format: 'svg' for paper map only, 'plate' for 3D-printable flat plate,
                    'both' for both (default).
        """
        try:
            require_state(state, bounds=True)
        except ValueError as e:
            return f"Error: {e}"
        if format not in ("svg", "plate", "both"):
            return "Error: format must be 'svg', 'plate', or 'both'."

        results = []

        if format in ("svg", "both"):
            generate_map_insert_svg(
                bounds=state.bounds,
                features=state.features,
                gpx_tracks=state.gpx_tracks,
                colors=state.colors,
            )
            results.append("SVG map insert generated")

        if format in ("plate", "both"):
            mp = state.model_params
            transform = GeoToModelTransform(bounds=state.bounds, model_width_mm=mp.width_mm)
            plate = generate_map_insert_plate(
                bounds=state.bounds,
                features=state.features,
                gpx_tracks=state.gpx_tracks,
                transform=transform,
                plate_thickness_mm=1.0,
            )
            state.map_insert_mesh = MeshData(
                vertices=plate.vertices,
                faces=plate.faces,
                name=plate.name,
                feature_type=plate.feature_type,
            )
            results.append("3D plate generated")

        return f"Map insert: {', '.join(results)}"
