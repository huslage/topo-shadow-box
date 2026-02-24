"""Generation tools: generate_model, generate_map_insert."""

from mcp.server.fastmcp import FastMCP

from ..state import state, MeshData
from ..core.mesh import generate_terrain_mesh, generate_feature_meshes, generate_gpx_track_mesh, _elevation_normalization
from ..core.map_insert import generate_map_insert_svg, generate_map_insert_plate
from ..core.coords import GeoToModelTransform


def register_generate_tools(mcp: FastMCP):

    @mcp.tool()
    def generate_model() -> str:
        """Generate the full 3D model from current state.

        Requires: area set + elevation fetched. Features and GPX tracks are optional.
        Generates: terrain mesh, feature meshes, and GPX track mesh.
        """
        if not state.bounds.is_set:
            return "Error: Set an area first."
        if not state.elevation.is_set:
            return "Error: Fetch elevation data first."

        b = state.bounds
        mp = state.model_params

        transform = GeoToModelTransform(
            bounds=b,
            model_width_mm=mp.width_mm,
        )

        norm = _elevation_normalization(state.elevation.grid)

        # Generate terrain
        terrain = generate_terrain_mesh(
            elevation=state.elevation,
            bounds=b,
            transform=transform,
            vertical_scale=mp.vertical_scale,
            base_height_mm=mp.base_height_mm,
            shape=mp.shape,
        )
        state.terrain_mesh = MeshData(
            vertices=terrain.vertices,
            faces=terrain.faces,
            name=terrain.name,
            feature_type=terrain.feature_type,
        )

        # Generate feature meshes
        state.feature_meshes = []
        if state.features and (state.features.roads or state.features.water or state.features.buildings):
            fmeshes = generate_feature_meshes(
                features=state.features,
                elevation=state.elevation,
                bounds=b,
                transform=transform,
                vertical_scale=mp.vertical_scale,
                shape=mp.shape,
                _norm=norm,
            )
            for fm in fmeshes:
                state.feature_meshes.append(MeshData(
                    vertices=fm.vertices,
                    faces=fm.faces,
                    name=fm.name,
                    feature_type=fm.feature_type,
                ))

        # Generate GPX track mesh
        state.gpx_mesh = None
        if state.gpx_tracks:
            gpx = generate_gpx_track_mesh(
                tracks=state.gpx_tracks,
                elevation=state.elevation,
                bounds=b,
                transform=transform,
                vertical_scale=mp.vertical_scale,
                shape=mp.shape,
                _norm=norm,
            )
            if gpx:
                state.gpx_mesh = MeshData(
                    vertices=gpx.vertices,
                    faces=gpx.faces,
                    name=gpx.name,
                    feature_type=gpx.feature_type,
                )

        # Summary
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

        Args:
            format: 'svg' for paper map, 'plate' for 3D-printable flat plate, 'both' for both.
        """
        if not state.bounds.is_set:
            return "Error: Set an area first."
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
