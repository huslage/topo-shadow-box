"""Tests for 3MF, OpenSCAD, and SVG exporters."""

import os
import zipfile
import pytest


def _minimal_mesh(name="terrain", mtype="terrain", color="#C8A882"):
    """Return a minimal valid mesh dict (single tetrahedron)."""
    return {
        "name": name,
        "type": mtype,
        "color": color,
        "vertices": [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "faces": [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ],
    }


class TestExport3MF:
    def test_output_is_valid_zip(self, tmp_path):
        from topo_shadow_box.exporters.threemf import export_3mf
        out = str(tmp_path / "test.3mf")
        export_3mf([_minimal_mesh()], out)
        assert zipfile.is_zipfile(out)

    def test_zip_contains_required_files(self, tmp_path):
        from topo_shadow_box.exporters.threemf import export_3mf
        out = str(tmp_path / "test.3mf")
        export_3mf([_minimal_mesh()], out)
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
        assert "[Content_Types].xml" in names
        assert "_rels/.rels" in names
        assert "3D/3dmodel.model" in names

    def test_model_xml_contains_mesh_data(self, tmp_path):
        from topo_shadow_box.exporters.threemf import export_3mf
        out = str(tmp_path / "test.3mf")
        export_3mf([_minimal_mesh()], out)
        with zipfile.ZipFile(out) as zf:
            xml = zf.read("3D/3dmodel.model").decode()
        assert "<vertices>" in xml
        assert "<triangles>" in xml
        assert "<vertex" in xml
        assert "<triangle" in xml

    def test_multi_material_has_one_object_per_mesh(self, tmp_path):
        from topo_shadow_box.exporters.threemf import export_3mf
        meshes = [
            _minimal_mesh("terrain", "terrain", "#C8A882"),
            _minimal_mesh("roads", "roads", "#D4C5A9"),
        ]
        out = str(tmp_path / "multi.3mf")
        result = export_3mf(meshes, out)
        assert result["objects"] == 2
        with zipfile.ZipFile(out) as zf:
            xml = zf.read("3D/3dmodel.model").decode()
        assert xml.count('<object ') == 2

    def test_entity_escaping_ampersand_lt_quot(self, tmp_path):
        from topo_shadow_box.exporters.threemf import export_3mf
        mesh = _minimal_mesh(name='Rock & Roll <>"', color="#FF0000")
        out = str(tmp_path / "escape.3mf")
        export_3mf([mesh], out)
        with zipfile.ZipFile(out) as zf:
            xml = zf.read("3D/3dmodel.model").decode()
        assert "&amp;" in xml
        assert "&lt;" in xml
        assert "&quot;" in xml
        assert ' name="Rock & Roll' not in xml


    def test_gt_entity_escaped_in_name(self, tmp_path):
        """'>' in mesh name should be escaped as '&gt;' in 3MF XML."""
        from topo_shadow_box.exporters.threemf import export_3mf
        mesh = _minimal_mesh(name="Height > 100m", color="#FF0000")
        out = str(tmp_path / "gt_escape.3mf")
        export_3mf([mesh], out)
        with zipfile.ZipFile(out) as zf:
            xml = zf.read("3D/3dmodel.model").decode()
        assert "&gt;" in xml, "Should escape '>' as '&gt;'"
        assert ' name="Height > 100m"' not in xml, "Should not have unescaped '>' in attribute"

    def test_raises_on_empty_mesh_list(self, tmp_path):
        from topo_shadow_box.exporters.threemf import export_3mf
        with pytest.raises(ValueError, match="No mesh data"):
            export_3mf([], str(tmp_path / "empty.3mf"))

    def test_returns_correct_filepath(self, tmp_path):
        from topo_shadow_box.exporters.threemf import export_3mf
        out = str(tmp_path / "check.3mf")
        result = export_3mf([_minimal_mesh()], out)
        assert result["filepath"] == out
        assert os.path.exists(out)


# ── OpenSCAD tests ────────────────────────────────────────────────────────────

class TestExportOpenSCAD:
    def _model_params(self):
        from topo_shadow_box.state import ModelParams
        return ModelParams(width_mm=200.0, vertical_scale=1.5, base_height_mm=10.0, shape="square")

    def test_output_file_is_created(self, tmp_path):
        from topo_shadow_box.exporters.openscad import export_openscad
        out = str(tmp_path / "test.scad")
        export_openscad([_minimal_mesh()], out, self._model_params())
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_contains_polyhedron_calls(self, tmp_path):
        from topo_shadow_box.exporters.openscad import export_openscad
        out = str(tmp_path / "test.scad")
        export_openscad([_minimal_mesh()], out, self._model_params())
        content = open(out).read()
        assert "polyhedron(" in content

    def test_contains_parameter_block(self, tmp_path):
        from topo_shadow_box.exporters.openscad import export_openscad
        out = str(tmp_path / "test.scad")
        export_openscad([_minimal_mesh()], out, self._model_params())
        content = open(out).read()
        assert "model_width" in content
        assert "vertical_scale" in content

    def test_contains_color_call(self, tmp_path):
        from topo_shadow_box.exporters.openscad import export_openscad
        out = str(tmp_path / "test.scad")
        export_openscad([_minimal_mesh()], out, self._model_params())
        content = open(out).read()
        assert "color(" in content

    def test_multiple_meshes_produce_multiple_polyhedrons(self, tmp_path):
        from topo_shadow_box.exporters.openscad import export_openscad
        meshes = [_minimal_mesh("a"), _minimal_mesh("b")]
        out = str(tmp_path / "multi.scad")
        export_openscad(meshes, out, self._model_params())
        content = open(out).read()
        assert content.count("polyhedron(") == 2


# ── SVG tests ─────────────────────────────────────────────────────────────────

class TestExportSVG:
    def _bounds(self):
        from topo_shadow_box.state import Bounds
        return Bounds(north=37.8, south=37.75, east=-122.4, west=-122.45, is_set=True)

    def _colors(self):
        from topo_shadow_box.state import Colors
        return Colors()

    def test_output_file_is_created(self, tmp_path):
        from topo_shadow_box.exporters.svg import export_svg
        from topo_shadow_box.core.models import OsmFeatureSet
        out = str(tmp_path / "test.svg")
        export_svg(
            bounds=self._bounds(),
            features=OsmFeatureSet(),
            gpx_tracks=[],
            colors=self._colors(),
            output_path=out,
        )
        assert os.path.exists(out)

    def test_output_is_valid_xml(self, tmp_path):
        from topo_shadow_box.exporters.svg import export_svg
        from topo_shadow_box.core.models import OsmFeatureSet
        import xml.etree.ElementTree as ET
        out = str(tmp_path / "test.svg")
        export_svg(
            bounds=self._bounds(),
            features=OsmFeatureSet(),
            gpx_tracks=[],
            colors=self._colors(),
            output_path=out,
        )
        tree = ET.parse(out)
        root = tree.getroot()
        assert "svg" in root.tag.lower()

    def test_output_is_nonempty(self, tmp_path):
        from topo_shadow_box.exporters.svg import export_svg
        from topo_shadow_box.core.models import OsmFeatureSet
        out = str(tmp_path / "test.svg")
        export_svg(
            bounds=self._bounds(),
            features=OsmFeatureSet(),
            gpx_tracks=[],
            colors=self._colors(),
            output_path=out,
        )
        assert os.path.getsize(out) > 0


def test_validate_output_path_rejects_outside_home():
    """_validate_output_path should raise ValueError for paths outside home directory."""
    from topo_shadow_box.tools.export import _validate_output_path
    import pytest

    # /etc/ is almost certainly outside home directory
    with pytest.raises(ValueError, match="outside"):
        _validate_output_path("/etc/topo_test.3mf")


def test_validate_output_path_accepts_home_subdirectory(tmp_path):
    """_validate_output_path should accept paths inside the home directory."""
    from topo_shadow_box.tools.export import _validate_output_path
    from pathlib import Path

    # tmp_path is inside the system temp dir; use a home subdir instead
    home_subpath = str(Path.home() / "topo_test_output.3mf")
    # Should not raise
    _validate_output_path(home_subpath)


def test_all_mesh_feature_types_have_colors():
    """Every mesh feature type used by export must have a color in Colors."""
    from topo_shadow_box.state import Colors

    colors = Colors()
    colors_dict = colors.as_dict()
    # Known feature types produced by mesh generation and used in _collect_meshes
    known_types = {"terrain", "roads", "water", "buildings", "gpx_track", "map_insert"}
    for ftype in known_types:
        assert ftype in colors_dict, (
            f"Feature type '{ftype}' has no color in Colors.as_dict(). "
            "Add it or remove from known_types if no longer used."
        )


def test_unknown_feature_type_logs_warning(caplog):
    """_collect_meshes should log a warning when a mesh has an unknown feature type."""
    import logging
    from topo_shadow_box.state import state, MeshData

    # Save and restore state
    original_feature_meshes = state.feature_meshes
    original_terrain = state.terrain_mesh

    state.terrain_mesh = None
    state.feature_meshes = [
        MeshData(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            faces=[[0, 1, 2]],
            name="Unknown Feature",
            feature_type="unknown_type_xyz",
        )
    ]

    from topo_shadow_box.tools.export import _collect_meshes
    with caplog.at_level(logging.WARNING, logger="topo_shadow_box.tools.export"):
        _collect_meshes()

    # Restore state
    state.feature_meshes = original_feature_meshes
    state.terrain_mesh = original_terrain

    assert any(
        r.name == "topo_shadow_box.tools.export"
        and r.levelno == logging.WARNING
        and "unknown_type_xyz" in r.message
        for r in caplog.records
    ), f"Expected WARNING about unknown_type_xyz. Got: {[(r.name, r.levelno, r.message) for r in caplog.records]}"
