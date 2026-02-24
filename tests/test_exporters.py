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
