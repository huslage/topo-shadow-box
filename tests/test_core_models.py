"""Tests for core return models."""
import pytest
from pydantic import ValidationError


class TestMeshResult:
    def test_valid_mesh(self):
        from topo_shadow_box.core.models import MeshResult
        m = MeshResult(
            vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            faces=[[0, 1, 2]],
        )
        assert len(m.vertices) == 3
        assert len(m.faces) == 1

    def test_vertex_must_have_3_components(self):
        from topo_shadow_box.core.models import MeshResult
        with pytest.raises(ValidationError):
            MeshResult(
                vertices=[[0.0, 0.0]],  # only 2 components
                faces=[[0, 0, 0]],
            )

    def test_face_must_have_3_indices(self):
        from topo_shadow_box.core.models import MeshResult
        with pytest.raises(ValidationError):
            MeshResult(
                vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                faces=[[0, 1]],  # only 2 indices
            )

    def test_face_index_must_be_non_negative(self):
        from topo_shadow_box.core.models import MeshResult
        with pytest.raises(ValidationError):
            MeshResult(
                vertices=[[0.0, 0.0, 0.0]],
                faces=[[-1, 0, 0]],
            )

    def test_face_index_must_be_valid_vertex_index(self):
        from topo_shadow_box.core.models import MeshResult
        with pytest.raises(ValidationError):
            MeshResult(
                vertices=[[0.0, 0.0, 0.0]],  # only 1 vertex (index 0)
                faces=[[0, 1, 2]],  # indices 1 and 2 out of range
            )

    def test_name_and_feature_type_default_empty(self):
        from topo_shadow_box.core.models import MeshResult
        m = MeshResult(vertices=[], faces=[])
        assert m.name == ""
        assert m.feature_type == ""

    def test_empty_mesh_is_valid(self):
        from topo_shadow_box.core.models import MeshResult
        m = MeshResult(vertices=[], faces=[])
        assert m.vertices == []
        assert m.faces == []


class TestElevationResult:
    def test_valid_elevation_result(self):
        import numpy as np
        from topo_shadow_box.core.models import ElevationResult
        e = ElevationResult(
            grid=np.zeros((10, 10)),
            lats=np.linspace(47.0, 47.1, 10),
            lons=np.linspace(-122.0, -121.9, 10),
            resolution=10,
            min_elevation=100.0,
            max_elevation=500.0,
        )
        assert e.resolution == 10

    def test_resolution_must_be_positive(self):
        import numpy as np
        from pydantic import ValidationError
        from topo_shadow_box.core.models import ElevationResult
        with pytest.raises(ValidationError):
            ElevationResult(
                grid=np.zeros((10, 10)),
                lats=np.zeros(10),
                lons=np.zeros(10),
                resolution=0,
                min_elevation=0.0,
                max_elevation=0.0,
            )

    def test_resolution_must_not_exceed_1000(self):
        import numpy as np
        from pydantic import ValidationError
        from topo_shadow_box.core.models import ElevationResult
        with pytest.raises(ValidationError):
            ElevationResult(
                grid=np.zeros((10, 10)),
                lats=np.zeros(10),
                lons=np.zeros(10),
                resolution=1001,
                min_elevation=0.0,
                max_elevation=0.0,
            )
