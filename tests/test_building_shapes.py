"""Tests for building shape generation."""

import pytest

from topo_shadow_box.core.building_shapes import BuildingShapeGenerator


class TestDetermineBuildingShape:
    def setup_method(self):
        self.gen = BuildingShapeGenerator()

    def test_church_returns_steeple(self):
        assert self.gen.determine_building_shape({"building": "church"}) == "steeple"

    def test_cathedral_returns_steeple(self):
        assert self.gen.determine_building_shape({"building": "cathedral"}) == "steeple"

    def test_chapel_returns_steeple(self):
        assert self.gen.determine_building_shape({"building": "chapel"}) == "steeple"

    def test_place_of_worship_christian_returns_steeple(self):
        tags = {"amenity": "place_of_worship", "religion": "christian", "building": "yes"}
        assert self.gen.determine_building_shape(tags) == "steeple"

    def test_place_of_worship_non_christian_falls_through(self):
        tags = {"amenity": "place_of_worship", "religion": "muslim", "building": "yes"}
        # Not christian, so falls through to building tag "yes" -> default -> flat_roof
        assert self.gen.determine_building_shape(tags) == "flat_roof"

    def test_house_returns_pitched_roof(self):
        assert self.gen.determine_building_shape({"building": "house"}) == "pitched_roof"

    def test_residential_returns_pitched_roof(self):
        assert self.gen.determine_building_shape({"building": "residential"}) == "pitched_roof"

    def test_detached_returns_pitched_roof(self):
        assert self.gen.determine_building_shape({"building": "detached"}) == "pitched_roof"

    def test_semidetached_returns_pitched_roof(self):
        assert self.gen.determine_building_shape({"building": "semidetached_house"}) == "pitched_roof"

    def test_warehouse_returns_gabled_roof(self):
        assert self.gen.determine_building_shape({"building": "warehouse"}) == "gabled_roof"

    def test_barn_returns_gabled_roof(self):
        assert self.gen.determine_building_shape({"building": "barn"}) == "gabled_roof"

    def test_commercial_returns_flat_roof(self):
        assert self.gen.determine_building_shape({"building": "commercial"}) == "flat_roof"

    def test_office_returns_flat_roof(self):
        assert self.gen.determine_building_shape({"building": "office"}) == "flat_roof"

    def test_retail_returns_flat_roof(self):
        assert self.gen.determine_building_shape({"building": "retail"}) == "flat_roof"

    def test_industrial_returns_flat_roof(self):
        assert self.gen.determine_building_shape({"building": "industrial"}) == "flat_roof"

    def test_apartments_returns_flat_roof(self):
        assert self.gen.determine_building_shape({"building": "apartments"}) == "flat_roof"

    def test_shop_tag_returns_flat_roof(self):
        tags = {"building": "yes", "shop": "bakery"}
        assert self.gen.determine_building_shape(tags) == "flat_roof"

    def test_default_unknown_type_returns_flat_roof(self):
        assert self.gen.determine_building_shape({"building": "yes"}) == "flat_roof"

    def test_empty_tags_returns_flat_roof(self):
        assert self.gen.determine_building_shape({}) == "flat_roof"

    def test_amenity_takes_precedence_over_building_tag(self):
        # amenity=place_of_worship + religion=christian should win over building=commercial
        tags = {"amenity": "place_of_worship", "religion": "christian", "building": "commercial"}
        assert self.gen.determine_building_shape(tags) == "steeple"


class TestFlatRoofMesh:
    def setup_method(self):
        self.gen = BuildingShapeGenerator()
        self.result = self.gen.generate_building_mesh(
            x1=0.0, x2=1.0, y_base=0.0, y_top=2.0, z1=0.0, z2=1.0,
            shape_type="flat_roof"
        )

    def test_returns_dict_with_required_keys(self):
        assert "vertices" in self.result
        assert "faces" in self.result
        assert "custom_color" in self.result

    def test_vertex_count(self):
        assert len(self.result["vertices"]) == 8

    def test_face_count(self):
        assert len(self.result["faces"]) == 12

    def test_custom_color_default_none(self):
        assert self.result["custom_color"] is None

    def test_vertices_contain_bounds(self):
        verts = self.result["vertices"]
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        zs = [v[2] for v in verts]
        assert min(xs) == 0.0
        assert max(xs) == 1.0
        assert min(ys) == 0.0
        assert max(ys) == 2.0
        assert min(zs) == 0.0
        assert max(zs) == 1.0

    def test_face_indices_valid(self):
        n_verts = len(self.result["vertices"])
        for face in self.result["faces"]:
            assert len(face) == 3
            for idx in face:
                assert 0 <= idx < n_verts


class TestPitchedRoofMesh:
    def setup_method(self):
        self.gen = BuildingShapeGenerator()
        self.result = self.gen.generate_building_mesh(
            x1=0.0, x2=2.0, y_base=0.0, y_top=3.0, z1=0.0, z2=2.0,
            shape_type="pitched_roof"
        )

    def test_vertex_count(self):
        # 8 box + 2 ridge = 10
        assert len(self.result["vertices"]) == 10

    def test_has_more_faces_than_flat(self):
        # Pitched roof has more geometry: 16 faces (8 walls/bottom + 4 roof + 2 gable + 2 bottom)
        assert len(self.result["faces"]) > 12

    def test_ridge_height_is_70_30_split(self):
        verts = self.result["vertices"]
        # Wall top should be at y_base + 0.7 * (y_top - y_base) = 2.1
        # Ridge should be at y_top = 3.0
        ys = sorted(set(v[1] for v in verts))
        assert pytest.approx(ys[0]) == 0.0    # base
        assert pytest.approx(ys[1]) == 2.1    # wall top
        assert pytest.approx(ys[2]) == 3.0    # ridge peak

    def test_ridge_is_centered_in_z(self):
        verts = self.result["vertices"]
        # Ridge vertices (8, 9) should have z = (z1+z2)/2 = 1.0
        ridge_verts = [v for v in verts if v[1] == pytest.approx(3.0)]
        for v in ridge_verts:
            assert v[2] == pytest.approx(1.0)

    def test_face_indices_valid(self):
        n_verts = len(self.result["vertices"])
        for face in self.result["faces"]:
            assert len(face) == 3
            for idx in face:
                assert 0 <= idx < n_verts


class TestGabledRoofMesh:
    def setup_method(self):
        self.gen = BuildingShapeGenerator()
        self.result = self.gen.generate_building_mesh(
            x1=0.0, x2=4.0, y_base=0.0, y_top=5.0, z1=0.0, z2=3.0,
            shape_type="gabled_roof"
        )

    def test_vertex_count(self):
        # 8 box + 2 ridge = 10
        assert len(self.result["vertices"]) == 10

    def test_has_more_faces_than_flat(self):
        assert len(self.result["faces"]) > 12

    def test_wall_roof_split_80_20(self):
        verts = self.result["vertices"]
        ys = sorted(set(v[1] for v in verts))
        # y_base=0, y_walls=0 + 0.8*5=4.0, y_peak=4.0 + 0.2*5=5.0
        assert pytest.approx(ys[0]) == 0.0
        assert pytest.approx(ys[1]) == 4.0
        assert pytest.approx(ys[2]) == 5.0

    def test_face_indices_valid(self):
        n_verts = len(self.result["vertices"])
        for face in self.result["faces"]:
            assert len(face) == 3
            for idx in face:
                assert 0 <= idx < n_verts


class TestSteepleMesh:
    def setup_method(self):
        self.gen = BuildingShapeGenerator()
        self.result = self.gen.generate_building_mesh(
            x1=0.0, x2=10.0, y_base=0.0, y_top=10.0, z1=0.0, z2=8.0,
            shape_type="steeple"
        )

    def test_vertex_count(self):
        # 8 body + 4 tower base + 4 tower top + 1 spire = 17
        assert len(self.result["vertices"]) == 17

    def test_has_more_faces_than_flat(self):
        assert len(self.result["faces"]) > 12

    def test_body_tower_spire_heights(self):
        verts = self.result["vertices"]
        ys = sorted(set(v[1] for v in verts))
        # y_base=0, y_body=6.0, y_tower=8.5, y_spire=10.0
        assert pytest.approx(ys[0]) == 0.0
        assert pytest.approx(ys[1]) == 6.0
        assert pytest.approx(ys[2]) == 8.5
        assert pytest.approx(ys[3]) == 10.0

    def test_spire_peak_is_centered(self):
        verts = self.result["vertices"]
        # Spire peak (vertex 16) should be at x_center, y_spire, tz_center
        spire = verts[16]
        x_center = (0.0 + 10.0) / 2
        assert spire[0] == pytest.approx(x_center)
        assert spire[1] == pytest.approx(10.0)  # y_spire

    def test_face_indices_valid(self):
        n_verts = len(self.result["vertices"])
        for face in self.result["faces"]:
            assert len(face) == 3
            for idx in face:
                assert 0 <= idx < n_verts


class TestCustomColor:
    def setup_method(self):
        self.gen = BuildingShapeGenerator()

    def test_custom_color_passthrough(self):
        result = self.gen.generate_building_mesh(
            x1=0.0, x2=1.0, y_base=0.0, y_top=1.0, z1=0.0, z2=1.0,
            shape_type="flat_roof", custom_color="#FF0000"
        )
        assert result["custom_color"] == "#FF0000"

    def test_custom_color_none_by_default(self):
        result = self.gen.generate_building_mesh(
            x1=0.0, x2=1.0, y_base=0.0, y_top=1.0, z1=0.0, z2=1.0,
            shape_type="flat_roof"
        )
        assert result["custom_color"] is None

    def test_custom_color_works_for_all_shapes(self):
        for shape in ["flat_roof", "pitched_roof", "gabled_roof", "steeple"]:
            result = self.gen.generate_building_mesh(
                x1=0.0, x2=1.0, y_base=0.0, y_top=1.0, z1=0.0, z2=1.0,
                shape_type=shape, custom_color="#00FF00"
            )
            assert result["custom_color"] == "#00FF00"


class TestDefaultShape:
    def setup_method(self):
        self.gen = BuildingShapeGenerator()

    def test_unknown_shape_type_defaults_to_flat_roof(self):
        result = self.gen.generate_building_mesh(
            x1=0.0, x2=1.0, y_base=0.0, y_top=1.0, z1=0.0, z2=1.0,
            shape_type="unknown_shape"
        )
        # Should fall through to flat_roof (else branch)
        assert len(result["vertices"]) == 8
        assert len(result["faces"]) == 12
