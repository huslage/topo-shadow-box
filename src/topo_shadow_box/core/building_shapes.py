"""Building shape generation for different building types.

Maps OSM building tags to 3D shape types and generates appropriate
meshes with correct face winding (CCW from outside) for 3D printing.
"""


class BuildingShapeGenerator:
    """Generate different 3D shapes for buildings based on their OSM tags.

    Supports various architectural styles:
    - Churches/cathedrals: steeple (body + tower + spire)
    - Houses/residential: pitched roof (gabled, 70% walls / 30% roof)
    - Warehouses/barns: gabled roof (80% walls / 20% roof)
    - Commercial/office/retail: flat roof (simple box)
    """

    def __init__(self):
        """Initialize building shape generator with tag-to-shape mapping."""
        self.shape_mapping = {
            "church": "steeple",
            "cathedral": "steeple",
            "chapel": "steeple",
            "house": "pitched_roof",
            "residential": "pitched_roof",
            "detached": "pitched_roof",
            "semidetached_house": "pitched_roof",
            "commercial": "flat_roof",
            "office": "flat_roof",
            "retail": "flat_roof",
            "warehouse": "gabled_roof",
            "barn": "gabled_roof",
            "industrial": "flat_roof",
            "apartments": "flat_roof",
            "default": "flat_roof",
        }

    def determine_building_shape(self, building_tags):
        """Determine which 3D shape to use for a building based on its OSM tags.

        Args:
            building_tags: Dict of OSM tags (building, amenity, shop, etc.)

        Returns:
            str: Shape type ('steeple', 'pitched_roof', 'gabled_roof', 'flat_roof')
        """
        # Check amenity first (place_of_worship + christian -> steeple)
        amenity = building_tags.get("amenity", "")
        if amenity == "place_of_worship":
            religion = building_tags.get("religion", "")
            if religion == "christian":
                return "steeple"

        # Check building tag
        building_type = building_tags.get("building", "yes")
        if building_type in self.shape_mapping:
            return self.shape_mapping[building_type]

        # Check for shop (usually commercial)
        if "shop" in building_tags:
            return "flat_roof"

        # Default
        return self.shape_mapping["default"]

    def generate_building_mesh(
        self, x1, x2, y_base, y_top, z1, z2, shape_type="flat_roof", custom_color=None
    ):
        """Generate a building mesh with the specified shape.

        Args:
            x1, x2: X bounds of building
            y_base: Base elevation
            y_top: Top elevation (height of main structure)
            z1, z2: Z bounds of building
            shape_type: Type of shape ('flat_roof', 'pitched_roof', 'gabled_roof', 'steeple')
            custom_color: Optional custom color hex string

        Returns:
            dict with keys 'vertices' (list of [x,y,z]), 'faces' (list of [i,j,k]),
            and 'custom_color' (str or None)
        """
        if shape_type == "steeple":
            vertices, faces = self._generate_steeple(x1, x2, y_base, y_top, z1, z2)
        elif shape_type == "pitched_roof":
            vertices, faces = self._generate_pitched_roof(x1, x2, y_base, y_top, z1, z2)
        elif shape_type == "gabled_roof":
            vertices, faces = self._generate_gabled_roof(x1, x2, y_base, y_top, z1, z2)
        else:
            vertices, faces = self._generate_flat_roof(x1, x2, y_base, y_top, z1, z2)

        return {
            "vertices": vertices,
            "faces": faces,
            "custom_color": custom_color,
        }

    def _generate_flat_roof(self, x1, x2, y_base, y_top, z1, z2):
        """Generate a simple box with flat roof (8 vertices, 12 faces)."""
        vertices = [
            # Bottom face (y = y_base)
            [x1, y_base, z1],  # 0
            [x2, y_base, z1],  # 1
            [x2, y_base, z2],  # 2
            [x1, y_base, z2],  # 3
            # Top face (y = y_top)
            [x1, y_top, z1],  # 4
            [x2, y_top, z1],  # 5
            [x2, y_top, z2],  # 6
            [x1, y_top, z2],  # 7
        ]

        faces = [
            # Bottom (CCW from below)
            [0, 1, 2],
            [0, 2, 3],
            # Top (CCW from above)
            [4, 7, 6],
            [4, 6, 5],
            # Front (z=z1)
            [0, 4, 5],
            [0, 5, 1],
            # Right (x=x2)
            [1, 5, 6],
            [1, 6, 2],
            # Back (z=z2)
            [2, 6, 7],
            [2, 7, 3],
            # Left (x=x1)
            [3, 7, 4],
            [3, 4, 0],
        ]

        return vertices, faces

    def _generate_pitched_roof(self, x1, x2, y_base, y_top, z1, z2):
        """Generate a building with pitched roof (10 vertices, 16 faces).

        Wall height is 70% of total, roof peak adds 30%.
        Ridge runs along X direction, centered in Z.
        """
        wall_height = (y_top - y_base) * 0.7
        y_walls = y_base + wall_height
        roof_peak_height = (y_top - y_base) * 0.3
        y_peak = y_walls + roof_peak_height

        z_center = (z1 + z2) / 2

        vertices = [
            # Bottom face
            [x1, y_base, z1],  # 0
            [x2, y_base, z1],  # 1
            [x2, y_base, z2],  # 2
            [x1, y_base, z2],  # 3
            # Wall top
            [x1, y_walls, z1],  # 4
            [x2, y_walls, z1],  # 5
            [x2, y_walls, z2],  # 6
            [x1, y_walls, z2],  # 7
            # Roof ridge (peak)
            [x1, y_peak, z_center],  # 8
            [x2, y_peak, z_center],  # 9
        ]

        faces = [
            # Bottom
            [0, 1, 2],
            [0, 2, 3],
            # Walls (4 sides)
            [0, 4, 5],
            [0, 5, 1],  # Front
            [1, 5, 6],
            [1, 6, 2],  # Right
            [2, 6, 7],
            [2, 7, 3],  # Back
            [3, 7, 4],
            [3, 4, 0],  # Left
            # Roof - front slope
            [4, 8, 9],
            [4, 9, 5],
            # Roof - back slope
            [7, 6, 9],
            [7, 9, 8],
            # Gable ends
            [4, 7, 8],  # Left gable
            [5, 9, 6],  # Right gable
        ]

        return vertices, faces

    def _generate_gabled_roof(self, x1, x2, y_base, y_top, z1, z2):
        """Generate a warehouse/barn with gabled roof (10 vertices, 16 faces).

        Wall height is 80% of total, roof peak adds 20%.
        Ridge runs along X direction, centered in Z.
        """
        wall_height = (y_top - y_base) * 0.8
        y_walls = y_base + wall_height
        roof_peak_height = (y_top - y_base) * 0.2
        y_peak = y_walls + roof_peak_height

        z_center = (z1 + z2) / 2

        vertices = [
            # Bottom face
            [x1, y_base, z1],  # 0
            [x2, y_base, z1],  # 1
            [x2, y_base, z2],  # 2
            [x1, y_base, z2],  # 3
            # Wall top
            [x1, y_walls, z1],  # 4
            [x2, y_walls, z1],  # 5
            [x2, y_walls, z2],  # 6
            [x1, y_walls, z2],  # 7
            # Roof ridge
            [x1, y_peak, z_center],  # 8
            [x2, y_peak, z_center],  # 9
        ]

        faces = [
            # Bottom
            [0, 1, 2],
            [0, 2, 3],
            # Walls
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
            # Roof
            [4, 8, 9],
            [4, 9, 5],
            [7, 6, 9],
            [7, 9, 8],
            # Gables
            [4, 7, 8],
            [5, 9, 6],
        ]

        return vertices, faces

    def _generate_steeple(self, x1, x2, y_base, y_top, z1, z2):
        """Generate a church with steeple (17 vertices, 30 faces).

        Body is 60% of total height, tower is 25%, spire is 15%.
        Tower sits at front-center with 30% width and 20% depth of the body.
        """
        body_height = (y_top - y_base) * 0.6
        y_body = y_base + body_height

        tower_height = (y_top - y_base) * 0.25
        y_tower = y_body + tower_height

        spire_height = (y_top - y_base) * 0.15
        y_spire = y_tower + spire_height

        # Tower footprint: 30% width, 20% depth, at front-center
        x_center = (x1 + x2) / 2
        z_front = z1
        tower_width = (x2 - x1) * 0.3
        tower_depth = (z2 - z1) * 0.2

        tx1 = x_center - tower_width / 2
        tx2 = x_center + tower_width / 2
        tz1 = z_front
        tz2 = z_front + tower_depth

        vertices = [
            # Main body - bottom
            [x1, y_base, z1],  # 0
            [x2, y_base, z1],  # 1
            [x2, y_base, z2],  # 2
            [x1, y_base, z2],  # 3
            # Main body - top
            [x1, y_body, z1],  # 4
            [x2, y_body, z1],  # 5
            [x2, y_body, z2],  # 6
            [x1, y_body, z2],  # 7
            # Tower - base (at y_body)
            [tx1, y_body, tz1],  # 8
            [tx2, y_body, tz1],  # 9
            [tx2, y_body, tz2],  # 10
            [tx1, y_body, tz2],  # 11
            # Tower - top (at y_tower)
            [tx1, y_tower, tz1],  # 12
            [tx2, y_tower, tz1],  # 13
            [tx2, y_tower, tz2],  # 14
            [tx1, y_tower, tz2],  # 15
            # Spire peak
            [x_center, y_spire, (tz1 + tz2) / 2],  # 16
        ]

        faces = [
            # Main body - bottom
            [0, 1, 2],
            [0, 2, 3],
            # Main body - top (with hole for tower)
            [4, 8, 9],
            [4, 9, 5],  # Front section
            [7, 6, 10],
            [7, 10, 11],  # Back section
            [4, 7, 11],
            [4, 11, 8],  # Left section
            [5, 9, 10],
            [5, 10, 6],  # Right section
            # Main body - walls
            [0, 4, 5],
            [0, 5, 1],  # Front
            [1, 5, 6],
            [1, 6, 2],  # Right
            [2, 6, 7],
            [2, 7, 3],  # Back
            [3, 7, 4],
            [3, 4, 0],  # Left
            # Tower - walls
            [8, 12, 13],
            [8, 13, 9],  # Front
            [9, 13, 14],
            [9, 14, 10],  # Right
            [10, 14, 15],
            [10, 15, 11],  # Back
            [11, 15, 12],
            [11, 12, 8],  # Left
            # Spire - 4 triangular faces from tower top to peak
            [12, 16, 13],  # Front
            [13, 16, 14],  # Right
            [14, 16, 15],  # Back
            [15, 16, 12],  # Left
        ]

        return vertices, faces
