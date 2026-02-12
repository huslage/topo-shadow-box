"""Shape clipping utilities for different model boundary types.

Each clipper defines a 2D boundary shape (circle, square, rectangle, hexagon)
and provides methods to test containment, clip linestrings/polygons, and
project points onto the boundary.
"""

import numpy as np
from abc import ABC, abstractmethod


class ShapeClipper(ABC):
    """Abstract base class for shape clipping operations.

    Subclasses implement containment tests and clipping for a specific
    boundary shape (circle, square, rectangle, hexagon).
    """

    def __init__(self, center_x: float, center_z: float, size: float):
        """Initialize shape clipper.

        Args:
            center_x: Center X coordinate in model space.
            center_z: Center Z coordinate in model space.
            size: Characteristic size (radius, half-width, etc.).
        """
        self.center_x = center_x
        self.center_z = center_z
        self.size = size

    @abstractmethod
    def is_inside(self, x, z):
        """Test if point(s) are inside the shape boundary.

        Args:
            x: X coordinate(s) - scalar or numpy array.
            z: Z coordinate(s) - scalar or numpy array.

        Returns:
            bool or numpy array of bools.
        """
        pass

    @abstractmethod
    def clip_linestring(self, points) -> list[np.ndarray]:
        """Clip a linestring to the shape boundary, preserving continuity.

        Args:
            points: Sequence of (x, z) pairs or numpy array of shape (N, 2).

        Returns:
            List of continuous path segments, each a numpy array of shape (M, 2).
        """
        pass

    @abstractmethod
    def clip_polygon(self, points) -> np.ndarray | None:
        """Clip a polygon to the shape boundary (all-or-nothing).

        Returns the polygon if all vertices are inside, else None.

        Args:
            points: Sequence of (x, z) pairs or numpy array of shape (N, 2).

        Returns:
            numpy array of vertices, or None if any vertex is outside.
        """
        pass

    @abstractmethod
    def project_to_boundary(self, x: float, z: float) -> tuple[float, float] | None:
        """Project a point to the nearest point on the shape boundary.

        Args:
            x: X coordinate.
            z: Z coordinate.

        Returns:
            (projected_x, projected_z) on the boundary, or None.
        """
        pass


class CircleClipper(ShapeClipper):
    """Circular boundary clipper with line-circle intersection."""

    def __init__(self, center_x: float, center_z: float, radius: float):
        super().__init__(center_x, center_z, radius)
        self.radius = radius

    def is_inside(self, x, z):
        dx = x - self.center_x
        dz = z - self.center_z
        return dx * dx + dz * dz <= self.radius * self.radius

    def _line_circle_intersection(self, p1, p2) -> list[tuple[float, float]]:
        """Find intersection points of a line segment with the circle.

        Returns a list of 0, 1, or 2 intersection points.
        """
        x1, z1 = p1
        x2, z2 = p2

        dx = x2 - x1
        dz = z2 - z1
        fx = x1 - self.center_x
        fz = z1 - self.center_z

        a = dx * dx + dz * dz
        b = 2 * (fx * dx + fz * dz)
        c = fx * fx + fz * fz - self.radius * self.radius

        if a < 1e-10:
            return []

        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return []

        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        intersections = []
        for t in [t1, t2]:
            if 0 <= t <= 1:
                intersections.append((x1 + t * dx, z1 + t * dz))

        return intersections

    def clip_linestring(self, points) -> list[np.ndarray]:
        if len(points) < 2:
            return []

        points = np.array(points)
        segments = []
        current_segment = []

        for i in range(len(points)):
            x, z = points[i]
            inside = self.is_inside(x, z)

            if inside:
                current_segment.append((x, z))

            if i < len(points) - 1:
                x_next, z_next = points[i + 1]
                inside_next = self.is_inside(x_next, z_next)

                if inside != inside_next:
                    intersections = self._line_circle_intersection(
                        (x, z), (x_next, z_next)
                    )
                    if intersections:
                        ix, iz = intersections[0]
                        if inside:
                            current_segment.append((ix, iz))
                            if len(current_segment) >= 2:
                                segments.append(np.array(current_segment))
                            current_segment = []
                        else:
                            current_segment = [(ix, iz)]

        if len(current_segment) >= 2:
            segments.append(np.array(current_segment))

        return segments

    def clip_polygon(self, points) -> np.ndarray | None:
        if len(points) == 0:
            return None

        points = np.array(points)
        inside_mask = self.is_inside(points[:, 0], points[:, 1])

        if np.all(inside_mask):
            return points
        return None

    def project_to_boundary(self, x: float, z: float) -> tuple[float, float]:
        dx = x - self.center_x
        dz = z - self.center_z
        dist = np.sqrt(dx * dx + dz * dz)

        if dist == 0:
            return (self.center_x + self.radius, self.center_z)

        scale = self.radius / dist
        return (self.center_x + dx * scale, self.center_z + dz * scale)


class SquareClipper(ShapeClipper):
    """Square boundary clipper."""

    def __init__(self, center_x: float, center_z: float, half_width: float):
        super().__init__(center_x, center_z, half_width)
        self.half_width = half_width

    def is_inside(self, x, z):
        return (np.abs(x - self.center_x) <= self.half_width) & \
               (np.abs(z - self.center_z) <= self.half_width)

    def _line_box_intersection(self, p1, p2) -> list[tuple[float, float]]:
        """Find intersection points of a line segment with the square boundary."""
        x1, z1 = p1
        x2, z2 = p2

        min_x = self.center_x - self.half_width
        max_x = self.center_x + self.half_width
        min_z = self.center_z - self.half_width
        max_z = self.center_z + self.half_width

        intersections = []

        # Left edge (x = min_x)
        if x1 != x2:
            t = (min_x - x1) / (x2 - x1)
            if 0 <= t <= 1:
                z = z1 + t * (z2 - z1)
                if min_z <= z <= max_z:
                    intersections.append((min_x, z))

        # Right edge (x = max_x)
        if x1 != x2:
            t = (max_x - x1) / (x2 - x1)
            if 0 <= t <= 1:
                z = z1 + t * (z2 - z1)
                if min_z <= z <= max_z:
                    intersections.append((max_x, z))

        # Bottom edge (z = min_z)
        if z1 != z2:
            t = (min_z - z1) / (z2 - z1)
            if 0 <= t <= 1:
                x = x1 + t * (x2 - x1)
                if min_x <= x <= max_x:
                    intersections.append((x, min_z))

        # Top edge (z = max_z)
        if z1 != z2:
            t = (max_z - z1) / (z2 - z1)
            if 0 <= t <= 1:
                x = x1 + t * (x2 - x1)
                if min_x <= x <= max_x:
                    intersections.append((x, max_z))

        # Remove duplicates (corner cases)
        unique = []
        for pt in intersections:
            if not any(np.allclose(pt, u, atol=1e-6) for u in unique):
                unique.append(pt)

        return unique

    def clip_linestring(self, points) -> list[np.ndarray]:
        if len(points) < 2:
            return []

        points = np.array(points)
        segments = []
        current_segment = []

        for i in range(len(points)):
            x, z = points[i]
            inside = self.is_inside(x, z)

            if inside:
                current_segment.append((x, z))

            if i < len(points) - 1:
                x_next, z_next = points[i + 1]
                inside_next = self.is_inside(x_next, z_next)

                if inside != inside_next:
                    intersections = self._line_box_intersection(
                        (x, z), (x_next, z_next)
                    )
                    if intersections:
                        ix, iz = intersections[0]
                        if inside:
                            current_segment.append((ix, iz))
                            if len(current_segment) >= 2:
                                segments.append(np.array(current_segment))
                            current_segment = []
                        else:
                            current_segment = [(ix, iz)]

        if len(current_segment) >= 2:
            segments.append(np.array(current_segment))

        return segments

    def clip_polygon(self, points) -> np.ndarray | None:
        if len(points) == 0:
            return None

        points = np.array(points)
        inside_mask = self.is_inside(points[:, 0], points[:, 1])

        if np.all(inside_mask):
            return points
        return None

    def project_to_boundary(self, x: float, z: float) -> tuple[float, float]:
        dx = x - self.center_x
        dz = z - self.center_z

        min_x = self.center_x - self.half_width
        max_x = self.center_x + self.half_width
        min_z = self.center_z - self.half_width
        max_z = self.center_z + self.half_width

        if abs(dx) >= abs(dz):
            if dx >= 0:
                return (max_x, z)
            else:
                return (min_x, z)
        else:
            if dz >= 0:
                return (x, max_z)
            else:
                return (x, min_z)


class RectangleClipper(ShapeClipper):
    """Rectangle boundary clipper with independent width and height."""

    def __init__(self, center_x: float, center_z: float,
                 half_width: float, half_height: float):
        super().__init__(center_x, center_z, max(half_width, half_height))
        self.half_width = half_width
        self.half_height = half_height

    def is_inside(self, x, z):
        return (np.abs(x - self.center_x) <= self.half_width) & \
               (np.abs(z - self.center_z) <= self.half_height)

    def _line_box_intersection(self, p1, p2) -> list[tuple[float, float]]:
        """Find intersection points of a line segment with the rectangle boundary."""
        x1, z1 = p1
        x2, z2 = p2

        min_x = self.center_x - self.half_width
        max_x = self.center_x + self.half_width
        min_z = self.center_z - self.half_height
        max_z = self.center_z + self.half_height

        intersections = []

        if x1 != x2:
            t = (min_x - x1) / (x2 - x1)
            if 0 <= t <= 1:
                z = z1 + t * (z2 - z1)
                if min_z <= z <= max_z:
                    intersections.append((min_x, z))

        if x1 != x2:
            t = (max_x - x1) / (x2 - x1)
            if 0 <= t <= 1:
                z = z1 + t * (z2 - z1)
                if min_z <= z <= max_z:
                    intersections.append((max_x, z))

        if z1 != z2:
            t = (min_z - z1) / (z2 - z1)
            if 0 <= t <= 1:
                x = x1 + t * (x2 - x1)
                if min_x <= x <= max_x:
                    intersections.append((x, min_z))

        if z1 != z2:
            t = (max_z - z1) / (z2 - z1)
            if 0 <= t <= 1:
                x = x1 + t * (x2 - x1)
                if min_x <= x <= max_x:
                    intersections.append((x, max_z))

        unique = []
        for pt in intersections:
            if not any(np.allclose(pt, u, atol=1e-6) for u in unique):
                unique.append(pt)

        return unique

    def clip_linestring(self, points) -> list[np.ndarray]:
        if len(points) < 2:
            return []

        points = np.array(points)
        segments = []
        current_segment = []

        for i in range(len(points)):
            x, z = points[i]
            inside = self.is_inside(x, z)

            if inside:
                current_segment.append((x, z))

            if i < len(points) - 1:
                x_next, z_next = points[i + 1]
                inside_next = self.is_inside(x_next, z_next)

                if inside != inside_next:
                    intersections = self._line_box_intersection(
                        (x, z), (x_next, z_next)
                    )
                    if intersections:
                        ix, iz = intersections[0]
                        if inside:
                            current_segment.append((ix, iz))
                            if len(current_segment) >= 2:
                                segments.append(np.array(current_segment))
                            current_segment = []
                        else:
                            current_segment = [(ix, iz)]

        if len(current_segment) >= 2:
            segments.append(np.array(current_segment))

        return segments

    def clip_polygon(self, points) -> np.ndarray | None:
        if len(points) == 0:
            return None

        points = np.array(points)
        inside_mask = self.is_inside(points[:, 0], points[:, 1])

        if np.all(inside_mask):
            return points
        return None

    def project_to_boundary(self, x: float, z: float) -> tuple[float, float]:
        dx = x - self.center_x
        dz = z - self.center_z

        min_x = self.center_x - self.half_width
        max_x = self.center_x + self.half_width
        min_z = self.center_z - self.half_height
        max_z = self.center_z + self.half_height

        ratio_x = abs(dx) / self.half_width if self.half_width > 0 else 0
        ratio_z = abs(dz) / self.half_height if self.half_height > 0 else 0

        if ratio_x >= ratio_z:
            if dx >= 0:
                return (max_x, z)
            else:
                return (min_x, z)
        else:
            if dz >= 0:
                return (x, max_z)
            else:
                return (x, min_z)


class HexagonClipper(ShapeClipper):
    """Hexagon boundary clipper (flat-top orientation)."""

    def __init__(self, center_x: float, center_z: float, radius: float):
        super().__init__(center_x, center_z, radius)
        self.radius = radius

        # Compute hexagon vertices (flat-top orientation)
        angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
        self.vertices = np.array([
            [center_x + radius * np.cos(a), center_z + radius * np.sin(a)]
            for a in angles
        ])

    def is_inside(self, x, z):
        """Test containment using ray-casting algorithm."""
        x = np.atleast_1d(x)
        z = np.atleast_1d(z)
        scalar_input = x.shape == (1,)

        result = np.zeros(x.shape, dtype=bool)

        for i in range(len(x)):
            px, pz = x[i], z[i]
            inside = False

            for j in range(6):
                k = (j + 1) % 6
                vx1, vz1 = self.vertices[j]
                vx2, vz2 = self.vertices[k]

                if ((vz1 > pz) != (vz2 > pz)) and \
                   (px < (vx2 - vx1) * (pz - vz1) / (vz2 - vz1) + vx1):
                    inside = not inside

            result[i] = inside

        return result[0] if scalar_input else result

    def clip_linestring(self, points) -> list[np.ndarray]:
        """Clip linestring to hexagon boundary.

        Uses simple inside/outside tracking without edge intersection
        (matching reference implementation).
        """
        if len(points) < 2:
            return []

        points = np.array(points)
        segments = []
        current_segment = []

        for i in range(len(points)):
            x, z = points[i]
            inside = self.is_inside(x, z)

            if inside:
                current_segment.append((x, z))
            elif len(current_segment) > 0:
                if len(current_segment) >= 2:
                    segments.append(np.array(current_segment))
                current_segment = []

        if len(current_segment) >= 2:
            segments.append(np.array(current_segment))

        return segments

    def clip_polygon(self, points) -> np.ndarray | None:
        if len(points) == 0:
            return None

        points = np.array(points)
        inside_mask = self.is_inside(points[:, 0], points[:, 1])

        if np.all(inside_mask):
            return points
        return None

    def project_to_boundary(self, x: float, z: float) -> tuple[float, float] | None:
        """Project point to nearest point on hexagon boundary."""
        min_dist_sq = float('inf')
        nearest_point = None

        for i in range(6):
            j = (i + 1) % 6
            v1 = self.vertices[i]
            v2 = self.vertices[j]

            edge = v2 - v1
            point_vec = np.array([x, z]) - v1
            edge_length_sq = np.dot(edge, edge)

            if edge_length_sq < 1e-12:
                projected = v1
            else:
                t = max(0, min(1, np.dot(point_vec, edge) / edge_length_sq))
                projected = v1 + t * edge

            dist_sq = (projected[0] - x) ** 2 + (projected[1] - z) ** 2

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest_point = projected

        return tuple(nearest_point) if nearest_point is not None else None
