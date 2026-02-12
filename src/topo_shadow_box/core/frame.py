"""Shadow box frame mesh generation."""

import math
import numpy as np

from ..state import Bounds
from .coords import GeoToModelTransform


def generate_frame_mesh(
    model_width_mm: float,
    frame_width_mm: float,
    frame_depth_mm: float,
    wall_thickness_mm: float,
    shape: str,
    bounds: Bounds,
    transform: GeoToModelTransform,
) -> dict:
    """Generate a shadow box frame as a mesh.

    The frame is a hollow rectangular/circular border that surrounds the terrain model.
    It's generated as the difference of two boxes (outer - inner).

    Returns dict with 'vertices' and 'faces'.
    """
    # Inner dimensions match the model
    inner_w = transform.model_width_x
    inner_h = transform.model_width_z

    # Outer dimensions include frame border
    outer_w = inner_w + 2 * frame_width_mm
    outer_h = inner_h + 2 * frame_width_mm

    # Frame origin: offset so inner aligns with model at (0, 0)
    ox = -frame_width_mm
    oz = -frame_width_mm

    # We build the frame as 4 wall segments + bottom plate
    # Y goes from -frame_depth_mm (bottom) to 0 (top, flush with terrain base)
    y_top = 0.0
    y_bottom = -frame_depth_mm

    vertices = []
    faces = []

    # Generate frame as 4 walls
    # Each wall is a rectangular prism

    # Front wall (south, high Z)
    _add_wall_segment(
        vertices, faces,
        x0=ox, x1=ox + outer_w,
        z0=oz + outer_h - frame_width_mm, z1=oz + outer_h,
        y_bottom=y_bottom, y_top=y_top,
    )

    # Back wall (north, low Z)
    _add_wall_segment(
        vertices, faces,
        x0=ox, x1=ox + outer_w,
        z0=oz, z1=oz + frame_width_mm,
        y_bottom=y_bottom, y_top=y_top,
    )

    # Left wall (west, low X) - only the inner portion (between front/back)
    _add_wall_segment(
        vertices, faces,
        x0=ox, x1=ox + frame_width_mm,
        z0=oz + frame_width_mm, z1=oz + outer_h - frame_width_mm,
        y_bottom=y_bottom, y_top=y_top,
    )

    # Right wall (east, high X) - only the inner portion
    _add_wall_segment(
        vertices, faces,
        x0=ox + outer_w - frame_width_mm, x1=ox + outer_w,
        z0=oz + frame_width_mm, z1=oz + outer_h - frame_width_mm,
        y_bottom=y_bottom, y_top=y_top,
    )

    # Bottom plate (covers entire frame footprint)
    _add_bottom_plate(
        vertices, faces,
        x0=ox, x1=ox + outer_w,
        z0=oz, z1=oz + outer_h,
        y=y_bottom,
        wall_thickness=wall_thickness_mm,
    )

    return {
        "vertices": vertices,
        "faces": faces,
    }


def _add_wall_segment(
    vertices: list, faces: list,
    x0: float, x1: float, z0: float, z1: float,
    y_bottom: float, y_top: float,
):
    """Add a rectangular wall segment (box) to the mesh."""
    vi = len(vertices)

    # 8 vertices of the box
    # Bottom face (y_bottom)
    vertices.append([x0, y_bottom, z0])  # 0
    vertices.append([x1, y_bottom, z0])  # 1
    vertices.append([x1, y_bottom, z1])  # 2
    vertices.append([x0, y_bottom, z1])  # 3
    # Top face (y_top)
    vertices.append([x0, y_top, z0])     # 4
    vertices.append([x1, y_top, z0])     # 5
    vertices.append([x1, y_top, z1])     # 6
    vertices.append([x0, y_top, z1])     # 7

    # 12 triangles (2 per face, 6 faces), CCW from outside
    # Top (+Y)
    faces.append([vi + 4, vi + 5, vi + 6])
    faces.append([vi + 4, vi + 6, vi + 7])
    # Bottom (-Y)
    faces.append([vi + 0, vi + 2, vi + 1])
    faces.append([vi + 0, vi + 3, vi + 2])
    # Front (+Z)
    faces.append([vi + 3, vi + 6, vi + 2])
    faces.append([vi + 3, vi + 7, vi + 6])
    # Back (-Z)
    faces.append([vi + 0, vi + 1, vi + 5])
    faces.append([vi + 0, vi + 5, vi + 4])
    # Left (-X)
    faces.append([vi + 0, vi + 4, vi + 7])
    faces.append([vi + 0, vi + 7, vi + 3])
    # Right (+X)
    faces.append([vi + 1, vi + 2, vi + 6])
    faces.append([vi + 1, vi + 6, vi + 5])


def _add_bottom_plate(
    vertices: list, faces: list,
    x0: float, x1: float, z0: float, z1: float,
    y: float, wall_thickness: float,
):
    """Add a thin bottom plate to the frame."""
    vi = len(vertices)
    y_top = y
    y_bot = y - wall_thickness

    vertices.append([x0, y_top, z0])  # 0
    vertices.append([x1, y_top, z0])  # 1
    vertices.append([x1, y_top, z1])  # 2
    vertices.append([x0, y_top, z1])  # 3
    vertices.append([x0, y_bot, z0])  # 4
    vertices.append([x1, y_bot, z0])  # 5
    vertices.append([x1, y_bot, z1])  # 6
    vertices.append([x0, y_bot, z1])  # 7

    # Top (+Y) - interior facing
    faces.append([vi + 0, vi + 2, vi + 1])
    faces.append([vi + 0, vi + 3, vi + 2])
    # Bottom (-Y) - exterior facing
    faces.append([vi + 4, vi + 5, vi + 6])
    faces.append([vi + 4, vi + 6, vi + 7])
    # Edges
    faces.append([vi + 0, vi + 1, vi + 5])
    faces.append([vi + 0, vi + 5, vi + 4])
    faces.append([vi + 1, vi + 2, vi + 6])
    faces.append([vi + 1, vi + 6, vi + 5])
    faces.append([vi + 2, vi + 3, vi + 7])
    faces.append([vi + 2, vi + 7, vi + 6])
    faces.append([vi + 3, vi + 0, vi + 4])
    faces.append([vi + 3, vi + 4, vi + 7])
