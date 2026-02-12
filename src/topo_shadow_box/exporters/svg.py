"""SVG map export for paper printing."""

from ..state import Bounds, Colors
from ..core.map_insert import generate_map_insert_svg


def export_svg(
    bounds: Bounds,
    features: dict,
    gpx_tracks: list,
    colors: Colors,
    output_path: str,
    model_width_mm: float = 200.0,
) -> None:
    """Export the map insert as an SVG file."""
    svg_content = generate_map_insert_svg(
        bounds=bounds,
        features=features,
        gpx_tracks=gpx_tracks,
        colors=colors,
    )
    with open(output_path, "w") as f:
        f.write(svg_content)
