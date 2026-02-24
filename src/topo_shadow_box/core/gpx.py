"""GPX file parsing."""

import gpxpy
from topo_shadow_box.models import GpxTrack, GpxPoint, GpxWaypoint


def parse_gpx_file(filepath: str) -> dict:
    """Parse a GPX file and extract tracks, waypoints, and bounds."""
    with open(filepath, "r") as f:
        gpx = gpxpy.parse(f)

    tracks = []
    for track in gpx.tracks:
        points = []
        for segment in track.segments:
            for point in segment.points:
                points.append(GpxPoint(
                    lat=point.latitude,
                    lon=point.longitude,
                    elevation=point.elevation if point.elevation else 0.0,
                ))
        if len(points) >= 2:
            tracks.append(GpxTrack(
                name=track.name or "Unnamed Track",
                points=points,
            ))

    waypoints = []
    for wp in gpx.waypoints:
        waypoints.append(GpxWaypoint(
            name=wp.name or "",
            lat=wp.latitude,
            lon=wp.longitude,
            elevation=wp.elevation if wp.elevation else 0.0,
            description=wp.description or "",
        ))

    bounds = gpx.get_bounds()
    bounds_dict = None
    if bounds:
        bounds_dict = {
            "north": bounds.max_latitude,
            "south": bounds.min_latitude,
            "east": bounds.max_longitude,
            "west": bounds.min_longitude,
        }

    return {
        "tracks": tracks,        # now list[GpxTrack]
        "waypoints": waypoints,  # now list[GpxWaypoint]
        "bounds": bounds_dict,
        "metadata": {
            "name": gpx.name,
            "description": gpx.description,
        },
    }
