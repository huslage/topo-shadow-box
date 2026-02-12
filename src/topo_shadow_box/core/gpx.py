"""GPX file parsing."""

import gpxpy


def parse_gpx_file(filepath: str) -> dict:
    """Parse a GPX file and extract tracks, waypoints, and bounds.

    Returns:
        dict with keys: tracks, waypoints, bounds, metadata
    """
    with open(filepath, "r") as f:
        gpx = gpxpy.parse(f)

    tracks = []
    for track in gpx.tracks:
        points = []
        for segment in track.segments:
            for point in segment.points:
                points.append({
                    "lat": point.latitude,
                    "lon": point.longitude,
                    "elevation": point.elevation if point.elevation else 0,
                })
        tracks.append({
            "name": track.name or "Unnamed Track",
            "points": points,
        })

    waypoints = []
    for wp in gpx.waypoints:
        waypoints.append({
            "name": wp.name,
            "lat": wp.latitude,
            "lon": wp.longitude,
            "elevation": wp.elevation if wp.elevation else 0,
            "description": wp.description,
        })

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
        "tracks": tracks,
        "waypoints": waypoints,
        "bounds": bounds_dict,
        "metadata": {
            "name": gpx.name,
            "description": gpx.description,
        },
    }
