"""Load GPX tracks and construct the internship project's original features."""

from datetime import datetime, tzinfo
from pathlib import Path

import geopandas as gpd
import gpxpy
import numpy as np
import pandas as pd
from dipy.tracking.streamline import set_number_of_points
from shapely.geometry import LineString, Point

GPX_CRS = "EPSG:4326"
MAP_CRS = "EPSG:3857"
RESAMPLED_POINTS = 13
MIN_TRACK_POINTS = 5
TIMESKIP_SECONDS = 30 * 60


def get_geo_data(path: str | Path, file: str) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load tracks, joining segments and marking time gaps longer than 30 minutes.

    Resampling includes timestamps, as in the original project. Tracks with fewer
    than five points are ignored. Invalid timestamps fail with file/track context.
    """
    source_path = Path(path) / file
    with source_path.open(encoding="utf-8") as source:
        gpx = gpxpy.parse(source)
    names, sampled, geometries = [], [], []
    jump_names, jump_hours, jump_minutes, jump_points = [], [], [], []
    for track_id, track in enumerate(gpx.tracks):
        coords: list[list[float]] = []
        gaps: list[tuple[Point, int, int]] = []
        for segment in track.segments:
            for point in segment.points:
                context = f"{file}, track {track_id}"
                if point.time is None:
                    raise ValueError(f"Missing timestamp in {context}")
                timestamp = point.time.timestamp()
                if not np.isfinite(timestamp) or timestamp < 0:
                    raise ValueError(f"Invalid or negative timestamp in {context}")
                if not (-180 <= point.longitude <= 180 and -90 <= point.latitude <= 90):
                    raise ValueError(f"Invalid longitude/latitude in {context}")
                if coords:
                    delta = timestamp - coords[-1][2]
                    if delta < 0:
                        raise ValueError(f"Timestamps are not chronological in {context}")
                    if delta > TIMESKIP_SECONDS:
                        gaps.append(
                            (
                                Point(point.longitude, point.latitude),
                                int(delta // 3600),
                                int(delta // 60 % 60),
                            )
                        )
                coords.append([point.longitude, point.latitude, timestamp])
        if len(coords) < MIN_TRACK_POINTS:
            continue
        name = f"{source_path.stem}_{track_id}"
        points = np.array(coords, dtype=np.float64)
        names.append(name)
        sampled.append(np.ascontiguousarray(set_number_of_points(points, RESAMPLED_POINTS)))
        geometries.append(LineString(points[:, :2]))
        for location, hours, minutes in gaps:
            jump_names.append(name)
            jump_hours.append(hours)
            jump_minutes.append(minutes)
            jump_points.append(location)
    tracks = gpd.GeoDataFrame(
        {"name": names, "13_pts": sampled, "geometry": geometries}, crs=GPX_CRS
    )
    jumps = gpd.GeoDataFrame(
        {
            "jump_name": jump_names,
            "jump_hours": jump_hours,
            "jump_minutes": jump_minutes,
            "geometry": jump_points,
        },
        crs=GPX_CRS,
    )
    return tracks, jumps


def load_directory(path: str | Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load a directory in stable filename order; do not traverse subdirectories."""
    directory = Path(path)
    if not directory.is_dir():
        raise ValueError(f"GPX directory does not exist: {directory}")
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".gpx")
    if not files:
        raise ValueError(f"No GPX files found in {directory}")
    track_frames, jump_frames = [], []
    for source in files:
        tracks, jumps = get_geo_data(directory, source.name)
        track_frames.append(tracks)
        jump_frames.append(jumps)
    tracks = gpd.GeoDataFrame(pd.concat(track_frames, ignore_index=True), crs=GPX_CRS)
    jumps = gpd.GeoDataFrame(pd.concat(jump_frames, ignore_index=True), crs=GPX_CRS)
    if tracks.empty:
        raise ValueError("No usable tracks: each track needs at least five timestamped points")
    return tracks, jumps


def _s2_to_cartesian(phi: float, theta: float) -> list[float]:
    """Preserve the original weekday/hour mapping onto a unit sphere."""
    return [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]


def _transform_time(df: gpd.GeoDataFrame, time_zone: tzinfo | None = None) -> gpd.GeoDataFrame:
    """Encode resampled times, then remove timestamps from spatial features.

    None retains the original host-local timezone behavior. Use an explicit
    timezone (e.g. UTC) for runs that must agree across machines.
    """
    result = df.copy()
    times = result["13_pts"].map(
        lambda points: [datetime.fromtimestamp(p[2], tz=time_zone) for p in points]
    )
    result["times"] = times.map(
        lambda values: [
            _s2_to_cartesian(2 * np.pi * t.weekday() / 7, 2 * np.pi * t.hour / 24) for t in values
        ]
    )
    result["13_pts"] = result["13_pts"].map(
        lambda points: np.ascontiguousarray(np.pad(points[:, :2], ((0, 0), (0, 1))))
    )
    return result
