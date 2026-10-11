"""Fixtures scoped to the later spatial analysis additions."""

from pathlib import Path

import geopandas as gpd
import gpxpy.gpx
import numpy as np
import pandas as pd
import pytest

from extensions.spatial_analysis.spatial import load_spatial_directory


@pytest.fixture
def spatial_directory(tmp_path: Path) -> Path:
    """Numbered tracks without timestamps, including a duplicate and a short track."""
    gpx = gpxpy.gpx.GPX()
    for number, lon, lat, count in (
        (2, 8, 50, 5),
        (4, 8, 50, 5),
        (7, 10, 52, 5),
        (11, 10.01, 52, 5),
        (13, 8, 50, 2),
    ):
        track = gpxpy.gpx.GPXTrack(number=number)
        segment = gpxpy.gpx.GPXTrackSegment()
        segment.points = [
            gpxpy.gpx.GPXTrackPoint(lat + i * 0.001, lon + i * 0.001, elevation=1000 + i * 100)
            for i in range(count)
        ]
        track.segments.append(segment)
        gpx.tracks.append(track)
    (tmp_path / "000001.gpx").write_text(gpx.to_xml(), encoding="utf-8")
    return tmp_path


@pytest.fixture
def feature_tracks(spatial_directory: Path) -> gpd.GeoDataFrame:
    """Spatial groups with controlled time features for application parity checks."""
    source = load_spatial_directory(spatial_directory).tracks
    tracks = gpd.GeoDataFrame(pd.concat([source] * 3, ignore_index=True), crs=source.crs)
    tracks["times"] = [np.tile([1.0, 0.0, 0.0], (13, 1)).tolist() for _ in range(len(tracks))]
    return tracks
