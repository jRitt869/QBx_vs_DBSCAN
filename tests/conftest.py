from datetime import UTC, datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString

from track_clusters.data import _s2_to_cartesian


@pytest.fixture
def feature_tracks() -> gpd.GeoDataFrame:
    """Two separated routes, with two time groups inside each."""
    spatial, temporal, geometry = [], [], []
    for track_id in range(24):
        group = track_id // 12
        points = np.column_stack(
            (np.linspace(group * 10, group * 10 + 1, 13), np.linspace(0, 1, 13), np.zeros(13))
        )
        hour = 8 if track_id % 12 < 6 else 18
        spatial.append(points)
        temporal.append([_s2_to_cartesian(0, 2 * np.pi * hour / 24)] * 13)
        geometry.append(LineString(points[:, :2]))
    return gpd.GeoDataFrame(
        {
            "name": [f"track_{i}" for i in range(24)],
            "13_pts": spatial,
            "times": temporal,
            "geometry": geometry,
        },
        crs="EPSG:4326",
    )


@pytest.fixture
def track_directory(tmp_path: Path) -> Path:
    """Known route/time groups and one gap for isolated pipeline regression checks."""
    namespace = "http://www.topografix.com/GPX/1/1"
    ET.register_namespace("", namespace)

    def tag(name: str) -> str:
        return f"{{{namespace}}}{name}"

    root = ET.Element(tag("gpx"), version="1.1", creator="GPX clustering regression fixture")
    for track_id in range(25):
        route = track_id // 12
        variant = track_id % 12
        start = datetime(2024, 1, 8, 8 if variant < 6 else 18, tzinfo=UTC)
        track = ET.SubElement(root, tag("trk"))
        ET.SubElement(track, tag("name")).text = f"route-{route}-{variant}"
        segment = ET.SubElement(track, tag("trkseg"))
        for point_id in range(13):
            longitude = 13.35 + route * 0.08 + point_id * 0.002 + variant * 0.00001
            latitude = 52.50 + route * 0.04 + (point_id % 5) * 0.001 + variant * 0.00001
            point = ET.SubElement(
                segment, tag("trkpt"), lat=f"{latitude:.6f}", lon=f"{longitude:.6f}"
            )
            gap_minutes = 40 if track_id == 0 and point_id >= 7 else 0
            moment = start + timedelta(minutes=point_id * 2 + gap_minutes)
            ET.SubElement(point, tag("time")).text = moment.isoformat().replace("+00:00", "Z")
    ET.indent(root)
    ET.ElementTree(root).write(tmp_path / "tracks.gpx", encoding="utf-8", xml_declaration=True)
    return tmp_path
