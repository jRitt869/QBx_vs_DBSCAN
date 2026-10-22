"""Controlled source data, including ignored GPX-number gaps and incomplete timestamps."""

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import geopandas as gpd
import gpxpy.gpx
import numpy as np
import pandas as pd
import pytest
from dipy.tracking.streamline import set_number_of_points

from extensions.analytics.storage import ClusterDatabase
from extensions.analytics.tests.evaluation_helpers import evaluate_sweeps as evaluate
from extensions.analytics.warehouse import ingest_database, read_config
from scripts.config import validate
from scripts.config.validate import IngestionInput


@pytest.fixture
def ingestion_input(monkeypatch):
    """Create model inputs whose key resolves to an isolated test database."""

    def create(directory, database, **settings):
        monkeypatch.setattr(validate, "OUTPUT_DIRECTORY", database.parent)
        settings.setdefault("dataset_key", database.stem)
        inputs = IngestionInput(source_folder=directory, **settings)
        assert inputs.database == database.resolve()
        return inputs

    return create


@pytest.fixture(autouse=True)
def output_directory(tmp_path, monkeypatch):
    """Keep key-based CLI imports in a separate temporary output directory per test."""
    output = tmp_path / "outputs"
    monkeypatch.setattr(validate, "OUTPUT_DIRECTORY", output)
    return output


@pytest.fixture(scope="module")
def sample_sources(tmp_path_factory):
    root = tmp_path_factory.mktemp("analytics_sources")
    data = root / "gpx"
    data.mkdir()
    gpx = gpxpy.gpx.GPX()
    labels = []
    for index in range(14):
        number = index * 3 + 2
        track = gpxpy.gpx.GPXTrack(number=number)
        count = 5 if index < 12 else (3 if index == 12 else 1)
        longitude = (8 if index < 6 else 10) + (max(0, index - 1) % 6) * 0.00001
        points = []
        for i in range(count):
            time = None
            if index == 0:
                time = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=[0, 1, 2, 60, 61][i])
            points.append(
                gpxpy.gpx.GPXTrackPoint(
                    50 + i * 0.001, longitude + i * 0.001, time=time, elevation=100 + i
                )
            )
        for group in (points[:3], points[3:]) if index == 0 else (points,):
            segment = gpxpy.gpx.GPXTrackSegment()
            segment.points = group
            track.segments.append(segment)
        gpx.tracks.append(track)
        if index < 13:
            labels.append(
                {
                    "file_name": "sample.gpx",
                    "id": index,
                    "cluster": (0 if index < 6 else 1) if index < 12 else -1,
                }
            )
    (data / "sample.gpx").write_text(gpx.to_xml(), encoding="utf-8")
    pd.DataFrame(labels).to_csv(root / "labels.csv", index=False)
    (root / "datasets.toml").write_text(
        """[[datasets]]
dataset_key = "sample"
name = "Controlled sample"
source_folder = "gpx"
labels = "labels.csv"
label_kind = "author"
label_description = "Controlled test labels"
""",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def source_copy(sample_sources, tmp_path):
    root = tmp_path / "sources"
    shutil.copytree(sample_sources, root)
    return root


@pytest.fixture
def analytics_database(source_copy):
    config = source_copy / "datasets.toml"
    source = read_config(config).datasets[0]
    database = source.database
    ingest_database(source, config)
    with ClusterDatabase(database, writable=True) as source_db:
        evaluate(source_db, qbx_eps=[0.1, 1], dbscan_eps=[0.1, 1], seeds=[3, 0, 1])
    return database


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
def feature_tracks(spatial_directory: Path, ingestion_input) -> gpd.GeoDataFrame:
    """Spatial groups with controlled time features for application parity checks."""
    path = spatial_directory / "features.sqlite"
    ingest_database(ingestion_input(spatial_directory, path))
    with ClusterDatabase(path) as database:
        source = database.load_tracks()
    source = source.loc[source.eligible == 1].reset_index(drop=True)
    source["13_pts"] = [
        np.pad(set_number_of_points(np.asarray(line.coords), 13), ((0, 0), (0, 1)))
        for line in source.geometry
    ]
    tracks = gpd.GeoDataFrame(pd.concat([source] * 3, ignore_index=True), crs=source.crs)
    tracks["times"] = [np.tile([1.0, 0.0, 0.0], (39, 1)).tolist() for _ in range(len(tracks))]
    return tracks
