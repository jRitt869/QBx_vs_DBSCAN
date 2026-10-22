"""Feature preparation and run storage for the maintained clustering pipeline."""

import numpy as np
import pytest

from extensions.analytics.evaluation import run_clustering
from extensions.analytics.storage import ClusterDatabase
from extensions.analytics.warehouse import ingest_database
from track_clusters.clustering import get_dbscan_clusters, get_qb_clusters
from track_clusters.validation import ClusteringSettings


@pytest.mark.parametrize("algorithm", ["dbscan", "qbx"])
def test_spatial_features_are_prepared_by_clustering(
    track_directory, tmp_path, algorithm, ingestion_input, monkeypatch
):
    from dipy.tracking.streamline import set_number_of_points

    import extensions.analytics.evaluation as pipeline

    path = tmp_path / "source.sqlite"
    ingest_database(ingestion_input(track_directory, path))
    with ClusterDatabase(path) as database:
        raw = database.load_tracks()
    expected = [set_number_of_points(np.asarray(line.coords), 13) for line in raw.geometry]
    center = np.mean(expected, axis=0)
    if algorithm == "qbx":
        distances = [
            np.linalg.norm(first - second, axis=1).mean()
            for index, first in enumerate(expected)
            for second in expected[index + 1 :]
        ]
        scale = np.median([value for value in distances if value > 0])
        expected = [np.pad((points - center) / scale, ((0, 0), (0, 1))) for points in expected]
    else:
        expected = [points - center for points in expected]
    cluster = get_dbscan_clusters if algorithm == "dbscan" else get_qb_clusters
    prepared = []

    def checked(tracks, *args, **kwargs):
        np.testing.assert_allclose(tracks["13_pts"].tolist(), expected, rtol=1e-12)
        assert "times" not in tracks and "timestamps" not in tracks
        prepared.append(True)
        return cluster(tracks, *args, **kwargs)

    monkeypatch.setattr(
        pipeline, "get_dbscan_clusters" if algorithm == "dbscan" else "get_qb_clusters", checked
    )
    with ClusterDatabase(path, writable=True) as database:
        run_clustering(database, ClusteringSettings(algorithm=algorithm, eps=0.5))
    assert prepared == [True]


def test_spatial_qbx_builds_one_hierarchy_for_all_layers(
    track_directory, tmp_path, ingestion_input, monkeypatch
):
    import track_clusters.clustering as clustering

    path = tmp_path / "spatial.sqlite"
    ingest_database(ingestion_input(track_directory, path))
    original = clustering.QuickBundlesX
    calls = []

    def recorded(*args, **kwargs):
        calls.append(kwargs["thresholds"])
        return original(*args, **kwargs)

    monkeypatch.setattr(clustering, "QuickBundlesX", recorded)
    with ClusterDatabase(path, writable=True) as database:
        run_id = run_clustering(database, ClusteringSettings(algorithm="qbx", eps=0.5))
        assert database.select_frame(
            "assignments",
            ("layer",),
            where={"run_id": run_id},
            distinct=True,
            order_by=("layer",),
        ).layer.tolist() == [0, 1, 2]
    assert calls == [[2.0, 1.0, 0.5]]


def test_database_map_without_timeskips(track_directory, tmp_path, ingestion_input):
    import gpxpy

    source = track_directory / "tracks.gpx"
    gpx = gpxpy.parse(source.read_text(encoding="utf-8"))
    gpx.tracks.pop(0)
    source.write_text(gpx.to_xml(), encoding="utf-8")
    path = tmp_path / "no-gaps.sqlite"
    ingest_database(ingestion_input(track_directory, path))
    with ClusterDatabase(path, writable=True) as database:
        run_id = run_clustering(
            database, ClusteringSettings(algorithm="dbscan", eps=0.5, feature_mode="spatiotemporal")
        )
        jumps = database.load_map_jumps(run_id)
    assert jumps.empty
    assert set(jumps.columns) == {"jump_name", "jump_hours", "jump_minutes", "x", "y"}
