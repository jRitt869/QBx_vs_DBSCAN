"""The isolated database adapter preserves the original numerical output."""

from datetime import UTC

import numpy as np
import pytest

from extensions.analytics.warehouse import ingest_database
from historical.clustering import get_dbscan_clusters, get_qb_clusters
from historical.data import MAP_CRS, _transform_time, load_directory
from historical.pipeline import run_clustering
from historical.storage import ClusterDatabase
from historical.viewer import map_track_frame


@pytest.mark.parametrize("algorithm,eps", [("dbscan", 0.5), ("qbx", 0.01)])
def test_database_adapter_matches_direct_clustering(
    track_directory, tmp_path, algorithm, eps, ingestion_input, monkeypatch
):
    import historical.pipeline as pipeline

    database = tmp_path / "source.sqlite"
    ingest_database(ingestion_input(track_directory, database))
    original, original_gaps = load_directory(track_directory)
    original = _transform_time(original, UTC)
    cluster = get_dbscan_clusters if algorithm == "dbscan" else get_qb_clusters

    def checked(tracks, *args):
        np.testing.assert_array_equal(
            np.asarray(original["13_pts"].tolist()), np.asarray(tracks["13_pts"].tolist())
        )
        np.testing.assert_array_equal(
            np.asarray(original.times.tolist()), np.asarray(tracks.times.tolist())
        )
        assert "timestamps" not in tracks
        return cluster(tracks, *args)

    monkeypatch.setattr(
        pipeline, "get_dbscan_clusters" if algorithm == "dbscan" else "get_qb_clusters", checked
    )
    expected = (
        get_dbscan_clusters(original, eps, 0.1, 5)
        if algorithm == "dbscan"
        else get_qb_clusters(original, eps, 0.1, 2, 0)
    )
    with ClusterDatabase(database, writable=True) as source_db:
        run_id = run_clustering(source_db, algorithm, eps, feature_mode="spatiotemporal")
    with ClusterDatabase(database) as source_db:
        tracks = map_track_frame(source_db.load_tracks(run_id))
        gaps = source_db.load_map_jumps(run_id)
        labels = source_db.load_layer_names(run_id)
    assert len(labels) == (4 if algorithm == "qbx" else 2)
    assert len(tracks) == 25 and len(gaps) == 1
    assert tracks["clusters0"].nunique() == 3
    for index in range(len(labels)):
        assert tracks[f"clusters{index}"].tolist() == expected[f"clusters{index}"].tolist()
        np.testing.assert_array_equal(
            tracks[f"colors{index}"].tolist(), expected[f"colors{index}"].tolist()
        )
    projected = expected.to_crs(MAP_CRS)
    np.testing.assert_allclose(
        tracks.x.tolist(), [list(line.coords.xy[0]) for line in projected.geometry]
    )
    np.testing.assert_allclose(
        tracks.y.tolist(), [list(line.coords.xy[1]) for line in projected.geometry]
    )
    assert gaps.jump_minutes.tolist() == original_gaps.jump_minutes.tolist()
    projected_gaps = original_gaps.to_crs(MAP_CRS)
    np.testing.assert_allclose(gaps.x, projected_gaps.geometry.x)
    np.testing.assert_allclose(gaps.y, projected_gaps.geometry.y)
