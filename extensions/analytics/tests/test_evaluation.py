from functools import partial
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from pydantic import ValidationError

from extensions.analytics.evaluation import run_clustering
from extensions.analytics.metrics import score_assignments
from extensions.analytics.storage import ClusterDatabase, Database
from extensions.analytics.temporal import TorusMetric, spatiotemporal_distance, torus_distance
from extensions.analytics.tests.evaluation_helpers import evaluate_sweeps as evaluate
from extensions.analytics.warehouse import ingest_database, read_labels
from track_clusters.clustering import get_dbscan_clusters, get_qb_clusters
from track_clusters.validation import ClusteringSettings


def labels_for_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {"file_name": ["000001.gpx"] * 5, "id": [0, 1, 2, 3, 4], "cluster": [0, 0, 1, 1, -1]}
    )


def test_spatial_loading_uses_track_indices_and_ignores_elevation(
    spatial_directory: Path, ingestion_input
) -> None:
    path = spatial_directory / "source.sqlite"
    ingest_database(ingestion_input(spatial_directory, path))
    with ClusterDatabase(path) as database:
        tracks = database.load_tracks()
        catalog = database.report("track_catalog")
    assert tracks.source_track_id.tolist() == [0, 1, 2, 3, 4]
    assert tracks.track_index.tolist() == [0, 1, 2, 3, 4]
    assert tracks.eligible.tolist() == [1, 1, 1, 1, 0]
    assert catalog.loc[catalog.eligible == 0, "source_track_id"].tolist() == [4]
    assert "times" not in tracks and "13_pts" not in tracks
    eligible = tracks.loc[tracks.eligible == 1]
    for index, (timestamps, geometry) in enumerate(
        zip(eligible.timestamps, eligible.geometry, strict=True)
    ):
        lon, lat = (8, 50) if index < 2 else (10 + (index == 3) * 0.01, 52)
        expected = np.array([[lon + i * 0.001, lat + i * 0.001] for i in range(5)])
        np.testing.assert_array_equal(geometry.coords, expected)
        assert timestamps.shape == (5,) and np.isnan(timestamps).all()
    assert len(tracks.iloc[-1].geometry.coords) == len(tracks.iloc[-1].timestamps) == 2


@pytest.mark.parametrize("algorithm,layers", [("dbscan", 1), ("qbx", 3)])
def test_loading_run_filters_tracks_and_aligns_all_saved_layers(
    spatial_directory, ingestion_input, algorithm, layers
):
    path = spatial_directory / "source.sqlite"
    ingest_database(ingestion_input(spatial_directory, path))
    with ClusterDatabase(path, writable=True) as database:
        raw = database.load_tracks()
        run_id = run_clustering(
            database, ClusteringSettings(algorithm=algorithm, eps=0.1, min_samples=2)
        )
        selected = database.load_tracks(run_id=run_id)
        assignments = database.select_frame("assignments", where={"run_id": run_id})
        with pytest.raises(ValueError, match="No cluster assignments"):
            database.load_tracks(run_id=run_id + 1)
    assert len(raw) == 5 and len(selected) == 4
    assert not raw.filter(regex=r"^(clusters|colors)\d+$").columns.tolist()
    assert len(selected.filter(regex=r"^(clusters|colors)\d+$").columns) == 2 * layers
    expected = raw.loc[raw.eligible == 1].reset_index(drop=True)
    assert selected.track_id.tolist() == expected.track_id.tolist()
    assert selected.geometry.equals(expected.geometry)
    assert selected.crs.to_epsg() == 4326
    np.testing.assert_array_equal(selected.timestamps.tolist(), expected.timestamps.tolist())
    for layer in range(layers):
        saved = (
            assignments.loc[assignments.layer == layer].set_index("track_id").loc[selected.track_id]
        )
        assert selected[f"clusters{layer}"].tolist() == saved.predicted_cluster.tolist()
        assert [
            "#{:02x}{:02x}{:02x}".format(*rgb) for rgb in selected[f"colors{layer}"]
        ] == saved.color.tolist()


def test_loading_all_tracks_preserves_empty_and_single_point_geometry(
    spatial_directory, ingestion_input
):
    import gpxpy
    from gpxpy.gpx import GPXTrack, GPXTrackPoint, GPXTrackSegment

    source = spatial_directory / "000001.gpx"
    gpx = gpxpy.parse(source.read_text(encoding="utf-8"))
    for count in (0, 1):
        track = GPXTrack()
        segment = GPXTrackSegment()
        segment.points = [GPXTrackPoint(latitude=50, longitude=8) for _ in range(count)]
        track.segments.append(segment)
        gpx.tracks.append(track)
    source.write_text(gpx.to_xml(), encoding="utf-8")
    path = spatial_directory / "source.sqlite"
    ingest_database(ingestion_input(spatial_directory, path))
    with ClusterDatabase(path) as database:
        tracks = database.load_tracks()
    assert len(tracks) == 7
    assert tracks.iloc[-2].geometry.is_empty and tracks.iloc[-2].timestamps.size == 0
    assert tracks.iloc[-1].geometry.geom_type == "Point"
    assert list(tracks.iloc[-1].geometry.coords) == [(8, 50)]
    assert tracks.iloc[-1].timestamps.shape == (1,)


@pytest.mark.parametrize("algorithm,layer", [("dbscan", 3), ("qbx", 1), ("qbx", 2), ("qbx", 3)])
def test_spatial_algorithms_match_application(
    feature_tracks: gpd.GeoDataFrame, algorithm: str, layer: int
) -> None:
    spatial = feature_tracks.drop(columns="times")
    clustered = (
        get_dbscan_clusters(spatial, 0.1, metric=partial(spatiotemporal_distance, temporal_scale=0))
        if algorithm == "dbscan"
        else get_qb_clusters(spatial, 0.1, seed=42, metric=AveragePointwiseEuclideanMetric())
    )
    assignments = clustered["clusters0" if algorithm == "dbscan" else f"clusters{layer - 1}"]
    original = (
        get_dbscan_clusters(
            feature_tracks,
            0.1,
            0.01,
            5,
            temporal_metric=torus_distance,
            metric=partial(spatiotemporal_distance, temporal_scale=0),
        )["clusters0"]
        if algorithm == "dbscan"
        else get_qb_clusters(
            feature_tracks,
            0.1,
            0.01,
            2,
            42,
            temporal_metric=TorusMetric(),
            metric=AveragePointwiseEuclideanMetric(),
        )[f"clusters{layer - 1}"]
    )
    np.testing.assert_array_equal(original, assignments)


def test_label_alignment_reports_short_tracks_and_rejects_mismatches(
    spatial_directory: Path, ingestion_input
) -> None:
    labels = labels_for_fixture()
    labels_path = spatial_directory / "labels.csv"
    labels.iloc[::-1].to_csv(labels_path, index=False)
    database = spatial_directory / "source.sqlite"
    ingest_database(ingestion_input(spatial_directory, database, labels=labels_path))
    with Database(database) as source_db:
        tracks = source_db.report("track_catalog")
    assert tracks.loc[tracks.eligible == 1, "reference_cluster"].tolist() == [0, 0, 1, 1]
    assert tracks.loc[tracks.eligible == 0, "source_track_id"].tolist() == [4]
    labels.iloc[1:].to_csv(labels_path, index=False)
    ingest_database(ingestion_input(spatial_directory, database, labels=labels_path))
    with Database(database) as source_db:
        row = source_db.report("track_catalog").iloc[0]
        assert pd.isna(row.provided_reference) and row.reference_origin == "unlabeled"
    pd.concat(
        [labels, pd.DataFrame({"file_name": ["000001.gpx"], "id": [99], "cluster": [0]})]
    ).to_csv(labels_path, index=False)
    with pytest.raises(ValueError, match="unknown track"):
        ingest_database(ingestion_input(spatial_directory, database, labels=labels_path))


@pytest.mark.parametrize(
    "bad_column,bad_value", [("id", 1.5), ("id", -1), ("cluster", -2), ("cluster", np.nan)]
)
def test_invalid_labels_are_rejected(tmp_path: Path, bad_column: str, bad_value: float) -> None:
    labels = labels_for_fixture().astype({bad_column: float})
    labels.loc[0, bad_column] = bad_value
    path = tmp_path / "labels.csv"
    labels.to_csv(path, index=False)
    with pytest.raises(ValueError):
        read_labels(path)


def test_duplicate_label_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    pd.concat([labels_for_fixture(), labels_for_fixture().iloc[:1]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        read_labels(path)


def test_pair_metrics_and_singleton_noise() -> None:
    # Six pairs: TP (0,1), FP (0,2)/(1,2), and three TN; no positive noise pair.
    scores = score_assignments([0, 0, -1, -1], [8, 8, 8, -1])
    assert [scores[f"pair_{key}"] for key in ("tp", "fp", "fn", "tn")] == [1, 2, 0, 3]
    assert scores["pair_precision"] == pytest.approx(1 / 3)
    assert scores["pair_recall"] == 1
    assert scores["pair_f1"] == pytest.approx(0.5)
    assert scores["pair_false_positive_rate"] == pytest.approx(2 / 5)
    assert scores["outlier_detection_recall"] == 0.5
    assert score_assignments([0, 0, -1, -1], [9, 9, -1, -1])["adjusted_rand"] == 1
    all_noise = score_assignments([-1, -1, -1], [-1, -1, -1])
    assert all_noise["pair_tn"] == 3
    assert all_noise["pair_false_negative_rate"] is None
    assert all_noise["adjusted_rand"] == 1
    with pytest.raises(ValueError, match="equal one-dimensional"):
        score_assignments([0, 1], [0])


def test_evaluation_stores_reproducible_results_only_in_sqlite(
    spatial_directory: Path, ingestion_input
) -> None:
    labels_path = spatial_directory / "labels.csv"
    labels_for_fixture().to_csv(labels_path, index=False)
    output = spatial_directory / "output"
    database = output / "source.sqlite"
    ingest_database(ingestion_input(spatial_directory, database, labels=labels_path))
    with ClusterDatabase(database, writable=True) as source_db:
        assert (
            evaluate(source_db, qbx_eps=[0.1], dbscan_eps=[0.1], seeds=[0, 7], min_samples=2)
            is None
        )
        runs = source_db.select_frame(
            "v_evaluation_runs", where={"dataset": "source"}, order_by=("evaluation_result_id",)
        )
    assert runs.algorithm.tolist() == ["dbscan", "qbx", "qbx"]
    assert runs.tracks.tolist() == [4, 4, 4]
    assert list(output.iterdir()) == [database]
    with Database(database) as source_db:
        tracks = source_db.report("track_catalog")
    assert tracks.loc[tracks.eligible == 1, "source_track_id"].tolist() == [0, 1, 2, 3]
    assert tracks.geometry_occurrence.sum() == 1
    assert tracks.loc[tracks.eligible == 0, "source_track_id"].tolist() == [4]
    with Database(database) as stored:
        connection = stored.connection
        assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 3
        assert connection.execute("SELECT COUNT(*) FROM assignments").fetchone()[0] == 28
        assert connection.execute("SELECT COUNT(*) FROM evaluation_results").fetchone()[0] == 3
    with ClusterDatabase(database, writable=True) as source_db:
        evaluate(source_db, qbx_eps=[0.1], dbscan_eps=[0.1], seeds=[0, 7], min_samples=2)
        repeated = source_db.select_frame(
            "v_evaluation_runs", where={"dataset": "source"}, order_by=("evaluation_result_id",)
        )
    assert repeated.run_id.tolist() == runs.run_id.tolist()
    assert repeated.evaluation_result_id.tolist() == runs.evaluation_result_id.tolist()


@pytest.mark.parametrize("thresholds,seeds", [([0.1, 0.1], [0]), ([0.1], [0, 0])])
def test_duplicate_sweep_values_are_rejected(spatial_directory, ingestion_input, thresholds, seeds):
    from extensions.analytics.evaluation import evaluate as evaluate_one

    path = spatial_directory / "source.sqlite"
    ingest_database(ingestion_input(spatial_directory, path))
    with ClusterDatabase(path) as database, pytest.raises(ValidationError, match="duplicates"):
        evaluate_one(
            database, ClusteringSettings(algorithm="qbx", eps=0.1), eps=thresholds, seeds=seeds
        )


def test_qbx_empty_seeds_are_rejected_before_loading_runs(spatial_directory, ingestion_input):
    from extensions.analytics.evaluation import evaluate as evaluate_one

    path = spatial_directory / "source.sqlite"
    ingest_database(ingestion_input(spatial_directory, path))
    with ClusterDatabase(path) as database, pytest.raises(ValueError, match="at least one seed"):
        evaluate_one(database, ClusteringSettings(algorithm="qbx", eps=0.1), eps=[0.1], seeds=[])


def test_dbscan_empty_seeds_use_one_seed_zero_run(spatial_directory, ingestion_input):
    from extensions.analytics.evaluation import evaluate as evaluate_one

    labels = spatial_directory / "labels.csv"
    labels_for_fixture().to_csv(labels, index=False)
    path = spatial_directory / "source.sqlite"
    ingest_database(ingestion_input(spatial_directory, path, labels=labels))
    settings = ClusteringSettings(algorithm="dbscan", eps=0.1, seed=42)
    with ClusterDatabase(path, writable=True) as database:
        evaluate_one(database, settings, eps=[0.1], seeds=[])
        scores = database.select_frame("v_evaluation_runs")
        assert len(scores) == 1 and scores.seed.tolist() == [0]
        before = database.row_counts()
        evaluate_one(database, settings, eps=[0.1], seeds=[0])
        assert database.row_counts() == before


@pytest.mark.parametrize("thresholds", [None, []])
def test_explicit_thresholds_are_required_before_loading_runs(
    spatial_directory, ingestion_input, thresholds
):
    from extensions.analytics.evaluation import evaluate as evaluate_one

    path = spatial_directory / "source.sqlite"
    ingest_database(ingestion_input(spatial_directory, path))
    with ClusterDatabase(path) as database, pytest.raises(ValidationError):
        evaluate_one(database, ClusteringSettings(algorithm="dbscan", eps=0.5), eps=thresholds)


def test_sweep_settings_are_revalidated_before_loading_inputs(
    spatial_directory, ingestion_input, monkeypatch
):
    from extensions.analytics.evaluation import evaluate as evaluate_one

    path = spatial_directory / "source.sqlite"
    ingest_database(ingestion_input(spatial_directory, path))
    settings = ClusteringSettings(algorithm="qbx", eps=0.1, factor=2)
    with ClusterDatabase(path, writable=True) as database:
        monkeypatch.setattr(
            database,
            "input_signature",
            lambda: pytest.fail("Invalid sweep settings must fail before database reads"),
        )
        with pytest.raises(ValidationError, match="finite number"):
            evaluate_one(database, settings, eps=(0.5, 1e308), seeds=(0,))
        assert database.select_frame("runs").empty
    assert settings.eps == 0.1 and settings.seed == 0
