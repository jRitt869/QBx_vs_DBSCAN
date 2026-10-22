"""Distance calibration, shared spatial geometry, and persisted metric identities."""

import hashlib
import json
from datetime import UTC, datetime
from functools import partial

import gpxpy
import numpy as np
import pandas as pd
import pytest
from bokeh.document import Document

from extensions.analytics.dashboard import build_document
from extensions.analytics.evaluation import ReferenceConfiguration, evaluate, run_clustering
from extensions.analytics.storage import ClusterDatabase
from extensions.analytics.temporal import (
    SpatiotemporalMetric,
    fit_distance_scales,
    prepare_temporal_features,
    spatiotemporal_distance,
)
from extensions.analytics.warehouse import ingest_database
from track_clusters.clustering import get_dbscan_clusters
from track_clusters.validation import ClusteringSettings


def _tracks(positions=(0, 1, 4)):
    start = datetime(2024, 1, 8, tzinfo=UTC).timestamp()
    return pd.DataFrame(
        {"13_pts": [np.tile([x, 2 * x, start + x * 3600], (13, 1)) for x in positions]}
    )


@pytest.mark.parametrize(
    "timed,p",
    [
        (False, 1),
        (False, 2),
        (False, 3),
        (True, 1),
        (True, 2),
        (True, 3),
    ],
)
def test_scales_match_median_nonzero_track_distances(timed, p):
    tracks = _tracks()
    scales = fit_distance_scales(tracks, joint=timed, p=p, time_zone=UTC)
    assert scales["spatial"] == pytest.approx(1 / (3 * np.sqrt(5)))
    assert scales["sampled_pairs"] == scales["spatial_nonzero_pairs"] == 3
    if timed:
        assert scales["temporal"] == pytest.approx(4 / (1 + 7**-p + 31**-p) ** (1 / p))
        assert scales["temporal_nonzero_pairs"] == 3
        packed = prepare_temporal_features(tracks, algorithm="qbx", time_zone=UTC, joint=True)
        metric = SpatiotemporalMetric(
            p, spatial_scale=scales["spatial"], temporal_scale=scales["temporal"]
        )
        assert metric.dist(*packed["13_pts"].iloc[:2]) == pytest.approx(2 ** (1 / p) / 3)
    else:
        assert scales["temporal"] == 0 and scales["temporal_nonzero_pairs"] == 0


def test_joint_metric_is_invariant_to_uniform_spatial_units():
    tracks = _tracks()
    transformed = tracks.copy()
    transformed["13_pts"] = tracks["13_pts"].map(
        lambda values: values * [1000, 1000, 1] + [50, -20, 0]
    )
    scales = [
        fit_distance_scales(frame, joint=True, p=2, time_zone=UTC)
        for frame in (tracks, transformed)
    ]
    assert scales[1]["spatial"] == pytest.approx(scales[0]["spatial"] / 1000)
    distances = []
    for frame, scale in zip((tracks, transformed), scales, strict=True):
        first, second = (
            frame["13_pts"]
            .iloc[:2]
            .map(lambda values: np.column_stack((values[:, :2], np.full((13, 3), 0.1))))
        )
        second[:, 2:] = 0.2
        distances.append(
            spatiotemporal_distance(
                first, second, spatial_scale=scale["spatial"], temporal_scale=scale["temporal"]
            )
        )
    assert distances[0] == pytest.approx(distances[1])


def test_dbscan_shared_scalar_preserves_coordinate_ratios(monkeypatch):
    import track_clusters.clustering as clustering

    tracks = _tracks(range(6))
    tracks["13_pts"] = tracks["13_pts"].map(lambda values: values[:, :2])
    scale = fit_distance_scales(tracks)["spatial"]
    captured = []
    metrics = []

    class ObservedDBSCAN:
        def __init__(self, **kwargs):
            metrics.append(kwargs["metric"])

        def fit_predict(self, values):
            captured.append(values)
            return np.zeros(len(values), dtype=int)

    def unexpected(*args, **kwargs):
        pytest.fail("Adaptive DBSCAN must not use per-coordinate StandardScaler")

    monkeypatch.setattr(clustering, "StandardScaler", unexpected, raising=False)
    monkeypatch.setattr(clustering, "DBSCAN", ObservedDBSCAN)
    get_dbscan_clusters(
        tracks, 0.5, metric=partial(spatiotemporal_distance, spatial_scale=scale, temporal_scale=0)
    )
    expected = np.stack(tracks["13_pts"]).reshape(6, -1)
    np.testing.assert_array_equal(captured[0], expected)
    assert np.std(captured[0][:, 1]) == pytest.approx(2 * np.std(captured[0][:, 0]))
    assert metrics[0](captured[0][0], captured[0][1]) == pytest.approx(np.sqrt(5) * scale)


@pytest.mark.parametrize("mode", ["spatial", "separate", "joint"])
def test_qbx_centering_preserves_normalized_distances_in_float32_centroids(
    track_directory, ingestion_input, monkeypatch, mode
):
    import extensions.analytics.evaluation as pipeline

    source = track_directory / "tracks.gpx"
    gpx = gpxpy.parse(source.read_text(encoding="utf-8"))
    for track_index, track in enumerate(gpx.tracks):
        for segment in track.segments:
            for point_index, point in enumerate(segment.points):
                point.longitude = 8 + point_index * 0.0001 + track_index * 1e-7
                point.latitude = 50
                point.time = datetime(2024, 1, 8, tzinfo=UTC)
    source.write_text(gpx.to_xml(), encoding="utf-8")
    path = track_directory / "precision.sqlite"
    ingest_database(ingestion_input(track_directory, path))
    real = pipeline.get_qb_clusters
    observed = []

    def checked(tracks, *args, **kwargs):
        metric = kwargs["metric"]
        first, second = tracks["13_pts"].iloc[[0, -1]]
        expected = (
            metric.dist(first, second)
            if mode == "joint"
            else np.linalg.norm(first - second, axis=1).mean()
        )
        assert expected > 0
        assert metric.dist(first.astype(np.float32), second.astype(np.float32)) == pytest.approx(
            expected, rel=1e-5
        )
        observed.append(True)
        return real(tracks, *args, **kwargs)

    monkeypatch.setattr(pipeline, "get_qb_clusters", checked)
    with ClusterDatabase(path, writable=True) as database:
        run_clustering(
            database,
            ClusteringSettings(
                algorithm="qbx",
                eps=0.5,
                feature_mode="spatial" if mode == "spatial" else "spatiotemporal",
                temporal_mode="joint" if mode == "joint" else "separate",
            ),
        )
    assert observed == [True]


@pytest.mark.parametrize("positions", [(0,), (0, 0, 0), (0, 0, 4)])
def test_zero_distances_have_explicit_finite_fallbacks(positions):
    tracks = _tracks(positions)
    scales = fit_distance_scales(tracks, joint=True, p=2, time_zone=UTC)
    if max(positions) == 0:
        assert scales["spatial"] == scales["temporal"] == 1
        assert scales["spatial_nonzero_pairs"] == scales["temporal_nonzero_pairs"] == 0
    else:
        assert scales["spatial_nonzero_pairs"] == scales["temporal_nonzero_pairs"] == 2
        assert scales["spatial"] == pytest.approx(1 / (4 * np.sqrt(5)))


def test_pair_sampling_is_deterministic_and_does_not_change_global_rng():
    rng = np.random.default_rng(42)
    tracks = pd.DataFrame({"13_pts": list(rng.normal(size=(1000, 13, 2)))})
    before = np.random.get_state()
    first = fit_distance_scales(tracks, pair_limit=73, seed=21)
    second = fit_distance_scales(tracks, pair_limit=73, seed=21)
    other = fit_distance_scales(tracks, pair_limit=73, seed=22)
    after = np.random.get_state()
    assert first == second and first["sampled_pairs"] == 73
    assert first["spatial"] != other["spatial"]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize("algorithm", ["qbx", "dbscan"])
@pytest.mark.parametrize("timed", [False, True])
def test_constant_tracks_have_finite_pipeline_scales(
    track_directory, ingestion_input, algorithm, timed
):
    source = track_directory / "tracks.gpx"
    gpx = gpxpy.parse(source.read_text(encoding="utf-8"))
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                point.longitude, point.latitude = 8, 50
                point.time = datetime(2024, 1, 8, tzinfo=UTC)
    source.write_text(gpx.to_xml(), encoding="utf-8")
    path = track_directory / "constant.sqlite"
    ingest_database(ingestion_input(track_directory, path))
    with ClusterDatabase(path, writable=True) as database:
        run_id = run_clustering(
            database,
            ClusteringSettings(
                algorithm=algorithm,
                eps=0.5,
                feature_mode="spatiotemporal" if timed else "spatial",
                temporal_mode="joint",
            ),
        )
        provenance = database.select_exec(
            "runs", ("provenance",), where={"run_id": run_id}
        ).fetchone()[0]
        scales = json.loads(provenance)["distance_scales"]
        assert scales["spatial"] == 1 and scales["spatial_nonzero_pairs"] == 0
        if timed:
            assert scales["temporal"] == 1 and scales["temporal_nonzero_pairs"] == 0
        assert set(database.load_tracks(run_id).clusters0) == {0}


@pytest.mark.parametrize("algorithm", ["dbscan", "qbx"])
def test_shared_seed_controls_calibration_and_qbx_ordering(
    track_directory, ingestion_input, monkeypatch, algorithm
):
    import extensions.analytics.evaluation as pipeline

    path = track_directory / "calibrated.sqlite"
    ingest_database(ingestion_input(track_directory, path))
    sampled_seeds, ordered_seeds = [], []
    fit = pipeline.fit_distance_scales
    cluster = pipeline.get_qb_clusters

    def observed_fit(*args, **kwargs):
        sampled_seeds.append(kwargs["seed"])
        return fit(*args, **kwargs)

    def observed_cluster(tracks, eps, eps_time, factor, seed, **kwargs):
        ordered_seeds.append(seed)
        return cluster(tracks, eps, eps_time, factor, seed, **kwargs)

    monkeypatch.setattr(pipeline, "fit_distance_scales", observed_fit)
    monkeypatch.setattr(pipeline, "get_qb_clusters", observed_cluster)
    with ClusterDatabase(path, writable=True) as database:
        ids = [
            run_clustering(
                database,
                ClusteringSettings(
                    algorithm=algorithm,
                    eps=eps,
                    feature_mode="spatiotemporal",
                    seed=seed,
                    scale_pairs=17,
                ),
            )
            for eps in (0.5, 1)
            for seed in (0, 7)
        ]
        assert len(set(ids)) == 4
        assert sampled_seeds == [0, 7, 0, 7]
        assert ordered_seeds == ([0, 7, 0, 7] if algorithm == "qbx" else [])
        runs = database.select_frame("runs", order_by=("run_id",))
        scales = [json.loads(value)["distance_scales"] for value in runs.provenance]
        assert scales[0] == scales[2] and scales[1] == scales[3]
        assert all(value["sampled_pairs"] == 17 and value["spatial"] > 0 for value in scales)
        settings = json.loads(runs.iloc[0].settings_json)
        assert settings["scale_pairs"] == 17 and settings["seed"] == 0
        assert (
            hashlib.sha256(runs.iloc[0].settings_json.encode()).hexdigest()
            == runs.iloc[0].config_key
        )

        def unexpected(*args, **kwargs):
            pytest.fail("Cached runs must return before recalibration or loading inputs")

        monkeypatch.setattr(database, "load_tracks", unexpected)
        assert run_clustering(database, ClusteringSettings(**settings)) == ids[0]
        assert sampled_seeds == [0, 7, 0, 7]


@pytest.mark.parametrize("algorithm", ["dbscan", "qbx"])
def test_calibration_pair_limits_are_separate_report_profiles(
    track_directory, ingestion_input, algorithm
):
    path = track_directory / "profiles.sqlite"
    ingest_database(ingestion_input(track_directory, path))
    reference = ReferenceConfiguration(algorithm="qbx", eps=0.25)
    with ClusterDatabase(path, writable=True) as database:
        for limit in (17, 31):
            evaluate(
                database,
                ClusteringSettings(algorithm=algorithm, scale_pairs=limit, eps=0.5),
                eps=(0.5,),
                seeds=(0,),
                reference=reference,
            )
        summary = database.report("parameter_sweep")
        assert set(summary.scale_pairs) == {17, 31}
        assert summary.runs.tolist() == [1, 1]
        selected = database.report("selected_models").iloc[0]
        scored = database.select_frame(
            "v_evaluation_runs",
            where={"evaluation_result_id": int(selected.representative_result_id)},
        ).iloc[0]
        assert selected.scale_pairs == scored.scale_pairs
        database.validate()
    app = build_document(Document(), path)
    labels = app.profile_selectors[algorithm].options
    assert any("17 calibration pairs" in label for label in labels)
    assert any("31 calibration pairs" in label for label in labels)
