"""Cyclic calendar metrics, circular centroids, and persisted run configurations."""

import json
import re
from datetime import UTC, datetime, timedelta
from functools import partial
from zoneinfo import ZoneInfo

import geopandas as gpd
import gpxpy
import numpy as np
import pandas as pd
import pytest
from bokeh.document import Document
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from pydantic import ValidationError

from extensions.analytics.dashboard import build_document
from extensions.analytics.evaluation import ReferenceConfiguration, evaluate, run_clustering
from extensions.analytics.storage import ClusterDatabase
from extensions.analytics.temporal import (
    SpatiotemporalMetric,
    TorusMetric,
    calendar_phases,
    prepare_temporal_features,
    resample_spatial_track,
    spatiotemporal_distance,
    torus_distance,
)
from extensions.analytics.warehouse import ingest_database
from scripts.config.parser import build_parser
from scripts.config.validate import ClusteringInput, EvaluationInput, validate_cli
from scripts.evaluate import main as evaluate_main
from track_clusters.clustering import get_dbscan_clusters, get_qb_clusters
from track_clusters.validation import ClusteringSettings


def test_spatial_resampling_aligns_epochs_without_changing_sample_positions():
    coordinates = [[0, 0], [1, 0], [1, 1], [3, 1]]
    start = datetime(2024, 1, 8, tzinfo=UTC).timestamp()
    first = resample_spatial_track(coordinates, start + np.array([0, 60, 600, 660]), 5)
    second = resample_spatial_track(coordinates, start + np.array([0, 600, 630, 660]), 5)
    np.testing.assert_allclose(first[:, :2], [[0, 0], [1, 0], [1, 1], [2, 1], [3, 1]])
    np.testing.assert_array_equal(first[:, :2], second[:, :2])
    np.testing.assert_array_equal(first[:, :2], resample_spatial_track(coordinates, point_count=5))
    np.testing.assert_allclose(first[:, 2] - start, [0, 60, 600, 630, 660], rtol=0, atol=1e-6)
    assert not np.array_equal(first[:, 2], second[:, 2])


def test_spatial_resampling_preserves_stationary_tracks_time_profile():
    coordinates = np.tile([8, 50], (5, 1))
    sampled = resample_spatial_track(coordinates, [0, 10, 20, 30, 40])
    np.testing.assert_array_equal(sampled[:, :2], np.tile([8, 50], (13, 1)))
    np.testing.assert_array_equal(sampled[:, :2], resample_spatial_track(coordinates))
    np.testing.assert_allclose(sampled[:, 2], np.linspace(0, 40, 13))


def test_spatial_resampling_handles_pauses_and_retains_endpoint_epochs():
    sampled = resample_spatial_track(
        [[0, 0], [0, 0], [1, 0], [1, 0], [2, 0]], [0, 20, 30, 40, 60], 5
    )
    np.testing.assert_allclose(sampled[:, 2], [0, 25, 40, 50, 60])


@pytest.mark.parametrize("pause_times", [[10, 40], [10, 20, 30, 40]])
def test_spatial_resampling_keeps_arrival_and_departure_sides_of_pause(pause_times):
    coordinates = [[0, 0], *([[1, 0]] * len(pause_times)), [2, 0]]
    sampled = resample_spatial_track(coordinates, [0, *pause_times, 50], 9)
    np.testing.assert_allclose(sampled[:, 0], np.linspace(0, 2, 9))
    np.testing.assert_allclose(sampled[:, 2], [0, 2.5, 5, 7.5, 40, 42.5, 45, 47.5, 50])
    np.testing.assert_array_equal(
        sampled[:, :2], resample_spatial_track(coordinates, point_count=9)
    )


def test_spatial_resampling_approaches_final_pause_at_arrival_time():
    sampled = resample_spatial_track([[0, 0], [1, 0], [2, 0], [2, 0]], [0, 10, 20, 60], 5)
    np.testing.assert_allclose(sampled[:, 2], [0, 5, 10, 15, 60])


def test_stationary_resampling_retains_irregular_recording_intervals():
    sampled = resample_spatial_track(np.tile([8, 50], (5, 1)), [0, 1, 11, 12, 40], 9)
    np.testing.assert_allclose(sampled[:, 2], [0, 0.5, 1, 6, 11, 11.5, 12, 26, 40])


@pytest.mark.parametrize("stationary", [False, True])
@pytest.mark.parametrize("algorithm", ["dbscan", "qbx"])
def test_separate_spatial_results_ignore_recording_schedule(
    track_directory, ingestion_input, monkeypatch, stationary, algorithm
):
    import extensions.analytics.evaluation as pipeline

    path = track_directory / "separated.sqlite"
    source = track_directory / "tracks.gpx"
    if stationary:
        gpx = gpxpy.parse(source.read_text(encoding="utf-8"))
        for point in gpx.tracks[0].segments[0].points:
            point.longitude, point.latitude = 13.35, 52.5
        source.write_text(gpx.to_xml(), encoding="utf-8")
    inputs = ingestion_input(track_directory, path)
    ingest_database(inputs)
    prepared, temporal = [], []
    real_cluster = (
        pipeline.get_dbscan_clusters if algorithm == "dbscan" else pipeline.get_qb_clusters
    )

    def observed(tracks, *args, **kwargs):
        prepared.append(np.stack(tracks["13_pts"]))
        if "times" in tracks:
            temporal.append(np.stack(tracks.times))
        return real_cluster(tracks, *args, **kwargs)

    monkeypatch.setattr(
        pipeline, "get_dbscan_clusters" if algorithm == "dbscan" else "get_qb_clusters", observed
    )
    with ClusterDatabase(path, writable=True) as database:
        spatial = run_clustering(database, ClusteringSettings(algorithm=algorithm, eps=0.5))
        timed = run_clustering(
            database,
            ClusteringSettings(algorithm=algorithm, eps=0.5, feature_mode="spatiotemporal"),
        )
        expected = database.load_tracks(spatial).clusters0.tolist()
        assert database.load_tracks(timed).clusters0.tolist() == expected
        provenance = database.select_exec(
            "runs", ("provenance",), where={"run_id": spatial}
        ).fetchone()[0]
        center = np.asarray(json.loads(provenance)["distance_scales"]["spatial_center"])
    gpx = gpxpy.parse(source.read_text(encoding="utf-8"))
    for track in gpx.tracks:
        points = [point for segment in track.segments for point in segment.points]
        start = points[0].time
        for index, point in enumerate(points):
            point.time = start + timedelta(seconds=index * index * 20 + index)
    source.write_text(gpx.to_xml(), encoding="utf-8")
    ingest_database(inputs, on_conflict="replace")
    with ClusterDatabase(path, writable=True) as database:
        timed = run_clustering(
            database,
            ClusteringSettings(algorithm=algorithm, eps=0.5, feature_mode="spatiotemporal", p=1),
        )
        assert database.load_tracks(timed).clusters0.tolist() == expected
    assert len(prepared) == 3
    assert all(values.shape == (25, 13, 2 if algorithm == "dbscan" else 3) for values in prepared)
    np.testing.assert_array_equal(prepared[0], prepared[1])
    np.testing.assert_array_equal(prepared[0], prepared[2])
    if stationary:
        np.testing.assert_allclose(
            prepared[0][0, :, :2] + center
            if algorithm == "dbscan"
            else prepared[0][0, :, :2] / json.loads(provenance)["distance_scales"]["spatial"]
            + center,
            np.tile([13.35, 52.5], (13, 1)),
        )
    assert not np.array_equal(temporal[0], temporal[1])


@pytest.mark.parametrize(
    "moment, expected",
    [
        (datetime(2024, 1, 1, tzinfo=UTC), [0, 0, 0]),
        (datetime(2024, 1, 31, 12, tzinfo=UTC), [0.5, 2.5 / 7, 30.5 / 31]),
        (datetime(2024, 2, 29, 12, tzinfo=UTC), [0.5, 3.5 / 7, 28.5 / 29]),
        (datetime(2024, 3, 1, tzinfo=UTC), [0, 4 / 7, 0]),
    ],
)
def test_calendar_phases_follow_actual_calendar_months(moment, expected):
    np.testing.assert_allclose(calendar_phases([moment.timestamp()], UTC)[0], expected)


def test_calendar_phases_use_local_wall_time_and_subseconds():
    moment = datetime(2024, 1, 7, 23, 30, 0, 500000, tzinfo=UTC)
    phases = calendar_phases([moment.timestamp()], ZoneInfo("Europe/Berlin"))[0]
    day = (1800 + 0.5) / 86400
    np.testing.assert_allclose(phases, [day, day / 7, (7 + day) / 31])
    # Both sides of the autumn clock change represent the same wall-clock phase.
    repeated = [datetime(2024, 10, 27, hour, 30, tzinfo=UTC).timestamp() for hour in (0, 1)]
    phases = calendar_phases(repeated, ZoneInfo("Europe/Berlin"))
    np.testing.assert_allclose(phases[0], phases[1])


@pytest.mark.parametrize("p", [1, 2, 3])
def test_joint_metric_matches_formula_and_wraps(p):
    first = np.tile([0, 0, 0.95, 0.95, 0.95], (13, 1))
    second = np.tile([3, 4, 0.05, 0.05, 0.05], (13, 1))
    expected = (5**p + 3 * 0.2**p) ** (1 / p)
    assert spatiotemporal_distance(first, second, p=p) == pytest.approx(expected)
    assert spatiotemporal_distance(second.ravel(), first.ravel(), p=p) == pytest.approx(expected)
    assert spatiotemporal_distance(first, first, p=p) == 0


@pytest.mark.parametrize("p", [1, 2, 3])
def test_torus_combines_normalized_angles_and_uses_all_samples(p):
    first = np.tile([0.99, 0.99, 0.99], (13, 1))
    second = np.tile([0.01, 0.01, 0.01], (13, 1))
    second[0] = first[0]
    expected = 12 / 13 * 0.04 * 3 ** (1 / p)
    assert torus_distance(first, second, p=p) == pytest.approx(expected)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_metrics_average_pointwise_distances(p):
    first = np.zeros((13, 5))
    second = first.copy()
    second[0, 0], second[1, 0] = 3, 4
    assert spatiotemporal_distance(first, second, p=p) == pytest.approx(7 / 13)
    # Combine space and time at each point before averaging across the track.
    second[:] = 0
    second[0, :2] = [3, 4]
    second[1, 2] = 0.5
    assert spatiotemporal_distance(first, second, p=p) == pytest.approx(6 / 13)
    assert spatiotemporal_distance(
        np.repeat(first, 2, axis=0), np.repeat(second, 2, axis=0), p=p
    ) == pytest.approx(6 / 13)
    first = np.zeros((13, 3))
    second = first.copy()
    second[0, 0], second[1, 1] = 0.1, 0.2
    assert torus_distance(first, second, p=p) == pytest.approx(0.6 / 13)
    assert torus_distance(
        np.repeat(first, 2, axis=0), np.repeat(second, 2, axis=0), p=p
    ) == pytest.approx(0.6 / 13)
    # Identical mean phase coordinates must not erase different sample orderings.
    reordered = second[::-1].copy()
    np.testing.assert_allclose(second.mean(axis=0), reordered.mean(axis=0))
    assert torus_distance(second, reordered, p=p) > 0


@pytest.mark.parametrize("p", [None, 1, 3])
@pytest.mark.parametrize("with_phases", [False, True])
def test_zero_temporal_weight_uses_only_spatial_distances(p, with_phases):
    first = np.zeros((13, 5 if with_phases else 2))
    second = first.copy()
    second[:, :2] = [3, 4]
    if with_phases:
        first[:, 2:], second[:, 2:] = np.nan, np.inf
    with np.errstate(all="raise"):
        assert spatiotemporal_distance(
            first, second, p=p, spatial_scale=0.2, temporal_scale=0
        ) == pytest.approx(1)


def test_metric_power_does_not_overflow():
    assert spatiotemporal_distance([0, 0, 0, 0, 0], [3, 4, 0.5, 0.5, 0.5], p=1e308) == 5
    assert torus_distance([0, 0, 0], [0.5, 0.5, 0.5], p=1e308) == 1


def test_qbx_circular_centroids_stay_near_wrap_boundary():
    moments = [datetime(2024, 1, 7, 23, 45, tzinfo=UTC), datetime(2024, 1, 8, 0, 15, tzinfo=UTC)]
    tracks = pd.DataFrame(
        {"13_pts": [np.tile([0, 0, moment.timestamp()], (13, 1)) for moment in moments]}
    )
    packed = prepare_temporal_features(tracks, algorithm="qbx", time_zone=UTC, joint=True)
    joint_metric, time_metric = SpatiotemporalMetric(2), TorusMetric(2)
    first, second = packed["13_pts"]
    midpoint = ((first + second) / 2).astype(np.float32)
    boundary = pd.DataFrame(
        {"13_pts": [np.tile([0, 0, datetime(2024, 1, 8, tzinfo=UTC).timestamp()], (13, 1))]}
    )
    target = prepare_temporal_features(boundary, algorithm="qbx", time_zone=UTC, joint=True)
    assert joint_metric.are_compatible(first.shape, second.shape)
    assert joint_metric.dist(midpoint, target["13_pts"].iloc[0]) < 1e-6
    temporal_midpoint = (first[13:] + second[13:]) / 2
    assert time_metric.dist(temporal_midpoint, target["13_pts"].iloc[0][13:]) < 1e-6
    assert time_metric.are_compatible((39, 3), (39, 3))
    assert not time_metric.are_compatible((1, 39), (1, 39))
    assert joint_metric.dist(first, second) == pytest.approx(
        np.sqrt(1 / 24**2 + 1 / 168**2 + 1 / 744**2)
    )
    # Circular directions with a zero resultant have a deterministic phase-zero tie.
    midpoint[13:] = 0
    phase_zero = target["13_pts"].iloc[0].copy()
    phase_zero[13:] = [1, 0, 0]
    assert joint_metric.dist(midpoint, phase_zero) == 0


@pytest.mark.parametrize("algorithm", ["dbscan", "qbx"])
@pytest.mark.parametrize("joint", [False, True])
def test_joint_mode_groups_by_time_without_creating_an_extra_layer(algorithm, joint):
    from extensions.analytics.temporal import fit_distance_scales

    start = datetime(2024, 1, 8, tzinfo=UTC).timestamp()
    # Identical paths and time-of-day, with different week/month phases.
    raw = gpd.GeoDataFrame(
        {"13_pts": [np.tile([0, 0, start + offset], (13, 1)) for offset in [0] * 6 + [86400] * 6]}
    )
    scales = fit_distance_scales(raw, joint=joint, time_zone=UTC)
    features = prepare_temporal_features(raw, algorithm=algorithm, time_zone=UTC, joint=joint)
    if algorithm == "qbx":
        assert features["13_pts"].iloc[0].shape == ((52, 3) if joint else (13, 3))
        result = get_qb_clusters(
            features,
            0.5,
            None if joint else 0.1,
            factor=1,
            metric=SpatiotemporalMetric(
                2, spatial_scale=scales["spatial"], temporal_scale=scales["temporal"]
            )
            if joint
            else AveragePointwiseEuclideanMetric(),
            temporal_metric=None if joint else TorusMetric(2),
        )
        primary_layers = 3
    else:
        assert features["13_pts"].iloc[0].shape == ((13, 5) if joint else (13, 2))
        result = get_dbscan_clusters(
            features,
            0.5,
            None if joint else 0.1,
            metric=partial(
                spatiotemporal_distance,
                spatial_scale=scales["spatial"],
                temporal_scale=scales["temporal"],
            ),
            temporal_metric=None if joint else torus_distance,
        )
        primary_layers = 1
    assert result.clusters0.nunique() == (2 if joint else 1)
    assert len([column for column in result if column.startswith("clusters")]) == primary_layers + (
        not joint
    )
    selected = result[f"clusters{primary_layers - 1 if joint else primary_layers}"]
    assert selected.iloc[:6].nunique() == selected.iloc[6:].nunique() == 1
    assert selected.iloc[0] != selected.iloc[6] and (selected >= 0).all()


@pytest.mark.parametrize("algorithm", ["dbscan", "qbx"])
def test_temporal_subdivision_uses_later_samples_for_both_algorithms(algorithm):
    start = datetime(2024, 1, 8, 8, tzinfo=UTC).timestamp()
    tracks = gpd.GeoDataFrame(
        {
            "13_pts": [
                np.column_stack((np.zeros((13, 2)), [start, *([start + offset] * 12)]))
                for offset in [0] * 6 + [10 * 3600] * 6
            ]
        }
    )
    features = prepare_temporal_features(tracks, algorithm=algorithm, time_zone=UTC)
    if algorithm == "qbx":
        result = get_qb_clusters(
            features,
            100,
            0.1,
            temporal_metric=TorusMetric(2),
            metric=AveragePointwiseEuclideanMetric(),
        )
        temporal = result.clusters3
    else:
        result = get_dbscan_clusters(
            features,
            100,
            0.1,
            temporal_metric=partial(torus_distance, p=2),
            metric=partial(spatiotemporal_distance, temporal_scale=0),
        )
        temporal = result.clusters1
    assert result.clusters0.nunique() == 1
    assert temporal.iloc[:6].nunique() == temporal.iloc[6:].nunique() == 1
    assert temporal.iloc[0] != temporal.iloc[6] and (temporal >= 0).all()


@pytest.mark.parametrize(
    "settings",
    [
        {"p": 0.5},
        {"p": 0},
        {"p": float("inf")},
        {"p": float("nan")},
        {"temporal_mode": "unknown"},
        {"temporal_mode": None},
        {"p": None},
    ],
)
def test_cyclic_settings_reject_invalid_parameters(settings):
    with pytest.raises(ValidationError):
        ClusteringSettings(algorithm="qbx", eps=0.5, feature_mode="spatiotemporal", **settings)


@pytest.mark.parametrize(
    "command,model,positionals",
    [("get_clusters", ClusteringInput, ["0.5"]), ("evaluate", EvaluationInput, [])],
)
def test_cli_exposes_joint_mode_and_power(command, model, positionals, tmp_path):
    path = tmp_path / "source.sqlite"
    path.touch()
    parser = build_parser(command)
    args = validate_cli(
        parser,
        model,
        parser.parse_args([str(path), *positionals, "--spatiotemporal", "--joint", "--p", "3"]),
    )
    assert args.temporal_mode == "joint" and args.p == 3
    assert "--period" not in parser.format_help()


@pytest.mark.parametrize("algorithm", ["dbscan", "qbx"])
def test_cyclic_run_keys_include_mode_and_power(track_directory, ingestion_input, algorithm):
    path = track_directory / "cyclic.sqlite"
    ingest_database(ingestion_input(track_directory, path))
    with ClusterDatabase(path, writable=True) as database:
        configurations = [("separate", 2), ("joint", 2), ("separate", 1), ("joint", 1)]
        ids = [
            run_clustering(
                database,
                ClusteringSettings(
                    algorithm=algorithm,
                    eps=0.5,
                    feature_mode="spatiotemporal",
                    temporal_mode=mode,
                    p=p,
                ),
            )
            for mode, p in configurations
        ]
        assert len(set(ids)) == 4
        for run_id, (mode, p) in zip(ids, configurations, strict=True):
            settings = json.loads(
                database.select_exec(
                    "runs", ("settings_json",), where={"run_id": run_id}
                ).fetchone()[0]
            )
            assert settings["temporal_mode"] == mode and settings["p"] == p
            assert settings["eps_time"] == (0.1 if mode == "separate" else None)
            layer_count = (1 if algorithm == "dbscan" else 3) + (mode == "separate")
            assert len(database.load_layer_names(run_id)) == layer_count
        assert (
            run_clustering(
                database,
                ClusteringSettings(algorithm=algorithm, eps=0.5, feature_mode="spatiotemporal"),
            )
            == ids[0]
        )
        assert (
            run_clustering(
                database,
                ClusteringSettings(
                    algorithm=algorithm,
                    eps=0.5,
                    feature_mode="spatiotemporal",
                    temporal_mode="joint",
                    eps_time=999,
                ),
            )
            == ids[1]
        )
        spatial = run_clustering(database, ClusteringSettings(algorithm=algorithm, eps=0.5))
        assert (
            run_clustering(
                database,
                ClusteringSettings(algorithm=algorithm, eps=0.5, temporal_mode="joint", p=3),
            )
            == spatial
        )


def test_evaluation_reports_and_dashboard_keep_metric_profiles_separate(
    track_directory, ingestion_input
):
    path = track_directory / "evaluated.sqlite"
    ingest_database(ingestion_input(track_directory, path))
    reference = ReferenceConfiguration(
        algorithm="qbx", eps=0.5, feature_mode="spatiotemporal", p=3, layer=4
    )
    with ClusterDatabase(path, writable=True) as database:
        for mode in ("separate", "joint"):
            evaluate(
                database,
                ClusteringSettings(
                    algorithm="dbscan", feature_mode="spatiotemporal", temporal_mode=mode, eps=0.5
                ),
                eps=(0.5,),
                reference=reference,
            )
        summary = database.report("parameter_sweep")
        assert len(summary) == 2 and summary.runs.tolist() == [1, 1]
        assert set(summary.temporal_mode) == {"separate", "joint"}
        assert set(summary.p) == {2}
        manifest = json.loads(database.select_exec("evaluations", ("manifest_json",)).fetchone()[0])
        assert manifest["reference_configuration"]["temporal_mode"] == "separate"
        assert manifest["reference_configuration"]["p"] == 3
    app = build_document(Document(), path)
    profiles = app.profile_selectors["dbscan"].options
    assert any("separate · p 2" in name for name in profiles)
    assert any("joint · p 2" in name for name in profiles)


@pytest.mark.parametrize("algorithm", ["dbscan", "qbx"])
@pytest.mark.parametrize("temporal_mode", ["separate", "joint"])
def test_evaluation_passes_settings_copies_to_clustering(
    track_directory, ingestion_input, monkeypatch, algorithm, temporal_mode
):
    import extensions.analytics.evaluation as module

    path = track_directory / "sweep.sqlite"
    ingest_database(ingestion_input(track_directory, path))
    settings = ClusteringSettings(
        algorithm=algorithm,
        eps=0.25,
        seed=42,
        feature_mode="spatiotemporal",
        temporal_mode=temporal_mode,
        timezone="Europe/Berlin",
        eps_time=0.3,
        p=3,
        min_samples=2,
        factor=3,
        scale_pairs=17,
    )
    original = settings.model_dump()
    reference = ReferenceConfiguration(algorithm="qbx", eps=0.25, seed=9, layer=1)
    reference_original = reference.model_dump()
    configurations = []
    real = module.run_clustering

    def recorded(database, configuration):
        assert isinstance(configuration, ClusteringSettings)
        configurations.append(configuration)
        return real(database, configuration)

    monkeypatch.setattr(module, "run_clustering", recorded)
    with ClusterDatabase(path, writable=True) as database:
        evaluate(database, settings, eps=(0.5, 1), seeds=(7, 3), reference=reference)
        assert configurations[0].model_dump() == reference.model_dump(exclude={"layer"})
        expected = [
            {**original, "eps": threshold, "seed": seed}
            for threshold in (0.5, 1)
            for seed in ((7, 3) if algorithm == "qbx" else (0,))
        ]
        assert [configuration.model_dump() for configuration in configurations[1:]] == expected
        stored = database.select_frame("runs", order_by=("run_id",))
        assert [json.loads(value) for value in stored.settings_json] == [
            reference.model_dump(exclude={"layer"}),
            *expected,
        ]
        assert real(database, configurations[0]) == stored.iloc[0].run_id
    assert settings.model_dump() == original
    assert reference.model_dump() == reference_original


def test_qbx_groups_across_midnight_with_circular_centroids():
    start = datetime(2024, 1, 8, tzinfo=UTC)
    moments = [start + timedelta(minutes=offset) for offset in [-15, 15, 0] * 4]
    tracks = gpd.GeoDataFrame(
        {"13_pts": [np.tile([0, 0, moment.timestamp()], (13, 1)) for moment in moments]}
    )
    prepared = prepare_temporal_features(tracks, algorithm="qbx", time_zone=UTC, joint=True)
    result = get_qb_clusters(prepared, 0.2, factor=1, metric=SpatiotemporalMetric(2))
    assert all((result[f"clusters{layer}"] == 0).all() for layer in range(3))


def test_timed_reports_and_dashboard_require_an_explicit_mode(track_directory, ingestion_input):
    path = track_directory / "recorded.sqlite"
    ingest_database(ingestion_input(track_directory, path))
    reference = ReferenceConfiguration(algorithm="qbx", eps=0.5)
    with ClusterDatabase(path, writable=True) as database:
        evaluate(
            database,
            ClusteringSettings(algorithm="dbscan", feature_mode="spatiotemporal", eps=0.5),
            eps=(0.5,),
            reference=reference,
        )
        recorded = database.select_frame("runs", where={"algorithm": "dbscan"}).iloc[0]
        settings = json.loads(recorded.settings_json)
        del settings["temporal_mode"]
        with database.transaction():
            database.update(
                "runs",
                {"settings_json": json.dumps(settings), "config_key": "recorded-without-mode"},
                where={"run_id": int(recorded.run_id)},
            )
        assert database.report("parameter_sweep").empty
        evaluate(
            database,
            ClusteringSettings(
                algorithm="dbscan", feature_mode="spatiotemporal", temporal_mode="joint", eps=0.5
            ),
            eps=(0.5,),
            reference=reference,
        )
        selected = database.report("parameter_sweep")
        assert selected.temporal_mode.tolist() == ["joint"]
        assert len(database.select_frame("runs", where={"algorithm": "dbscan"})) == 2
    document = Document()
    app = build_document(document, path)
    assert "dbscan" in app.kinds
    selector = document.select_one({"name": "dbscan_run"})
    assert len(selector.options) == 1
    assert selector.options[0][0] != str(recorded.run_id)
    assert "With time · joint · ε" in selector.options[0][1]
    assert "day/week/month" not in selector.options[0][1]
    document.to_json()


def test_report_view_refresh_preserves_records_and_is_idempotent(track_directory, ingestion_input):
    path = track_directory / "reports.sqlite"
    ingest_database(ingestion_input(track_directory, path))
    with ClusterDatabase(path, writable=True) as database:
        run_clustering(database, ClusteringSettings(algorithm="dbscan", eps=0.5))
        tables = [
            row[0]
            for row in database.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        ]
        before = {name: database.select_frame(name) for name in tables}
        # Model a persisted source whose report definitions do not expose cyclic settings.
        with database.transaction():
            for name in ("v_evaluation_runs", "v_parameter_summary", "v_selected_runs"):
                definition = database.connection.execute(
                    "SELECT sql FROM sqlite_master WHERE name=?", (name,)
                ).fetchone()[0]
                for field in ("temporal_mode", "p"):
                    definition = definition.replace(
                        f"       json_extract(r.settings_json, '$.{field}') AS {field},\n", ""
                    )
                definition = re.sub(r", r\.(?:temporal_mode|p)\b", "", definition)
                definition = re.sub(
                    r"^\s*AND r\.(?:temporal_mode|p)\b[^\n]*$",
                    "",
                    definition,
                    flags=re.MULTILINE,
                )
                database.connection.execute(f'DROP VIEW "{name}"')
                database.connection.execute(definition)
        with database.transaction():
            database.refresh_views()
        assert {"temporal_mode", "p"} <= set(database.table("v_evaluation_runs").c.keys())
        for name, expected in before.items():
            pd.testing.assert_frame_equal(database.select_frame(name), expected)
        database.validate()
        contents = path.read_bytes()
        with database.transaction():
            database.refresh_views()
        assert path.read_bytes() == contents


def test_evaluation_cli_applies_independent_target_and_reference_metrics(
    track_directory, ingestion_input
):
    path = track_directory / "cli.sqlite"
    ingest_database(ingestion_input(track_directory, path))
    config = track_directory / "reference.toml"
    config.write_text(
        'algorithm = "qbx"\neps = 0.5\nfeature_mode = "spatiotemporal"\n'
        'temporal_mode = "separate"\np = 3\nlayer = 4\n',
        encoding="utf-8",
    )
    assert (
        evaluate_main(
            [
                str(path),
                "--spatiotemporal",
                "--joint",
                "--p",
                "1",
                "--eps",
                "0.5",
                "--reference-config",
                str(config),
            ]
        )
        == 0
    )
    with ClusterDatabase(path) as database:
        settings = {
            row.algorithm: json.loads(row.settings_json)
            for row in database.select_frame("runs").itertuples()
        }
        assert settings["dbscan"]["temporal_mode"] == "joint" and settings["dbscan"]["p"] == 1
        assert (settings["qbx"]["temporal_mode"], settings["qbx"]["p"]) == ("separate", 3)
        assert len(database.report("parameter_sweep")) == 1
