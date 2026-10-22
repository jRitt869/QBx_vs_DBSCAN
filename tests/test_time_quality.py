"""Schema-defined timestamp readiness guards numerical work and direct run writes."""

import sqlite3

import numpy as np
import pytest

from extensions.analytics.evaluation import run_clustering
from extensions.analytics.storage import ClusterDatabase, Database
from extensions.analytics.warehouse import ingest_database
from track_clusters.validation import ClusteringSettings


def temporal_run():
    return {
        "algorithm": "dbscan",
        "feature_mode": "spatiotemporal",
        "config_key": "direct-temporal",
        "settings_json": "{}",
        "created_at_utc": "2026-10-19T00:00:00+00:00",
        "clustering_seconds": 0,
        "provenance": "Schema constraint test",
    }


@pytest.mark.parametrize(
    "epoch,flag",
    [
        (None, "missing_times"),
        (float("nan"), "missing_times"),  # SQLite stores NaN as NULL.
        (float("inf"), "nonfinite_times"),
        (float("-inf"), "nonfinite_times"),
        (-1, "negative_times"),
        (0, "decreasing_times"),
    ],
)
def test_invalid_times_remain_loadable_but_cannot_create_temporal_runs(
    track_directory, ingestion_input, monkeypatch, epoch, flag
):
    import extensions.analytics.evaluation as pipeline

    path = track_directory / "source.sqlite"
    ingest_database(ingestion_input(track_directory, path))
    with ClusterDatabase(path, writable=True) as database:
        with database.transaction():
            database.update(
                "points", {"recorded_at_epoch": epoch}, where={"track_id": 1, "point_index": 5}
            )
        quality = database.select_frame("v_track_time_quality", where={"track_id": 1}).iloc[0]
        assert quality.point_count == 13 and quality[flag] > 0 and quality.temporal_valid == 0
        assert not database.supports_time()
        raw = database.load_tracks()
        assert "13_pts" not in raw and "times" not in raw
        actual = raw.timestamps.iloc[0][5]
        if epoch is None or np.isnan(epoch):
            assert np.isnan(actual)
        else:
            assert actual == epoch
        # Bad timestamps do not prevent spatial clustering.
        spatial_run = run_clustering(database, ClusteringSettings(algorithm="dbscan", eps=0.5))

        def unexpected(*args, **kwargs):
            pytest.fail("Invalid temporal inputs reached numerical preparation")

        monkeypatch.setattr(pipeline, "set_number_of_points", unexpected)
        with pytest.raises(
            ValueError, match="complete, finite, nonnegative, chronological timestamps"
        ):
            run_clustering(
                database,
                ClusteringSettings(algorithm="dbscan", eps=0.5, feature_mode="spatiotemporal"),
            )
        # Bypassing the pipeline still cannot insert or switch to a temporal run.
        with pytest.raises(sqlite3.IntegrityError, match="Temporal clustering requires"):
            with database.transaction():
                database.insert("runs", temporal_run())
        with pytest.raises(sqlite3.IntegrityError, match="Temporal clustering requires"):
            with database.transaction():
                database.update(
                    "runs", {"feature_mode": "spatiotemporal"}, where={"run_id": spatial_run}
                )
        assert database.select_exec("runs", ("feature_mode",)).fetchall() == [("spatial",)]


def test_equal_zero_timestamps_are_valid(track_directory, ingestion_input):
    path = track_directory / "source.sqlite"
    ingest_database(ingestion_input(track_directory, path))
    with ClusterDatabase(path, writable=True) as database:
        with database.transaction():
            database.update("points", {"recorded_at_epoch": 0}, where=None)
        assert database.supports_time()
        quality = database.select_frame("v_track_time_quality")
        assert quality.temporal_valid.eq(1).all()
        assert (
            quality[["missing_times", "nonfinite_times", "negative_times", "decreasing_times"]]
            .eq(0)
            .all()
            .all()
        )
        run_id = run_clustering(
            database, ClusteringSettings(algorithm="dbscan", eps=0.5, feature_mode="spatiotemporal")
        )
        assert database.select_exec("runs", ("run_id",)).fetchall() == [(run_id,)]
        # A direct mode change also accepts valid data.
        with database.transaction():
            database.update("runs", {"feature_mode": "spatial"}, where={"run_id": run_id})
            database.update("runs", {"feature_mode": "spatiotemporal"}, where={"run_id": run_id})


def test_untimed_short_tracks_do_not_block_temporal_clustering(track_directory, ingestion_input):
    import gpxpy

    source = track_directory / "tracks.gpx"
    gpx = gpxpy.parse(source.read_text(encoding="utf-8"))
    short = gpx.tracks[0].clone()
    short.segments[0].points = short.segments[0].points[:3]
    for point in short.segments[0].points:
        point.time = None
    gpx.tracks.append(short)
    source.write_text(gpx.to_xml(), encoding="utf-8")
    path = track_directory / "source.sqlite"
    ingest_database(ingestion_input(track_directory, path))
    with ClusterDatabase(path, writable=True) as database:
        excluded = database.select_frame("v_track_time_quality", where={"eligible": 0}).iloc[0]
        assert excluded.missing_times == 3 and excluded.temporal_valid == 0
        assert database.supports_time()
        run_id = run_clustering(
            database, ClusteringSettings(algorithm="dbscan", eps=0.5, feature_mode="spatiotemporal")
        )
        assert len(database.load_tracks()) == 26
        assert len(database.load_tracks(run_id)) == 25
        assert (
            database.select_frame(
                "assignments", ("track_id",), where={"run_id": run_id}, distinct=True
            ).track_id.nunique()
            == 25
        )


def test_empty_source_cannot_claim_temporal_readiness(tmp_path):
    path = tmp_path / "empty.sqlite"
    Database.initialize(path)
    with ClusterDatabase(path, writable=True) as database:
        assert not database.supports_time()
        with pytest.raises(ValueError, match="No usable tracks"):
            run_clustering(
                database,
                ClusteringSettings(algorithm="dbscan", eps=0.5, feature_mode="spatiotemporal"),
            )
        with pytest.raises(sqlite3.IntegrityError, match="Temporal clustering requires"):
            with database.transaction():
                database.insert("runs", temporal_run())


def test_eligible_track_without_points_is_not_temporally_ready(track_directory, ingestion_input):
    path = track_directory / "source.sqlite"
    ingest_database(ingestion_input(track_directory, path))
    with ClusterDatabase(path, writable=True) as database:
        with database.transaction():
            database.delete("points", where={"track_id": 1})
        quality = database.select_frame("v_track_time_quality", where={"track_id": 1}).iloc[0]
        assert quality.eligible == 1 and quality.point_count == 0 and quality.temporal_valid == 0
        assert not database.supports_time()
        with pytest.raises(
            ValueError, match="complete, finite, nonnegative, chronological timestamps"
        ):
            run_clustering(
                database,
                ClusteringSettings(algorithm="dbscan", eps=0.5, feature_mode="spatiotemporal"),
            )
