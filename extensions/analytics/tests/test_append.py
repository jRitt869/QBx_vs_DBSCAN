"""Append-only ingestion, explicit file conflicts, and atomic new-database publication."""

import shutil

import gpxpy
import gpxpy.gpx
import pandas as pd
import pytest
from pydantic import ValidationError

from extensions.analytics.evaluation import run_clustering
from extensions.analytics.storage import ClusterDatabase
from extensions.analytics.warehouse import create_database, ingest_database
from scripts.load_data import main
from track_clusters.validation import ClusteringSettings


def test_appending_from_another_directory_preserves_existing_inputs_and_labels(
    analytics_database, source_copy, ingestion_input, tmp_path
):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    shutil.copyfile(source_copy / "gpx/sample.gpx", incoming / "more.gpx")
    csv = incoming / "labels.csv"
    labels = pd.read_csv(source_copy / "labels.csv")
    labels["file_name"] = "more.gpx"
    labels.to_csv(csv, index=False)
    with ClusterDatabase(analytics_database) as database:
        tracks = database.select_frame("tracks", order_by=("track_id",))
        points = database.select_frame("points", order_by=("track_id", "point_index"))
        old_labels = database.select_frame("labels", order_by=("track_id",))
    result = ingest_database(ingestion_input(incoming, analytics_database, labels=csv))
    assert result["files_added"] == 1 and result["files_replaced"] == 0
    assert result["source_files"] == 2 and result["tracks"] == 28
    with ClusterDatabase(analytics_database) as database:
        assert database.select_frame(
            "source_files", order_by=("file_name",)
        ).file_name.tolist() == ["more.gpx", "sample.gpx"]
        pd.testing.assert_frame_equal(
            database.select_frame(
                "tracks",
                where=database.table("tracks").c.track_id.in_(tracks.track_id),
                order_by=("track_id",),
            ),
            tracks,
        )
        pd.testing.assert_frame_equal(
            database.select_frame(
                "points",
                where=database.table("points").c.track_id.in_(tracks.track_id),
                order_by=("track_id", "point_index"),
            ),
            points,
        )
        pd.testing.assert_frame_equal(
            database.select_frame(
                "labels",
                where=database.table("labels").c.track_id.in_(old_labels.track_id),
                order_by=("track_id",),
            ),
            old_labels,
        )
        assert len(database.select_frame("labels")) == 26
        artifacts = database.select_frame("artifacts")
        assert len(artifacts) == 2 and set(artifacts.source_path) == {
            str(csv.resolve()).replace("\\", "/"),
            str((source_copy / "labels.csv").resolve()).replace("\\", "/"),
        }
        assert database.select_frame("runs").empty and database.select_frame("evaluations").empty
        database.validate()
    # The original directory now contains only a subset, which must not remove more.gpx.
    again = ingest_database(
        ingestion_input(source_copy / "gpx", analytics_database, labels=source_copy / "labels.csv")
    )
    assert again["tracks"] == 28 and again["files_skipped"] == 1
    with ClusterDatabase(analytics_database) as database:
        assert len(database.select_frame("labels")) == 26


@pytest.mark.parametrize("policy", ["skip", "replace", "error"])
def test_changed_filename_obeys_explicit_policy(
    analytics_database, source_copy, ingestion_input, policy
):
    path = source_copy / "gpx/sample.gpx"
    document = gpxpy.parse(path.read_text(encoding="utf-8"))
    document.tracks.append(document.tracks[0].clone())
    path.write_text(document.to_xml(), encoding="utf-8")
    inputs = ingestion_input(source_copy / "gpx", analytics_database)
    before = analytics_database.read_bytes()
    if policy == "error":
        with pytest.raises(ValueError, match="already exists with different contents"):
            ingest_database(inputs, on_conflict=policy)
        assert analytics_database.read_bytes() == before
    else:
        result = ingest_database(inputs, on_conflict=policy)
        assert result["tracks"] == (15 if policy == "replace" else 14)
        assert result["runs"] == (0 if policy == "replace" else 8)
        assert result["files_replaced"] == (policy == "replace")
        assert result["files_skipped"] == (policy == "skip")
        with ClusterDatabase(analytics_database) as database:
            database.validate()


@pytest.mark.parametrize("policy", ["skip", "replace", "error"])
def test_identical_files_reuse_runs_under_every_policy(
    analytics_database, source_copy, ingestion_input, policy
):
    with ClusterDatabase(analytics_database) as database:
        runs = database.select_frame("runs", order_by=("run_id",))
        assignments = database.select_frame("assignments", order_by=("run_id", "track_id", "layer"))
    result = ingest_database(
        ingestion_input(source_copy / "gpx", analytics_database), on_conflict=policy
    )
    assert result["reused_inputs"] and result["files_skipped"] == 1
    with ClusterDatabase(analytics_database) as database:
        pd.testing.assert_frame_equal(database.select_frame("runs", order_by=("run_id",)), runs)
        pd.testing.assert_frame_equal(
            database.select_frame("assignments", order_by=("run_id", "track_id", "layer")),
            assignments,
        )
        assert database.row_counts()["evaluation_results"] == 8


def test_no_input_files_does_not_remove_stored_tracks(
    analytics_database, tmp_path, ingestion_input
):
    directory = tmp_path / "empty"
    directory.mkdir()
    result = ingest_database(ingestion_input(directory, analytics_database))
    assert result["tracks"] == 14 and result["runs"] == 8 and result["reused_inputs"]


def test_short_track_additions_preserve_eligible_clustering(
    analytics_database, tmp_path, ingestion_input
):
    directory = tmp_path / "short"
    directory.mkdir()
    document = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack()
    segment = gpxpy.gpx.GPXTrackSegment()
    segment.points = [gpxpy.gpx.GPXTrackPoint(50, 8), gpxpy.gpx.GPXTrackPoint(50.001, 8.001)]
    track.segments.append(segment)
    document.tracks.append(track)
    (directory / "short.gpx").write_text(document.to_xml(), encoding="utf-8")
    with ClusterDatabase(analytics_database) as database:
        expected = database.select_frame("assignments", order_by=("run_id", "track_id", "layer"))
    result = ingest_database(ingestion_input(directory, analytics_database))
    assert result["tracks"] == 15 and result["runs"] == 8 and result["evaluation_results"] == 8
    with ClusterDatabase(analytics_database, writable=True) as database:
        pd.testing.assert_frame_equal(
            database.select_frame("assignments", order_by=("run_id", "track_id", "layer")), expected
        )
        assert (
            run_clustering(database, ClusteringSettings(algorithm="dbscan", eps=0.1))
            in expected.run_id.tolist()
        )


def test_failed_addition_and_reference_import_roll_back_everything(
    analytics_database, source_copy, tmp_path, ingestion_input
):
    directory = tmp_path / "incoming"
    directory.mkdir()
    shutil.copyfile(source_copy / "gpx/sample.gpx", directory / "new.gpx")
    labels = directory / "bad.csv"
    labels.write_text("file_name,id,cluster\nnew.gpx,999,0\n", encoding="utf-8")
    before = analytics_database.read_bytes()
    with pytest.raises(ValueError, match="unknown track"):
        ingest_database(ingestion_input(directory, analytics_database, labels=labels))
    assert analytics_database.read_bytes() == before
    labels.unlink()
    (directory / "zz_invalid.gpx").write_text("not GPX", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot parse GPX"):
        ingest_database(ingestion_input(directory, analytics_database))
    assert analytics_database.read_bytes() == before


def test_failed_first_ingestion_does_not_publish_an_empty_database(
    source_copy, tmp_path, ingestion_input, monkeypatch
):
    import extensions.analytics.warehouse as warehouse

    path = tmp_path / "source.sqlite"
    load = warehouse.load_gpx_sources
    borrowed = []

    def fail(database, *args, **kwargs):
        borrowed.append(database)
        load(database, *args, **kwargs)
        raise RuntimeError("after loading")

    monkeypatch.setattr(warehouse, "load_gpx_sources", fail)
    with pytest.raises(RuntimeError, match="after loading"):
        ingest_database(ingestion_input(source_copy / "gpx", path))
    assert not path.exists() and not list(tmp_path.glob("*.building"))
    import sqlite3

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        borrowed[0].connection.execute("SELECT 1")


def test_creation_never_overwrites_an_existing_or_concurrently_created_file(
    source_copy, tmp_path, ingestion_input
):
    path = tmp_path / "source.sqlite"
    inputs = ingestion_input(source_copy / "gpx", path)
    with pytest.raises(FileExistsError):
        with create_database(inputs, {}) as database:
            assert not path.exists()
            assert database.select_exec("datasets", ("dataset_key",)).fetchone()[0] == path.stem
            path.write_bytes(b"Created by another process")
    assert path.read_bytes() == b"Created by another process"
    with pytest.raises(FileExistsError):
        with create_database(inputs, {}):
            pytest.fail("Existing path reached the creation body")
    assert not list(tmp_path.glob("*.building"))


def test_conflict_policy_validation_and_cli_forwarding(
    source_copy, output_directory, capsys, ingestion_input
):
    path = output_directory / "sample.sqlite"
    inputs = ingestion_input(source_copy / "gpx", path)
    with pytest.raises(ValidationError):
        ingest_database(inputs, on_conflict="unknown")
    assert not path.exists()
    assert main(["single", str(source_copy / "gpx"), "--database-key", "sample"]) == 0
    document = gpxpy.parse((source_copy / "gpx/sample.gpx").read_text(encoding="utf-8"))
    document.tracks.append(document.tracks[0].clone())
    (source_copy / "gpx/sample.gpx").write_text(document.to_xml(), encoding="utf-8")
    assert (
        main(
            [
                "single",
                str(source_copy / "gpx"),
                "--database-key",
                "sample",
                "--on-conflict",
                "replace",
            ]
        )
        == 0
    )
    capsys.readouterr()
    with ClusterDatabase(path) as database:
        assert database.row_counts()["tracks"] == 15
