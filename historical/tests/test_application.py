"""The historical map and clustering use only the explicitly selected SQLite source."""

import hashlib
import json

import numpy as np
import pytest
from bokeh.core.validation import check_integrity
from bokeh.document import Document

from extensions.analytics.warehouse import ingest_database
from historical.get_raw_data import main
from historical.pipeline import run_clustering
from historical.storage import ClusterDatabase, Database
from historical.viewer import build_document, map_track_frame


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--seed", "1.5", "{database}", "1", "1"],
        ["{database}", "nan", "1"],
        ["{database}", "1", "0"],
        ["--factor", "0.5", "{database}", "1", "1"],
        ["--min_s", "0", "{database}", "1", "1"],
        ["--seed", "-1", "{database}", "1", "1"],
        ["--timezone", "invalid/timezone", "{database}", "1", "1"],
    ],
)
def test_cli_invalid_arguments(arguments, tmp_path):
    database = tmp_path / "untouched.sqlite"
    database.write_bytes(b"Invalid inputs must not open this file")
    with pytest.raises(SystemExit) as error:
        main([argument.format(database=database) for argument in arguments])
    assert error.value.code == 2
    assert database.read_bytes() == b"Invalid inputs must not open this file"


@pytest.mark.parametrize("algorithm", ["dbscan", "qbx"])
def test_historical_database_round_trip_without_raw_files(
    track_directory, tmp_path, algorithm, ingestion_input
):
    directory = tmp_path / "outputs"
    database = directory / "selected-source.sqlite"
    ingest_database(
        ingestion_input(
            track_directory, database, dataset_key="selected-source", name="Selected source"
        )
    )
    (track_directory / "tracks.gpx").unlink()
    arguments = [str(database), "0.5", "0.1"] + (["-q"] if algorithm == "qbx" else [])
    assert main(arguments) == 0
    with Database(database) as stored:
        connection = stored.connection
        run_id, mode, settings = connection.execute(
            "SELECT run_id, feature_mode, settings_json FROM runs"
        ).fetchone()
    assert mode == "spatiotemporal"
    assert json.loads(settings)["timezone"] == "UTC"
    assert list(directory.iterdir()) == [database]
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    document = Document()
    view = build_document(document, database, algorithm=algorithm)
    with ClusterDatabase(database) as source_db:
        stored = map_track_frame(source_db.load_tracks(run_id))
        gaps = source_db.load_map_jumps(run_id)
        labels = source_db.load_layer_names(run_id)
    assert view.labels == labels
    assert view.track_source.data["name"] == stored.name.tolist()
    assert view.track_source.data["clusters0"] == stored.clusters0.tolist()
    np.testing.assert_allclose(view.track_source.data["x"], stored.x.tolist())
    assert view.jump_source.data["jump_minutes"] == gaps.jump_minutes.tolist() == [42]
    assert (
        "selected-source / Selected source"
        in document.select_one({"name": "historical_source"}).text
    )
    assert not check_integrity(document.models).error
    document.to_json()
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


def test_historical_viewer_selects_runs_within_the_requested_source(
    track_directory, tmp_path, ingestion_input
):
    first = tmp_path / "first.sqlite"
    second = tmp_path / "second.sqlite"
    ingest_database(ingestion_input(track_directory, first, dataset_key="first"))
    ingest_database(ingestion_input(track_directory, second, dataset_key="second"))
    with ClusterDatabase(first, writable=True) as source_db:
        first_run = run_clustering(source_db, "dbscan", 0.5, feature_mode="spatiotemporal")
    with ClusterDatabase(second, writable=True) as source_db:
        second_run = run_clustering(source_db, "qbx", 0.5, feature_mode="spatiotemporal")
    # IDs deliberately overlap: a run must always be resolved inside its source.
    assert first_run == second_run
    first_view = build_document(Document(), first, run_id=first_run)
    second_view = build_document(Document(), second, run_id=second_run)
    assert len(first_view.labels) == 2
    assert len(second_view.labels) == 4
    first_view.hide_toggle.active = True
    assert not second_view.hide_toggle.active


def test_historical_viewer_run_and_algorithm_selection(track_directory, tmp_path, ingestion_input):
    database = tmp_path / "source.sqlite"
    ingest_database(ingestion_input(track_directory, database))
    with ClusterDatabase(database, writable=True) as source_db:
        dbscan = run_clustering(source_db, "dbscan", 0.5, feature_mode="spatiotemporal")
    with ClusterDatabase(database, writable=True) as source_db:
        qbx = run_clustering(source_db, "qbx", 0.5, feature_mode="spatiotemporal")
    with ClusterDatabase(database, writable=True) as source_db:
        run_clustering(
            source_db, "dbscan", 0.1, feature_mode="spatial"
        )  # A newer spatial-only run is not the historical map.
    assert build_document(Document(), database).labels == ["Layer 1", "Layer 2", "Layer 3", "Time"]
    assert len(build_document(Document(), database, algorithm="dbscan").labels) == 2
    assert len(build_document(Document(), database, run_id=dbscan).labels) == 2
    assert len(build_document(Document(), database, run_id=qbx).labels) == 4
    document = Document()
    assert build_document(document, database, run_id=dbscan, algorithm="qbx") is None
    assert "No matching" in document.roots[0].text


def test_historical_viewer_does_not_create_missing_results(
    track_directory, tmp_path, ingestion_input
):
    database = tmp_path / "source.sqlite"
    ingest_database(ingestion_input(track_directory, database))
    with ClusterDatabase(database, writable=True) as source_db:
        spatial = run_clustering(source_db, "dbscan", 0.5, feature_mode="spatial")
    before = database.read_bytes()
    for run_id in (None, spatial, 999):
        document = Document()
        assert build_document(document, database, run_id=run_id) is None
        assert "No matching timestamp-inclusive" in document.roots[0].text
    assert database.read_bytes() == before
