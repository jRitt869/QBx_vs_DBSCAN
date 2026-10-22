"""Standalone source ingestion contracts, independent of visualization."""

import hashlib
import json
import subprocess
import sys

import gpxpy.gpx
import pandas as pd
import pytest

from extensions.analytics.storage import PROJECT, Database
from extensions.analytics.warehouse import ingest_database, read_config
from scripts.load_data import main


def test_track_indices_ignore_duplicate_or_missing_numbers_and_reset_per_file(
    tmp_path, ingestion_input
):
    source = tmp_path / "gpx"
    source.mkdir()
    document = gpxpy.gpx.GPX()
    for number, count in ((42, 2), (42, 5), (None, 5)):
        track = gpxpy.gpx.GPXTrack(number=number)
        segment = gpxpy.gpx.GPXTrackSegment()
        segment.points = [
            gpxpy.gpx.GPXTrackPoint(50 + point * 0.001, 8 + point * 0.001) for point in range(count)
        ]
        track.segments.append(segment)
        document.tracks.append(track)
    originals = {}
    for name in ("first.gpx", "second.gpx"):
        path = source / name
        path.write_text(document.to_xml(), encoding="utf-8")
        originals[name] = path.read_bytes()
    labels = tmp_path / "labels.csv"
    pd.DataFrame(
        [
            {"file_name": name, "id": index, "cluster": label}
            for name in originals
            for index, label in enumerate((-1, 7, 8))
        ]
    ).to_csv(labels, index=False)
    path = tmp_path / "source.sqlite"
    ingest_database(ingestion_input(source, path, labels=labels))
    with Database(path) as database:
        catalog = database.report("track_catalog")
    assert catalog.source_track_id.tolist() == [0, 1, 2, 0, 1, 2]
    assert catalog.track_index.tolist() == catalog.source_track_id.tolist()
    assert catalog.provided_reference.tolist() == [-1, 7, 8, -1, 7, 8]
    assert catalog.loc[catalog.eligible == 1, "source_track_id"].tolist() == [1, 2, 1, 2]
    assert all((source / name).read_bytes() == raw for name, raw in originals.items())


def test_standalone_ingestion_does_not_require_labels_or_evaluation(source_copy, ingestion_input):
    source = source_copy / "gpx"
    before = (source / "sample.gpx").read_bytes()
    database = source_copy / "raw.sqlite"
    result = ingest_database(ingestion_input(source, database))
    assert (result["tracks"], result["points"], result["runs"], result["assignments"]) == (
        14,
        64,
        0,
        0,
    )
    with Database(database) as source_db:
        inventory = source_db.report("dataset_inventory").iloc[0]
    assert inventory.label_kind == "unlabeled"
    assert inventory.unlabeled_eligible_tracks == 12
    assert inventory.missing_eligible_labels == 0
    assert inventory.reference_outliers == 0
    assert inventory.labeled_tracks == 0
    with Database(database) as source_db:
        assert set(source_db.report("track_catalog").status) == {
            "Eligible: not labeled",
            "Excluded: fewer than five points",
        }
    with Database(database) as source_db:
        assert source_db.report("selected_models").empty
    with Database(database) as source_db:
        assert len(source_db.report("time_gaps")) == 1
    assert (source / "sample.gpx").read_bytes() == before
    rebuild = ingest_database(ingestion_input(source, database))
    assert rebuild["points"] == 64


def test_invalid_source_leaves_existing_snapshot_intact(source_copy, ingestion_input):
    database = source_copy / "raw.sqlite"
    ingest_database(ingestion_input(source_copy / "gpx", database))
    before = database.read_bytes()
    (source_copy / "gpx/sample.gpx").write_text("not GPX", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot parse GPX"):
        ingest_database(ingestion_input(source_copy / "gpx", database), on_conflict="replace")
    assert database.read_bytes() == before
    assert not list(source_copy.glob("*.building"))


def test_raw_script_avoids_evaluation_imports(source_copy):
    database = source_copy / "independent.sqlite"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; "
            "from extensions.analytics.warehouse import ingest_database; "
            "from scripts.config import validate; "
            "validate.OUTPUT_DIRECTORY = Path(sys.argv[2]).parent; "
            "ingest_database(validate.IngestionInput(source_folder=Path(sys.argv[1]), "
            "dataset_key=Path(sys.argv[2]).stem)); "
            "assert 'extensions.analytics.evaluation' not in sys.modules; "
            "assert 'sklearn' not in sys.modules; assert 'dipy' not in sys.modules",
            str(source_copy / "gpx"),
            str(database),
        ],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    with Database(database) as source_db:
        assert len(source_db.report("track_catalog")) == 14


def test_ingestion_cli(source_copy, capsys, output_directory):
    database = output_directory / "new-data.sqlite"
    assert (
        main(
            [
                "single",
                str(source_copy / "gpx"),
                "--database-key",
                "new-data",
                "--name",
                "New GPX source",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)[0]["tracks"] == 14
    with Database(database) as source_db:
        assert source_db.report("dataset_inventory").iloc[0].dataset == "new-data"
        metadata = dict(source_db.connection.execute("SELECT key, value FROM build_metadata"))
    assert all(metadata[key] == "" for key in ("config_path", "config_toml", "config_sha256"))


def test_single_reuses_configured_database_key_and_preserves_results(
    analytics_database, source_copy, capsys
):
    with Database(analytics_database) as database:
        before = database.report("track_catalog")
    assert main(["single", str(source_copy / "gpx"), "--database-key", "sample"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert len(result) == 1 and result[0]["database"] == str(analytics_database.resolve())
    assert result[0]["runs"] == 8 and result[0]["assignments"] == 240
    assert result[0]["evaluation_results"] == 8
    with Database(analytics_database) as database:
        pd.testing.assert_frame_equal(before, database.report("track_catalog"))
    assert list(analytics_database.parent.glob("*.sqlite")) == [analytics_database]


def test_reused_ingestion_updates_config_provenance_and_preserves_results(
    analytics_database, source_copy
):
    config = source_copy / "datasets.toml"
    inputs = read_config(config).datasets[0]
    with config.open("a", encoding="utf-8") as target:
        target.write("\n# Updated configuration provenance\n")
    result = ingest_database(inputs, config)
    assert result["reused_inputs"]
    assert result["datasets"] == result["source_files"] == 1
    assert result["tracks"] == 14 and result["runs"] > 0
    assert result["runs"] == 8 and result["assignments"] == 240
    assert result["evaluation_results"] == 8
    with Database(analytics_database) as database:
        metadata = dict(database.connection.execute("SELECT key, value FROM build_metadata"))
    assert metadata["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert metadata["config_toml"] == config.read_text(encoding="utf-8")

    repeated = ingest_database(inputs)
    assert repeated == result
    with Database(analytics_database) as database:
        metadata = dict(database.connection.execute("SELECT key, value FROM build_metadata"))
    assert all(metadata[key] == "" for key in ("config_path", "config_toml", "config_sha256"))


def test_unreadable_config_preserves_existing_snapshot(analytics_database, source_copy):
    inputs = read_config(source_copy / "datasets.toml").datasets[0]
    before = analytics_database.read_bytes()
    with pytest.raises(FileNotFoundError):
        ingest_database(inputs, source_copy / "missing.toml")
    assert analytics_database.read_bytes() == before


def test_single_and_batch_ingest_identical_source_contents(source_copy, capsys, output_directory):
    database = output_directory / "sample.sqlite"
    assert main(["batch", "--config", str(source_copy / "datasets.toml")]) == 0
    with Database(database) as configured:
        reports = {
            name: configured.report(name)
            for name in ("dataset_inventory", "track_catalog", "time_gaps")
        }
    assert (
        main(
            [
                "single",
                str(source_copy / "gpx"),
                "--database-key",
                "sample",
                "--name",
                "Controlled sample",
                "--labels",
                str(source_copy / "labels.csv"),
                "--label-description",
                "Controlled test labels",
            ]
        )
        == 0
    )
    capsys.readouterr()
    with Database(database) as direct:
        for name, report in reports.items():
            pd.testing.assert_frame_equal(direct.report(name), report)
    assert list(output_directory.iterdir()) == [database]


def test_source_folder_without_tracks_is_rejected(tmp_path, ingestion_input):
    (tmp_path / "empty.gpx").write_text(
        '<gpx version="1.1" creator="test"></gpx>', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="No GPX tracks"):
        ingest_database(ingestion_input(tmp_path, tmp_path / "raw.sqlite"))
    assert not (tmp_path / "raw.sqlite").exists()
