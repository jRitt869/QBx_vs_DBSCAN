"""Sequential setup, early failure, and the Bokeh process handoff."""

import json
import subprocess
from pathlib import Path

import gpxpy
import pandas as pd
import pytest

from extensions.analytics.storage import DEFAULT_CONFIG, PROJECT, ClusterDatabase, Database
from extensions.analytics.warehouse import ingest_database
from scripts import load_data, start
from scripts.config import validate
from scripts.config.validate import SourceConfiguration


def specification(directory, key="sample"):
    return {
        "dataset_key": key,
        "name": "Controlled source",
        "source_folder": directory,
        "labels": directory / "labels.csv",
        "label_kind": "author",
        "label_description": "Controlled author labels",
    }


@pytest.mark.parametrize(
    "label_mode,temporal_ready",
    [
        ("full", True),
        ("full", False),
        ("missing", True),
        ("missing", False),
        ("partial", True),
        ("partial", False),
    ],
)
def test_prepare_runs_real_entry_points_and_leaves_only_a_source_database(
    track_directory, monkeypatch, label_mode, temporal_ready
):
    source = track_directory / "tracks.gpx"
    if not temporal_ready:
        gpx = gpxpy.parse(source.read_text(encoding="utf-8"))
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    point.time = None
        source.write_text(gpx.to_xml(), encoding="utf-8")
    labels = track_directory / "labels.csv"
    if label_mode != "missing":
        count = 25 if label_mode == "full" else 24
        pd.DataFrame(
            {
                "file_name": ["tracks.gpx"] * count,
                "id": range(count),
                "cluster": [index // 12 if index < 24 else -1 for index in range(count)],
            }
        ).to_csv(labels, index=False)
    config = track_directory / "datasets.toml"
    config.write_text(
        '[[datasets]]\ndataset_key="sample"\nname="Controlled source"\nsource_folder="."\n'
        + (
            'labels="labels.csv"\nlabel_kind="author"\nlabel_description="Controlled author labels"\n'
            if label_mode != "missing"
            else ""
        ),
        encoding="utf-8",
    )
    before = (source.read_bytes(), labels.read_bytes() if labels.exists() else None)
    output = track_directory / "outputs"
    monkeypatch.setattr(validate, "OUTPUT_DIRECTORY", output)
    run_script = start._run
    commands = []

    def run(module, arguments):
        commands.append(module)
        if module == "load_data":
            assert load_data.main(list(map(str, arguments))) == 0
        else:
            run_script(module, arguments)

    monkeypatch.setattr(start, "_run", run)
    assert start.main(["--config", str(config), "--prepare-only"]) == 0
    assert commands == ["load_data"] + ["evaluate"] * (4 if temporal_ready else 2)
    path = output / "sample.sqlite"
    assert list(output.iterdir()) == [path]
    with Database(path) as database:
        inventory = database.report("dataset_inventory").iloc[0]
        assert inventory.eligible_tracks == 25
        assert inventory.evaluation_runs == (54 if temporal_ready else 41)
        if label_mode != "missing":
            assert inventory.label_description == "Controlled author labels"
        assert database.frame(
            "SELECT DISTINCT scope FROM evaluations ORDER BY scope"
        ).scope.tolist() == (["spatial", "spatiotemporal"] if temporal_ready else ["spatial"])
        assert database.select_frame(
            "runs", ("feature_mode",), distinct=True, order_by=("feature_mode",)
        ).feature_mode.tolist() == (
            ["spatial", "spatiotemporal"] if temporal_ready else ["spatial"]
        )
        if temporal_ready:
            assert database.frame(
                "SELECT algorithm, COUNT(*) AS layers FROM assignments a JOIN runs r USING(run_id) WHERE r.feature_mode='spatiotemporal' GROUP BY algorithm"
            ).layers.tolist() == [300, 700]
            sweep = database.report("parameter_sweep")
            for algorithm in ("dbscan", "qbx"):
                spatial = sweep.loc[(sweep.algorithm == algorithm) & (sweep.scope == "spatial")]
                temporal = sweep.loc[
                    (sweep.algorithm == algorithm) & (sweep.scope == "spatiotemporal")
                ]
                assert temporal.eps.tolist() == spatial.eps.tolist()
                assert (temporal.runs == 1).all()
        assert database.row_counts()["runs"] == (54 if temporal_ready else 41)
        assert set(database.select_frame("runs", ("algorithm",)).algorithm) == {"dbscan", "qbx"}
        assert (
            len(database.select_frame("labels"))
            == {"full": 25, "partial": 24, "missing": 0}[label_mode]
        )
        evaluations = database.select_frame("evaluations")
        assert {json.loads(value)["reference_kind"] for value in evaluations.manifest_json} == {
            "labels" if label_mode == "full" else "run"
        }
        if label_mode == "full":
            assert evaluations.reference_run_id.isna().all()
        else:
            assert evaluations.reference_run_id.nunique() == 1
            reference_layer = 3 if temporal_ready else 2
            assert evaluations.reference_layer.tolist() == [reference_layer] * len(evaluations)
            reference = database.select_frame(
                "runs", where={"run_id": int(evaluations.iloc[0].reference_run_id)}
            ).iloc[0]
            settings = json.loads(reference.settings_json)
            assert reference.algorithm == "qbx"
            assert reference.feature_mode == ("spatiotemporal" if temporal_ready else "spatial")
            assert (settings["eps"], settings["factor"], settings["seed"]) == (0.5, 2, 0)
            assert settings["eps_time"] == (0.1 if temporal_ready else None)
            assert settings["timezone"] == ("UTC" if temporal_ready else None)
            for manifest in evaluations.manifest_json:
                configuration = json.loads(manifest)["reference_configuration"]
                assert configuration == {**settings, "layer": reference_layer + 1}
    assert before == (source.read_bytes(), labels.read_bytes() if labels.exists() else None)


def test_failed_setup_stops_before_later_steps_or_server(track_directory, monkeypatch):
    (track_directory / "labels.csv").write_text(
        "file_name,id,cluster\ntracks.gpx,0,0\n", encoding="utf-8"
    )
    sources = SourceConfiguration(datasets=[specification(track_directory)])
    monkeypatch.setattr(start, "read_config", lambda path: sources)
    monkeypatch.setattr(validate, "OUTPUT_DIRECTORY", track_directory / "outputs")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs["check"] and kwargs["cwd"] == PROJECT
        if len(commands) == 1:
            ingest_database(sources.datasets[0])
        if len(commands) == 2:
            raise subprocess.CalledProcessError(7, command)

    monkeypatch.setattr(start.subprocess, "run", run)
    monkeypatch.setattr(
        start.os, "execv", lambda *args: pytest.fail("Server started after failure")
    )
    assert start.main([]) == 7
    assert [command[2] for command in commands] == ["scripts.load_data", "scripts.evaluate"]
    assert commands[0][3] == "batch"


def test_server_handoff_follows_setup_and_forwards_options(monkeypatch):
    events = []
    monkeypatch.setattr(
        start,
        "prepare_sources",
        lambda config: events.append(("prepared", config)),
    )
    monkeypatch.setattr(start.os, "chdir", lambda path: events.append(("directory", path)))
    monkeypatch.setattr(
        start.os,
        "execv",
        lambda executable, command: events.append(("server", command)),
    )
    assert start.main(["--allow-websocket-origin", "example.test:5006"]) == 0
    assert [event[0] for event in events] == ["prepared", "directory", "server"]
    assert events[0][1] == DEFAULT_CONFIG
    command = events[-1][1]
    assert command[1:4] == ["-m", "bokeh", "serve"]
    assert Path(command[4]) == PROJECT / "scripts/run_server.py"
    assert command[-2:] == ["--allow-websocket-origin", "example.test:5006"]


def test_prepare_only_skips_the_server(monkeypatch):
    calls = []
    monkeypatch.setattr(start, "prepare_sources", lambda *args: calls.append(args))
    monkeypatch.setattr(start.os, "execv", lambda *args: pytest.fail("Prepare-only started server"))
    assert start.main(["--prepare-only"]) == 0
    assert len(calls) == 1


@pytest.mark.parametrize("epoch", [None, float("inf"), float("-inf"), -1, 0])
def test_invalid_timestamps_do_not_select_temporal_clustering(
    track_directory, epoch, ingestion_input
):
    path = track_directory / "source.sqlite"
    ingest_database(ingestion_input(track_directory, path))
    assert start._supports_time(path)
    with ClusterDatabase(path, writable=True) as database, database.transaction():
        database.connection.execute(
            "UPDATE points SET recorded_at_epoch=? WHERE track_id=1 AND point_index=5", (epoch,)
        )
    assert not start._supports_time(path)


@pytest.mark.parametrize("key", ["../outside", "CON"])
def test_nonportable_source_keys_are_rejected_before_running_scripts(tmp_path, monkeypatch, key):
    (tmp_path / "labels.csv").write_text("file_name,id,cluster\n", encoding="utf-8")
    config = tmp_path / "config.toml"
    config.write_text(
        f'[[datasets]]\ndataset_key="{key}"\nname="Controlled source"\nsource_folder="."\n'
        'labels="labels.csv"\nlabel_kind="author"\nlabel_description="Controlled author labels"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        start, "_run", lambda *args: pytest.fail("Unsafe source path reached a script")
    )
    with pytest.raises(ValueError, match="key"):
        start.prepare_sources(config)
