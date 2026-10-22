"""Assignment edits invalidate runs; empty evaluations are cleaned up atomically."""

import json

import pandas as pd
import pytest

from extensions.analytics.evaluation import run_clustering
from extensions.analytics.storage import ClusterDatabase
from extensions.analytics.tests.evaluation_helpers import evaluate_sweeps as evaluate
from extensions.analytics.warehouse import ingest_database
from track_clusters.validation import ClusteringSettings


@pytest.fixture
def cleanup_database(source_copy, ingestion_input):
    path = source_copy / "cleanup.sqlite"
    ingest_database(ingestion_input(source_copy / "gpx", path, labels=source_copy / "labels.csv"))
    with ClusterDatabase(path, writable=True) as database:
        evaluate(database, dbscan_eps=[0.1, 1], qbx_eps=[])
        yield database


def test_assignment_deletion_invalidates_run_and_frees_its_settings_key(cleanup_database):
    database = cleanup_database
    runs = database.select_frame("runs", order_by=("run_id",))
    run_id = int(runs.iloc[0].run_id)
    settings = json.loads(runs.iloc[0].settings_json)
    original_inputs = database.select_frame("tracks", order_by=("track_id",))
    assignments = database.select_frame("assignments", where={"run_id": run_id})
    selected = assignments.iloc[0]
    other_runs = runs[runs.run_id != run_id].reset_index(drop=True)
    with database.transaction():
        assert (
            database.delete(
                "assignments",
                where={
                    "run_id": run_id,
                    "track_id": int(selected.track_id),
                    "layer": int(selected.layer),
                },
            )
            == 1
        )
    assert database.select_exec("runs", where={"run_id": run_id}).fetchone() is None
    assert database.select_frame("assignments", where={"run_id": run_id}).empty
    assert database.select_frame("evaluation_results", where={"run_id": run_id}).empty
    assert (
        len(
            database.select_frame(
                "evaluations", ("evaluation_id", "scope"), order_by=("evaluation_id",)
            )
        )
        == 1
    )
    pd.testing.assert_frame_equal(database.select_frame("runs", order_by=("run_id",)), other_runs)
    pd.testing.assert_frame_equal(
        database.select_frame("tracks", order_by=("track_id",)), original_inputs
    )
    replacement = run_clustering(database, ClusteringSettings(**settings))
    assert len(database.select_frame("runs", order_by=("run_id",))) == 2
    assert len(database.select_frame("assignments", where={"run_id": replacement})) == len(
        assignments
    )
    assert database.select_frame("evaluation_results", where={"run_id": replacement}).empty
    database.validate()


@pytest.mark.parametrize("recursive", [False, True])
@pytest.mark.parametrize(
    "column", ["predicted_cluster", "color", "layer_name", "layer", "track_id"]
)
def test_assignment_value_changes_invalidate_only_their_run(cleanup_database, recursive, column):
    database = cleanup_database
    database.connection.execute(f"PRAGMA recursive_triggers = {int(recursive)}")
    runs = database.select_frame("runs", order_by=("run_id",))
    run_id = int(runs.iloc[0].run_id)
    assignment = database.select_frame("assignments", where={"run_id": run_id}).iloc[0]
    short_track = database.select_exec("tracks", ("track_id",), where={"eligible": 0}).fetchone()[0]
    changes = {
        "predicted_cluster": int(assignment.predicted_cluster) + 1,
        "color": "#123456",
        "layer_name": "Edited layer",
        "layer": 3,
        "track_id": short_track,
    }
    with database.transaction():
        database.update(
            "assignments",
            {column: changes[column]},
            where={
                "run_id": run_id,
                "track_id": int(assignment.track_id),
                "layer": int(assignment.layer),
            },
        )
    assert database.select_exec("runs", where={"run_id": run_id}).fetchone() is None
    assert database.select_frame("assignments", where={"run_id": run_id}).empty
    assert database.select_frame("evaluation_results", where={"run_id": run_id}).empty
    pd.testing.assert_frame_equal(
        database.select_frame("runs", order_by=("run_id",)),
        runs[runs.run_id != run_id].reset_index(drop=True),
    )
    database.validate()


@pytest.mark.parametrize("recursive", [False, True])
def test_moving_an_assignment_invalidates_both_runs(cleanup_database, recursive):
    database = cleanup_database
    database.connection.execute(f"PRAGMA recursive_triggers = {int(recursive)}")
    first, second = database.select_frame("runs", order_by=("run_id",)).run_id.tolist()
    assignment = database.select_frame("assignments", where={"run_id": first}).iloc[0]
    with database.transaction():
        database.update(
            "assignments",
            {"run_id": second, "layer": 3},
            where={
                "run_id": first,
                "track_id": int(assignment.track_id),
                "layer": int(assignment.layer),
            },
        )
    assert all(
        database.select_frame(name).empty
        for name in ("runs", "assignments", "evaluations", "evaluation_results")
    )
    database.validate()


def test_unchanged_assignment_values_preserve_runs_and_scores(cleanup_database):
    database = cleanup_database
    tables = ("runs", "assignments", "evaluations", "evaluation_results")
    before = {name: database.select_frame(name) for name in tables}
    assignment = before["assignments"].iloc[0].to_dict()
    with database.transaction():
        database.update(
            "assignments",
            assignment,
            where={name: assignment[name] for name in ("run_id", "track_id", "layer")},
        )
    for name, expected in before.items():
        pd.testing.assert_frame_equal(database.select_frame(name), expected)
    database.validate()


def test_last_score_removes_evaluation_but_preserves_clustering(cleanup_database):
    database = cleanup_database
    scores = database.select_frame("evaluation_results", order_by=("evaluation_result_id",))
    runs = database.select_frame("runs", order_by=("run_id",))
    assignments = database.select_frame("assignments", order_by=("run_id", "track_id", "layer"))
    with database.transaction():
        database.delete(
            "evaluation_results",
            where={"evaluation_result_id": int(scores.iloc[0].evaluation_result_id)},
        )
        assert (
            len(
                database.select_frame(
                    "evaluations", ("evaluation_id", "scope"), order_by=("evaluation_id",)
                )
            )
            == 1
        )
        database.delete("evaluation_results", where=None)
    assert database.select_frame("evaluations").empty
    pd.testing.assert_frame_equal(database.select_frame("runs", order_by=("run_id",)), runs)
    pd.testing.assert_frame_equal(
        database.select_frame("assignments", order_by=("run_id", "track_id", "layer")), assignments
    )
    database.validate()


@pytest.mark.parametrize("recursive", [False, True])
@pytest.mark.parametrize(
    "operation", ["delete_assignments", "update_assignments", "delete_runs", "delete_evaluations"]
)
def test_bulk_changes_cascade_without_removing_inputs(cleanup_database, recursive, operation):
    database = cleanup_database
    database.connection.execute(f"PRAGMA recursive_triggers = {int(recursive)}")
    inputs = {
        name: database.select_frame(name)
        for name in ("datasets", "source_files", "tracks", "points", "labels")
    }
    with database.transaction():
        action, table = operation.split("_", 1)
        if action == "update":
            database.update(table, {"color": "#123456"}, where=None)
        else:
            database.delete(table, where=None)
    assert database.select_frame("evaluations").empty
    assert database.select_frame("evaluation_results").empty
    for name, expected in inputs.items():
        pd.testing.assert_frame_equal(database.select_frame(name), expected)
    if operation == "delete_evaluations":
        assert len(database.select_frame("runs", order_by=("run_id",))) == 2
        assert not database.select_frame("assignments").empty
    else:
        assert database.select_frame("runs", order_by=("run_id",)).empty
        assert database.select_frame("assignments").empty
    database.validate()


@pytest.mark.parametrize("operation", ["delete", "update"])
def test_invalidation_is_rolled_back_with_the_original_change(cleanup_database, operation):
    database = cleanup_database
    tables = ("runs", "assignments", "evaluations", "evaluation_results")
    before = {name: database.select_frame(name) for name in tables}
    with pytest.raises(RuntimeError, match="abort"):
        with database.transaction():
            if operation == "delete":
                database.delete("assignments", where=None)
            else:
                database.update("assignments", {"color": "#123456"}, where=None)
            assert all(database.select_frame(name).empty for name in tables)
            raise RuntimeError("abort")
    for name, expected in before.items():
        pd.testing.assert_frame_equal(database.select_frame(name), expected)
    database.validate()
