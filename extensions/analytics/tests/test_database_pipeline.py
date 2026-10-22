"""The database is the only boundary between ingestion, clustering, evaluation, and UI."""

import json
import sqlite3

import pandas as pd
import pytest

from extensions.analytics.evaluation import run_clustering
from extensions.analytics.storage import ClusterDatabase, Database
from extensions.analytics.tests.evaluation_helpers import evaluate_sweeps as evaluate
from extensions.analytics.warehouse import ingest_database
from track_clusters.validation import ClusteringSettings


def counts(database):
    with Database(database) as stored:
        connection = stored.connection
        return {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "tracks",
                "points",
                "labels",
                "runs",
                "assignments",
                "evaluation_results",
            )
        }


def test_database_only_evaluation_reuses_labels_and_creates_missing_algorithm(
    source_copy, tmp_path, monkeypatch, ingestion_input
):
    output = tmp_path / "output"
    path = output / "sample.sqlite"
    ingest_database(ingestion_input(source_copy / "gpx", path, labels=source_copy / "labels.csv"))
    with ClusterDatabase(path, writable=True) as source_db:
        original = run_clustering(source_db, ClusteringSettings(algorithm="dbscan", eps=0.1))
    with ClusterDatabase(path, writable=True) as source_db:
        assert (
            run_clustering(source_db, ClusteringSettings(algorithm="dbscan", eps=0.1)) == original
        )
    source_copy.rename(source_copy.with_name("offline_source"))
    import extensions.analytics.evaluation as module

    calls = []
    real = module.run_clustering

    def recorded(database, settings):
        calls.append(settings.algorithm)
        return real(database, settings)

    monkeypatch.setattr(module, "run_clustering", recorded)
    with ClusterDatabase(path, writable=True) as source_db:
        assert evaluate(source_db, dbscan_eps=[0.1], qbx_eps=[0.1], seeds=[0]) is None
        result = source_db.select_frame(
            "v_evaluation_runs", where={"dataset": "sample"}, order_by=("evaluation_result_id",)
        )
    assert calls == ["dbscan", "qbx"]
    assert result.algorithm.tolist() == ["dbscan", "qbx"]
    before = counts(path)
    with ClusterDatabase(path, writable=True) as source_db:
        evaluate(source_db, dbscan_eps=[0.1], qbx_eps=[0.1], seeds=[0])
    assert counts(path) == before
    assert calls == ["dbscan", "qbx", "dbscan", "qbx"]
    assert list(output.iterdir()) == [path]
    with ClusterDatabase(path) as source_db:
        tracks = source_db.load_tracks()
        assert len(tracks) == 14 and tracks.eligible.sum() == 12


def test_reference_update_preserves_clustering_and_invalidates_scores(
    source_copy, tmp_path, ingestion_input
):
    path = tmp_path / "sample.sqlite"
    ingest_database(ingestion_input(source_copy / "gpx", path, labels=source_copy / "labels.csv"))
    with ClusterDatabase(path, writable=True) as source_db:
        evaluate(source_db, dbscan_eps=[0.1], qbx_eps=[0.1], seeds=[0])
    before = counts(path)
    ingest_database(ingestion_input(source_copy / "gpx", path))
    assert (
        counts(path) == before
    )  # Omitting labels on a repeat import preserves supplied references.
    changed = pd.read_csv(source_copy / "labels.csv")
    changed.loc[0, "cluster"] = -1
    changed.to_csv(source_copy / "labels.csv", index=False)
    ingest_database(ingestion_input(source_copy / "gpx", path, labels=source_copy / "labels.csv"))
    after = counts(path)
    assert after["evaluation_results"] == 0
    assert after["runs"] == before["runs"]
    assert after["assignments"] == before["assignments"]
    with ClusterDatabase(path, writable=True) as source_db:
        evaluate(source_db, dbscan_eps=[0.1], qbx_eps=[0.1], seeds=[0])
    assert counts(path)["evaluation_results"] == 2


def test_failed_reference_update_and_reclustering_are_transactional(
    source_copy, tmp_path, ingestion_input
):
    path = tmp_path / "sample.sqlite"
    ingest_database(ingestion_input(source_copy / "gpx", path, labels=source_copy / "labels.csv"))
    with ClusterDatabase(path, writable=True) as source_db:
        run_clustering(source_db, ClusteringSettings(algorithm="dbscan", eps=0.1))
    before = path.read_bytes()
    labels = pd.read_csv(source_copy / "labels.csv")
    labels.loc[0, "cluster"] = -2
    labels.to_csv(source_copy / "labels.csv", index=False)
    with pytest.raises(ValueError, match="cluster must contain"):
        ingest_database(
            ingestion_input(source_copy / "gpx", path, labels=source_copy / "labels.csv")
        )
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="timestamp"):
        with ClusterDatabase(path, writable=True) as source_db:
            run_clustering(
                source_db,
                ClusteringSettings(algorithm="qbx", eps=0.1, feature_mode="spatiotemporal"),
            )
    assert path.read_bytes() == before


def test_requested_layers_are_populated_and_profiles_are_not_connected(analytics_database):
    from bokeh.document import Document

    from extensions.analytics.dashboard import build_document

    with ClusterDatabase(analytics_database, writable=True) as source_db:
        evaluate(
            source_db,
            qbx_eps=[0.1, 1],
            dbscan_eps=[],
            seeds=[3, 0, 1],
            algorithms=("qbx",),
            layer=1,
        )
    with Database(analytics_database) as stored:
        connection = stored.connection
        assert {
            r[0]
            for r in connection.execute(
                "SELECT DISTINCT layer FROM assignments a JOIN runs r USING(run_id) WHERE r.algorithm='qbx'"
            )
        } == {0, 1, 2}
    app = build_document(Document(), analytics_database)
    app.tabs.active = app.kinds.index("analysis")
    app.analysis_select.value = str(
        next(int(value) for value, label in app.analysis_select.options if "QuickBundlesX" in label)
    )
    profiles = app.profile_selectors["qbx"]
    assert len(profiles.options) == 2
    evaluation_label = dict(app.analysis_select.options)[app.analysis_select.value]
    assert all(profile in evaluation_label for profile in profiles.options)
    assert app.document.select_one({"name": "evaluation_profiles"}).visible
    for profile in profiles.options:
        profiles.value = profile
        eps = app.sweep_sources["qbx"].data["eps"]
        assert len(eps) == len(set(eps))
    app.metric_select.active = app.metric_select.labels.index("Pair F1")
    assert len(app.sweep_sources["qbx"].data["mean"]) == 2


def test_evaluations_share_all_layer_assignments_with_direct_clustering(
    source_copy, tmp_path, ingestion_input
):
    path = tmp_path / "sample.sqlite"
    ingest_database(ingestion_input(source_copy / "gpx", path, labels=source_copy / "labels.csv"))
    with ClusterDatabase(path, writable=True) as database:
        run_id = run_clustering(database, ClusteringSettings(algorithm="qbx", eps=0.1, seed=7))
        before = database.frame("SELECT * FROM assignments ORDER BY run_id, track_id, layer")
        assert len(before) == 36
        assert set(before.layer) == {0, 1, 2}
        assert before.color.str.fullmatch("#[0-9a-f]{6}").all()
        sweep = {"qbx_eps": [0.1], "dbscan_eps": [], "seeds": [7], "algorithms": ("qbx",)}
        assert evaluate(database, **sweep, layer=1) is None
        coarse = database.select_frame(
            "v_evaluation_runs", where={"dataset": "sample"}, order_by=("evaluation_result_id",)
        ).query("layer == 0")
        evaluate(database, **sweep, layer=3)
        fine = database.select_frame(
            "v_evaluation_runs", where={"dataset": "sample"}, order_by=("evaluation_result_id",)
        ).query("layer == 2")
        evaluate(database, **sweep, layer=1)
        repeated = database.select_frame(
            "v_evaluation_runs", where={"dataset": "sample"}, order_by=("evaluation_result_id",)
        ).query("layer == 0")
        # Neither scoring another layer nor re-scoring creates more predictions.
        pd.testing.assert_frame_equal(
            database.frame("SELECT * FROM assignments ORDER BY run_id, track_id, layer"), before
        )
        assert coarse.run_id.tolist() == fine.run_id.tolist() == [run_id]
        assert coarse.layer.tolist() == [0] and fine.layer.tolist() == [2]
        assert coarse.evaluation_result_id.tolist() == repeated.evaluation_result_id.tolist()
        assert coarse.evaluation_result_id.tolist() != fine.evaluation_result_id.tolist()
        assert database.row_counts()["runs"] == 1
        assert database.row_counts()["evaluation_results"] == 2
        selected = database.report("selected_models").iloc[0]
        details = database.assignment_details(
            "sample",
            int(selected.representative_run_id),
            int(selected.representative_layer),
            int(selected.evaluation_id),
        )
        expected = before[before.layer == selected.representative_layer].set_index("track_id")
        assert (
            details.predicted_cluster.tolist()
            == expected.loc[details.track_id].predicted_cluster.tolist()
        )
        assert database.report("cluster_composition").tracks.sum() == 12
        # All layers share the same stored configuration, including on inactive options.
        assert (
            run_clustering(
                database, ClusteringSettings(algorithm="qbx", eps=0.1, seed=7, min_samples=99)
            )
            == run_id
        )
        assert database.row_counts()["evaluation_results"] == 2


def test_layer_statistics_match_assignments_without_requiring_evaluation(
    source_copy, tmp_path, ingestion_input
):
    path = tmp_path / "sample.sqlite"
    ingest_database(ingestion_input(source_copy / "gpx", path))
    with ClusterDatabase(path, writable=True) as database:
        run_clustering(database, ClusteringSettings(algorithm="dbscan", eps=0.1))
        run_clustering(database, ClusteringSettings(algorithm="qbx", eps=0.1))
        assignments = database.frame("SELECT * FROM assignments")
        statistics = database.frame("SELECT * FROM v_run_layers")
        assert len(statistics) == 4 and database.row_counts()["evaluation_results"] == 0
        for row in statistics.itertuples():
            labels = assignments.loc[
                (assignments.run_id == row.run_id) & (assignments.layer == row.layer),
                "predicted_cluster",
            ]
            assert row.tracks == len(labels)
            assert row.clusters == len(set(labels) - {-1})
            assert row.outliers == (labels == -1).sum()
            assert row.outlier_fraction == pytest.approx((labels == -1).mean())


def test_evaluation_uses_only_requested_configurations(analytics_database, monkeypatch):
    with ClusterDatabase(analytics_database, writable=True) as database:
        unrelated = run_clustering(database, ClusteringSettings(algorithm="qbx", eps=2, seed=42))
        original_runs = database.select_frame("runs", order_by=("run_id",))
        original_scores = database.frame(
            "SELECT * FROM evaluation_results ORDER BY evaluation_result_id"
        )
        select_frame = database.select_frame

        def select_requested(table, *args, **kwargs):
            if table == "runs" and not kwargs.get("where"):
                pytest.fail("Evaluation must not select configurations from all stored runs")
            return select_frame(table, *args, **kwargs)

        monkeypatch.setattr(database, "select_frame", select_requested)
        before = database.row_counts()
        evaluate(database, qbx_eps=[0.1, 1], dbscan_eps=[0.1, 1], seeds=[3, 0, 1])
        assert database.row_counts() == before
        pd.testing.assert_frame_equal(select_frame("runs", order_by=("run_id",)), original_runs)
        pd.testing.assert_frame_equal(
            database.frame("SELECT * FROM evaluation_results ORDER BY evaluation_result_id"),
            original_scores,
        )
        assert (
            unrelated
            not in database.select_frame(
                "v_evaluation_runs", where={"dataset": "sample"}, order_by=("evaluation_result_id",)
            ).run_id.tolist()
        )


@pytest.mark.parametrize("whole_layer", [False, True])
def test_evaluation_recreates_run_invalidated_by_assignment_deletion(
    analytics_database, whole_layer
):
    with ClusterDatabase(analytics_database, writable=True) as database:
        original_runs = database.select_frame("runs", order_by=("run_id",))
        original = original_runs.query("algorithm == 'qbx'").iloc[0]
        run_id = int(original.run_id)
        with database.transaction():
            database.connection.execute(
                "DELETE FROM assignments WHERE run_id=? AND layer=0 AND (? OR track_id=(SELECT MIN(track_id) FROM assignments WHERE run_id=?))",
                (run_id, whole_layer, run_id),
            )
        assert database.select_exec("runs", where={"run_id": run_id}).fetchone() is None
        assert database.select_frame("assignments", where={"run_id": run_id}).empty
        evaluate(database, qbx_eps=[0.1], dbscan_eps=[], seeds=[3], algorithms=("qbx",), layer=1)
        replacement = database.select_exec(
            "runs", ("run_id",), where={"config_key": original.config_key}
        ).fetchone()[0]
        assert replacement != run_id
        assert len(database.select_frame("runs", order_by=("run_id",))) == len(original_runs)
        assert len(database.select_frame("assignments", where={"run_id": replacement})) == 3 * len(
            database.eligible_references()
        )
        assert database.select_frame(
            "evaluation_results", where={"run_id": replacement}
        ).layer.tolist() == [0]
        database.validate()


def test_shared_run_constraints_cascades_and_cached_clustering(analytics_database):
    with ClusterDatabase(analytics_database, writable=True) as database:
        runs = database.select_frame("runs", order_by=("run_id",))
        first = runs.iloc[0].to_dict()
        first.pop("run_id")
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            with database.transaction():
                database.insert("runs", first)
        result = (
            database.frame("SELECT * FROM evaluation_results ORDER BY evaluation_result_id")
            .iloc[0]
            .to_dict()
        )
        result.pop("evaluation_result_id")
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            with database.transaction():
                database.insert("evaluation_results", result)
        target = int(runs.iloc[0].run_id)
        before = database.row_counts()
        settings = json.loads(runs.iloc[0].settings_json)
        assert run_clustering(database, ClusteringSettings(**settings)) == target
        after = database.row_counts()
        assert after["runs"] == before["runs"]
        assert after["assignments"] == before["assignments"]
        assert after["evaluation_results"] == before["evaluation_results"]
        # Deleting a run removes its scores and every layer, but keeps other runs.
        other = int(runs.iloc[1].run_id)
        with database.transaction():
            database.connection.execute("DELETE FROM runs WHERE run_id=?", (other,))
        assert database.frame(
            "SELECT * FROM assignments WHERE run_id=:run_id", {"run_id": other}
        ).empty
        assert database.frame(
            "SELECT * FROM evaluation_results WHERE run_id=:run_id", {"run_id": other}
        ).empty
        assert database.row_counts()["runs"] == before["runs"] - 1
        database.validate()


def test_clustering_cli_reuses_run_without_changing_metadata_or_scores(analytics_database):
    from scripts.get_clusters import main

    with ClusterDatabase(analytics_database) as database:
        before = database.row_counts()
        runs = database.select_frame("runs", order_by=("run_id",))
    contents = analytics_database.read_bytes()
    assert main([str(analytics_database), "0.1"]) == 0
    with ClusterDatabase(analytics_database) as database:
        after = database.row_counts()
        assert after["runs"] == before["runs"]
        assert after["assignments"] == before["assignments"]
        assert after["evaluation_results"] == before["evaluation_results"]
        pd.testing.assert_frame_equal(database.select_frame("runs", order_by=("run_id",)), runs)
    assert analytics_database.read_bytes() == contents


@pytest.mark.parametrize("algorithm", ["dbscan", "qbx"])
def test_clustering_looks_up_key_once_and_skips_stored_configurations(
    analytics_database, monkeypatch, algorithm
):
    with ClusterDatabase(analytics_database, writable=True) as database:
        select = database.select
        keys = []

        def recorded(table, *args, **kwargs):
            if table == "runs" and "config_key" in (kwargs.get("where") or {}):
                keys.append(kwargs["where"]["config_key"])
            return select(table, *args, **kwargs)

        monkeypatch.setattr(database, "select", recorded)
        run_id = run_clustering(database, ClusteringSettings(algorithm=algorithm, eps=2.5))
        assert len(keys) == 1
        before = analytics_database.read_bytes()

        def unexpected(*args, **kwargs):
            pytest.fail("An existing settings key must return before loading or writing inputs")

        monkeypatch.setattr(database, "load_tracks", unexpected)
        monkeypatch.setattr(database, "input_signature", unexpected)
        monkeypatch.setattr(database, "transaction", unexpected)
        assert run_clustering(database, ClusteringSettings(algorithm=algorithm, eps=2.5)) == run_id
        assert len(keys) == 2 and keys[0] == keys[1]
        assert analytics_database.read_bytes() == before
