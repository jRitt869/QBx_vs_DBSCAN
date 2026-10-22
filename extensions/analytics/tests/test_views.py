"""Report views are self-contained and keep evaluation scopes separate."""

from datetime import UTC, datetime, timedelta

import gpxpy
import pandas as pd

from extensions.analytics.evaluation import run_clustering
from extensions.analytics.storage import ClusterDatabase, Database, available_queries
from extensions.analytics.tests.evaluation_helpers import evaluate_sweeps as evaluate
from extensions.analytics.warehouse import ingest_database
from track_clusters.validation import ClusteringSettings


def test_reports_are_database_views(analytics_database):
    with Database(analytics_database) as stored:
        connection = stored.connection
        views = {
            row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='view'")
        }
        assert set(available_queries()) <= views
        for name in available_queries():
            direct = pd.read_sql_query(f'SELECT * FROM "{name}"', connection)
            with Database(analytics_database) as source_db:
                assert len(direct) == len(source_db.report(name))


def test_result_ids_preserve_first_recorded_representative_when_rescoring(
    source_copy, tmp_path, ingestion_input
):
    path = tmp_path / "result_order.sqlite"
    ingest_database(ingestion_input(source_copy / "gpx", path, labels=source_copy / "labels.csv"))
    with ClusterDatabase(path, writable=True) as database:
        first_run = run_clustering(database, ClusteringSettings(algorithm="qbx", eps=0.1, seed=0))
        second_run = run_clustering(database, ClusteringSettings(algorithm="qbx", eps=0.1, seed=7))
        assert first_run < second_run
        evaluate(database, dbscan_eps=[], qbx_eps=[0.1], seeds=[7, 0])
        results = database.select_frame(
            "v_evaluation_runs", where={"dataset": path.stem}, order_by=("evaluation_result_id",)
        )
        assert results.run_id.tolist() == [second_run, first_run]
        assert results.evaluation_result_id.is_monotonic_increasing
        selected = database.report("selected_models").iloc[0]
        assert selected.representative_run_id == second_run
        assert selected.representative_seed == 7
        evaluate(database, dbscan_eps=[], qbx_eps=[0.1], seeds=[0, 7])
        pd.testing.assert_frame_equal(
            database.select_frame(
                "v_evaluation_runs",
                where={"dataset": path.stem},
                order_by=("evaluation_result_id",),
            ),
            results,
        )
        repeated = database.report("selected_models").iloc[0]
        assert repeated.representative_result_id == selected.representative_result_id
        assert repeated.representative_seed == 7


def test_selected_models_and_overlap_stay_within_evaluation_scope(
    analytics_database, source_copy, ingestion_input
):
    source = source_copy / "gpx/sample.gpx"
    document = gpxpy.parse(source.read_text(encoding="utf-8"))
    for track in document.tracks:
        for index, point in enumerate(
            point for segment in track.segments for point in segment.points
        ):
            point.time = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=index)
    source.write_text(document.to_xml(), encoding="utf-8")
    ingest_database(
        ingestion_input(source_copy / "gpx", analytics_database, labels=source_copy / "labels.csv"),
        on_conflict="replace",
    )
    for mode in ("spatial", "spatiotemporal"):
        with ClusterDatabase(analytics_database, writable=True) as source_db:
            evaluate(source_db, feature_mode=mode, dbscan_eps=[0.1], qbx_eps=[0.1], seeds=[0])
    with Database(analytics_database) as source_db:
        selected = source_db.report("selected_models")
    assert len(selected) == 4
    assert not selected.duplicated(["evaluation_id", "algorithm"]).any()
    for evaluation_id in selected.evaluation_id.unique():
        with Database(analytics_database) as source_db:
            scoped = source_db.report("selected_models", evaluation_id=int(evaluation_id))
        assert len(scoped) == 1
        with Database(analytics_database) as stored:
            connection = stored.connection
            for result_id in scoped.representative_result_id:
                assert (
                    connection.execute(
                        "SELECT evaluation_id FROM evaluation_results WHERE evaluation_result_id=?",
                        (int(result_id),),
                    ).fetchone()[0]
                    == evaluation_id
                )
        with Database(analytics_database) as source_db:
            overlap = source_db.report("cluster_composition", evaluation_id=int(evaluation_id))
        assert overlap.evaluation_id.unique().tolist() == [evaluation_id]
        assert overlap.groupby("algorithm").tracks.sum().tolist() == [12]
