"""Contracts for configured ingestion, SQL aggregation, and update safety."""

import hashlib
import json
import sqlite3
from contextlib import closing

import numpy as np
import pandas as pd
import pytest

from extensions.analytics.storage import Database, available_queries
from extensions.analytics.warehouse import ingest_database, read_config
from scripts.load_data import main


def test_raw_inventory_keeps_excluded_and_unlabeled_short_tracks(analytics_database):
    with Database(analytics_database) as source_db:
        inventory = source_db.report("dataset_inventory").iloc[0]
    assert (inventory.raw_tracks, inventory.eligible_tracks, inventory.excluded_tracks) == (
        14,
        12,
        2,
    )
    assert (inventory.labeled_tracks, inventory.excluded_labeled_tracks) == (13, 1)
    assert (inventory.points, inventory.timestamped_points) == (64, 5)
    assert inventory.missing_eligible_labels == 0
    with Database(analytics_database) as source_db:
        tracks = source_db.report("track_catalog")
    assert tracks.source_track_id.tolist() == list(range(14))
    assert tracks.iloc[0].segment_count == 2
    assert pd.isna(tracks.iloc[-1].provided_reference)
    assert pd.isna(tracks.iloc[-1].reference_cluster)
    assert tracks.iloc[-1].reference_origin == "unlabeled"
    with Database(analytics_database) as source_db:
        gaps = source_db.report("time_gaps")
    assert gaps.gap_seconds.tolist() == [58 * 60]
    assert gaps.segment_index.tolist() == [1]
    with closing(sqlite3.connect(analytics_database)) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_sql_aggregates_match_independent_pandas_statistics(analytics_database):
    with Database(analytics_database) as source_db:
        runs = source_db.frame("SELECT * FROM v_evaluation_runs ORDER BY evaluation_result_id")
    grouped = runs.groupby(["algorithm", "eps"]).adjusted_rand
    expected = (
        grouped.agg(adjusted_rand_mean="mean", adjusted_rand_std="std").fillna(0).reset_index()
    )
    with Database(analytics_database) as source_db:
        actual = source_db.report("parameter_sweep")
    merged = actual.merge(expected, on=["algorithm", "eps"], suffixes=("_sql", "_pandas"))
    assert len(merged) == len(expected)
    np.testing.assert_allclose(merged.adjusted_rand_mean_sql, merged.adjusted_rand_mean_pandas)
    np.testing.assert_allclose(
        np.sqrt(merged.adjusted_rand_variance), merged.adjusted_rand_std, atol=1e-7
    )
    with Database(analytics_database) as source_db:
        selected = source_db.report("selected_models")
    assert selected[selected.algorithm == "qbx"].iloc[0].representative_seed == 3
    with Database(analytics_database) as source_db:
        assert source_db.report("cluster_composition").groupby(
            "algorithm"
        ).tracks.sum().tolist() == [12, 12]


def test_rebuild_is_idempotent_and_preserves_inputs(analytics_database, source_copy):
    with Database(analytics_database) as source_db:
        before = source_db.report("track_catalog")
    source = source_copy / "gpx/sample.gpx"
    fingerprint = source.read_bytes()
    config = source_copy / "datasets.toml"
    counts = ingest_database(read_config(config).datasets[0], config)
    assert counts["tracks"] == 14
    assert counts["runs"] == 8
    assert counts["assignments"] == 240
    assert counts["evaluation_results"] == 8
    with Database(analytics_database) as source_db:
        pd.testing.assert_frame_equal(before, source_db.report("track_catalog"))
    assert source.read_bytes() == fingerprint


@pytest.mark.parametrize("failure", ["source", "invalid_label", "unknown_label"])
def test_failed_rebuild_preserves_previous_database(analytics_database, source_copy, failure):
    before = hashlib.sha256(analytics_database.read_bytes()).hexdigest()
    config = source_copy / "datasets.toml"
    with config.open("a", encoding="utf-8") as target:
        target.write("\n# Config provenance must roll back with failed ingestion.\n")
    if failure == "source":
        path = source_copy / "gpx/sample.gpx"
        path.write_text("not GPX", encoding="utf-8")
    else:
        path = source_copy / "labels.csv"
        labels = pd.read_csv(path)
        if failure == "invalid_label":
            labels.loc[0, "cluster"] = -2
        else:
            labels.loc[len(labels)] = ["unknown.gpx", 99, 0]
        labels.to_csv(path, index=False)
    with pytest.raises(ValueError):
        ingest_database(read_config(config).datasets[0], config, on_conflict="replace")
    assert hashlib.sha256(analytics_database.read_bytes()).hexdigest() == before
    assert not list(analytics_database.parent.glob("*.building"))


def test_frames_support_ctes_and_bound_parameters(analytics_database):
    with Database(analytics_database) as source_db:
        frame = source_db.frame(
            """
            WITH selected AS (SELECT * FROM datasets WHERE dataset_key = :dataset)
            SELECT dataset_key FROM selected
        """,
            {"dataset": "sample"},
        )
    assert frame.dataset_key.tolist() == ["sample"]
    with Database(analytics_database) as source_db:
        assert source_db.report("track_catalog", "sample' OR 1=1 --").empty


def test_other_sqlite_databases_are_not_overwritten(source_copy):
    config = source_copy / "datasets.toml"
    inputs = read_config(config).datasets[0].model_copy(update={"dataset_key": "other"})
    database = inputs.database
    database.parent.mkdir(parents=True)
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("CREATE TABLE user_records (value TEXT)")
        connection.commit()
    before = database.read_bytes()
    with pytest.raises(ValueError, match="not an analytics snapshot"):
        ingest_database(inputs, config)
    assert database.read_bytes() == before


def test_cli_batch_imports_configured_sources(source_copy, capsys, output_directory):
    output = output_directory
    assert main(["batch", "--config", str(source_copy / "datasets.toml")]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert len(summary) == 1 and summary[0]["tracks"] == 14
    assert summary[0]["runs"] == summary[0]["evaluation_results"] == 0
    assert list(output.iterdir()) == [output / "sample.sqlite"]
    config = source_copy / "datasets.toml"
    with Database(output / "sample.sqlite") as source_db:
        metadata = dict(source_db.connection.execute("SELECT key, value FROM build_metadata"))
    assert metadata["config_toml"] == config.read_text(encoding="utf-8")
    assert metadata["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert metadata["config_path"] == config.resolve().as_posix()


def test_all_saved_queries_execute(analytics_database):
    for name in available_queries():
        with Database(analytics_database) as source_db:
            assert isinstance(source_db.report(name), pd.DataFrame)


def test_build_ingests_inputs_without_running_evaluation(source_copy):
    config = source_copy / "datasets.toml"
    source = read_config(config).datasets[0].model_copy(update={"dataset_key": "inputs"})
    database = source.database
    result = ingest_database(source, config)
    assert result["tracks"] == 14
    assert result["runs"] == result["evaluation_results"] == result["assignments"] == 0
    with Database(database) as source_db:
        assert source_db.report("selected_models").empty
