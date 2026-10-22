"""Hand-made references and explicit run configurations replace generated labels."""

import json
import sqlite3

import pandas as pd
import pytest
from bokeh.document import Document
from pydantic import ValidationError

from extensions.analytics.dashboard import build_document
from extensions.analytics.evaluation import ReferenceConfiguration, evaluate
from extensions.analytics.metrics import score_assignments
from extensions.analytics.storage import ClusterDatabase
from extensions.analytics.warehouse import ingest_database, replace_references
from scripts.evaluate import main
from track_clusters.validation import ClusteringSettings


def source_database(directory, ingestion_input, labels=False):
    csv = directory / "labels.csv"
    if labels:
        pd.DataFrame(
            {"file_name": ["000001.gpx"] * 4, "id": range(4), "cluster": [8, 9, -1, 9]}
        ).to_csv(csv, index=False)
    path = directory / "source.sqlite"
    ingest_database(ingestion_input(directory, path, labels=csv if labels else None))
    return path


def test_unlabeled_source_requires_an_explicit_reference(spatial_directory, ingestion_input):
    path = source_database(spatial_directory, ingestion_input)
    with ClusterDatabase(path, writable=True) as database:
        assert database.select_frame("labels").empty
        assert "regions" not in database.metadata.tables
        assert "synthetic_reference" not in database.table("labels").c
        with pytest.raises(ValueError, match="hand-made labels"):
            evaluate(database, ClusteringSettings(algorithm="dbscan", eps=0.1), eps=[0.1])
        assert database.select_frame("runs").empty


@pytest.mark.parametrize("handmade", [False, True])
def test_explicit_reference_run_drives_scores_and_dashboard(
    spatial_directory, ingestion_input, handmade
):
    path = source_database(spatial_directory, ingestion_input, labels=handmade)
    reference = ReferenceConfiguration(algorithm="dbscan", eps=0.1, min_samples=2)
    with ClusterDatabase(path, writable=True) as database:
        evaluate(
            database,
            ClusteringSettings(algorithm="qbx", eps=0.1),
            eps=[0.1],
            seeds=[7],
            reference=reference,
        )
        header = database.select_frame("evaluations").iloc[0]
        result = database.select_frame("v_evaluation_runs").iloc[0]
        assert header.algorithm == "qbx" and header.reference_layer == 0
        assert json.loads(header.manifest_json)["reference_kind"] == "run"
        truth = database.select_frame(
            "assignments",
            where={"run_id": int(header.reference_run_id), "layer": 0},
            order_by=("track_id",),
        )
        predicted = database.select_frame(
            "assignments",
            where={"run_id": int(result.run_id), "layer": int(result.layer)},
            order_by=("track_id",),
        )
        expected = score_assignments(
            truth.predicted_cluster.to_numpy(), predicted.predicted_cluster.to_numpy()
        )
        assert result.adjusted_rand == expected["adjusted_rand"]
        assert result.pair_f1 == expected["pair_f1"]
        before = database.row_counts()
        evaluate(
            database,
            ClusteringSettings(algorithm="qbx", eps=0.1),
            eps=[0.1],
            seeds=[7],
            reference=reference,
        )
        assert database.row_counts() == before
        assert database.report("cluster_composition").tracks.sum() == 4
        details = database.assignment_details(
            "source", int(result.run_id), int(result.layer), int(header.evaluation_id)
        )
        assert details.reference_cluster.tolist() == truth.predicted_cluster.tolist()
        evaluate(
            database,
            ClusteringSettings(algorithm="dbscan", min_samples=2, eps=0.1),
            eps=[0.1],
            reference=reference,
        )
    document = Document()
    app = build_document(document, path)
    app.tabs.active = app.kinds.index("analysis")
    assert any("Reference run" in label for _, label in app.analysis_select.options)
    assert sum(app.composition_source.data["tracks"]) == 4
    reference_details = document.select_one({"name": "overview_reference_source"}).text
    assert "Recorded evaluation reference:" not in reference_details
    assert reference_details.count("Configuration:") == 1
    assert "eps=0.1" in reference_details and "min samples=2" in reference_details
    assert "Configuration:" in reference_details
    if handmade:
        assert "Displayed reference: hand-made labels" in reference_details
        assert (spatial_directory / "labels.csv").as_posix() in reference_details
        displayed = pd.Series([8, 9, -1, 9])
    else:
        assert "no hand-made labels supplied" in reference_details
        assert "Displayed reference: Reference run" in reference_details
        assert (
            f"Reference run {int(header.reference_run_id)} (dbscan), layer 1" in reference_details
        )
        displayed = truth.predicted_cluster
    distribution = document.select_one({"name": "reference_distribution"}).data
    expected_counts = {
        f"Reference cluster {label}": int(count)
        for label, count in displayed[displayed >= 0].value_counts().items()
    }
    if (displayed == -1).any():
        expected_counts["Reference outliers"] = int((displayed == -1).sum())
    expected_counts["Excluded short tracks"] = 1
    assert dict(zip(distribution["label"], distribution["tracks"])) == expected_counts
    assert sum(distribution["share"]) == pytest.approx(1)
    assert document.select_one({"name": "stat_Reference clusters"}).text == str(
        displayed[displayed >= 0].nunique()
    )
    assert document.select_one({"name": "stat_Eligible reference outliers"}).text == str(
        int((displayed == -1).sum())
    )
    document.to_json()


def test_overview_follows_selected_run_reference_without_changing_supplied_labels(
    spatial_directory, ingestion_input
):
    path = source_database(spatial_directory, ingestion_input)
    with ClusterDatabase(path, writable=True) as database:
        for minimum in (2, 5):
            evaluate(
                database,
                ClusteringSettings(algorithm="qbx", eps=0.1),
                eps=[0.1],
                seeds=[7],
                reference=ReferenceConfiguration(algorithm="dbscan", eps=0.1, min_samples=minimum),
            )
        evaluations = database.select_frame("evaluations", order_by=("evaluation_id",))
        expected = {
            int(row.evaluation_id): database.select_frame(
                "v_evaluation_references", where={"evaluation_id": int(row.evaluation_id)}
            ).reference_cluster
            for row in evaluations.itertuples()
        }
    document = Document()
    app = build_document(document, path)
    assert document.select_one({"name": "stat_Reference clusters"}).text == "2"
    assert document.select_one({"name": "stat_Eligible reference outliers"}).text == "0"
    app.status_select.value = "Excluded: fewer than five points"
    for evaluation_id in (*expected, int(evaluations.iloc[0].evaluation_id)):
        app.analysis_select.value = str(evaluation_id)
        labels = expected[evaluation_id]
        assert document.select_one({"name": "stat_Reference clusters"}).text == str(
            labels[labels >= 0].nunique()
        )
        assert document.select_one({"name": "stat_Eligible reference outliers"}).text == str(
            int((labels == -1).sum())
        )
        data = document.select_one({"name": "reference_distribution"}).data
        assert sum(data["tracks"]) == 5
        assert sum(data["share"]) == pytest.approx(1)
        assert "Unlabeled eligible tracks" not in data["label"]
        assert app.status_select.value == "Excluded: fewer than five points"
        assert len(document.select_one({"name": "track_catalog"}).source.data["c0"]) == 1
    with ClusterDatabase(path) as database:
        assert database.select_frame("labels").empty
        assert database.report("dataset_inventory").iloc[0].reference_groups == 0
    document.to_json()


def test_default_uses_handmade_labels_and_separates_algorithms(spatial_directory, ingestion_input):
    path = source_database(spatial_directory, ingestion_input, labels=True)
    with ClusterDatabase(path, writable=True) as database:
        for algorithm in ("dbscan", "qbx"):
            evaluate(
                database,
                ClusteringSettings(algorithm=algorithm, min_samples=2, eps=0.1),
                eps=[0.1],
                seeds=[0],
            )
        headers = database.select_frame("evaluations", order_by=("evaluation_id",))
        assert headers.algorithm.tolist() == ["dbscan", "qbx"]
        assert headers.reference_run_id.isna().all()
        scores = database.select_frame("v_evaluation_runs", order_by=("evaluation_result_id",))
        prediction = database.select_frame(
            "assignments",
            where={"run_id": int(scores.iloc[0].run_id), "layer": 0},
            order_by=("track_id",),
        )
        assert (
            scores.iloc[0].adjusted_rand
            == score_assignments([8, 9, -1, 9], prediction.predicted_cluster.to_numpy())[
                "adjusted_rand"
            ]
        )
        with pytest.raises(sqlite3.IntegrityError, match="exactly one algorithm"):
            with database.transaction():
                database.update(
                    "evaluation_results",
                    {"evaluation_id": int(headers.iloc[0].evaluation_id)},
                    where={"run_id": int(scores.iloc[1].run_id)},
                )


def test_reference_layers_have_separate_evaluations_and_cascade(spatial_directory, ingestion_input):
    path = source_database(spatial_directory, ingestion_input, labels=True)
    with ClusterDatabase(path, writable=True) as database:
        for layer in (1, 3):
            evaluate(
                database,
                ClusteringSettings(algorithm="dbscan", eps=0.1),
                eps=[0.1],
                reference=ReferenceConfiguration(algorithm="qbx", eps=0.1, layer=layer),
            )
        evaluate(database, ClusteringSettings(algorithm="dbscan", eps=0.1), eps=[0.1])
        headers = database.select_frame("evaluations")
        assert len(headers) == 3 and headers.reference_layer.dropna().tolist() == [0, 2]
        with database.transaction():
            replace_references(database, None)
        assert len(database.select_frame("evaluations")) == 2
        reference_id = int(headers.iloc[0].reference_run_id)
        with database.transaction():
            database.delete("runs", where={"run_id": reference_id})
        assert database.select_frame("evaluations").empty
        assert database.select_frame("evaluation_results").empty
        assert len(database.select_frame("runs")) == 1
        database.validate()


def test_partial_handmade_labels_are_not_silently_filled(spatial_directory, ingestion_input):
    path = source_database(spatial_directory, ingestion_input, labels=True)
    with ClusterDatabase(path, writable=True) as database:
        with database.transaction():
            database.delete("labels", where={"track_id": 1})
        with pytest.raises(ValueError, match="reference-config"):
            evaluate(database, ClusteringSettings(algorithm="dbscan", eps=0.1), eps=[0.1])
        assert database.select_frame("runs").empty


def test_reference_config_cli_runs_without_raw_files(spatial_directory, ingestion_input, capsys):
    path = source_database(spatial_directory, ingestion_input)
    config = spatial_directory / "reference.toml"
    config.write_text('algorithm="qbx"\neps=0.1\nseed=7\nlayer=1\n', encoding="utf-8")
    (spatial_directory / "000001.gpx").unlink()
    assert main([str(path), "--eps", "0.1", "--reference-config", str(config)]) == 0
    with ClusterDatabase(path) as database:
        manifest = json.loads(database.select_exec("evaluations", ("manifest_json",)).fetchone()[0])
        assert manifest["reference_configuration"]["layer"] == 1
        assert manifest["reference_configuration"]["seed"] == 7
    before = path.read_bytes()
    config.write_text('algorithm="qbx"\neps=-1\n', encoding="utf-8")
    with pytest.raises(SystemExit):
        main([str(path), "--reference-config", str(config)])
    error = capsys.readouterr().err
    assert "reference-config: eps" in error and "errors.pydantic.dev" not in error
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "settings",
    [
        {"algorithm": "dbscan", "eps": 0.1, "layer": 2},
        {"algorithm": "qbx", "eps": 0.1, "layer": 4},
        {"algorithm": "qbx", "eps": 0},
        {"algorithm": "qbx", "eps": 0.1, "unknown": True},
    ],
)
def test_reference_configuration_validation(settings):
    with pytest.raises(ValidationError):
        ReferenceConfiguration.model_validate(settings)
