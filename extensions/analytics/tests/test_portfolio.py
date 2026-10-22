"""The real sources are evaluated through SQLite with separate label conventions."""

import shutil

import pandas as pd
import pytest
from bokeh.core.validation import check_integrity
from bokeh.document import Document

from extensions.analytics.dashboard import build_document
from extensions.analytics.evaluation import ReferenceConfiguration
from extensions.analytics.storage import ClusterDatabase, Database, available_queries
from extensions.analytics.tests.evaluation_helpers import evaluate_sweeps as evaluate
from scripts.config import validate
from scripts.load_data import main


@pytest.fixture(scope="module")
def portfolio_sources(tmp_path_factory):
    output = tmp_path_factory.mktemp("portfolio")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(validate, "OUTPUT_DIRECTORY", output)
        assert main(["batch"]) == 0
    inventories = []
    for source in ("003", "dump"):
        with ClusterDatabase(output / f"{source}.sqlite", writable=True) as source_db:
            inventories.append(source_db.report("dataset_inventory").iloc[0])
            assert (
                evaluate(
                    source_db,
                    dbscan_eps=[0.5],
                    qbx_eps=[0.5],
                    seeds=[0, 1],
                    reference=ReferenceConfiguration(algorithm="qbx", eps=0.5)
                    if source == "dump"
                    else None,
                )
                is None
            )
            assert source_db.row_counts()["evaluation_results"] == 3
    assert sum(r.raw_tracks for r in inventories) == 617
    assert sum(r.points for r in inventories) == 353072
    assert sum(r.evaluation_runs for r in inventories) == 0
    assert {p.name for p in output.iterdir()} == {"003.sqlite", "dump.sqlite"}
    return output


def test_real_inventory_and_query_aggregates(portfolio_sources):
    for source, raw, eligible in (("003", 466, 267), ("dump", 151, 125)):
        database = portfolio_sources / f"{source}.sqlite"
        with Database(database) as source_db:
            inventory = source_db.report("dataset_inventory")
        assert len(inventory) == 1
        assert inventory.iloc[0].dataset == source
        assert inventory.iloc[0].raw_tracks == raw
        assert inventory.iloc[0].eligible_tracks == eligible
        for name in available_queries():
            with Database(database) as source_db:
                assert isinstance(source_db.report(name), pd.DataFrame)
        with Database(database) as source_db:
            actual = source_db.report("parameter_sweep")
        assert actual.algorithm.tolist() == ["dbscan", "qbx"]
        assert actual.runs.tolist() == [1, 2]
        assert (actual.adjusted_rand_mean.between(-1, 1)).all()
        assert (actual.pair_f1_mean.between(0, 1)).all()
    with Database(portfolio_sources / "dump.sqlite") as source_db:
        assert len(source_db.report("time_gaps")) == 31


@pytest.mark.parametrize("source", ["003", "dump"])
def test_core_trace_sampling_matches_sql_sampling(portfolio_sources, source):
    with ClusterDatabase(portfolio_sources / f"{source}.sqlite") as database:
        tracks = database.report("track_catalog")
        track_id = int(tracks.loc[tracks.point_count.idxmax(), "track_id"])
        expected = pd.read_sql_query(
            """WITH ordered AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY segment_index ORDER BY point_index
                ) AS segment_point,
                COUNT(*) OVER (PARTITION BY segment_index) AS segment_points,
                COUNT(*) OVER () AS total_points
                FROM points WHERE track_id = :track_id
            )
            SELECT point_index, segment_index, longitude, latitude FROM ordered
            WHERE point_index % (total_points / 3000 + 1) = 0
               OR segment_point IN (1, segment_points)
            ORDER BY point_index LIMIT 4000""",
            database.connection,
            params={"track_id": track_id},
        )
        pd.testing.assert_frame_equal(database.track_points(track_id), expected)


def test_real_dashboard_switches_sources_without_stale_results(portfolio_sources):
    document = Document()
    app = build_document(document, source_dir=portfolio_sources)
    app.tabs.active = app.kinds.index("overview")
    assert document.select_one({"name": "stat_Raw tracks"}).text == "466"
    app.dataset_select.value = "dump"
    app.tabs.active = app.kinds.index("overview")
    assert document.select_one({"name": "stat_Raw tracks"}).text == "151"
    with ClusterDatabase(portfolio_sources / "dump.sqlite") as database:
        truth = database.select_frame(
            "v_evaluation_references", where={"evaluation_id": app._analysis_id}
        ).reference_cluster
    assert document.select_one({"name": "stat_Reference clusters"}).text == str(
        truth[truth >= 0].nunique()
    )
    assert document.select_one({"name": "stat_Eligible reference outliers"}).text == str(
        int((truth == -1).sum())
    )
    distribution = document.select_one({"name": "reference_distribution"}).data
    assert sum(distribution["tracks"]) == 151
    assert "Unlabeled eligible tracks" not in distribution["label"]
    app.tabs.active = app.kinds.index("analysis")
    app.analysis_select.value = str(
        next(int(value) for value, label in app.analysis_select.options if "QuickBundlesX" in label)
    )
    assert sum(app.composition_source.data["tracks"]) == 125
    app.dataset_select.value = "003"
    app.evaluation_run_select.value = app.evaluation_run_select.options[-1][0]
    with Database(portfolio_sources / "003.sqlite") as source_db:
        result = source_db.select_frame(
            "evaluation_results",
            where={"evaluation_result_id": int(app.evaluation_run_select.value)},
        ).iloc[0]
    assert float(document.select_one({"name": "stat_ARI"}).text) == pytest.approx(
        result.adjusted_rand, abs=0.00005
    )
    for index in range(len(app.panels)):
        app.tabs.active = index
        assert "003" in app.dataset_note.text
    assert not check_integrity(document.models).error
    document.to_json()


def test_dump_overview_uses_the_selected_temporal_reference_layer(portfolio_sources, tmp_path):
    path = tmp_path / "dump.sqlite"
    shutil.copy2(portfolio_sources / "dump.sqlite", path)
    with ClusterDatabase(path, writable=True) as database:
        evaluate(
            database,
            dbscan_eps=[0.5],
            reference=ReferenceConfiguration(
                algorithm="qbx", eps=0.5, feature_mode="spatiotemporal", layer=4
            ),
        )
        evaluation = database.select_frame("evaluations", order_by=("evaluation_id",)).iloc[-1]
        truth = database.select_frame(
            "assignments",
            where={"run_id": int(evaluation.reference_run_id), "layer": 3},
        ).predicted_cluster
    document = Document()
    app = build_document(document, path)
    app.analysis_select.value = str(evaluation.evaluation_id)
    assert document.select_one({"name": "stat_Reference clusters"}).text == str(
        truth[truth >= 0].nunique()
    )
    assert document.select_one({"name": "stat_Eligible reference outliers"}).text == str(
        int((truth == -1).sum())
    )
    data = document.select_one({"name": "reference_distribution"}).data
    expected = {
        f"Reference cluster {label}": int(count)
        for label, count in truth[truth >= 0].value_counts().items()
    }
    if (truth == -1).any():
        expected["Reference outliers"] = int((truth == -1).sum())
    expected["Excluded short tracks"] = 26
    assert dict(zip(data["label"], data["tracks"])) == expected
    assert sum(data["share"]) == pytest.approx(1)
    details = document.select_one({"name": "overview_reference_source"}).text
    assert f"Reference run {int(evaluation.reference_run_id)} (qbx), layer 4" in details
    assert "feature mode=spatiotemporal" in details
    assert not check_integrity(document.models).error
    document.to_json()
