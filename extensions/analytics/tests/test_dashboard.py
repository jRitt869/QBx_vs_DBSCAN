"""Conditional source-specific Bokeh views and independent session state."""

import sqlite3
from datetime import UTC, datetime, timedelta

import gpxpy
import numpy as np
import pandas as pd
import pytest
from bokeh.core.validation import check_integrity
from bokeh.document import Document
from bokeh.models import Button, Div, FileInput, HelpButton, RadioButtonGroup, TextAreaInput

from extensions.analytics.dashboard import build_document
from extensions.analytics.evaluation import ReferenceConfiguration, evaluate, run_clustering
from extensions.analytics.storage import ClusterDatabase
from extensions.analytics.warehouse import ingest_database
from track_clusters.validation import ClusteringSettings


def activate(app, kind):
    app.tabs.active = app.kinds.index(kind)


def test_source_tabs_and_analysis(analytics_database):
    document = Document()
    app = build_document(document, analytics_database)
    assert [panel.title for panel in app.tabs.tabs] == [
        "Overview",
        "DBSCAN",
        "QuickBundlesX",
        "Analysis",
    ]
    assert app.kinds[app.tabs.active] == "overview"
    assert "12 tracks" in app.cluster_views["dbscan"].summary.text
    for kind in app.kinds:
        activate(app, kind)
        assert app.dataset in app.dataset_note.text
        assert not any(
            "Source:" in div.text for div in app.panels[app.tabs.active].child.select({"type": Div})
        )
    activate(app, "analysis")
    assert isinstance(app.metric_select, RadioButtonGroup)
    app.metric_select.active = app.metric_select.labels.index("Pair F1")
    assert app.sweep_plots["dbscan"].yaxis.axis_label == "Pair F1"
    assert not app.sweep_bands["dbscan"].visible
    app.analysis_select.value = str(
        next(int(value) for value, label in app.analysis_select.options if "QuickBundlesX" in label)
    )
    assert app.metric_select.labels[app.metric_select.active] == "Pair F1"
    assert document.select_one({"name": "performance_measure"}) is app.metric_select
    assert app.profile_selectors["qbx"].document is document
    assert app.sweep_plots["qbx"].yaxis.axis_label == "Pair F1"
    assert not app.sweep_bands["qbx"].visible
    with ClusterDatabase(analytics_database) as database:
        expected = database.report("parameter_sweep", evaluation_id=app._analysis_id)
        composition = database.report("cluster_composition", evaluation_id=app._analysis_id)
    np.testing.assert_allclose(app.sweep_sources["qbx"].data["mean"], expected.pair_f1_mean)
    assert app.composition_source.data["tracks"] == composition.tracks.tolist()
    app.metric_select.active = 0
    assert app.sweep_bands["qbx"].visible
    assert document.select_one({"name": "assignment_track"}) is None
    assert document.select_one({"name": "assignment_route"}) is None
    activate(app, "overview")
    assert document.select_one({"name": "stat_Raw tracks"}).text == "14"
    assert all(button.tooltip.content for button in document.select({"type": HelpButton}))
    assert not list(document.select({"type": FileInput}))
    assert not list(document.select({"type": TextAreaInput}))
    assert not check_integrity(document.models).error
    document.to_json()


@pytest.mark.parametrize("include_run_reference", [False, True])
def test_run_statistics_follow_selected_result_and_evaluation(
    analytics_database, include_run_reference
):
    if include_run_reference:
        with ClusterDatabase(analytics_database, writable=True) as database:
            for algorithm in ("dbscan", "qbx"):
                evaluate(
                    database,
                    ClusteringSettings(algorithm=algorithm, eps=0.1),
                    eps=(0.1, 1),
                    seeds=(3, 0, 1),
                    reference=ReferenceConfiguration(algorithm="dbscan", eps=1),
                )
    document = Document()
    app = build_document(document, analytics_database)
    activate(app, "analysis")
    metrics = {
        "ARI": "adjusted_rand",
        "AMI": "adjusted_mutual_information",
        "F1": "pair_f1",
        "Pair precision": "pair_precision",
        "Pair recall": "pair_recall",
    }
    for evaluation_id, _ in app.analysis_select.options:
        app.analysis_select.value = evaluation_id
        with ClusterDatabase(analytics_database) as database:
            results = database.select_frame(
                "evaluation_results", where={"evaluation_id": int(evaluation_id)}
            )
            selected = database.report("selected_models", evaluation_id=int(evaluation_id)).iloc[0]
            overlaps = {}
            for result in results.itertuples():
                details = database.assignment_details(
                    app.dataset, int(result.run_id), int(result.layer), int(evaluation_id)
                )
                overlaps[result.evaluation_result_id] = pd.crosstab(
                    details.reference_cluster, details.predicted_cluster
                ).stack()
        assert app.evaluation_run_select.value == str(int(selected.representative_result_id))
        assert {int(value) for value, _ in app.evaluation_run_select.options} == set(
            results.evaluation_result_id
        )
        for result in results.itertuples():
            app.evaluation_run_select.value = str(result.evaluation_result_id)
            for label, field in metrics.items():
                value = document.select_one({"name": f"stat_{label}"}).text
                assert float(value) == pytest.approx(getattr(result, field), abs=0.00005)
            data = app.composition_source.data
            assert {
                (int(reference), int(predicted)): tracks
                for reference, predicted, tracks in zip(
                    data["reference"], data["predicted"], data["tracks"], strict=True
                )
            } == {
                pair: count
                for pair, count in overlaps[result.evaluation_result_id].items()
                if count
            }
            assert set(data["run_key"]) == {
                f"{selected.algorithm}_run{result.run_id}_layer{result.layer}"
            }
    assert not check_integrity(document.models).error
    document.to_json()


def test_no_results_disclaimer_and_only_available_algorithm_tabs(
    source_copy, tmp_path, ingestion_input
):
    path = tmp_path / "raw.sqlite"
    ingest_database(ingestion_input(source_copy / "gpx", path))
    document = Document()
    app = build_document(document, path)
    assert app.kinds == ["overview", "analysis"]
    assert "No hand-made reference labels" in app.dataset_note.text
    activate(app, "analysis")
    assert document.select_one({"name": "evaluation_disclaimer"}) is not None
    with ClusterDatabase(path, writable=True) as source_db:
        run_clustering(source_db, ClusteringSettings(algorithm="dbscan", eps=0.1))
    document = Document()
    app = build_document(document, path)
    assert app.kinds == ["overview", "dbscan", "analysis"]
    activate(app, "analysis")
    assert document.select_one({"name": "evaluation_disclaimer"}) is not None
    assert not check_integrity(document.models).error


@pytest.mark.parametrize("labels_kind", ["complete", "partial", "absent"])
def test_overview_reference_distribution_accounts_for_every_raw_track(
    source_copy, tmp_path, ingestion_input, labels_kind
):
    labels = source_copy / "labels.csv"
    frame = pd.read_csv(labels)
    frame.loc[frame.id == 11, "cluster"] = -1
    if labels_kind == "partial":
        frame = frame.loc[frame.id != 10]
    frame.to_csv(labels, index=False)
    path = tmp_path / "overview.sqlite"
    ingest_database(
        ingestion_input(
            source_copy / "gpx",
            path,
            labels=labels if labels_kind != "absent" else None,
        )
    )
    document = Document()
    app = build_document(document, path)
    overview = app.panels[app.kinds.index("overview")].child
    first_row = overview.children[1]
    assert [
        next(div.name for div in card.select({"type": Div}) if div.name)
        for card in first_row.children
    ] == [
        "stat_Raw tracks",
        "stat_Eligible tracks",
        "stat_Coordinate points",
        "stat_Source files",
    ]
    data = document.select_one({"name": "reference_distribution"}).data
    counts = dict(zip(data["label"], data["tracks"]))
    expected = {"Excluded short tracks": 2}
    if labels_kind == "absent":
        expected["Unlabeled eligible tracks"] = 12
    else:
        expected.update(
            {"Reference cluster 0": 6, "Reference cluster 1": 5, "Reference outliers": 1}
        )
        if labels_kind == "partial":
            expected["Reference cluster 1"] = 4
            expected["Unlabeled eligible tracks"] = 1
    assert counts == expected
    assert sum(counts.values()) == 14
    assert np.sum(data["share"]) == pytest.approx(1)
    assert data["end_angle"][-1] == pytest.approx(2 * np.pi)
    assert document.select_one({"name": "stat_Reference clusters"}).text == (
        "0" if labels_kind == "absent" else "2"
    )
    reference_details = document.select_one({"name": "overview_reference_source"}).text
    if labels_kind == "absent":
        assert "no hand-made labels supplied" in reference_details
        assert str(labels) not in reference_details
    else:
        assert "Displayed reference: hand-made labels" in reference_details
        assert labels.as_posix() in reference_details
    assert "No evaluations have been recorded" in reference_details
    assert document.select_one({"name": "stat_Eligible geometry duplicates"}) is None
    assert (
        "Eligible geometry duplicates: 1"
        in document.select_one({"name": "eligible_geometry_duplicates"}).text
    )
    catalog = document.select_one({"name": "track_catalog"})
    assert not {"geometry_occurrence", "geometry_sha256"} & {c.title for c in catalog.columns}
    assert not any(
        "Exact geometry duplicates" in div.text for div in overview.select({"type": Div})
    )
    assert not check_integrity(document.models).error
    document.to_json()


def test_catalog_filter_updates_visible_rows_and_download_without_database_reads(
    analytics_database, monkeypatch
):
    document = Document()
    app = build_document(document, analytics_database)
    activate(app, "overview")
    catalog = document.select_one({"name": "track_catalog"})
    download = next(
        button for button in document.select({"type": Button}) if button.label == "Download CSV"
    )
    payload = download.js_event_callbacks["button_click"][0].args["payload"]
    with ClusterDatabase(analytics_database) as database:
        tracks = database.report("track_catalog").drop(
            columns=["geometry_occurrence", "geometry_sha256"]
        )
    monkeypatch.setattr(app, "read", lambda *args, **kwargs: pytest.fail("Filter queried SQLite"))
    for status in (app.status_select.options[1], "All statuses"):
        app.status_select.value = status
        expected = tracks if status == "All statuses" else tracks[tracks.status == status]
        assert len(catalog.source.data["c0"]) == len(expected)
        assert payload.data["content"] == [expected.to_csv(index=False)]
        assert payload.data["filename"] == [f"{app.dataset}_track_catalog.csv"]


def test_empty_output_directory_has_ingestion_instructions(tmp_path, monkeypatch):
    import extensions.analytics.dashboard as dashboard

    monkeypatch.setattr(dashboard, "OUTPUT_DIRECTORY", tmp_path)
    document = Document()
    app = build_document(document)
    assert app.database_path == tmp_path / "missing.sqlite"
    assert app.dataset_select.disabled
    assert "python -m scripts.load_data single" in app.panels[0].child.children[0].text


def test_sessions_keep_analysis_and_map_selections_independent(analytics_database):
    first = build_document(Document(), analytics_database)
    second = build_document(Document(), analytics_database)
    activate(first, "analysis")
    activate(second, "analysis")
    first.metric_select.active = first.metric_select.labels.index("Pair F1")
    first.analysis_select.value = str(
        next(
            int(value) for value, label in first.analysis_select.options if "QuickBundlesX" in label
        )
    )
    assert second.metric_select.labels[second.metric_select.active] == "Adjusted Rand index"
    assert set(first.sweep_sources) == {"qbx"}
    assert set(second.sweep_sources) == {"dbscan"}
    first.cluster_views["dbscan"].hide_toggle.active = True
    assert not second.cluster_views["dbscan"].hide_toggle.active
    assert first.layout is not second.layout


def test_first_tab_switches_reuse_preloaded_views_without_database_reads(
    analytics_database, monkeypatch
):
    import extensions.analytics.dashboard as dashboard

    connections = []

    class ObservedDatabase(ClusterDatabase):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            connections.append(self)

    monkeypatch.setattr(dashboard, "ClusterDatabase", ObservedDatabase)
    document = Document()
    app = build_document(document, analytics_database)
    assert app.kinds[app.tabs.active] == "overview"
    assert set(app.cluster_views) == {"dbscan", "qbx"}
    assert document.select_one({"name": "stat_Raw tracks"}).text == "14"
    assert document.select_one({"name": "evaluation_scope"}) is not None
    ready = [panel.child for panel in app.panels]
    before = len(connections)
    for index in range(len(app.kinds)):
        app.tabs.active = index
        assert app.panels[index].child is ready[index]
    assert len(connections) == before
    assert app.database is None


@pytest.mark.parametrize("algorithm", ["dbscan", "qbx"])
def test_source_switch_attaches_a_complete_map_with_working_controls(
    source_copy, tmp_path, ingestion_input, algorithm
):
    source = source_copy / "gpx/sample.gpx"
    gpx = gpxpy.parse(source.read_text(encoding="utf-8"))
    for track in gpx.tracks:
        points = (point for segment in track.segments for point in segment.points)
        for index, point in enumerate(points):
            if point.time is None:
                point.time = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index)
    source.write_text(gpx.to_xml(), encoding="utf-8")
    directory = tmp_path / "databases"
    for key in ("first", "second"):
        path = directory / f"{key}.sqlite"
        ingest_database(ingestion_input(source.parent, path, dataset_key=key))
        with ClusterDatabase(path, writable=True) as database:
            run_clustering(database, ClusteringSettings(algorithm=algorithm, eps=0.1))
            run_clustering(
                database,
                ClusteringSettings(algorithm=algorithm, eps=0.1, feature_mode="spatiotemporal"),
            )
    document = Document()
    app = build_document(document, source_dir=directory)
    attached = []

    def check_ready(attr, old, new):
        panel = new[0].tabs[app.kinds.index(algorithm)]
        assert len(list(panel.child.select({"name": "layer_selector"}))) == 1
        assert len(list(panel.child.select({"name": "hide_outliers"}))) == 1
        assert len(list(panel.child.select({"name": "tracks"}))) == 1
        assert len(list(new[0].select({"name": "stat_Raw tracks"}))) == 1
        assert len(list(new[0].select({"name": "evaluation_disclaimer"}))) == 1
        attached.append(app.dataset)

    app.tab_container.on_change("children", check_ready)
    for key in ("second", "first"):
        previous = app.cluster_views[algorithm]
        previous_tabs = app.tabs
        app.dataset_select.value = key
        assert app.tabs is not previous_tabs and previous_tabs.document is None
        view = app.cluster_views[algorithm]
        assert view is not previous and previous.hide_toggle.document is None
        assert view.hide_toggle.document is document
        view.layer_selector.active = view.labels.index("Time")
        assert "<b>Time</b>" in view.summary.text
        assert view.track_source.data["cluster"] == view.track_source.data[f"clusters{view.layer}"]
        view.hide_toggle.active = True
        assert view.track_source.data["is_visible"] == [
            0.0 if label == -1 else 1.0 for label in view.track_source.data["cluster"]
        ]
        view.color_toggle.active = False
        assert view.track_source.data["color"] == [
            "#ff0000" if label == -1 else "#00c800" for label in view.track_source.data["cluster"]
        ]
        assert not view.jump_toggle.disabled
        view.jump_toggle.active = True
        assert view.jump_source.data["is_visible"] == [1.0]
        selector = document.select_one({"name": f"{algorithm}_run"})
        assert any("With time · separate · ε" in label for _, label in selector.options)
        assert all("day/week/month" not in label for _, label in selector.options)
        selector.value = next(value for value, label in selector.options if "· Spatial ·" in label)
        replacement = app.cluster_views[algorithm]
        assert replacement is not view and view.hide_toggle.document is None
        replacement.hide_toggle.active = True
        replacement.color_toggle.active = False
        assert replacement.hide_toggle.document is document
        assert "12 tracks" in replacement.summary.text
    assert attached == ["second", "first"]
    assert not check_integrity(document.models).error
    document.to_json()


def test_callbacks_share_and_release_connections_before_source_replacement(
    source_copy, tmp_path, monkeypatch, ingestion_input
):
    import extensions.analytics.dashboard as dashboard

    instances = []

    class ObservedDatabase(ClusterDatabase):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.frame_count = 0
            instances.append(self)

        def frame(self, *args, **kwargs):
            self.frame_count += 1
            return super().frame(*args, **kwargs)

    directory = tmp_path / "outputs"
    first, second = directory / "first.sqlite", directory / "second.sqlite"
    ingest_database(ingestion_input(source_copy / "gpx", first, dataset_key="first"))
    ingest_database(ingestion_input(source_copy / "gpx", second, dataset_key="second"))
    with ClusterDatabase(first, writable=True) as database:
        run_clustering(database, ClusteringSettings(algorithm="dbscan", eps=0.1))
    monkeypatch.setattr(dashboard, "ClusterDatabase", ObservedDatabase)
    document = Document()
    app = build_document(document, source_dir=directory)
    activate(app, "analysis")
    assert any(database.frame_count > 1 for database in instances)
    assert app.database is None
    for database in instances:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            database.connection.execute("SELECT 1")
    with pytest.raises(RuntimeError, match="callback failed"):
        with app.database_scope():
            raise RuntimeError("callback failed")
    assert app.database is None
    # New files append in place; an idle dashboard must release its read connection.
    source = source_copy / "gpx/sample.gpx"
    source.with_name("more.gpx").write_bytes(source.read_bytes())
    ingest_database(ingestion_input(source_copy / "gpx", first, dataset_key="first"))
    activate(app, "overview")
    assert document.select_one({"name": "stat_Raw tracks"}).text == "28"
    app.dataset_select.value = "second"
    assert document.select_one({"name": "stat_Raw tracks"}).text == "14"
    assert app.database is None
