"""SQL-backed Bokeh analysis with one independent state per browser session."""

import json
import sqlite3
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.palettes import Category20
from sqlalchemy import func

from track_clusters.viewer import ClusterView

from .layouts import (
    ANALYSIS_TITLE,
    AnalysisLayout,
    ClusteringLayout,
    DashboardLayout,
    EvaluationLayout,
    OverviewLayout,
    clustering_map,
    message_panel,
    source_tabs,
    tab_panels,
)
from .presentation import (
    ALGORITHMS,
    MEASURES,
    RUN_METRICS,
    clustering_run_name,
    evaluation_name,
    evaluation_run_name,
    map_track_frame,
    note_html,
    profile_name,
    set_table,
)
from .storage import OUTPUT_DIRECTORY, ClusterDatabase
from .warehouse import discover_databases


def _current_runs(runs):
    """Select calibrated runs with an explicit supported mode when time is enabled."""
    return func.json_extract(runs.c.settings_json, "$.scale_pairs").is_not(None) & (
        (runs.c.feature_mode == "spatial")
        | func.json_extract(runs.c.settings_json, "$.temporal_mode").in_(("separate", "joint"))
    )


class AnalyticsDashboard:
    """Own source selections, analysis views, and callbacks for one Bokeh session."""

    def __init__(self, document, database=None, source_dir=None):
        self.document = document
        self._cache, self._rendered = {}, {}
        self._analysis_id = None
        self.database = None
        self._changing_source = False
        directory = Path(source_dir) if source_dir is not None else OUTPUT_DIRECTORY
        self.sources = (
            discover_databases(directory)
            if database is None
            else discover_databases(Path(database))
            if Path(database).is_dir()
            else {}
        )
        if database is not None and not Path(database).is_dir() and Path(database).is_file():
            with ClusterDatabase(database) as source:
                inventory = source.report("dataset_inventory")
            if len(inventory) != 1:
                raise ValueError("Each database must contain exactly one source")
            self.sources = {inventory.iloc[0].dataset: Path(database)}
        self.database_path = next(iter(self.sources.values()), directory / "missing.sqlite")
        view = DashboardLayout(list(self.sources))
        self.dataset_select = view.dataset_select
        self.dataset_note = view.dataset_note
        self.database_note = view.database_note
        self.tab_container = view.tab_container
        self.layout = view.layout
        self.dataset_select.on_change("value", self._dataset_changed)
        self._activate_source()
        document.title = "GPX Trajectory Analytics"
        document.add_root(self.layout)

    @property
    def dataset(self):
        return self.dataset_select.value

    @contextmanager
    def database_scope(self):
        """Share one read connection within a render or callback, then release it.

        No connection is kept while the browser is idle, allowing atomic ingestion
        to replace source databases on Windows. Nested callbacks borrow the handle.
        """
        if self.database is not None:
            yield self.database
            return
        with ClusterDatabase(self.database_path) as database:
            self.database = database
            try:
                yield database
            finally:
                self.database = None

    def read(self, operation, **parameters):
        revision = self._source_revision()
        key = (revision, operation, json.dumps(parameters, sort_keys=True))
        if key not in self._cache:
            self._cache = {k: v for k, v in self._cache.items() if k[0] == revision}
            self._cache[key] = getattr(self.database, operation)(**parameters)
        return self._cache[key].copy()

    def _source_revision(self):
        info = self.database_path.stat()
        return str(self.database_path.resolve()), info.st_mtime_ns, info.st_size

    def _evaluations(self):
        return self.read(
            "select_frame",
            table="evaluations",
            columns=("evaluation_id", "scope", "algorithm", "manifest_json"),
            order_by=("evaluation_id",),
        )

    def _fetch(self, name, **parameters):
        if name in {"parameter_sweep", "selected_models", "cluster_composition"}:
            parameters.setdefault("evaluation_id", self._analysis_id)
        return self.read("report", name=name, **parameters)

    def _activate_source(self, active_kind=None):
        self._changing_source = True
        self._cache.clear()
        self._rendered.clear()
        self._analysis_id = None
        self.cluster_views = {}
        algorithms = []
        if self.sources:
            self.database_path = self.sources[self.dataset]
            with self.database_scope():
                self.inventory = self._fetch("dataset_inventory", dataset=self.dataset)
                selected = self.inventory.iloc[0]
                self.source_name = f"{self.dataset} / {selected['name']}"
                self.dataset_note.text = note_html(
                    f"{self.source_name} — {selected.label_description}"
                )
                self.database_note.text = note_html(f"Database: {self.database_path.name}")
                available = self.database.select_frame(
                    "runs",
                    ("algorithm",),
                    where=_current_runs(self.database.table("runs")),
                    distinct=True,
                    order_by=("algorithm",),
                )
                algorithms = available.algorithm.tolist()
        else:
            self.source_name = "No data source selected"
            self.dataset_note.text = note_html(
                "Ingest a source database before opening the dashboard."
            )
            self.database_note.text = ""
        self.kinds = ["overview", *algorithms, "analysis"]
        self.panels = tab_panels(self.kinds)
        active = self.kinds.index(active_kind) if active_kind in self.kinds else 0
        # Prepare every tab once, sharing a connection for the selected source.
        with self.database_scope() if self.sources else nullcontext():
            for index in range(len(self.kinds)):
                self._render(index)
        # Replace the whole tab group to avoid overlapping Bokeh child-view rebuilds.
        self.tabs = source_tabs(self.panels, active)
        self.tabs.on_change("active", self._tab_changed)
        self.tab_container.children = [self.tabs]
        self._changing_source = False

    def _dataset_changed(self, attr, old, new):
        active_kind = self.kinds[self.tabs.active]
        self._activate_source(active_kind)

    def _tab_changed(self, attr, old, new):
        if not self._changing_source:
            self._render(new)

    def _render(self, index):
        if not self.sources:
            self.panels[index].child = message_panel(
                "No source databases are available. Run python -m scripts.load_data single SOURCE --database-key KEY --labels REFERENCES.csv, then reload this page."
            )
            return
        try:
            kind = self.kinds[index]
            key = (*self._source_revision(), kind)
            if self._rendered.get(kind) == key:
                return
            with self.database_scope():
                content = (
                    self._clustering(kind)
                    if kind in ALGORITHMS
                    else {
                        "overview": self._overview,
                        "analysis": self._analysis,
                    }[kind]()
                )
            self.panels[index].child = content
            self._rendered[kind] = key
        except (OSError, ValueError, sqlite3.Error) as error:
            self.panels[index].child = message_panel(f"Could not read source results: {error}")

    def _clustering(self, algorithm):
        stored_runs = self.database.table("runs")
        runs = self.database.select_frame(
            "runs",
            where=(stored_runs.c.algorithm == algorithm) & _current_runs(stored_runs),
            order_by=(
                (stored_runs.c.feature_mode == "spatiotemporal").desc(),
                stored_runs.c.run_id.desc(),
            ),
        )
        best = self.read(
            "select_frame",
            table="v_selected_runs",
            columns=("run_id",),
            where={"algorithm": algorithm},
            order_by=("evaluation_id",),
        )
        first = int(runs.iloc[0].run_id)
        if not (runs.feature_mode == "spatiotemporal").any() and not best.empty:
            first = int(best.iloc[0].run_id)
        panel = ClusteringLayout(
            algorithm,
            [(str(run.run_id), clustering_run_name(run)) for run in runs.itertuples()],
            str(first),
        )

        def _show_run(attr, old, new):
            with self.database_scope():
                tracks = map_track_frame(self.database.load_tracks(int(new)))
                jumps = self.database.load_map_jumps(int(new))
                labels = self.database.load_layer_names(int(new))
                view = ClusterView(tracks, jumps, labels=labels, algorithm=ALGORITHMS[algorithm])
                settings = json.loads(runs.loc[runs.run_id == int(new), "settings_json"].iloc[0])
                view.layout = clustering_map(
                    view, has_time="Time" in labels, joint=settings.get("temporal_mode") == "joint"
                )
                view.jump_toggle.disabled = jumps.empty
                self.cluster_views[algorithm] = view
                panel.body.children = [view.layout]

        panel.run_select.on_change("value", _show_run)
        _show_run("value", None, str(first))
        return panel.layout

    def _analysis(self):
        evaluations = self._evaluations()
        if evaluations.empty:
            return message_panel(
                "No evaluation has been run for this source. Clustering results alone do not establish agreement with reference labels. Run scripts/evaluate.py on this source database to populate this tab.",
                title=ANALYSIS_TITLE,
                name="evaluation_disclaimer",
                warning=True,
            )
        self._analysis_id = int(evaluations.iloc[0].evaluation_id)
        options = []
        for record in evaluations.itertuples():
            summary = self._fetch(
                "parameter_sweep", dataset=self.dataset, evaluation_id=int(record.evaluation_id)
            )
            profiles = sorted({profile_name(row) for row in summary.itertuples()})
            options.append((str(record.evaluation_id), evaluation_name(record, profiles)))
        view = AnalysisLayout(options, str(self._analysis_id))
        self.analysis_select = view.analysis_select
        self.metric_select = view.metric_select
        self._analysis_view = view
        view.content.children = [self._evaluation()]

        def _select_analysis(attr, old, new):
            with self.database_scope():
                self._analysis_id = int(new)
                view.content.children = [self._evaluation()]
                if not self.inventory.iloc[0].labeled_tracks:
                    status = self.status_select.value
                    overview = self.kinds.index("overview")
                    self.panels[overview].child = self._overview()
                    self.status_select.value = status

        def _select_measure(attr, old, new):
            for algorithm in self.sweep_sources:
                self._refresh_sweep(algorithm)

        self.analysis_select.on_change("value", _select_analysis)
        self.metric_select.on_change("active", _select_measure)
        return view.layout

    def _overview(self):
        inventory = self._fetch("dataset_inventory", dataset=self.dataset).iloc[0]
        tracks = self._fetch("track_catalog", dataset=self.dataset).drop(
            columns=["geometry_occurrence", "geometry_sha256"]
        )
        evaluations = self._evaluations()
        reference_evaluation = None
        reference_clusters = tracks.reference_cluster
        reference_caption = (
            "Supplied reference labels; tracks without a label are shown separately."
        )
        if not inventory.labeled_tracks and not evaluations.empty:
            evaluation_id = self._analysis_id or int(evaluations.iloc[0].evaluation_id)
            reference_evaluation = evaluations.loc[evaluations.evaluation_id == evaluation_id].iloc[
                0
            ]
            truth = self.read(
                "select_frame",
                table="v_evaluation_references",
                where={"evaluation_id": evaluation_id},
            )
            reference_clusters = tracks.track_id.map(truth.set_index("track_id").reference_cluster)
            reference_caption = (
                "Reference labels from the selected evaluation in Analysis; "
                "excluded short tracks are shown separately."
            )
        eligible = tracks.loc[tracks.eligible == 1].assign(reference_cluster=reference_clusters)
        groups = eligible.loc[eligible.reference_cluster >= 0].groupby("reference_cluster").size()
        labels = [f"Reference cluster {int(label)}" for label in groups.index]
        sizes = groups.tolist()
        colors = [Category20[20][index % 20] for index in range(len(groups))]
        for label, count, color in (
            ("Reference outliers", int((eligible.reference_cluster == -1).sum()), "#C4514E"),
            ("Excluded short tracks", int(inventory.excluded_tracks), "#ADB6B1"),
            ("Unlabeled eligible tracks", int(eligible.reference_cluster.isna().sum()), "#E0B45C"),
        ):
            if count:
                labels.append(label)
                sizes.append(count)
                colors.append(color)
        shares = np.asarray(sizes, dtype=float) / len(tracks) if len(tracks) else np.array([])
        angles = np.concatenate(([0.0], np.cumsum(shares) * 2 * np.pi))
        reference_notes = []
        if inventory.labeled_tracks:
            artifacts = self.read(
                "select_frame",
                table="artifacts",
                columns=("role", "source_path"),
                order_by=("role",),
            )
            paths = artifacts.loc[artifacts.role.str.startswith("labels"), "source_path"]
            reference_notes.append("Displayed reference: hand-made labels.")
            if not paths.empty:
                reference_notes.append(f"Label source: {', '.join(paths)}")
            reference_notes.append(inventory.label_description)
        else:
            reference_notes.append("Reference source: no hand-made labels supplied.")
            if reference_evaluation is None:
                reference_notes.append("Displayed reference: none.")
            else:
                manifest = json.loads(reference_evaluation.manifest_json)
                reference_notes.append(
                    f"Displayed reference: {manifest['reference_description']}. "
                    "These are clustering predictions, not hand-made labels."
                )
        seen_references = set()
        for evaluation in evaluations.itertuples():
            manifest = json.loads(evaluation.manifest_json)
            key = (manifest["reference_run_id"], manifest["reference_layer"])
            if key in seen_references:
                continue
            seen_references.add(key)
            if manifest["reference_kind"] == "labels":
                reference_notes.append("Recorded evaluations use these hand-made labels.")
                continue
            config = manifest["reference_configuration"]
            settings = " · ".join(
                f"{key.replace('_', ' ')}={value:g}"
                if isinstance(value, float)
                else f"{key.replace('_', ' ')}={value}"
                for key, value in config.items()
                if key not in {"algorithm", "layer"} and value is not None
            )
            reference_notes.append(f"Configuration: {settings}")
        if evaluations.empty:
            reference_notes.append("No evaluations have been recorded for this source.")

        gaps = self._fetch("time_gaps", dataset=self.dataset)
        view = OverviewLayout(
            totals=zip(
                ["Raw tracks", "Eligible tracks", "Coordinate points", "Source files"],
                [
                    f"{int(value):,}"
                    for value in (
                        inventory.raw_tracks,
                        inventory.eligible_tracks,
                        inventory.points,
                        inventory.files,
                    )
                ],
            ),
            references=[
                ("Reference clusters", len(groups)),
                ("Eligible reference outliers", int((eligible.reference_cluster == -1).sum())),
                ("Excluded short tracks", int(inventory.excluded_tracks)),
            ],
            reference_details="\n".join(reference_notes),
            reference_caption=reference_caption,
            tracks=tracks,
            statuses=["All statuses", *sorted(tracks.status.unique())],
            gaps=gaps[["file_name", "source_track_id", "gap_seconds"]],
            gap_count=len(gaps),
            has_timestamps=inventory.timestamped_points != 0,
            duplicate_count=int(inventory.duplicate_eligible_tracks),
        )
        view.reference_source.data = dict(
            label=labels,
            tracks=sizes,
            share=shares,
            start_angle=angles[:-1],
            end_angle=angles[1:],
            color=colors,
        )
        self.status_select = view.status_select
        filename = f"{self.dataset}_track_catalog.csv"
        catalog_payload = ColumnDataSource(
            dict(content=[tracks.to_csv(index=False)], filename=[filename])
        )
        view.catalog_download.js_on_click(
            CustomJS(
                args=dict(payload=catalog_payload),
                code="""
                const blob = new Blob([payload.data.content[0]], {type: 'text/plain;charset=utf-8'});
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = payload.data.filename[0];
                link.click();
                setTimeout(() => URL.revokeObjectURL(url), 1000);
                """,
            )
        )

        def _filter_catalog(attr, old, new):
            visible = tracks if new == "All statuses" else tracks[tracks.status == new]
            set_table(view.catalog, visible)
            catalog_payload.data = dict(content=[visible.to_csv(index=False)], filename=[filename])

        self.status_select.on_change("value", _filter_catalog)
        return view.layout

    def _refresh_sweep(self, algorithm):
        selected = self.profile_selectors[algorithm].value
        frame = self.sweep_frames[algorithm]
        frame = frame[frame.profile == selected]
        measure = self.metric_select.labels[self.metric_select.active]
        self.sweep_sources[algorithm].data = dict(
            eps=frame.eps.tolist(),
            mean=frame[MEASURES[measure]].tolist(),
            low=frame.adjusted_rand_min.tolist(),
            high=frame.adjusted_rand_max.tolist(),
            runs=frame.runs.tolist(),
        )
        self.sweep_plots[algorithm].yaxis.axis_label = measure
        self.sweep_bands[algorithm].visible = measure == "Adjusted Rand index"

    def _evaluation(self):
        summary = self._fetch("parameter_sweep", dataset=self.dataset)
        self.sweep_sources = {}
        self.metric_select.disabled = summary.empty
        self._analysis_view.sweep_controls.visible = not summary.empty
        self._analysis_view.profile_controls.children = []
        self._analysis_view.profile_controls.visible = False
        if summary.empty:
            return message_panel(
                "No evaluation scores are available for this source. Run scripts/evaluate.py on its database, then reload the dashboard."
            )
        # The schema guarantees that an evaluation contains exactly one algorithm.
        algorithm = summary.algorithm.iloc[0]
        best = self._fetch("selected_models", dataset=self.dataset).iloc[0]
        summary["profile"] = [profile_name(record) for record in summary.itertuples()]
        profiles = sorted(summary.profile.unique())
        filters = {"dataset": self.dataset}
        if self._analysis_id is not None:
            filters["evaluation_id"] = self._analysis_id
        runs = self.read(
            "select_frame",
            table="v_evaluation_runs",
            where=filters,
            order_by=("evaluation_result_id",),
        )
        view = EvaluationLayout(
            algorithm=algorithm,
            profiles=profiles,
            selected_profile=profile_name(best),
            run_options=[
                (str(record.evaluation_result_id), evaluation_run_name(record))
                for record in runs.itertuples()
            ],
            selected_run=str(int(best.representative_result_id)),
        )
        self._analysis_view.profile_controls.children = [view.profile_select]
        self._analysis_view.profile_controls.visible = view.profile_select.visible
        self._analysis_view.sweep_heading.children = [view.sweep_heading]
        self.sweep_frames = {algorithm: summary}
        self.profile_selectors = {algorithm: view.profile_select}
        self.sweep_sources = {algorithm: view.sweep_source}
        self.sweep_plots = {algorithm: view.sweep_plot}
        self.sweep_bands = {algorithm: view.sweep_band}
        self._refresh_sweep(algorithm)

        def _select_profile(attr, old, new):
            self._refresh_sweep(algorithm)

        view.profile_select.on_change("value", _select_profile)
        self.composition_source = view.composition_source
        self.evaluation_run_select = view.run_select

        def _select_run(attr, old, new):
            selected = runs.loc[runs.evaluation_result_id == int(new)].iloc[0]
            for label, field in RUN_METRICS.items():
                view.run_values[label].text = f"{selected[field]:.4f}"
            with self.database_scope():
                assignments = self.read(
                    "assignment_details",
                    dataset=self.dataset,
                    run_id=int(selected.run_id),
                    layer=int(selected.layer),
                    evaluation_id=self._analysis_id,
                )
            composition = (
                assignments.groupby(["reference_cluster", "predicted_cluster"])
                .size()
                .reset_index(name="tracks")
            )
            view.composition_source.data = dict(
                predicted=composition.predicted_cluster.astype(str).tolist(),
                reference=composition.reference_cluster.astype(str).tolist(),
                tracks=composition.tracks.tolist(),
                run_key=[selected.run_key] * len(composition),
            )
            view.heatmap.x_range.factors = [
                str(value) for value in sorted(composition.predicted_cluster.unique())
            ]
            view.heatmap.y_range.factors = [
                str(value) for value in sorted(composition.reference_cluster.unique(), reverse=True)
            ]
            view.color_mapper.high = max(1, int(composition.tracks.max()))

        self.evaluation_run_select.on_change("value", _select_run)
        _select_run("value", None, self.evaluation_run_select.value)
        return view.layout


def build_document(document, database=None, source_dir=None):
    return AnalyticsDashboard(document, database, source_dir)
