"""Dashboard visual groups built from prepared data, without queries or callbacks.

Each instance owns fresh Bokeh models. The controller fills their data sources,
registers interactions, and attaches the completed layouts to the document.
"""

from bokeh.layouts import column, row
from bokeh.models import (
    Band,
    Button,
    ColorBar,
    ColumnDataSource,
    Div,
    FactorRange,
    GridBox,
    HoverTool,
    LinearColorMapper,
    RadioButtonGroup,
    Select,
    TabPanel,
    Tabs,
)
from bokeh.palettes import Blues9
from bokeh.plotting import figure

from .presentation import (
    ALGORITHMS,
    BACKGROUND,
    BLUE,
    GREEN,
    MEASURES,
    RUN_METRICS,
    analysis_figure,
    heading,
    note,
    separator,
    statistic,
    stats,
    style_plot,
    table,
)

ANALYSIS_TITLE = "How do the clustering choices behave?"


def message_panel(text, *, title=None, name=None, warning=False):
    children = [heading(title, level=1)] if title is not None else []
    return column(*children, note(text, name=name, warning=warning), sizing_mode="stretch_width")


def tab_panels(kinds):
    names = {**ALGORITHMS, "overview": "Overview", "analysis": "Analysis"}
    return [TabPanel(title=names[kind], child=column(Div())) for kind in kinds]


def source_tabs(panels, active):
    return Tabs(
        tabs=panels,
        active=active,
        sizing_mode="stretch_width",
        name="dashboard_tabs",
        css_classes=["dashboard-tabs"],
        # Hidden cached panels must not reserve the tallest visited tab's height.
        stylesheets=[':host > [style*="visibility: hidden"] { display: none !important; }'],
    )


class DashboardLayout:
    """Page header, source controls, and tab container."""

    def __init__(self, sources):
        self.dataset_select = Select(
            title="Data source",
            options=sources,
            value=next(iter(sources), ""),
            disabled=not sources,
            name="data_source",
            width=300,
        )
        self.dataset_note = note("", name="dataset_provenance")
        self.database_note = note("", name="source_database")
        self.tab_container = column(sizing_mode="stretch_width")
        self.layout = column(
            row(
                column(
                    Div(
                        text='<h1 style="margin: 0">GPX Trajectory Analytics</h1>',
                        sizing_mode="stretch_width",
                    ),
                    self.dataset_note,
                    sizing_mode="stretch_width",
                    align="start",
                ),
                column(self.dataset_select, self.database_note, width=310, align="start"),
                sizing_mode="stretch_width",
            ),
            self.tab_container,
            sizing_mode="stretch_width",
            styles={"padding": "16px", "background": BACKGROUND, "color": "#213E33"},
        )


class ClusteringLayout:
    """Run selector and replaceable map area for a clustering tab."""

    def __init__(self, algorithm, options, selected):
        self.run_select = Select(
            title=f"{ALGORITHMS[algorithm]} run",
            options=options,
            value=selected,
            sizing_mode="stretch_width",
            name=f"{algorithm}_run",
        )
        self.body = column(sizing_mode="stretch_width")
        self.layout = column(
            heading("GPX Track Clustering", level=1),
            self.run_select,
            self.body,
            sizing_mode="stretch_width",
        )


def clustering_map(view, *, has_time, joint=False):
    """Arrange the existing map widgets without changing their interactions."""
    description = (
        "Explore joint spatial/time clustering layers using day, week, and month. "
        "Red tracks are outliers. Select a layer and hover over a track for details."
        if joint
        else "Explore spatial clusters and their temporal subdivisions. "
        "Red tracks are outliers. Select a layer and hover over a track for details."
        if has_time
        else "Explore spatial clustering layers. Red tracks are outliers. "
        "Select a layer and hover over a track for details. "
        "This run contains no temporal subdivision."
    )
    view.layer_selector.align = "center"
    return column(
        note(description),
        row(
            view.layer_selector,
            row(view.hide_toggle, view.color_toggle, view.jump_toggle, align="center"),
            spacing=15,
            sizing_mode="stretch_width",
        ),
        view.summary,
        view.plot,
        sizing_mode="stretch_width",
    )


class OverviewLayout:
    """Input statistics, reference distribution, catalog, and timestamp-gap groups."""

    def __init__(
        self,
        *,
        totals,
        references,
        reference_details,
        reference_caption,
        tracks,
        statuses,
        gaps,
        gap_count,
        has_timestamps,
        duplicate_count,
    ):
        self.reference_source = ColumnDataSource(
            dict(label=[], tracks=[], share=[], start_angle=[], end_angle=[], color=[]),
            name="reference_distribution",
        )
        distribution = style_plot(
            figure(
                x_range=(-1.1, 1.1),
                y_range=(-1.1, 1.1),
                height=470,
                sizing_mode="stretch_width",
                tools="hover,save",
                tooltips=[
                    ("Reference", "@label"),
                    ("Tracks", "@tracks"),
                    ("Share", "@share{0.0%}"),
                ],
                name="reference_distribution_plot",
                css_classes=["reference-pie"],
            )
        )
        distribution.wedge(
            x=0,
            y=0,
            radius=170,
            radius_units="screen",
            start_angle="start_angle",
            end_angle="end_angle",
            fill_color="color",
            line_color="white",
            legend_field="label",
            source=self.reference_source,
        )
        distribution.axis.visible = False
        distribution.grid.visible = False
        distribution.add_layout(distribution.legend[0], "right")
        distribution.legend.label_text_font_size = "11px"
        self.status_select = Select(
            title="Track status",
            options=statuses,
            value="All statuses",
            name="track_status",
            width=260,
            align=("end", "center"),
            css_classes=["track-status-control"],
        )
        self.catalog = table(tracks, name="track_catalog", height=350)
        self.catalog.css_classes = ["catalog-table"]
        self.catalog_download = Button(label="Download CSV", width=150)
        gap_stat = statistic("Gaps longer than 30 minutes", gap_count)
        gap_stat.margin = (5, 0, 0, 0)
        gap_stat.css_classes = ["timestamp-gap-count"]
        self.gap_table = table(gaps, name="timestamp_gaps", height=30 + 25 * len(gaps))
        self.gap_table.min_height = 30
        self.gap_table.margin = 0
        self.gap_table.align = ("start", "center")
        self.gap_table.css_classes = ["timestamp-gap-table"]
        reference_stats = stats(references)
        reference_stats.css_classes = ["reference-stats"]
        details = note(reference_details, name="overview_reference_source")
        details.max_height = 140
        details.styles.update(
            {"white-space": "pre-line", "font-size": "12px", "overflow-y": "auto"}
        )
        self.layout = column(
            heading("What is in the sample?", level=1),
            stats(totals),
            GridBox(
                children=[
                    (
                        column(
                            heading("What does the reference look like?", level=2),
                            note(reference_caption),
                            sizing_mode="stretch_width",
                            align=("start", "end"),
                        ),
                        0,
                        0,
                    ),
                    (reference_stats, 1, 0),
                    (details, 2, 0),
                    (distribution, 0, 1, 3, 1),
                ],
                rows=["1fr", "max-content", "1fr"],
                cols=["1fr", "1fr"],
                height=distribution.height,
                sizing_mode="stretch_width",
            ),
            separator("overview_separator", margin=(12, 0, 8, 0)),
            GridBox(
                children=[
                    (
                        row(
                            heading("Track catalog", level=2),
                            self.status_select,
                            spacing=16,
                            margin=0,
                            sizing_mode="stretch_width",
                        ),
                        0,
                        0,
                    ),
                    (heading("Timestamp gaps", level=2), 0, 1),
                    (
                        column(
                            self.catalog,
                            sizing_mode="stretch_width",
                            align="start",
                            css_classes=["track-catalog-panel"],
                        ),
                        1,
                        0,
                    ),
                    (
                        column(
                            gap_stat,
                            note(
                                "Adjacent timed points within eligible tracks, including joined GPX segments. Missing times stay NULL; timestamps are UTC."
                                if has_timestamps
                                else "This sample has no timestamps. Temporal analysis is unavailable; no times are invented."
                            ),
                            GridBox(
                                children=[(self.gap_table, 0, 0)],
                                rows=["minmax(0, 1fr)"],
                                cols=["1fr"],
                                sizing_mode="stretch_both",
                                min_height=30,
                                margin=5,
                                css_classes=["timestamp-gap-table-area"],
                            ),
                            sizing_mode="stretch_both",
                            css_classes=["timestamp-gaps-panel"],
                        ),
                        1,
                        1,
                    ),
                    (
                        note(
                            f"Eligible geometry duplicates: {duplicate_count:,}",
                            name="eligible_geometry_duplicates",
                        ),
                        2,
                        0,
                    ),
                    (self.catalog_download, 3, 0),
                ],
                cols=["2fr", "1fr"],
                spacing=(0, 32),
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        )


class AnalysisLayout:
    """Full-width evaluation selection below the title and controls above the results."""

    def __init__(self, options, selected):
        self.analysis_select = Select(
            title="Select evaluation run",
            options=options,
            value=selected,
            name="evaluation_scope",
            sizing_mode="stretch_width",
        )
        self.metric_select = RadioButtonGroup(
            labels=list(MEASURES),
            active=0,
            name="performance_measure",
            width=460,
            margin=0,
            stylesheets=[
                ".bk-btn { min-height: calc(var(--line-height-computed) "
                "+ 2 * var(--padding-vertical) + 2px); }"
            ],
        )
        self.profile_controls = column(
            sizing_mode="stretch_width", name="evaluation_profiles", visible=False
        )
        self.sweep_heading = column(sizing_mode="stretch_width", margin=0)
        self.sweep_controls = column(
            self.sweep_heading,
            self.profile_controls,
            column(
                Div(text="Performance measure", margin=0),
                self.metric_select,
                width=460,
                spacing=0,
                margin=(0, 5, 5, 5),
            ),
            sizing_mode="stretch_width",
        )
        self.content = column(sizing_mode="stretch_width")
        self.layout = column(
            heading(ANALYSIS_TITLE, level=1),
            self.analysis_select,
            note(
                "Compare threshold sensitivity and seed variation. Each method keeps its own feature space and distance convention, so epsilon values are interpreted within each method."
            ),
            self.sweep_controls,
            self.content,
            sizing_mode="stretch_width",
        )


class EvaluationLayout:
    """A wide epsilon chart above centered run scores and their overlap chart."""

    def __init__(self, *, algorithm, profiles, selected_profile, run_options, selected_run):
        self.profile_select = Select(
            title=f"{ALGORITHMS[algorithm]} parameters",
            options=profiles,
            value=selected_profile,
            visible=len(profiles) > 1,
            sizing_mode="stretch_width",
        )
        self.sweep_heading = heading(
            "Evaluation performance across primary epsilon", algorithm, level=2
        )
        self.sweep_heading.margin = (5, 5, 0, 5)
        self.sweep_heading.select_one({"type": Div}).stylesheets = ["h2 { margin-bottom: 0; }"]
        self.sweep_source = ColumnDataSource(dict(eps=[], mean=[], low=[], high=[], runs=[]))
        color = GREEN if algorithm == "dbscan" else BLUE
        self.sweep_plot = analysis_figure(
            sizing_mode="stretch_width",
            x_axis_type="log",
            x_axis_label="Primary epsilon",
            y_axis_label="Adjusted Rand index",
            tools="pan,wheel_zoom,save,reset",
        )
        self.sweep_band = Band(
            base="eps",
            lower="low",
            upper="high",
            source=self.sweep_source,
            fill_color=color,
            fill_alpha=0.12,
            line_alpha=0,
        )
        self.sweep_plot.add_layout(self.sweep_band)
        self.sweep_plot.line("eps", "mean", source=self.sweep_source, color=color, line_width=2)
        dots = self.sweep_plot.scatter("eps", "mean", source=self.sweep_source, color=color, size=7)
        self.sweep_plot.add_tools(
            HoverTool(
                renderers=[dots],
                tooltips=[("Epsilon", "@eps"), ("Mean", "@mean{0.0000}"), ("Runs", "@runs")],
            )
        )
        self.composition_source = ColumnDataSource(
            dict(predicted=[], reference=[], tracks=[], run_key=[])
        )
        self.color_mapper = LinearColorMapper(palette=list(reversed(Blues9)), low=0, high=1)
        self.heatmap = analysis_figure(
            x_range=FactorRange(),
            y_range=FactorRange(),
            sizing_mode="stretch_width",
            tools="hover,save,reset",
            tooltips=[
                ("Reference", "@reference"),
                ("Predicted", "@predicted"),
                ("Tracks", "@tracks"),
                ("Run", "@run_key"),
            ],
            x_axis_label="Predicted cluster (−1 = noise)",
            y_axis_label="Reference group (−1 = outlier)",
            name="evaluation_overlap",
        )
        self.heatmap.rect(
            "predicted",
            "reference",
            width=1,
            height=1,
            source=self.composition_source,
            fill_color={"field": "tracks", "transform": self.color_mapper},
            line_color="white",
        )
        self.heatmap.add_layout(ColorBar(color_mapper=self.color_mapper), "right")
        self.run_select = Select(
            options=run_options,
            value=selected_run,
            name="evaluation_run",
            sizing_mode="stretch_width",
        )
        run_statistics = stats((label, "") for label in RUN_METRICS)
        self.run_values = {
            label: run_statistics.select_one({"name": f"stat_{label}"}) for label in RUN_METRICS
        }
        run_description = note(
            "Select a run to view its agreement scores against the reference used in this evaluation."
        )
        run_description.stylesheets = ["p { margin: 0; }"]
        overlap_heading = heading(
            "Reference / prediction overlap", "Reference / prediction overlap"
        )
        overlap_heading.select_one({"type": Div}).styles["text-align"] = "center"
        self.layout = column(
            GridBox(
                children=[
                    (self.sweep_plot, 0, 0),
                    (
                        note(
                            "Lines show seed means. For ARI, shaded bands show the observed seed minimum–maximum, not confidence intervals. DBSCAN has one deterministic run per threshold."
                        ),
                        1,
                        0,
                    ),
                ],
                cols=["1fr"],
                sizing_mode="stretch_width",
                name="epsilon_chart_panel",
            ),
            separator("analysis_separator2", margin=(12, 0, 8, 0)),
            GridBox(
                children=[
                    (overlap_heading, 0, 1),
                    (
                        GridBox(
                            children=[
                                (
                                    column(
                                        heading("Per-run performance of this evaluation", level=2),
                                        run_description,
                                        self.run_select,
                                        sizing_mode="stretch_width",
                                        align=("start", "end"),
                                    ),
                                    0,
                                    0,
                                ),
                                (run_statistics, 1, 0),
                            ],
                            rows=["minmax(max-content, 1fr)", "max-content", "minmax(0, 1fr)"],
                            cols=["1fr"],
                            height=self.heatmap.height,
                            sizing_mode="stretch_width",
                            name="evaluation_run_summary",
                        ),
                        1,
                        0,
                    ),
                    (self.heatmap, 1, 1),
                    (
                        note(
                            "Cluster IDs are arbitrary; the diagonal is not an accuracy measure. This shows overlap for the selected run against this evaluation's reference."
                        ),
                        2,
                        1,
                    ),
                ],
                cols=["2fr", "1fr"],
                spacing=(0, 48),
                sizing_mode="stretch_width",
                name="evaluation_run_details",
            ),
            sizing_mode="stretch_width",
        )
