"""Shared display primitives, labels, and data adapters; no application callbacks."""

import json
from html import escape

import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource,
    DataTable,
    Div,
    HelpButton,
    TableColumn,
    Tooltip,
)
from bokeh.plotting import figure

BACKGROUND = "#F7F9F6"
GREEN = "#147D64"
BLUE = "#5279B8"
ALGORITHMS = {"dbscan": "DBSCAN", "qbx": "QuickBundlesX"}
MEASURES = {
    "Adjusted Rand index": "adjusted_rand_mean",
    "Pair F1": "pair_f1_mean",
    "Adjusted mutual information": "adjusted_mutual_information_mean",
}
RUN_METRICS = {
    "ARI": "adjusted_rand",
    "AMI": "adjusted_mutual_information",
    "F1": "pair_f1",
    "Pair precision": "pair_precision",
    "Pair recall": "pair_recall",
}
STAT_HELP = {
    "ARI": "Adjusted Rand index for this run against the evaluation reference, adjusted for chance.",
    "AMI": "Adjusted mutual information for this run against the evaluation reference, adjusted for chance.",
    "F1": "Harmonic mean of pair precision and pair recall for this run.",
    "Pair precision": "Share of track pairs clustered together that also share a reference group.",
    "Pair recall": "Share of track pairs in the same reference group that are clustered together.",
    "Source files": "GPX files in the selected dataset, including files with no tracks.",
    "Raw tracks": "All original GPX tracks, including tracks too short for clustering.",
    "Eligible tracks": "Tracks with at least five recorded points.",
    "Reference clusters": "Number of distinct clusters among eligible tracks in the reference.",
    "Eligible reference outliers": "Eligible outlier tracks in the reference.",
    "Coordinate points": "All recorded track points, including points in excluded short tracks.",
    "Excluded short tracks": "Tracks under five points in the data set; excluded from clustering.",
    "Gaps longer than 30 minutes": "Number of 30-minute (or greater) gaps between adjacent points in eligible tracks.",
}
SECTION_HELP = {
    "dbscan": "Impact of the parameter on the performance measure; deterministic.",
    "qbx": "Impact of the parameter on the performance measure, for a fixed cluster level.",
    "Reference / prediction overlap": "2D-Histogram over all tracks within the specified run.",
}


def map_track_frame(tracks):
    """Project loaded geometry and retain only the serializable track-map columns."""
    from track_clusters.data import MAP_CRS

    projected = tracks.to_crs(MAP_CRS)
    columns = ["track_id", "name", *tracks.filter(regex=r"^(clusters|colors)\d+$").columns]
    frame = pd.DataFrame(tracks[columns])
    frame["x"] = [list(geometry.coords.xy[0]) for geometry in projected.geometry]
    frame["y"] = [list(geometry.coords.xy[1]) for geometry in projected.geometry]
    return frame


def note_html(text):
    """Escape note text consistently for initial rendering and later updates."""
    return f"<p>{escape(str(text))}</p>"


def note(text, *, name=None, warning=False):
    return Div(
        text=note_html(text),
        name=name,
        sizing_mode="stretch_width",
        styles={"color": "#785116" if warning else "#52625C"},
    )


def heading(text, help_key=None, level=3):
    title = Div(text=f"<h{level}>{escape(text)}</h{level}>", sizing_mode="stretch_width")
    if help_key is None:
        return title
    return row(
        title,
        HelpButton(
            tooltip=Tooltip(content=SECTION_HELP[help_key], position="bottom"), align="center"
        ),
        sizing_mode="stretch_width",
    )


def statistic(label, value):
    return column(
        row(
            Div(text=escape(label), sizing_mode="stretch_width"),
            HelpButton(tooltip=Tooltip(content=STAT_HELP[label], position="bottom")),
            sizing_mode="stretch_width",
        ),
        Div(text=escape(str(value)), name=f"stat_{label}", styles={"font-size": "28px"}),
        sizing_mode="stretch_width",
        min_width=130,
        styles={"background": "white", "padding": "12px", "border": "1px solid #E2E8E1"},
    )


def stats(items):
    return row(*(statistic(label, value) for label, value in items), sizing_mode="stretch_width")


def _table_data(frame):
    """Serialize table values into positional Bokeh fields."""
    # Positional fields support arbitrary SQL aliases, including duplicate column names.
    return {f"c{i}": frame.iloc[:, i].tolist() for i in range(len(frame.columns))}


def _table_columns(labels):
    """Build display columns matching the positional table-data fields."""
    return [TableColumn(field=f"c{i}", title=str(label)) for i, label in enumerate(labels)]


def table(frame, *, name=None, height=240):
    return DataTable(
        source=ColumnDataSource(_table_data(frame)),
        columns=_table_columns(frame.columns),
        index_position=None,
        sizing_mode="stretch_width",
        height=height,
        name=name,
        editable=False,
        autosize_mode="fit_columns",
    )


def set_table(widget, frame):
    widget.columns = _table_columns(frame.columns)
    widget.source.data = _table_data(frame)


def separator(name, *, margin):
    """Use the same visible rule for overview and analysis sections."""
    return Div(
        height=2,
        min_height=2,
        styles={"background-color": "#CFCFCF"},
        margin=margin,
        sizing_mode="stretch_width",
        name=name,
        css_classes=[name.replace("_", "-")],
    )


def analysis_figure(**kwargs):
    """Keep both analysis plot areas the same height and vertical bounds."""
    return style_plot(figure(height=350, min_border_top=10, min_border_bottom=50, **kwargs))


def style_plot(plot):
    plot.background_fill_color = BACKGROUND
    plot.border_fill_color = BACKGROUND
    plot.outline_line_color = None
    plot.axis.axis_label_text_font_style = "normal"
    plot.axis.axis_label_text_color = "#31453D"
    plot.axis.major_label_text_color = "#52625C"
    plot.grid.grid_line_color = "#E7EDE9"
    return plot


def _temporal_profile_name(temporal_mode, p):
    """Describe the primary-feature choice and temporal metric without conflating saved runs."""
    return f"{temporal_mode} · p {p:g}"


def _calibration_name(pairs):
    """Describe the pair limit when the run includes distance calibration."""
    return f" · max {pairs:g} calibration pairs" if pd.notna(pairs) else ""


def clustering_run_name(record):
    settings = json.loads(record.settings_json)
    mode = (
        f"With time · {settings['temporal_mode']}"
        if record.feature_mode == "spatiotemporal"
        else "Spatial"
    )
    details = (
        f" · factor {settings['factor']:g}"
        if record.algorithm == "qbx"
        else f" · min samples {settings['min_samples']}"
    )
    if settings.get("seed") is not None:
        details += f" · seed {settings['seed']}"
    if record.feature_mode == "spatiotemporal":
        details += f" · p {settings['p']:g}"
    details += _calibration_name(settings.get("scale_pairs"))
    return f"Run {record.run_id} · {mode} · ε {settings['eps']:g}{details}"


def evaluation_name(record, profiles):
    manifest = json.loads(record.manifest_json)
    reference = (
        "Hand-made labels"
        if manifest["reference_kind"] == "labels"
        else f"Reference run {manifest['reference_run_id']} "
        f"({manifest['reference_configuration']['algorithm']})"
    )
    title = f"{ALGORITHMS[record.algorithm]} · {record.scope} · {reference}"
    return f"{title} · {' / '.join(profiles)}" if profiles else title


def profile_name(record):
    spatial = (
        f"factor {record.factor:g} · spatial layer {int(record.layer) + 1}"
        if record.algorithm == "qbx"
        else f"min samples {int(record.min_samples)}"
    )
    spatial += _calibration_name(getattr(record, "scale_pairs", None))
    if record.scope != "spatiotemporal":
        return spatial
    cyclic = _temporal_profile_name(
        getattr(record, "temporal_mode", None),
        getattr(record, "p", None),
    )
    temporal_threshold = f" · time ε {record.eps_time:g}" if pd.notna(record.eps_time) else ""
    return f"{spatial}{temporal_threshold} · {record.timezone} · {cyclic}"


def evaluation_run_name(record):
    details = profile_name(record)
    if pd.notna(record.seed):
        details += f" · seed {int(record.seed)}"
    if record.algorithm == "dbscan":
        details += f" · layer {int(record.layer) + 1}"
    return (
        f"Run {record.run_id} · {ALGORITHMS[record.algorithm]} · {record.scope} · "
        f"ε {record.eps:g} · {details}"
    )
