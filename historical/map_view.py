"""Bokeh document construction with independent state for every browser session."""

from collections.abc import Iterable

import numpy as np
import pandas as pd
import xyzservices.providers as xyz
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource,
    Div,
    HoverTool,
    RadioButtonGroup,
    Range1d,
    SaveTool,
    Toggle,
)
from bokeh.plotting import figure

LABELS_QBX = ["Layer 1", "Layer 2", "Layer 3", "Time"]
LABELS_DBSCAN = ["Layer 1", "Time"]


def _css_colors(values: Iterable[Iterable[int]]) -> list[str]:
    """Convert stored RGB triplets into hexadecimal Bokeh colors."""
    return ["#{:02x}{:02x}{:02x}".format(*(int(v) for v in rgb)) for rgb in values]


def _range(values: pd.Series) -> Range1d:
    """Fit a map range around all supplied coordinates with a visible margin."""
    coordinates = np.concatenate([np.asarray(v, dtype=float) for v in values])
    lower, upper = float(coordinates.min()), float(coordinates.max())
    margin = max((upper - lower) * 0.1, 500)
    return Range1d(start=lower - margin, end=upper + margin)


class ClusterView:
    """Own the map sources, widgets, and callbacks for one Bokeh document."""

    def __init__(
        self, tracks: pd.DataFrame, jumps: pd.DataFrame, *, labels=None, algorithm=None
    ) -> None:
        """Build a session-local map, display sources, and clustering controls.

        Args:
            tracks: Nonempty frame with projected x/y sequences, track names, and one
                clustersN/colorsN pair per layer; colors are stored RGB triplets.
            jumps: Projected x/y markers with jump_name, jump_hours, and jump_minutes.
            labels: Ordered layer names; None infers the historical DBSCAN/QBx names.
            algorithm: Display name for the plot title; None infers it from the layer count.
        """
        if tracks.empty:
            raise ValueError("The track table is empty; preprocess GPX files first")
        self.layer = 0
        data = {name: tracks[name].tolist() for name in tracks.columns}
        self.labels = (
            labels
            if labels is not None
            else (LABELS_QBX if "colors2" in tracks.columns else LABELS_DBSCAN)
        )
        for layer in range(len(self.labels)):
            data[f"colors{layer}"] = _css_colors(data[f"colors{layer}"])
        data.update(
            color=data["colors0"], cluster=data["clusters0"], is_visible=[1.0] * len(tracks)
        )
        self.track_source = ColumnDataSource(data, name="tracks")
        jump_data = {name: jumps[name].tolist() for name in jumps.columns}
        jump_data["is_visible"] = [0.0] * len(jumps)
        self.jump_source = ColumnDataSource(jump_data, name="timeskips")
        self.layer_selector = RadioButtonGroup(labels=self.labels, active=0, name="layer_selector")
        self.hide_toggle = Toggle(label="Hide Outliers", name="hide_outliers")
        self.color_toggle = Toggle(label="Show Clusters", active=True, name="show_clusters")
        self.jump_toggle = Toggle(label="Show Timeskips", name="show_timeskips")
        self.summary = Div(name="cluster_summary")
        self.layer_selector.on_change("active", self._change_layer)
        self.hide_toggle.on_change("active", self._refresh_callback)
        self.color_toggle.on_change("active", self._refresh_callback)
        self.jump_toggle.on_change("active", self._change_jumps)

        algorithm = algorithm or ("QuickBundlesX" if len(self.labels) == 4 else "DBSCAN")
        self.plot = figure(
            title=f"GPX track clustering · {algorithm}",
            x_range=_range(tracks["x"]),
            y_range=_range(tracks["y"]),
            x_axis_type="mercator",
            y_axis_type="mercator",
            height=650,
            sizing_mode="stretch_width",
            active_scroll="wheel_zoom",
            tools="pan,wheel_zoom,save,reset",
            match_aspect=True,
        )
        self.plot.add_tile(xyz.BaseMapDE.Grey)
        self.plot.select_one(SaveTool).filename = "gpx-track-clusters"
        lines = self.plot.multi_line(
            "x", "y", source=self.track_source, color="color", line_width=3, alpha="is_visible"
        )
        circles = self.plot.scatter(
            "x",
            "y",
            source=self.jump_source,
            marker="circle",
            size=10,
            line_color="black",
            fill_color="white",
            alpha="is_visible",
        )
        self.plot.add_tools(
            HoverTool(renderers=[lines], tooltips=[("track", "@name"), ("cluster", "@cluster")]),
            HoverTool(
                renderers=[circles],
                tooltips=[
                    ("track", "@jump_name"),
                    ("hours", "@jump_hours"),
                    ("minutes", "@jump_minutes"),
                ],
            ),
        )
        self.layout = column(
            Div(
                text="<h1>GPX Track Clustering</h1><p>Explore spatial clusters and their temporal subdivisions. "
                "Red tracks are outliers. Select a layer and hover over a track for details.</p>"
            ),
            self.layer_selector,
            self.summary,
            self.plot,
            row(self.hide_toggle, self.color_toggle, self.jump_toggle, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )
        self._refresh()

    def _change_layer(self, attr: str, old: int | None, new: int | None) -> None:
        """Switch the displayed layer when Bokeh reports a layer-selector change."""
        if new is not None:
            self.layer = new
            self._refresh()

    def _refresh_callback(self, attr: str, old: bool, new: bool) -> None:
        """Refresh tracks after an outlier or color toggle changes."""
        self._refresh()

    def _refresh(self) -> None:
        """Apply current layer/color/outlier controls and update the cluster summary."""
        data = dict(self.track_source.data)
        clusters = data[f"clusters{self.layer}"]
        colors = (
            data[f"colors{self.layer}"]
            if self.color_toggle.active
            else ["#ff0000" if label == -1 else "#00c800" for label in clusters]
        )
        visible = [0.0 if label == -1 and self.hide_toggle.active else 1.0 for label in clusters]
        data.update(color=colors, cluster=clusters, is_visible=visible)
        self.track_source.data = data
        outliers = sum(label == -1 for label in clusters)
        count = len({label for label in clusters if label >= 0})
        self.summary.text = (
            f"<p><b>{self.labels[self.layer]}</b> · {len(clusters)} tracks · "
            f"{count} clusters · {outliers} outliers</p>"
        )

    def _change_jumps(self, attr: str, old: bool, new: bool) -> None:
        """Show or hide the stored time-gap markers in response to their toggle."""
        data = dict(self.jump_source.data)
        data["is_visible"] = [float(new)] * len(data["x"])
        self.jump_source.data = data
