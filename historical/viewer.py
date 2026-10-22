"""SQLite loading for the original internship map; no browser-side data creation."""

import sqlite3
from html import escape
from pathlib import Path

import pandas as pd
from bokeh.document import Document
from bokeh.models import Div
from sqlalchemy import func

from .data import MAP_CRS
from .map_view import ClusterView
from .storage import ClusterDatabase


def map_track_frame(tracks):
    """Project stored tracks into the original map's serializable data columns.

    Args:
        tracks: Source GeoDataFrame with per-layer predictions and RGB colors.
    """
    projected = tracks.to_crs(MAP_CRS)
    columns = ["track_id", "name", *tracks.filter(regex=r"^(clusters|colors)\d+$").columns]
    frame = pd.DataFrame(tracks[columns])
    frame["x"] = [list(geometry.coords.xy[0]) for geometry in projected.geometry]
    frame["y"] = [list(geometry.coords.xy[1]) for geometry in projected.geometry]
    return frame


def _historical_run(database: ClusterDatabase, run_id=None, algorithm=None):
    """Find the latest run using the internship's timestamp-inclusive metrics."""
    filters = {"feature_mode": "spatiotemporal"}
    if run_id is not None:
        filters["run_id"] = run_id
    if algorithm is not None:
        filters["algorithm"] = algorithm
    runs = database.table("runs")
    condition = func.json_extract(runs.c.settings_json, "$.period").is_(None)
    condition &= func.json_extract(runs.c.settings_json, "$.p").is_(None)
    for name, value in filters.items():
        condition &= runs.c[name] == value
    return database.select_exec(
        "runs",
        ("run_id", "algorithm"),
        where=condition,
        order_by=(database.table("runs").c.run_id.desc(),),
        limit=1,
    ).fetchone()


def build_document(
    document: Document,
    data_source: str | Path,
    *,
    run_id: int | None = None,
    algorithm: str | None = None,
) -> ClusterView | None:
    """Load a source's stored spatial/time run, preserving the historical map.

    Without a run ID, select that source's latest timestamp-inclusive run,
    optionally restricted to one algorithm. Spatial-only runs never silently
    substitute for the historical workflow. Every connection is read-only.

    Args:
        document: Bokeh document receiving the map or an actionable source-error message.
        data_source: Existing SQLite source database from which to load stored results.
        run_id: Specific timestamp-inclusive run ID, or None for the latest match.
        algorithm: Optional "dbscan" or "qbx" restriction on the selected run.

    Returns:
        The attached ClusterView, or None when the document displays an error message.
    """
    database = Path(data_source)
    source_name = database.name
    document.title = "GPX Track Clustering"
    try:
        if algorithm not in (None, "dbscan", "qbx"):
            raise ValueError("Algorithm must be dbscan or qbx")
        with ClusterDatabase(database) as source:
            sources = source.select_exec("datasets", ("dataset_key", "name")).fetchall()
            if len(sources) != 1:
                raise ValueError("Each database must contain exactly one track source")
            source_name = f"{sources[0][0]} / {sources[0][1]}"
            run = _historical_run(source, run_id, algorithm)
            if run is None:
                raise ValueError(
                    "No matching timestamp-inclusive clustering run exists in this source database. "
                    "Run python -m historical.get_raw_data DATABASE EPS EPS_TIME first, "
                    "or select an existing spatial/time run."
                )
            tracks = map_track_frame(source.load_tracks(run[0]))
            jumps = source.load_map_jumps(run[0])
            labels = source.load_layer_names(run[0])
        expected = (
            ["Layer 1", "Time"] if run[1] == "dbscan" else ["Layer 1", "Layer 2", "Layer 3", "Time"]
        )
        if labels != expected:
            raise ValueError(
                "The stored spatial/time run is incomplete; rerun its clustering command"
            )
        view = ClusterView(
            tracks,
            jumps,
            labels=labels,
            algorithm="DBSCAN" if run[1] == "dbscan" else "QuickBundlesX",
        )
    except (OSError, ValueError, sqlite3.Error) as error:
        document.add_root(
            Div(
                text=f"<h1>GPX Track Clustering</h1><h2>Source: {escape(source_name)}</h2>"
                f"<p>{escape(str(error))}</p>",
                name="historical_source_message",
            )
        )
        return None
    view.layout.children.insert(
        0,
        Div(
            text=f"<h2>Source: {escape(source_name)}</h2>"
            f"<p>Database: {escape(database.name)} · Run {run[0]}</p>",
            name="historical_source",
        ),
    )
    document.add_root(view.layout)
    return view
