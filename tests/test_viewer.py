from pathlib import Path

import geopandas as gpd
import pandas as pd
from bokeh.core.validation import check_integrity
from bokeh.document import Document

from track_clusters.clustering import get_dbscan_clusters, get_qb_clusters
from track_clusters.viewer import ClusterView, build_document


def _table(tracks: gpd.GeoDataFrame) -> pd.DataFrame:
    table = pd.DataFrame(tracks.drop(columns="geometry"))
    table["x"] = [[0.0, 1000.0]] * len(tracks)
    table["y"] = [[0.0, 1000.0]] * len(tracks)
    return table


def _jumps() -> pd.DataFrame:
    return pd.DataFrame(
        {"jump_name": ["track_0"], "jump_hours": [0], "jump_minutes": [42], "x": [0.0], "y": [0.0]}
    )


def test_controls_visibility_and_serialization(feature_tracks: gpd.GeoDataFrame) -> None:
    view = ClusterView(_table(get_dbscan_clusters(feature_tracks.iloc[:2], 0.1, 0.01, 5)), _jumps())
    assert view.layer_selector.active == 0
    assert view.track_source.data["is_visible"] == [1.0, 1.0]
    assert view.track_source.data["color"] == ["#ff0000", "#ff0000"]
    view.hide_toggle.active = True
    assert view.track_source.data["is_visible"] == [0.0, 0.0]
    view.hide_toggle.active = False
    view.layer_selector.active = 1
    assert view.layer == 1
    view.jump_toggle.active = True
    assert view.jump_source.data["is_visible"] == [1.0]
    assert "2 outliers" in view.summary.text
    document = Document()
    document.add_root(view.layout)
    assert not check_integrity(document.models).error
    document.to_json()


def test_layers_colors_and_session_independence(feature_tracks: gpd.GeoDataFrame) -> None:
    table = _table(get_qb_clusters(feature_tracks, 0.1, 0.01, 2, 7))
    first, second = ClusterView(table, _jumps()), ClusterView(table, _jumps())
    first.layer_selector.active = 3
    assert first.track_source.data["cluster"] == table["clusters3"].tolist()
    assert first.layer == 3 and second.layer == 0
    assert "4 clusters" in first.summary.text
    first.color_toggle.active = False
    assert set(first.track_source.data["color"]) == {"#00c800"}
    assert second.color_toggle.active
    first.color_toggle.active = True
    assert first.track_source.data["color"] == first.track_source.data["colors3"]


def test_missing_data(tmp_path: Path) -> None:
    document = Document()
    assert build_document(document, tmp_path) is None
    assert "Preprocess" in document.roots[0].text
