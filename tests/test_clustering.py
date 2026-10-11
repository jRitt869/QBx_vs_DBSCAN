import geopandas as gpd
import numpy as np
import pytest

from track_clusters.clustering import TemporalCosineMetric, get_dbscan_clusters, get_qb_clusters


@pytest.mark.parametrize(
    "second, expected", [([1.0, 0.0, 0.0], 0.0), ([0.0, 1.0, 0.0], 0.5), ([-1.0, 0.0, 0.0], 1.0)]
)
def test_temporal_metric_matches_normalized_angular_distance(
    second: list[float], expected: float
) -> None:
    metric = TemporalCosineMetric()
    first = np.tile([1.0, 0.0, 0.0], (13, 1))
    assert metric.are_compatible(first.shape, first.shape)
    assert not metric.are_compatible(first.shape, (1, 39))
    assert metric.dist(first, np.tile(second, (13, 1))) == pytest.approx(expected)


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_qbx_assignments_follow_track_order(feature_tracks: gpd.GeoDataFrame, seed: int) -> None:
    result = get_qb_clusters(feature_tracks, 0.1, 0.01, 2, seed)
    assert result["name"].tolist() == feature_tracks["name"].tolist()
    assert result.geometry.equals(feature_tracks.geometry)
    for layer in range(3):
        labels = result[f"clusters{layer}"].to_numpy()
        assert len(set(labels[:12])) == len(set(labels[12:])) == 1
        assert labels[0] != labels[12]
        assert (labels >= 0).all()
    assert result["clusters3"].nunique() == 4
    for start in (0, 6, 12, 18):
        assert result["clusters3"].iloc[start : start + 6].nunique() == 1
    assert "13_pts" in feature_tracks


def test_qbx_repeatability_and_global_rng(feature_tracks: gpd.GeoDataFrame) -> None:
    before = np.random.get_state()
    first = get_qb_clusters(feature_tracks, 0.1, 0.01, 2, 42)
    second = get_qb_clusters(feature_tracks, 0.1, 0.01, 2, 42)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    for layer in range(4):
        np.testing.assert_array_equal(first[f"clusters{layer}"], second[f"clusters{layer}"])
        np.testing.assert_array_equal(
            np.stack(first[f"colors{layer}"]), np.stack(second[f"colors{layer}"])
        )


def test_qbx_uses_all_time_samples(feature_tracks: gpd.GeoDataFrame) -> None:
    tracks = feature_tracks.iloc[:12].copy()
    # Identical first samples must not erase different times later in each track.
    tracks["times"] = tracks["times"].map(lambda times: [[0.0, 0.0, 1.0], *times[1:]])
    assert get_qb_clusters(tracks, 0.1, 0.01, 2, 0)["clusters3"].nunique() == 2


def test_dbscan_shared_temporal_labels_and_nondefault_index(
    feature_tracks: gpd.GeoDataFrame,
) -> None:
    tracks = feature_tracks.copy()
    tracks.index = np.arange(100, 124)
    result = get_dbscan_clusters(tracks, 0.1, 0.01, 5)
    assert result.index.equals(tracks.index)
    assert result["clusters0"].nunique() == 2
    assert result["clusters1"].nunique() == 4
    for start in (0, 6, 12, 18):
        assert result["clusters1"].iloc[start : start + 6].nunique() == 1
    assert "clusters0" not in tracks


@pytest.mark.parametrize("algorithm", ["qbx", "dbscan"])
def test_all_outliers_are_red(feature_tracks: gpd.GeoDataFrame, algorithm: str) -> None:
    tracks = feature_tracks.iloc[:2]
    result = (
        get_qb_clusters(tracks, 0.1, 0.01, 2, 0)
        if algorithm == "qbx"
        else get_dbscan_clusters(tracks, 0.1, 0.01, 5)
    )
    for column in result:
        if column.startswith("clusters"):
            assert (result[column] == -1).all()
        if column.startswith("colors"):
            np.testing.assert_array_equal(np.stack(result[column]), [[255, 0, 0], [255, 0, 0]])


def test_only_temporal_outliers(feature_tracks: gpd.GeoDataFrame) -> None:
    result = get_dbscan_clusters(feature_tracks.iloc[:2], 0.1, 0.01, 1)
    assert (result["clusters0"] >= 0).all()
    assert (result["clusters1"] == -1).all()
