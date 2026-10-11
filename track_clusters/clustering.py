"""QuickBundlesX and DBSCAN, retaining the original spatial/time feature choices."""

import geopandas as gpd
import numpy as np
from dipy.segment.clustering import Cluster, ClusterMap, QuickBundlesX
from dipy.segment.featurespeed import IdentityFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric, Metric
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from .colors import OUTLIER_HSB, TEMPORAL_MIN_SIZE, _get_color, _qb_to_colors, hue_ranges


class TemporalCosineMetric(Metric):
    """DIPY's normalized angular distance over all encoded time samples.

    QuickBundlesX expects three-dimensional streamlines. Keep the 13 x 3 shape
    for its centroid buffers, and flatten only inside the distance calculation.
    Passing 1 x 39 vectors directly to its compiled implementation can crash.
    """

    def __init__(self) -> None:
        super().__init__(feature=IdentityFeature())

    def are_compatible(self, shape1: tuple[int, int], shape2: tuple[int, int]) -> bool:
        return shape1 == shape2 and shape1[1] == 3

    def dist(self, features1: NDArray[np.floating], features2: NDArray[np.floating]) -> float:
        first, second = features1.ravel(), features2.ravel()
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        if denominator == 0:
            return 0.0 if np.array_equal(first, second) else 1.0
        similarity = np.clip(np.dot(first, second) / denominator, -1.0, 1.0)
        return float(np.arccos(similarity) / np.pi)


def get_qb_clusters(
    df: gpd.GeoDataFrame, eps: float, eps_time: float, factor: float, seed: int
) -> gpd.GeoDataFrame:
    """Cluster three spatial layers and subdivide the leaves by time.

    A local RandomState preserves the original seeded permutation without
    changing NumPy's global random state. Output rows retain their input order.
    """
    result = df.copy()
    ordering = np.random.RandomState(seed).permutation(len(df))
    tracks = [df["13_pts"].iloc[i] for i in ordering]
    times = [np.ascontiguousarray(df["times"].iloc[i], dtype=np.float64) for i in ordering]
    tree = QuickBundlesX(
        thresholds=[factor**2 * eps, factor * eps, eps],
        metric=AveragePointwiseEuclideanMetric(),
    ).cluster(tracks)
    temporal_clusters = ClusterMap()
    for spatial_cluster in tree.get_clusters(3):
        vectors = [times[i] for i in spatial_cluster]
        clusters = (
            QuickBundlesX(thresholds=[eps_time], metric=TemporalCosineMetric())
            .cluster(vectors)
            .get_clusters(1)
        )
        for temporal_cluster in clusters:
            temporal_clusters.add_cluster(
                Cluster(indices=[spatial_cluster[i] for i in temporal_cluster])
            )
    colors, labels = _qb_to_colors(tree, temporal_clusters)
    inverse_order = np.argsort(ordering)
    for layer, (rgb, assignments) in enumerate(zip(colors, labels, strict=True)):
        result[f"colors{layer}"] = list(rgb[inverse_order])
        result[f"clusters{layer}"] = assignments[inverse_order]
    return result.drop(columns=["13_pts", "times"])


def get_dbscan_clusters(
    df: gpd.GeoDataFrame, eps: float, eps_time: float, min_s: int
) -> gpd.GeoDataFrame:
    """Standardize flattened spatial tracks; cluster their first time samples."""
    result = df.copy()
    features = np.array([np.asarray(track)[:, :2].ravel() for track in df["13_pts"]])
    features = StandardScaler().fit_transform(features)
    spatial_labels = DBSCAN(eps=eps, min_samples=min_s).fit_predict(features)
    result["clusters0"] = spatial_labels
    n_clusters = int(spatial_labels.max()) + 1
    hsb = np.tile(OUTLIER_HSB, (len(df), 1))
    if n_clusters:
        hues = np.array([int((r[0] + r[-1]) / 2) for r in hue_ranges(n_clusters)])
        assigned = spatial_labels >= 0
        hsb[assigned, 0] = hues[spatial_labels[assigned]]
        hsb[assigned, 1] = 0.5 * spatial_labels[assigned] / n_clusters + 0.5
    result["colors0"] = list(_get_color(hsb))
    temporal_labels = np.full(len(df), -1, dtype=np.int_)
    next_label = 0
    for cluster_id in range(n_clusters):
        positions = np.flatnonzero(spatial_labels == cluster_id)
        times = np.array([df["times"].iloc[i][0] for i in positions], dtype=np.float64)
        labels = DBSCAN(eps=eps_time, min_samples=TEMPORAL_MIN_SIZE, metric="cosine").fit_predict(
            times
        )
        count = int(labels.max()) + 1
        for local_label in range(count):
            members = positions[labels == local_label]
            temporal_labels[members] = next_label
            hsb[members, 2] = local_label / count / 3 + 2 / 3
            next_label += 1
        hsb[positions[labels == -1]] = OUTLIER_HSB
    result["clusters1"] = temporal_labels
    result["colors1"] = list(_get_color(hsb))
    return result.drop(columns=["13_pts", "times"])
