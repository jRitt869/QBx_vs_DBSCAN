"""QuickBundlesX and DBSCAN, retaining the original spatial/time feature choices."""

import geopandas as gpd
import numpy as np
from dipy.segment.clustering import Cluster, ClusterMap, QuickBundlesX
from dipy.segment.featurespeed import IdentityFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric, Metric
from numpy.typing import NDArray
from pydantic import ConfigDict, validate_call
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from .colors import OUTLIER_HSB, TEMPORAL_MIN_SIZE, _get_color, _qb_to_colors, hue_ranges
from .validation import Factor, PositiveNumber, Seed, qbx_thresholds


class TemporalCosineMetric(Metric):
    """DIPY's normalized angular distance over all encoded time samples.

    QuickBundlesX expects three-dimensional streamlines. Keep the 13 x 3 shape
    for its centroid buffers, and flatten only inside the distance calculation.
    Passing 1 x 39 vectors directly to its compiled implementation can crash.
    """

    def __init__(self) -> None:
        """Initialize DIPY's metric with identity features that retain 13-by-3 samples."""
        super().__init__(feature=IdentityFeature())

    def are_compatible(self, shape1: tuple[int, int], shape2: tuple[int, int]) -> bool:
        """Check the feature-shape compatibility required by DIPY's metric interface.

        Args:
            shape1: First feature array's (sample_count, coordinate_count) shape.
            shape2: Second feature array's shape in the same convention.

        Returns:
            True for equal shapes with three coordinates per sample.
        """
        return shape1 == shape2 and shape1[1] == 3

    def dist(self, features1: NDArray[np.floating], features2: NDArray[np.floating]) -> float:
        """Compute normalized angular distance over all encoded temporal samples.

        Args:
            features1: First compatible array of encoded weekday/hour vectors.
            features2: Second compatible array, flattened only for this calculation.

        Returns:
            Angular distance divided by pi in [0, 1]. Equal zero vectors have distance
            zero; an otherwise zero denominator yields distance one.
        """
        first, second = features1.ravel(), features2.ravel()
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        if denominator == 0:
            return 0.0 if np.array_equal(first, second) else 1.0
        similarity = np.clip(np.dot(first, second) / denominator, -1.0, 1.0)
        return float(np.arccos(similarity) / np.pi)


@validate_call(config=ConfigDict(arbitrary_types_allowed=True, validate_default=True))
def get_qb_clusters(
    df: gpd.GeoDataFrame,
    eps: PositiveNumber,
    eps_time: PositiveNumber | None = None,
    factor: Factor = 2,
    seed: Seed = 0,
) -> gpd.GeoDataFrame:
    """Cluster three primary layers, optionally subdividing the leaves by time.

    eps_time=None skips temporal clustering and does not require a times column.
    Spatial inputs retain their padded three-coordinate representation.
    A local RandomState preserves the original seeded permutation without
    changing NumPy's global random state. Output rows retain their input order.
    Pydantic validates numerical constraints from the parameter annotations.

    Args:
        df: Nonempty track frame with padded 13_pts samples and, when enabled, times arrays.
        eps: Positive finest spatial threshold applied to the supplied coordinate units.
        eps_time: Temporal angular threshold, or None to omit temporal subdivision.
        factor: Spatial threshold multiplier, producing [factor**2 * eps, factor * eps, eps].
        seed: Local input-permutation seed, preserving the original seeded hierarchy.

    Returns:
        A copied frame in input row order with clustersN/colorsN columns and the
        temporary 13_pts/times feature columns removed.
    """
    if df.empty:
        raise ValueError("Cannot cluster an empty dataset")
    result = df.copy()
    ordering = np.random.RandomState(seed).permutation(len(df))
    tracks = [df["13_pts"].iloc[i] for i in ordering]
    tree = QuickBundlesX(
        thresholds=qbx_thresholds(eps, factor),
        metric=AveragePointwiseEuclideanMetric(),
    ).cluster(tracks)
    temporal_clusters = None
    if eps_time is not None:
        times = [np.ascontiguousarray(df["times"].iloc[i], dtype=np.float64) for i in ordering]
        temporal_clusters = ClusterMap()
        for spatial_cluster in tree.get_clusters(3):
            vectors = [times[i] for i in spatial_cluster]
            clusters = (
                QuickBundlesX(
                    thresholds=[eps_time],
                    metric=TemporalCosineMetric(),
                )
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
    return result.drop(columns=["13_pts", "times"], errors="ignore")


def get_dbscan_clusters(
    df: gpd.GeoDataFrame,
    eps: float,
    eps_time: float | None = None,
    min_s: int = 5,
) -> gpd.GeoDataFrame:
    """Cluster scaled spatial coordinates, then optionally subdivide by time.

    Spatial clustering uses flattened Euclidean distance. Temporal subdivision
    uses the original first-sample cosine metric.
    Each 13_pts array can have any spatial coordinate dimension.

    Args:
        df: Track frame with 13_pts samples and, when enabled, encoded times arrays.
        eps: Spatial neighborhood threshold after flattening and scaling.
        eps_time: Temporal metric threshold, or None for spatial-only clustering.
        min_s: Minimum spatial DBSCAN neighborhood size; temporal clustering uses three.

    Returns:
        A copied frame with spatial and optional temporal clustersN/colorsN columns,
        preserving row order and removing the temporary feature columns.
    """
    result = df.copy()
    features = np.array([np.asarray(track).ravel() for track in df["13_pts"]])
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
    if eps_time is None:
        return result.drop(columns=["13_pts", "times"], errors="ignore")
    temporal_labels = np.full(len(df), -1, dtype=np.int_)
    next_label = 0
    for cluster_id in range(n_clusters):
        positions = np.flatnonzero(spatial_labels == cluster_id)
        times = np.array(
            [df["times"].iloc[i][0] for i in positions],
            dtype=np.float64,
        )
        labels = DBSCAN(
            eps=eps_time,
            min_samples=TEMPORAL_MIN_SIZE,
            metric="cosine",
        ).fit_predict(times)
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
