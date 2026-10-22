"""QuickBundlesX and DBSCAN kernels for calibrated spatial and cyclic-time features."""

import geopandas as gpd
import numpy as np
from dipy.segment.clustering import Cluster, ClusterMap, QuickBundlesX
from dipy.segment.metric import Metric
from pydantic import ConfigDict, validate_call
from sklearn.cluster import DBSCAN

from .colors import OUTLIER_HSB, TEMPORAL_MIN_SIZE, _get_color, _qb_to_colors, hue_ranges
from .validation import Factor, PositiveNumber, Seed, qbx_thresholds


@validate_call(config=ConfigDict(arbitrary_types_allowed=True, validate_default=True))
def get_qb_clusters(
    df: gpd.GeoDataFrame,
    eps: PositiveNumber,
    eps_time: PositiveNumber | None = None,
    factor: Factor = 2,
    seed: Seed = 0,
    *,
    metric: Metric,
    temporal_metric: Metric | None = None,
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
        metric: Explicit DIPY metric for the supplied primary feature arrays.
        temporal_metric: Optional DIPY metric for temporal subdivision features.

    Returns:
        A copied frame in input row order with clustersN/colorsN columns and the
        temporary 13_pts/times feature columns removed.
    """
    if eps_time is not None and temporal_metric is None:
        raise ValueError("Temporal clustering requires a temporal metric")
    if df.empty:
        raise ValueError("Cannot cluster an empty dataset")
    result = df.copy()
    ordering = np.random.RandomState(seed).permutation(len(df))
    tracks = [df["13_pts"].iloc[i] for i in ordering]
    tree = QuickBundlesX(
        thresholds=qbx_thresholds(eps, factor),
        metric=metric,
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
                    metric=temporal_metric,
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
    *,
    metric,
    temporal_metric=None,
) -> gpd.GeoDataFrame:
    """Cluster scaled spatial coordinates, then optionally subdivide by time.

    Primary clustering uses flattened features and the supplied metric. Temporal subdivision
    uses the supplied distance callable over every temporal sample.
    Each 13_pts array can have any spatial coordinate dimension.

    Args:
        df: Track frame with 13_pts samples and, when enabled, encoded times arrays.
        eps: Spatial neighborhood threshold after flattening and scaling.
        eps_time: Temporal metric threshold, or None for spatial-only clustering.
        min_s: Minimum primary DBSCAN neighborhood size; temporal clustering uses three.
        metric: Primary distance callable receiving two point arrays with the same
            shape as each track's 13_pts array. Scaling belongs to this metric.
        temporal_metric: Optional distance callable using every temporal sample.

    Returns:
        A copied frame with spatial and optional temporal clustersN/colorsN columns,
        preserving row order and removing the temporary feature columns.
    """
    if eps_time is not None and temporal_metric is None:
        raise ValueError("Temporal clustering requires a temporal metric")
    if df.empty:
        raise ValueError("Cannot cluster an empty dataset")
    result = df.copy()
    features = np.array([np.asarray(track).ravel() for track in df["13_pts"]])
    shape = np.asarray(df["13_pts"].iloc[0]).shape
    spatial_labels = DBSCAN(
        eps=eps,
        min_samples=min_s,
        metric=lambda first, second: metric(first.reshape(shape), second.reshape(shape)),
    ).fit_predict(features)
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
            [np.asarray(df["times"].iloc[i]).ravel() for i in positions],
            dtype=np.float64,
        )
        labels = DBSCAN(
            eps=eps_time,
            min_samples=TEMPORAL_MIN_SIZE,
            metric=temporal_metric,
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
