"""The original hierarchical hue/saturation/brightness color scheme."""

from collections.abc import Callable, Iterable

import numpy as np
from dipy.segment.clustering import Cluster, TreeClusterMap
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int_]

SPATIAL_MIN_SIZE = 5
TEMPORAL_MIN_SIZE = 3
OUTLIER_HSB = [0.0, 1.0, 1.0]


def _raw_rgb(param: int) -> list[int]:
    """Map a hue in [0, 1530) onto the original six-part color wheel."""
    if not 0 <= param < 1530:
        raise ValueError("Color parameter must be between 0 and 1529")
    if param < 255:
        return [255, param, 0]
    if param < 510:
        return [510 - param, 255, 0]
    if param < 765:
        return [0, 255, param - 510]
    if param < 1020:
        return [0, 1020 - param, 255]
    if param < 1275:
        return [param - 1020, 0, 255]
    return [255, 0, 1530 - param]


_HUE_RGB = np.array([_raw_rgb(p) for p in range(1530)])
_BASE_RANGE = np.arange(170, 1350)


def _get_color(hsb: FloatArray) -> IntArray:
    """Convert the original custom hue/saturation/brightness representation to RGB.

    Args:
        hsb: Array of shape (tracks, 3), with hue indices into the 1530-color wheel
            and fractional saturation/brightness channels.

    Returns:
        An integer RGB array of shape (tracks, 3).
    """
    rgb = _HUE_RGB[hsb[:, 0].astype(np.intp)] * hsb[:, 2, None]
    maximum = np.max(rgb, axis=1, keepdims=True)
    rgb += (maximum - rgb) * (1 - hsb[:, 1, None])
    return rgb.astype(np.int_)


def hue_ranges(count: int) -> list[IntArray]:
    """Allocate palette intervals, cycling hues when clusters outnumber available hues.

    Args:
        count: Number of clusters needing colors; zero keeps one fallback hue range.

    Returns:
        Integer hue-index arrays spanning the original palette's usable range.
    """
    ranges = np.array_split(_BASE_RANGE, max(1, count))
    if count <= len(_BASE_RANGE):
        return ranges
    return [np.array([_BASE_RANGE[i % len(_BASE_RANGE)]]) for i in range(count)]


def _tree_to_hsb(
    clusters: Iterable[Cluster],
    parent_levels: IntArray,
    min_size: int,
    rule: Callable[[int, int, int], float],
) -> tuple[FloatArray, IntArray]:
    """Assign child cluster labels and one HSB channel using their parent hierarchy."""
    levels = np.full(len(parent_levels), -1, dtype=np.int_)
    channel = np.zeros(len(parent_levels))
    eligible = [c for c in clusters if len(c) >= min_size and parent_levels[c[0]] >= 0]
    parents = np.array([parent_levels[c[0]] for c in eligible], dtype=np.int_)
    counts = np.bincount(parents)
    positions = np.zeros(len(counts), dtype=np.int_)
    for label, (cluster, parent) in enumerate(zip(eligible, parents, strict=True)):
        indices = np.fromiter(cluster, dtype=np.intp, count=len(cluster))
        channel[indices] = rule(int(parent), int(positions[parent]), int(counts[parent]))
        levels[indices] = label
        positions[parent] += 1
    return channel, levels


def _qb_to_colors(
    tree: TreeClusterMap, temporal_clusters: Iterable[Cluster] | None = None
) -> tuple[tuple[IntArray, ...], tuple[IntArray, ...]]:
    """Apply the original hierarchical colors to three spatial layers and optional time.

    Args:
        tree: Three-level QuickBundlesX tree in the clustering input's track ordering.
        temporal_clusters: Optional leaf subdivisions using indices in that same ordering.

    Returns:
        A tuple of RGB arrays and a tuple of assignment arrays, ordered by layer.
        Each layer has one entry per track; excluded clusters are red with label -1.
    """
    track_count = len(tree.root)
    levels1 = np.full(track_count, -1, dtype=np.int_)
    eligible = [c for c in tree.get_clusters(1) if len(c) >= SPATIAL_MIN_SIZE]
    for label, cluster in enumerate(eligible):
        levels1[np.fromiter(cluster, dtype=np.intp, count=len(cluster))] = label
    ranges = hue_ranges(len(eligible))
    hues = np.array([int((r[0] + r[-1]) / 2) for r in ranges])
    hsb = np.tile([0.0, 1.0, 0.9], (track_count, 1))
    hsb[:, 0] = hues[np.maximum(levels1, 0)]
    hsb[levels1 == -1] = OUTLIER_HSB
    colors1 = _get_color(hsb)

    hue, levels2 = _tree_to_hsb(
        tree.get_clusters(2),
        levels1,
        SPATIAL_MIN_SIZE,
        lambda p, k, n: ranges[p][0] + int((ranges[p][-1] - ranges[p][0]) / n * k),
    )
    hsb[:, 0] = hue
    hsb[levels2 == -1] = OUTLIER_HSB
    colors2 = _get_color(hsb)

    brightness, levels3 = _tree_to_hsb(
        tree.get_clusters(3),
        levels2,
        SPATIAL_MIN_SIZE,
        lambda p, k, n: 0.3 * (k / n) + 0.7,
    )
    hsb[:, 2] = brightness
    hsb[levels3 == -1] = OUTLIER_HSB
    colors3 = _get_color(hsb)

    if temporal_clusters is None:
        return (colors1, colors2, colors3), (levels1, levels2, levels3)

    saturation, levels4 = _tree_to_hsb(
        temporal_clusters,
        levels3,
        TEMPORAL_MIN_SIZE,
        lambda p, k, n: 3 * (k / n) / 4 + 0.25,
    )
    hsb[:, 1] = saturation
    hsb[levels4 == -1] = OUTLIER_HSB
    return (colors1, colors2, colors3, _get_color(hsb)), (levels1, levels2, levels3, levels4)
