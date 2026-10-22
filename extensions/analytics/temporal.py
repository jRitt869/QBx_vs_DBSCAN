"""Calendar phases and cyclic metrics for database-backed spatiotemporal runs."""

import calendar
from datetime import datetime, tzinfo

import numpy as np
from dipy.segment.featurespeed import IdentityFeature
from dipy.segment.metric import Metric
from dipy.tracking.streamline import set_number_of_points

from track_clusters.data import RESAMPLED_POINTS


def resample_spatial_track(
    coordinates, timestamps=None, point_count=RESAMPLED_POINTS
) -> np.ndarray:
    """Resample spatial arc length, optionally aligning epochs to the same positions.

    Args:
        coordinates: Ordered (n, 2) longitude/latitude coordinates.
        timestamps: Aligned chronological Unix epochs, or None for spatial samples only.
            The caller validates timestamps before resampling.
        point_count: Number of output samples, normally thirteen.

    Returns:
        A (point_count, 2) array of spatial samples, with a third epoch column when supplied.
        Time jumps from arrival to departure at pauses; an interior sample exactly
        at a pause uses its departure epoch. Endpoint epochs remain the recorded
        first/last values. Fully stationary tracks retain their fixed location
        and interpolate epochs by recorded point index.
    """
    coordinates = np.asarray(coordinates, dtype=np.float64)
    distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(coordinates, axis=0), axis=1))]
    stationary = distances[-1] == 0
    sampled = (
        np.repeat(coordinates[:1], point_count, axis=0)
        if stationary
        else set_number_of_points(coordinates, point_count)
    )
    if timestamps is None:
        return np.ascontiguousarray(sampled)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if stationary:
        epochs = np.interp(
            np.linspace(0, len(timestamps) - 1, point_count), np.arange(len(timestamps)), timestamps
        )
    else:
        positions = np.linspace(0, distances[-1], point_count)[1:-1]
        # Approach a repeated distance at its first epoch and leave at its last.
        following = np.searchsorted(distances, positions, side="right")
        previous = following - 1
        fraction = (positions - distances[previous]) / (distances[following] - distances[previous])
        epochs = np.empty(point_count, dtype=np.float64)
        epochs[1:-1] = timestamps[previous] + fraction * (
            timestamps[following] - timestamps[previous]
        )
    epochs[0], epochs[-1] = timestamps[0], timestamps[-1]
    return np.ascontiguousarray(np.column_stack((sampled, epochs)))


def calendar_phases(timestamps, time_zone: tzinfo | None) -> np.ndarray:
    """Map Unix epochs to local day, Monday-based week, and calendar-month phases.

    Args:
        timestamps: One-dimensional sequence of finite Unix timestamps.
        time_zone: Calendar timezone, or None for the host's local timezone.

    Returns:
        An (n, 3) array in [0, 1), retaining subsecond precision. Month lengths
        follow the calendar, including leap years; phases use wall-clock time.
    """
    phases = []
    for timestamp in timestamps:
        moment = datetime.fromtimestamp(float(timestamp), tz=time_zone)
        day = (
            moment.hour * 3600 + moment.minute * 60 + moment.second + moment.microsecond / 1e6
        ) / 86400
        phases.append(
            (
                day,
                (moment.weekday() + day) / 7,
                (moment.day - 1 + day) / calendar.monthrange(moment.year, moment.month)[1],
            )
        )
    return np.asarray(phases, dtype=np.float64).reshape(-1, 3)


def _circle_distance(first, second):
    """Return shortest phase separations, measured in turns."""
    difference = np.abs(first - second) % 1
    return np.minimum(difference, 1 - difference)


def _lp_norm(values, p):
    """Evaluate Lp norms on the last axis without overflowing for large exponents."""
    scale = np.max(values, axis=-1)
    normalized = np.divide(
        values, scale[..., None], out=np.zeros_like(values), where=scale[..., None] != 0
    )
    return scale * np.sum(normalized**p, axis=-1) ** (1 / p)


def _torus_distances(first, second, p):
    """Return pointwise Lp norms of normalized day/week/month angular differences."""
    return _lp_norm(2 * _circle_distance(first, second), p)


def _sample_pairs(count, limit, seed):
    """Sample distinct unordered track pairs without allocating a quadratic matrix."""
    total = count * (count - 1) // 2
    indices = (
        np.arange(total)
        if total <= limit
        else np.random.default_rng(seed).choice(total, size=limit, replace=False)
    )
    all_first = np.arange(count, dtype=np.int64)
    first_offset = all_first * (2 * count - all_first - 1) // 2
    random_first = np.searchsorted(first_offset, indices, side="right") - 1
    random_second = random_first + 1 + indices - first_offset[random_first]
    return random_first, random_second


def _inverse_positive_median(values):
    """Fit a multiplicative weight, falling back to one for constant data."""
    positive = values[values > 0]
    return (1 / float(np.median(positive)) if len(positive) else 1.0), len(positive)


def fit_distance_scales(
    tracks,
    *,
    joint=False,
    p=2,
    time_zone=None,
    pair_limit=10000,
    seed=0,
):
    """Calibrate multiplicative distance weights from sampled track pairs.

    Args:
        tracks: Nonempty frame with raw 13_pts longitude/latitude and optional epoch arrays.
        joint: Whether primary clustering combines space and all three calendar phases.
        p: Validated cyclic Lp exponent for the temporal component in joint mode.
            Spatial distances are Euclidean and independent of this exponent.
        time_zone: Timezone for calendar phases, or None for host-local time.
        pair_limit: Maximum number of distinct unordered pairs, normally ten thousand.
        seed: Local pair-sampling seed; QBx uses the same value for its input permutation.

    Returns:
        JSON-compatible spatial/temporal weights and pair counts. Each active weight is
        the reciprocal median nonzero component distance on the same sampled pairs.
        Time has weight zero outside joint mode. Constant components or fewer than two
        tracks use weight one. The spatial weight is shared across coordinates/samples.
        The shared mean spatial profile is also returned for centering before clustering.
    """
    points = np.stack(tracks["13_pts"]).astype(np.float64)
    first, second = _sample_pairs(len(points), pair_limit, seed)
    spatial = np.linalg.norm(points[first, :, :2] - points[second, :, :2], axis=-1)
    spatial = spatial.mean(axis=-1)
    spatial_scale, spatial_nonzero = _inverse_positive_median(spatial)
    temporal_scale, temporal_nonzero = 0.0, 0
    if joint:
        phases = np.stack([calendar_phases(values[:, 2], time_zone) for values in points])
        temporal = _torus_distances(phases[first], phases[second], p).mean(axis=-1)
        temporal_scale, temporal_nonzero = _inverse_positive_median(temporal)
    return {
        "spatial": spatial_scale,
        "temporal": temporal_scale,
        "spatial_center": points[:, :, :2].mean(axis=0).tolist(),
        "sampled_pairs": len(first),
        "spatial_nonzero_pairs": spatial_nonzero,
        "temporal_nonzero_pairs": temporal_nonzero,
    }


def spatiotemporal_distance(
    first, second, *, p=2, spatial_scale=1, temporal_scale=1, spatial_dimensions=2
) -> float:
    """Average pointwise joint distances over corresponding track samples.

    Args:
        first: Point array containing spatial coordinates and optional day/week/month
            phases. Flattened joint samples are also accepted.
        second: Samples in the same shape and order as first.
        p: Finite Lp exponent for temporal distances and the space/time combination.
        spatial_scale: Multiplicative spatial weight; default one.
        temporal_scale: Multiplicative temporal weight; zero ignores temporal coordinates.
        spatial_dimensions: Number of spatial coordinates per point; normally two.

    Returns:
        Arithmetic mean of the pointwise Lp combinations of scaled Euclidean spatial
        and torus temporal distances. All corresponding samples contribute equally.
    """
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if first.ndim == 1:
        first = first.reshape(-1, spatial_dimensions + 3)
        second = second.reshape(-1, spatial_dimensions + 3)
    spatial = (
        np.linalg.norm(first[:, :spatial_dimensions] - second[:, :spatial_dimensions], axis=1)
        * spatial_scale
    )
    if temporal_scale == 0:
        return float(spatial.mean())
    temporal = (
        _torus_distances(first[:, spatial_dimensions:], second[:, spatial_dimensions:], p)
        * temporal_scale
    )
    return float(_lp_norm(np.column_stack((spatial, temporal)), p).mean())


def torus_distance(first, second, *, p=2) -> float:
    """Average pointwise day/week/month torus Lp distances across track samples.

    Args:
        first: Flattened or (n, 3) day/week/month phases, each in [0, 1).
        second: Corresponding phases in the same sample order.
        p: Finite Lp exponent greater than or equal to one.

    Returns:
        Arithmetic mean of the per-sample torus Lp norms. Angles are divided by pi,
        so antipodal phases contribute one in each temporal coordinate.
    """
    first = np.asarray(first, dtype=np.float64).reshape(-1, 3)
    second = np.asarray(second, dtype=np.float64).reshape(-1, 3)
    return float(_torus_distances(first, second, p).mean())


def _circle_rows(phases):
    """Lift phases into three-coordinate rows for DIPY's centroid buffers."""
    angles = 2 * np.pi * np.asarray(phases).ravel()
    return np.ascontiguousarray(
        np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))
    )


def _row_phases(rows):
    """Read circular centroid directions, choosing phase zero for a zero resultant."""
    phases = (np.arctan2(rows[:, 1], rows[:, 0]) / (2 * np.pi)) % 1
    return np.where(np.hypot(rows[:, 0], rows[:, 1]) > 1e-7, phases, 0)


def prepare_temporal_features(tracks, *, algorithm, time_zone, joint=False):
    """Prepare algorithm-specific primary features and separate torus time samples.

    Args:
        tracks: Track frame whose 13_pts arrays contain resampled x, y, and Unix epoch.
        algorithm: "dbscan" for raw phases or "qbx" for circular centroid features.
        time_zone: Calendar timezone, or None for host-local time.
        joint: Include all day/week/month phases in primary features; otherwise keep
            them in a separate temporal buffer for subdivision.

    Returns:
        A copied frame with primary 13_pts and, in separate mode, times arrays.
        Joint DBSCAN uses 13-by-5 samples. Joint QBx uses 52-by-3 feature buffers:
        13 spatial rows followed by 39 day/week/month circular rows. Separate QBx
        uses 13 padded spatial rows and 39 circular rows in its times buffer.
    """
    result = tracks.copy()
    primary, temporal = [], []
    for points in tracks["13_pts"]:
        phases = calendar_phases(points[:, 2], time_zone)
        if algorithm == "qbx":
            spatial = np.pad(points[:, :2], ((0, 0), (0, 1)))
            circular = _circle_rows(phases)
            primary.append(
                np.ascontiguousarray(np.concatenate((spatial, circular))) if joint else spatial
            )
            temporal.append(circular)
        else:
            primary.append(
                np.ascontiguousarray(np.column_stack((points[:, :2], phases)))
                if joint
                else np.ascontiguousarray(points[:, :2])
            )
            temporal.append(np.ascontiguousarray(phases))
    result["13_pts"] = primary
    if not joint:
        result["times"] = temporal
    return result


def _are_compatible(shape1, shape2, blocks):
    """Check matching three-coordinate buffers contain complete feature blocks."""
    return shape1 == shape2 and shape1[1] == 3 and shape1[0] % blocks == 0


class SpatiotemporalMetric(Metric):
    """Joint track metric with circular means for QBx's temporal centroids."""

    def __init__(self, p=2, *, spatial_scale=1, temporal_scale=1):
        """Configure fixed block scales before QBx starts comparing tracks or centroids.

        Args:
            p: Validated finite cyclic Lp exponent of at least one.
            spatial_scale: Shared multiplicative spatial weight, fitted before clustering.
            temporal_scale: Multiplicative temporal weight, fitted on the same track pairs.
        """
        super().__init__(feature=IdentityFeature())
        self.p = p
        self.spatial_scale = spatial_scale
        self.temporal_scale = temporal_scale

    def are_compatible(self, shape1, shape2):
        return _are_compatible(shape1, shape2, 4)

    def dist(self, features1, features2):
        """Measure two packed spatial/circular feature arrays.

        Args:
            features1: Spatial rows followed by corresponding sine/cosine rows.
            features2: Matching feature array or arithmetic QBx centroid buffer.

        Returns:
            The arithmetic mean of pointwise joint spatial/torus Lp distances.
        """
        count = len(features1) // 4
        first = np.column_stack(
            (features1[:count, :2], _row_phases(features1[count:]).reshape(count, 3))
        )
        second = np.column_stack(
            (features2[:count, :2], _row_phases(features2[count:]).reshape(count, 3))
        )
        return spatiotemporal_distance(
            first,
            second,
            p=self.p,
            spatial_scale=self.spatial_scale,
            temporal_scale=self.temporal_scale,
        )


class TorusMetric(Metric):
    """Day/week/month angular metric with circular means for QBx centroids."""

    def __init__(self, p=2):
        """Configure the exponent and retain circular feature rows.

        Args:
            p: Validated finite Lp exponent greater than or equal to one.
        """
        super().__init__(feature=IdentityFeature())
        self.p = p

    def are_compatible(self, shape1, shape2):
        return _are_compatible(shape1, shape2, 3)

    def dist(self, features1, features2):
        """Measure two packed three-circle feature arrays.

        Args:
            features1: Sine/cosine rows ordered by sample, then day/week/month.
            features2: Matching feature array or arithmetic QBx centroid buffer.

        Returns:
            Mean pointwise torus Lp distance, with angular components normalized by pi.
        """
        return torus_distance(_row_phases(features1), _row_phases(features2), p=self.p)
