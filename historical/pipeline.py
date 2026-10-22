"""Database orchestration for the original internship clustering methods."""

import colorsys
import hashlib
import json
from datetime import UTC, datetime
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np
from dipy.tracking.streamline import set_number_of_points

from .clustering import get_dbscan_clusters, get_qb_clusters
from .data import RESAMPLED_POINTS, _transform_time
from .storage import ClusterDatabase
from .validation import ClusteringSettings


def _color_for_label(label):
    """Retain the stored spatial-only color convention."""
    if label == -1:
        return "#ff0000"
    rgb = colorsys.hsv_to_rgb((label * 0.61803398875 + 0.35) % 1, 0.75, 0.8)
    return "#{:02x}{:02x}{:02x}".format(*(int(value * 255) for value in rgb))


def run_clustering(
    database: ClusterDatabase,
    algorithm,
    eps,
    *,
    feature_mode="spatiotemporal",
    eps_time=0.1,
    min_samples=5,
    factor=2.0,
    seed=0,
    timezone="UTC",
):
    """Store one run using timestamp-inclusive resampling and the original metrics.

    Args:
        database: Open writable source database; this function owns its write transaction.
        algorithm: "dbscan" or "qbx".
        eps: Positive original spatial threshold.
        feature_mode: "spatiotemporal" by default, or "spatial" to omit time subdivision.
        eps_time: Original temporal threshold; inactive in spatial mode.
        min_samples: DBSCAN's spatial minimum neighborhood size.
        factor: QBx multiplier for its three spatial thresholds.
        seed: QBx input-permutation seed; inactive for DBSCAN.
        timezone: IANA timezone or "local" for weekday/hour encoding.

    Returns:
        The cached or newly inserted run ID. Source points and other runs are preserved.
    """
    parameters = ClusteringSettings(
        algorithm=algorithm,
        eps=eps,
        feature_mode=feature_mode,
        eps_time=eps_time,
        min_samples=min_samples,
        factor=factor,
        seed=seed,
        timezone=timezone,
    )
    settings = parameters.model_dump()
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    existing = database.select_exec("runs", ("run_id",), where={"config_key": key}).fetchone()
    if existing:
        return existing[0]
    signature = database.input_signature()
    with_time = parameters.feature_mode == "spatiotemporal"
    if with_time and not database.supports_time():
        raise ValueError(
            "No usable tracks for temporal clustering: eligible tracks require complete, "
            "finite, nonnegative, chronological timestamps; use spatial mode for untimed data"
        )
    tracks = database.load_tracks()
    tracks = tracks.loc[tracks.eligible == 1].reset_index(drop=True)
    if tracks.empty:
        raise ValueError("No usable tracks: each track needs at least five points")
    sampled = []
    for geometry, timestamps in zip(tracks.geometry, tracks.timestamps, strict=True):
        coordinates = np.asarray(geometry.coords)
        values = np.column_stack((coordinates, timestamps)) if with_time else coordinates
        points = np.ascontiguousarray(set_number_of_points(values, RESAMPLED_POINTS))
        if not with_time and parameters.algorithm == "qbx":
            points = np.pad(points, ((0, 0), (0, 1)))
        sampled.append(points)
    tracks["13_pts"] = sampled
    tracks = tracks.drop(columns="timestamps")
    if with_time:
        tracks = _transform_time(
            tracks, None if parameters.timezone == "local" else ZoneInfo(parameters.timezone)
        )
    started = perf_counter()
    result = (
        get_dbscan_clusters(tracks, parameters.eps, parameters.eps_time, parameters.min_samples)
        if parameters.algorithm == "dbscan"
        else get_qb_clusters(
            tracks, parameters.eps, parameters.eps_time, parameters.factor, parameters.seed
        )
    )
    elapsed = perf_counter() - started
    spatial_layers = 1 if parameters.algorithm == "dbscan" else 3
    with database.transaction():
        if database.input_signature() != signature:
            raise ValueError("Source inputs changed during clustering; retry the run")
        run_id = database.insert(
            "runs",
            {
                "algorithm": parameters.algorithm,
                "feature_mode": parameters.feature_mode,
                "config_key": key,
                "settings_json": json.dumps(settings, sort_keys=True),
                "created_at_utc": datetime.now(UTC).isoformat(),
                "clustering_seconds": elapsed,
                "provenance": "historical internship algorithms; stored source points",
            },
        )
        for layer in range(spatial_layers + int(with_time)):
            labels = result[f"clusters{layer}"].tolist()
            colors = (
                [
                    "#{:02x}{:02x}{:02x}".format(*(int(value) for value in rgb))
                    for rgb in result[f"colors{layer}"]
                ]
                if with_time
                else [_color_for_label(label) for label in labels]
            )
            name = (
                "Time"
                if layer == spatial_layers
                else "Spatial"
                if parameters.algorithm == "dbscan" and not with_time
                else f"Layer {layer + 1}"
            )
            database.insert_many(
                "assignments",
                (
                    {
                        "run_id": run_id,
                        "track_id": int(track_id),
                        "layer": layer,
                        "layer_name": name,
                        "predicted_cluster": int(label),
                        "color": color,
                    }
                    for track_id, label, color in zip(tracks.track_id, labels, colors, strict=True)
                ),
            )
    return run_id
