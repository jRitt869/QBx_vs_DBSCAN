"""Cluster stored tracks and score predictions against supplied labels or reference runs."""

import colorsys
import hashlib
import json
from datetime import UTC, datetime
from functools import partial
from time import perf_counter
from typing import Literal, Self
from zoneinfo import ZoneInfo

import numpy as np
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.tracking.streamline import set_number_of_points
from pydantic import ConfigDict, Field, model_validator, validate_call

from track_clusters.clustering import get_dbscan_clusters, get_qb_clusters
from track_clusters.data import RESAMPLED_POINTS
from track_clusters.validation import (
    ClusteringSettings,
    Seeds,
    Thresholds,
)

from .metrics import score_assignments
from .storage import ClusterDatabase
from .temporal import (
    SpatiotemporalMetric,
    TorusMetric,
    fit_distance_scales,
    prepare_temporal_features,
    resample_spatial_track,
    spatiotemporal_distance,
    torus_distance,
)


def _color_for_label(label):
    """Choose a stable display color for a spatial assignment."""
    if label == -1:
        return "#ff0000"
    rgb = colorsys.hsv_to_rgb((label * 0.61803398875 + 0.35) % 1, 0.75, 0.8)
    return "#{:02x}{:02x}{:02x}".format(*(int(v * 255) for v in rgb))


def run_clustering(
    database: ClusterDatabase,
    settings: ClusteringSettings,
):
    """Reuse a normalized settings key, or cluster stored tracks and insert one run.

    Args:
        database: Open writable source database; this function owns the write transaction.
        settings: Validated algorithm, thresholds, feature preparation, and sampling settings.

    Returns:
        The existing or newly inserted run ID, with predictions and colors for every layer.

    Existing keys return before loading inputs or writing. New runs use the shared
    clustering functions and fail if eligible inputs change during computation.
    """
    with_time = settings.feature_mode == "spatiotemporal"
    joint = settings.temporal_mode == "joint"
    separate_time = with_time and not joint
    settings_json = json.dumps(settings.model_dump(), sort_keys=True)
    key = hashlib.sha256(settings_json.encode()).hexdigest()
    existing = database.select_exec("runs", ("run_id",), where={"config_key": key}).fetchone()
    if existing:
        return existing[0]
    source_signature = database.input_signature()
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
        if settings.algorithm == "dbscan" or not joint:
            points = resample_spatial_track(coordinates, timestamps if with_time else None)
            if settings.algorithm == "qbx" and not with_time:
                points = np.pad(points, ((0, 0), (0, 1)))
        else:
            values = np.column_stack((coordinates, timestamps))
            points = np.ascontiguousarray(
                np.repeat(values[:1], RESAMPLED_POINTS, axis=0)
                if np.all(values == values[0])
                else set_number_of_points(values, RESAMPLED_POINTS)
            )
        sampled.append(points)
    tracks["13_pts"] = sampled
    tracks = tracks.drop(columns="timestamps")
    time_zone = None if settings.timezone in (None, "local") else ZoneInfo(settings.timezone)
    scales = fit_distance_scales(
        tracks,
        joint=joint,
        p=settings.p,
        time_zone=time_zone,
        pair_limit=settings.scale_pairs,
        seed=settings.seed,
    )
    center = np.asarray(scales["spatial_center"])
    tracks["13_pts"] = tracks["13_pts"].map(
        lambda points: np.ascontiguousarray(
            np.column_stack((points[:, :2] - center, points[:, 2:]))
        )
    )
    if settings.algorithm == "qbx" and not joint:
        tracks["13_pts"] = tracks["13_pts"].map(
            lambda points: np.column_stack((points[:, :2] * scales["spatial"], points[:, 2:]))
        )
    if with_time:
        tracks = prepare_temporal_features(
            tracks,
            algorithm=settings.algorithm,
            joint=joint,
            time_zone=time_zone,
        )
    started = perf_counter()
    if settings.algorithm == "dbscan":
        result = get_dbscan_clusters(
            tracks,
            settings.eps,
            settings.eps_time,
            settings.min_samples,
            metric=partial(
                spatiotemporal_distance,
                p=settings.p,
                spatial_scale=scales["spatial"],
                temporal_scale=scales["temporal"],
            ),
            temporal_metric=partial(torus_distance, p=settings.p) if separate_time else None,
        )
    else:
        result = get_qb_clusters(
            tracks,
            settings.eps,
            settings.eps_time,
            settings.factor,
            settings.seed,
            metric=SpatiotemporalMetric(
                settings.p, spatial_scale=scales["spatial"], temporal_scale=scales["temporal"]
            )
            if joint
            else AveragePointwiseEuclideanMetric(),
            temporal_metric=TorusMetric(settings.p) if separate_time else None,
        )
    primary_layers = 1 if settings.algorithm == "dbscan" else 3
    layers = []
    for layer in range(primary_layers + separate_time):
        labels = result[f"clusters{layer}"].tolist()
        if with_time:
            colors = [
                "#{:02x}{:02x}{:02x}".format(*(int(v) for v in rgb))
                for rgb in result[f"colors{layer}"]
            ]
        else:
            colors = [_color_for_label(label) for label in labels]
        if layer == primary_layers:
            name = "Time"
        elif settings.algorithm == "dbscan":
            name = "Spatial/time" if joint else "Spatial"
        else:
            name = f"{'Joint layer' if joint else 'Layer'} {layer + 1}"
        layers.append((layer, name, labels, colors))
    elapsed = perf_counter() - started
    with database.transaction():
        if database.input_signature() != source_signature:
            raise ValueError("Source inputs changed during clustering; retry the run")
        database.refresh_views()
        run_id = database.insert(
            "runs",
            {
                "algorithm": settings.algorithm,
                "feature_mode": settings.feature_mode,
                "config_key": key,
                "settings_json": settings_json,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "clustering_seconds": elapsed,
                "provenance": json.dumps(
                    {
                        "implementation": "track_clusters algorithms; stored source points",
                        "distance_scales": scales,
                    },
                    sort_keys=True,
                ),
            },
        )
        for layer, name, labels, colors in layers:
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


NOISE_POLICY = "each -1 is a separate singleton in both truth and predictions"
SELECTION_NOTE = "exploratory in-sample threshold selection by mean ARI; no held-out validation"


class ReferenceConfiguration(ClusteringSettings):
    """One reference run and its one-based layer, defaulting to the finest primary layer."""

    layer: Literal[1, 2, 3, 4] | None = None

    @model_validator(mode="after")
    def _reference_layer(self) -> Self:
        """Default to the finest primary layer and reject layers unavailable in this mode."""
        primary_layers = 1 if self.algorithm == "dbscan" else 3
        maximum = primary_layers + (self.temporal_mode == "separate")
        if self.layer is None:
            self.layer = primary_layers
        if self.layer > maximum:
            raise ValueError(
                f"{self.algorithm} in {self.feature_mode} mode has only {maximum} layers"
            )
        return self


def _require_references(truth):
    """Require complete labels for at least two eligible tracks before scoring."""
    if len(truth) < 2 or truth.reference_cluster.isna().any():
        raise ValueError(
            "Scoring requires at least two eligible tracks, all with hand-made labels; supply --reference-config to use a reference run instead"
        )


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def evaluate(
    database: ClusterDatabase,
    settings: ClusteringSettings,
    *,
    eps: Thresholds,
    reference: ReferenceConfiguration | None = None,
    seeds: Seeds = Field(default_factory=lambda: [0, 1, 2, 3, 4]),
    layer: Literal[1, 2, 3] | None = None,
) -> None:
    """Persist one algorithm's agreement scores against labels or an explicit reference run.

    Args:
        database: Open writable source database; clustering and scoring manage transactions.
        settings: Base clustering settings; each sweep replaces eps and seed in a validated copy.
        eps: Nonempty sequence of distinct positive primary thresholds to evaluate.
        reference: Reference run configuration; None requires complete hand-made labels.
        seeds: Distinct QBx ordering/calibration seeds; nonempty for QBx. DBSCAN accepts
            an empty sequence and always uses seed zero.
        layer: One-based primary scoring layer; None selects DBSCAN 1 or QBx 3.

    Missing target and reference runs are populated through run_clustering(). Scores
    and the evaluation manifest are stored in the database; no result is returned.
    """
    if settings.algorithm == "dbscan" and layer not in (None, 1):
        raise ValueError("DBSCAN has one spatial layer")
    if settings.algorithm == "qbx" and not seeds:
        raise ValueError("QuickBundlesX requires at least one seed")
    configurations = [
        ClusteringSettings.model_validate({**settings.model_dump(), "eps": threshold, "seed": seed})
        for threshold in eps
        for seed in (seeds if settings.algorithm == "qbx" else (0,))
    ]
    signature = database.input_signature()
    reference_id = reference_layer = None
    truth = database.eligible_references()
    if reference is None:
        _require_references(truth)
        reference_description = "Hand-made labels from the source database"
    else:
        reference_settings = ClusteringSettings.model_validate(
            reference.model_dump(exclude={"layer"})
        )
        reference_id = run_clustering(database, reference_settings)
        reference_layer = reference.layer - 1
        reference_description = (
            f"Reference run {reference_id} ({reference.algorithm}), layer {reference.layer}"
        )
        reference_labels = database.select_frame(
            "assignments",
            ("track_id", "predicted_cluster"),
            where={"run_id": reference_id, "layer": reference_layer},
            order_by=("track_id",),
        )
        if len(truth) < 2 or reference_labels.track_id.tolist() != truth.track_id.tolist():
            raise ValueError("Reference run must assign every eligible track in the selected layer")
    requested = [run_clustering(database, configuration) for configuration in configurations]
    with database.transaction():
        if database.input_signature() != signature:
            raise ValueError("Source inputs changed during evaluation; retry")
        database.refresh_views()
        dataset_id = database.select_exec("datasets", ("dataset_id",)).fetchone()[0]
        identity = {
            "dataset_id": dataset_id,
            "scope": settings.feature_mode,
            "algorithm": settings.algorithm,
            "reference_run_id": reference_id,
            "reference_layer": reference_layer,
        }
        existing = database.select_exec(
            "evaluations", ("evaluation_id",), where=identity
        ).fetchone()
        manifest = json.dumps(
            {
                "scope": settings.feature_mode,
                "algorithm": settings.algorithm,
                "source": "database inputs and stored cluster assignments",
                "reference_kind": "labels" if reference is None else "run",
                "reference_description": reference_description,
                "reference_run_id": reference_id,
                "reference_layer": reference_layer,
                "reference_configuration": reference.model_dump()
                if reference is not None
                else None,
                "noise_policy": NOISE_POLICY,
                "selection_note": SELECTION_NOTE,
            },
            sort_keys=True,
        )
        if existing:
            evaluation_id = existing[0]
            database.update(
                "evaluations", {"manifest_json": manifest}, where={"evaluation_id": evaluation_id}
            )
        else:
            evaluation_id = database.insert(
                "evaluations",
                {
                    **identity,
                    "manifest_json": manifest,
                    "selection_note": SELECTION_NOTE,
                    "noise_policy": NOISE_POLICY,
                },
            )
        truth = database.select_frame(
            "v_evaluation_references",
            ("track_id", "reference_cluster"),
            where={"evaluation_id": evaluation_id},
            order_by=("track_id",),
        )
        if reference is None:
            _require_references(truth)
        if len(truth) < 2 or truth.reference_cluster.isna().any():
            raise ValueError("Reference run no longer has complete assignments")
        selected_layer = 0 if settings.algorithm == "dbscan" else (layer or 3) - 1
        for run_id in requested:
            predictions = database.select_frame(
                "assignments",
                ("track_id", "predicted_cluster"),
                where={"run_id": run_id, "layer": selected_layer},
                order_by=("track_id",),
            )
            if predictions.track_id.tolist() != truth.track_id.tolist():
                raise ValueError(
                    f"Run {run_id} does not have complete assignments for the requested layer"
                )
            scores = score_assignments(
                truth.reference_cluster.to_numpy(dtype=np.int64),
                predictions.predicted_cluster.to_numpy(dtype=np.int64),
            )
            values = {
                "layer": selected_layer,
                **{
                    key: value
                    for key, value in scores.items()
                    if key not in {"tracks", "clusters", "outliers", "outlier_fraction"}
                },
            }
            identity = {"evaluation_id": evaluation_id, "run_id": run_id, "layer": selected_layer}
            existing = database.select_exec(
                "evaluation_results", ("evaluation_result_id",), where=identity
            ).fetchone()
            if existing:
                database.update(
                    "evaluation_results", values, where={"evaluation_result_id": existing[0]}
                )
            else:
                database.insert("evaluation_results", {**identity, **values})
