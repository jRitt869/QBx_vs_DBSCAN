"""Timestamp-independent GPX loading and the application's spatial algorithms."""

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import gpxpy
import numpy as np
import pandas as pd
from dipy.segment.clustering import QuickBundlesX
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.tracking.streamline import set_number_of_points
from numpy.typing import NDArray
from shapely.geometry import LineString
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from track_clusters.colors import SPATIAL_MIN_SIZE
from track_clusters.data import GPX_CRS, MIN_TRACK_POINTS, RESAMPLED_POINTS


@dataclass
class SpatialDataset:
    tracks: gpd.GeoDataFrame
    skipped: pd.DataFrame
    files: list[Path]


def load_spatial_directory(path: str | Path, *, id_source: str = "gpx-number") -> SpatialDataset:
    """Load >=5-point tracks, preserving each GPX track's explicit number.

    Timestamps/elevation are ignored. Resample longitude/latitude by spatial
    arc length, then pad a zero z coordinate for DIPY. Segments remain joined,
    consistent with the existing application. Skipped tracks are reported.
    """
    directory = Path(path)
    if id_source not in ("gpx-number", "index"):
        raise ValueError("id_source must be gpx-number or index")
    if not directory.is_dir():
        raise ValueError(f"GPX directory does not exist: {directory}")
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".gpx")
    if not files:
        raise ValueError(f"No GPX files found in {directory}")
    records, skipped = [], []
    for file in files:
        try:
            with file.open(encoding="utf-8") as source:
                gpx = gpxpy.parse(source)
        except gpxpy.gpx.GPXException as error:
            raise ValueError(f"Cannot parse {file.name}: {error}") from error
        used_ids = set()
        for track_index, track in enumerate(gpx.tracks):
            track_id = (
                track.number
                if id_source == "gpx-number" and track.number is not None
                else track_index
            )
            if track_id in used_ids:
                raise ValueError(
                    f"Duplicate track ID {track_id} in {file.name}; use --id-source index if appropriate"
                )
            used_ids.add(track_id)
            coords = np.array(
                [(p.longitude, p.latitude) for segment in track.segments for p in segment.points],
                dtype=np.float64,
            )
            key = {"file_name": file.name, "id": track_id, "track_index": track_index}
            if len(coords) < MIN_TRACK_POINTS:
                skipped.append(
                    {**key, "point_count": len(coords), "reason": "fewer than five points"}
                )
                continue
            if (
                not np.isfinite(coords).all()
                or (np.abs(coords[:, 0]) > 180).any()
                or (np.abs(coords[:, 1]) > 90).any()
            ):
                raise ValueError(f"Invalid coordinates in {file.name}, track {track_id}")
            padded = np.pad(coords, ((0, 0), (0, 1)))
            sampled = np.ascontiguousarray(set_number_of_points(padded, RESAMPLED_POINTS))
            records.append(
                {
                    **key,
                    "name": f"{file.stem}_{track_id}",
                    "point_count": len(coords),
                    "13_pts": sampled,
                    "geometry": LineString(coords),
                }
            )
    if not records:
        raise ValueError("No usable tracks: each track needs at least five coordinate points")
    return SpatialDataset(
        gpd.GeoDataFrame(records, crs=GPX_CRS),
        pd.DataFrame(skipped, columns=["file_name", "id", "track_index", "point_count", "reason"]),
        files,
    )


def cluster_spatial(
    tracks: gpd.GeoDataFrame,
    algorithm: str,
    eps: float,
    *,
    min_samples: int = 5,
    factor: float = 2,
    seed: int = 0,
    layer: int = 3,
) -> NDArray[np.int_]:
    """Return spatial assignments only; preserve each algorithm's feature space."""
    if tracks.empty:
        raise ValueError("Cannot cluster an empty dataset")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a finite positive number")
    if algorithm == "dbscan":
        if not isinstance(min_samples, int) or min_samples < 1:
            raise ValueError("min_samples must be a positive integer")
        flattened = np.array([np.asarray(t)[:, :2].ravel() for t in tracks["13_pts"]])
        return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(
            StandardScaler().fit_transform(flattened)
        )
    if algorithm != "qbx":
        raise ValueError("algorithm must be qbx or dbscan")
    if not np.isfinite(factor) or factor < 1 or not np.isfinite(factor**2 * eps):
        raise ValueError("factor must be finite and at least 1, with finite thresholds")
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 4294967295")
    if layer not in (1, 2, 3):
        raise ValueError("layer must be 1, 2, or 3")
    ordering = np.random.RandomState(seed).permutation(len(tracks))
    values = [tracks["13_pts"].iloc[i] for i in ordering]
    tree = QuickBundlesX(
        thresholds=[factor**2 * eps, factor * eps, eps], metric=AveragePointwiseEuclideanMetric()
    ).cluster(values)
    assignments = np.full(len(tracks), -1, dtype=np.int_)
    label = 0
    for cluster in tree.get_clusters(layer):
        if len(cluster) >= SPATIAL_MIN_SIZE:
            assignments[ordering[np.asarray(cluster.indices, dtype=int)]] = label
            label += 1
    return assignments
