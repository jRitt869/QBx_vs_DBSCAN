"""Prepare the two Feather tables consumed by the Bokeh application."""

import json
from datetime import tzinfo
from pathlib import Path

import numpy as np
import pandas as pd

from .clustering import get_dbscan_clusters, get_qb_clusters
from .data import MAP_CRS, _transform_time, load_directory

TRACKS_FILE = "raw_data"
JUMPS_FILE = "raw_data_jumps"


def validate_parameters(eps: float, eps_time: float, min_s: int, factor: float, seed: int) -> None:
    """Reject invalid inputs before loading data or starting a server."""
    for name, value in (("eps", eps), ("eps_time", eps_time), ("factor", factor)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number")
    if factor < 1:
        raise ValueError("factor must be at least 1 to keep thresholds in descending order")
    if not isinstance(min_s, int) or min_s < 1:
        raise ValueError("min_s must be a positive integer")
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 4294967295")


def gen_clusters(
    path: str | Path,
    is_QB: bool,
    eps: float,
    eps_time: float,
    min_s: int,
    factor: float,
    seed: int,
    *,
    output_dir: str | Path = ".",
    time_zone: tzinfo | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the original pipeline and export map coordinates in Web Mercator."""
    validate_parameters(eps, eps_time, min_s, factor, seed)
    tracks, jumps = load_directory(path)
    features = _transform_time(tracks, time_zone)
    if is_QB:
        clustered = get_qb_clusters(features, eps, eps_time, factor, seed)
    else:
        clustered = get_dbscan_clusters(features, eps, eps_time, min_s)
    clustered = clustered.to_crs(MAP_CRS)
    jumps = jumps.to_crs(MAP_CRS)
    clustered["x"] = [list(line.coords.xy[0]) for line in clustered.geometry]
    clustered["y"] = [list(line.coords.xy[1]) for line in clustered.geometry]
    jumps["x"] = [point.x for point in jumps.geometry]
    jumps["y"] = [point.y for point in jumps.geometry]
    track_table = pd.DataFrame(clustered.drop(columns="geometry"))
    jump_table = pd.DataFrame(jumps.drop(columns="geometry"))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    track_table.to_feather(destination / TRACKS_FILE)
    jump_table.to_feather(destination / JUMPS_FILE)
    metadata = {
        "algorithm": "QuickBundlesX" if is_QB else "DBSCAN",
        "eps": eps,
        "eps_time": eps_time,
        "min_s": min_s,
        "factor": factor,
        "seed": seed,
        "timezone": str(time_zone) if time_zone else "host-local",
        "map_crs": MAP_CRS,
        "tracks": len(track_table),
        "timeskips": len(jump_table),
    }
    (destination / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return track_table, jump_table
