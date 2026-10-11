"""Optional spatial evaluation against file/GPX-number labels."""

import argparse
import hashlib
import json
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, pair_confusion_matrix

from .spatial import SpatialDataset, cluster_spatial, load_spatial_directory


def read_labels(path: str | Path) -> pd.DataFrame:
    """Read the original CSV format; ignore optional annotation/index columns."""
    labels = pd.read_csv(path, dtype={"file_name": str})
    required = ["file_name", "id", "cluster"]
    if missing := set(required) - set(labels):
        raise ValueError(f"Label CSV is missing columns: {', '.join(sorted(missing))}")
    labels = labels[required].copy()
    if labels.empty or labels.isna().any().any():
        raise ValueError("Label CSV must contain nonempty file_name, id, and cluster values")
    for column, minimum in (("id", 0), ("cluster", -1)):
        values = pd.to_numeric(labels[column], errors="raise")
        if not np.isfinite(values).all() or (values % 1 != 0).any() or (values < minimum).any():
            raise ValueError(f"{column} must contain integers >= {minimum}")
        labels[column] = values.astype(np.int64)
    if labels.duplicated(["file_name", "id"]).any():
        raise ValueError("Label CSV contains duplicate (file_name, id) keys")
    return labels


def align_labels(
    dataset: SpatialDataset, labels: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Require labels for every eligible track; report labels for skipped tracks."""
    aligned = dataset.tracks.merge(
        labels, on=["file_name", "id"], how="left", validate="one_to_one"
    )
    if aligned["cluster"].isna().any():
        examples = (
            aligned.loc[aligned["cluster"].isna(), ["file_name", "id"]].head().to_dict("records")
        )
        raise ValueError(f"Missing labels for eligible tracks: {examples}")
    all_keys = pd.concat(
        [dataset.tracks[["file_name", "id"]], dataset.skipped[["file_name", "id"]]]
    )
    unknown = labels.merge(all_keys, on=["file_name", "id"], how="left", indicator=True)
    if (unknown["_merge"] == "left_only").any():
        examples = (
            unknown.loc[unknown["_merge"] == "left_only", ["file_name", "id"]]
            .head()
            .to_dict("records")
        )
        raise ValueError(f"Labels reference unknown tracks: {examples}; check --id-source")
    excluded = labels.merge(
        dataset.skipped, on=["file_name", "id"], how="inner", validate="one_to_one"
    )
    return aligned, excluded


def separate_noise(labels: Sequence[int]) -> np.ndarray:
    """Treat every -1 as a singleton rather than one giant noise cluster."""
    result = np.asarray(labels, dtype=np.int64).copy()
    noise = result == -1
    start = int(result.max(initial=-1)) + 1
    result[noise] = np.arange(start, start + int(noise.sum()))
    return result


def score_assignments(
    truth: Sequence[int], predictions: Sequence[int]
) -> dict[str, int | float | None]:
    """Pairwise confusion counts are unordered, distinct track pairs."""
    truth = np.asarray(truth, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if truth.shape != predictions.shape or truth.ndim != 1 or len(truth) < 2:
        raise ValueError("Scoring requires equal one-dimensional arrays of at least two tracks")
    normalized_truth, normalized_predictions = separate_noise(truth), separate_noise(predictions)
    tn, fp, fn, tp = (
        int(v) // 2 for v in pair_confusion_matrix(normalized_truth, normalized_predictions).ravel()
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tracks": len(truth),
        "clusters": len(set(predictions) - {-1}),
        "outliers": int((predictions == -1).sum()),
        "outlier_fraction": float((predictions == -1).mean()),
        "adjusted_rand": adjusted_rand_score(normalized_truth, normalized_predictions),
        "adjusted_mutual_information": adjusted_mutual_info_score(
            normalized_truth, normalized_predictions
        ),
        "pair_tp": tp,
        "pair_fp": fp,
        "pair_fn": fn,
        "pair_tn": tn,
        "pair_precision": precision,
        "pair_recall": recall,
        "pair_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "pair_false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "pair_false_negative_rate": fn / (fn + tp) if fn + tp else None,
        "outlier_detection_precision": (
            float(((predictions == -1) & (truth == -1)).sum() / (predictions == -1).sum())
            if (predictions == -1).any()
            else 0.0
        ),
        "outlier_detection_recall": (
            float(((predictions == -1) & (truth == -1)).sum() / (truth == -1).sum())
            if (truth == -1).any()
            else None
        ),
    }


def _fingerprint(path: Path) -> dict[str, str]:
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def evaluate(
    data_dir: Path,
    labels_path: Path,
    output_dir: Path,
    *,
    qbx_eps: Sequence[float],
    dbscan_eps: Sequence[float],
    seeds: Sequence[int],
    factor: float = 2.0,
    min_samples: int = 5,
    layer: int = 3,
    id_source: str = "gpx-number",
) -> pd.DataFrame:
    """Sweep thresholds and seeds; save individual and aggregate scores."""
    if not qbx_eps and not dbscan_eps:
        raise ValueError("Provide at least one threshold to evaluate")
    if qbx_eps and not seeds:
        raise ValueError("QuickBundlesX requires at least one seed")
    for name, values in (("qbx_eps", qbx_eps), ("dbscan_eps", dbscan_eps), ("seeds", seeds)):
        if len(set(values)) != len(values):
            raise ValueError(f"{name} must not contain duplicates")
    dataset = load_spatial_directory(data_dir, id_source=id_source)
    labels = read_labels(labels_path)
    tracks, excluded = align_labels(dataset, labels)
    truth = tracks["cluster"].to_numpy(dtype=np.int64)
    records, predictions = [], {}
    for algorithm, thresholds in (("dbscan", dbscan_eps), ("qbx", qbx_eps)):
        for eps in thresholds:
            for seed in seeds if algorithm == "qbx" else [0]:
                started = perf_counter()
                assignments = cluster_spatial(
                    tracks,
                    algorithm,
                    eps,
                    min_samples=min_samples,
                    factor=factor,
                    seed=seed,
                    layer=layer,
                )
                elapsed = perf_counter() - started
                key = f"{algorithm}_eps{float(eps)}_seed{seed}"
                predictions[key] = assignments
                records.append(
                    {
                        "run": key,
                        "algorithm": algorithm,
                        "eps": eps,
                        "seed": seed if algorithm == "qbx" else None,
                        "factor": factor if algorithm == "qbx" else None,
                        "layer": layer if algorithm == "qbx" else None,
                        "min_samples": min_samples if algorithm == "dbscan" else None,
                        "seconds": elapsed,
                        **score_assignments(truth, assignments),
                    }
                )
    results = pd.DataFrame(records)
    # Choose parameters by mean ARI across seeds, never by the best individual seed.
    metrics = [
        "adjusted_rand",
        "adjusted_mutual_information",
        "pair_f1",
        "pair_false_positive_rate",
        "pair_false_negative_rate",
        "outlier_fraction",
    ]
    summary = results.groupby(["algorithm", "eps"], sort=False)[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    summary["runs"] = results.groupby(["algorithm", "eps"], sort=False).size().to_numpy()
    for column in summary:
        if column.endswith("_std"):
            summary[column] = summary[column].fillna(0.0)
    best = {}
    for algorithm in results["algorithm"].unique():
        best_eps = (
            summary[summary.algorithm == algorithm]
            .sort_values(["adjusted_rand_mean", "eps"], ascending=[False, True])
            .iloc[0]
            .eps
        )
        representative = results[(results.algorithm == algorithm) & (results.eps == best_eps)].iloc[
            0
        ]
        best[algorithm] = representative["run"]

    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    assignment_table = tracks[["file_name", "id", "track_index", "point_count", "cluster"]].copy()
    for key, assignments in predictions.items():
        assignment_table[key] = assignments
    assignment_table.to_csv(output_dir / "assignments.csv", index=False)
    excluded.to_csv(output_dir / "excluded_labels.csv", index=False)
    dataset.skipped.to_csv(output_dir / "skipped_tracks.csv", index=False)
    duplicate_records = _duplicates(tracks)
    duplicate_records.to_csv(output_dir / "duplicate_tracks.csv", index=False)
    manifest = {
        "scope": "spatial only; timestamps and elevation ignored",
        "resampling": "13 points in longitude/latitude arc length, zero-padded z",
        "id_source": id_source,
        "minimum_points": 5,
        "input_files": [_fingerprint(file) for file in dataset.files],
        "labels": _fingerprint(labels_path),
        "label_rows": len(labels),
        "eligible_tracks": len(tracks),
        "excluded_label_rows": len(excluded),
        "skipped_tracks": len(dataset.skipped),
        "exact_geometry_duplicates": len(duplicate_records),
        "versions": {
            package: version(package)
            for package in ("numpy", "pandas", "dipy", "scikit-learn", "gpxpy", "matplotlib")
        },
        "code": [
            _fingerprint(Path(__file__)),
            _fingerprint(Path(__file__).with_name("spatial.py")),
        ],
        "parameters": {
            "qbx_eps": list(qbx_eps),
            "dbscan_eps": list(dbscan_eps),
            "seeds": list(seeds),
            "factor": factor,
            "min_samples": min_samples,
            "layer": layer,
        },
        "noise_policy": "each -1 is a separate singleton in both truth and predictions",
        "best_parameters": best,
        "selection_note": "exploratory in-sample threshold selection by mean ARI; no held-out validation",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    from .plots import plot_comparison, plot_metric_sweep

    plot_comparison(
        tracks,
        truth,
        {a: (r, predictions[r]) for a, r in best.items()},
        output_dir / "comparison.png",
    )
    plot_metric_sweep(summary, output_dir / "parameter_sweep.png")
    lines = [
        "# Spatial evaluation",
        "",
        f"Scored **{len(tracks)} tracks** against {len(labels)} label rows; "
        f"{len(excluded)} labeled short tracks were excluded.",
        f"Reference labels contain {len(set(truth) - {-1})} clusters and {(truth == -1).sum()} outliers.",
        f"Found {len(duplicate_records)} exact duplicate geometries; all records are retained.",
        "",
        "Timestamps are ignored. Both methods use the same 13-point spatial resampling; "
        "QBx still uses coordinate degrees and DBSCAN standardized flattened coordinates.",
        "",
        "Every -1 track is treated as its own singleton for ARI, AMI, and pairwise metrics. "
        "Pair false-positive rate is FP/(FP+TN); false-negative rate is FN/(TP+FN). "
        "Undefined rates are blank. Precision/recall/F1 use 0 when their denominator is zero.",
        "",
        "Thresholds are selected on this dataset for exploration. These are not held-out performance "
        "estimates, and the two epsilon scales are different.",
        "",
        "| Algorithm | Epsilon | Mean ARI | ARI std | Mean pair F1 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for algorithm, run in best.items():
        chosen = results[results.run == run].iloc[0]
        aggregate = summary[(summary.algorithm == algorithm) & (summary.eps == chosen.eps)].iloc[0]
        lines.append(
            f"| {algorithm} | {chosen.eps:g} | {aggregate.adjusted_rand_mean:.4f} | "
            f"{aggregate.adjusted_rand_std:.4f} | {aggregate.pair_f1_mean:.4f} |"
        )
    lines += [
        "",
        "The comparison plot uses the first requested seed at the selected QBx threshold, "
        "rather than the most favorable seed. See metrics.csv for all runs, summary.csv for "
        "seed means/std, assignments.csv for track-level results, and manifest.json for checksums/settings.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


def _duplicates(tracks: pd.DataFrame) -> pd.DataFrame:
    first_seen, records = {}, []
    for _, track in tracks.iterrows():
        digest = hashlib.sha256(
            np.asarray(track.geometry.coords, dtype="<f8").tobytes()
        ).hexdigest()
        key = f"{track.file_name}:{track.id}"
        if digest in first_seen:
            records.append(
                {
                    "file_name": track.file_name,
                    "id": track.id,
                    "duplicate_of": first_seen[digest],
                    "geometry_sha256": digest,
                }
            )
        else:
            first_seen[digest] = key
    return pd.DataFrame(records, columns=["file_name", "id", "duplicate_of", "geometry_sha256"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("labels", type=Path, help="CSV with file_name,id,cluster columns")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--qbx-eps", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.5, 1, 2, 5])
    parser.add_argument("--dbscan-eps", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1, 2, 5])
    parser.add_argument("--algorithms", choices=["both", "qbx", "dbscan"], default="both")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--factor", type=float, default=2)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--qbx-layer", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--id-source", choices=["gpx-number", "index"], default="gpx-number")
    args = parser.parse_args(argv)
    try:
        results = evaluate(
            args.data_dir,
            args.labels,
            args.output_dir,
            qbx_eps=args.qbx_eps if args.algorithms != "dbscan" else [],
            dbscan_eps=args.dbscan_eps if args.algorithms != "qbx" else [],
            seeds=args.seeds,
            factor=args.factor,
            min_samples=args.min_samples,
            layer=args.qbx_layer,
            id_source=args.id_source,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Evaluated {len(results)} runs; results and plots saved to {args.output_dir}")
    return 0
