"""Optional geographic annotations from explicit reviewed regions."""

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import _duplicates, _fingerprint
from .plots import plot_geographic_labels
from .spatial import load_spatial_directory


def assign_regions(coords: np.ndarray, regions: Sequence[dict]) -> tuple[int, str]:
    """Require full-resolution containment; never infer labels from a tested algorithm."""
    candidates = []
    for region in regions:
        xmin, ymin, xmax, ymax = region["bounds"]
        inside = (
            (coords[:, 0] >= xmin)
            & (coords[:, 0] <= xmax)
            & (coords[:, 1] >= ymin)
            & (coords[:, 1] <= ymax)
        )
        if inside.all():
            candidates.append(region["cluster"])
    if len(candidates) == 1:
        return int(candidates[0]), "contained"
    return -1, "ambiguous_regions" if candidates else "outside_defined_areas"


def read_regions(path: Path) -> list[dict]:
    regions = json.loads(path.read_text(encoding="utf-8"))["regions"]
    ids = set()
    for region in regions:
        cluster = region["cluster"]
        bounds = np.asarray(region["bounds"], dtype=float)
        if type(cluster) is not int or cluster < 0 or cluster in ids:
            raise ValueError("Region cluster IDs must be unique nonnegative integers")
        if bounds.shape != (4,) or not np.isfinite(bounds).all():
            raise ValueError(
                "Region bounds must be four finite [min_lon,min_lat,max_lon,max_lat] values"
            )
        if not (-180 <= bounds[0] < bounds[2] <= 180 and -90 <= bounds[1] < bounds[3] <= 90):
            raise ValueError("Invalid geographic region bounds")
        ids.add(cluster)
    if not regions:
        raise ValueError("Define at least one region")
    return regions


def annotate(data_dir: Path, region_path: Path) -> pd.DataFrame:
    """Write CSV labels, plots, and a validation audit alongside the GPX tracks."""
    dataset = load_spatial_directory(data_dir)
    regions = read_regions(region_path)
    records = []
    for _, track in dataset.tracks.iterrows():
        coords = np.asarray(track.geometry.coords)
        cluster, reason = assign_regions(coords, regions)
        xmin, ymin, xmax, ymax = track.geometry.bounds
        digest = hashlib.sha256(np.asarray(coords, dtype="<f8").tobytes()).hexdigest()
        records.append(
            {
                "file_name": track.file_name,
                "id": track.id,
                "track_index": track.track_index,
                "cluster": cluster,
                "point_count": track.point_count,
                "min_lon": xmin,
                "min_lat": ymin,
                "max_lon": xmax,
                "max_lat": ymax,
                "status": reason,
                "stationary": bool(xmin == xmax and ymin == ymax),
                "geometry_sha256": digest,
            }
        )
    audit = pd.DataFrame(records)
    # Explicitly record short tracks as excluded/unassigned; they are never scored.
    excluded = dataset.skipped.assign(cluster=-1, status="excluded_short_track", stationary=False)
    all_audit = pd.concat([audit, excluded], ignore_index=True)
    labels = all_audit[["file_name", "id", "cluster"]].sort_values(["file_name", "id"])
    eligible_labels = audit["cluster"].to_numpy(dtype=int)
    counts = audit[audit.cluster >= 0].groupby("cluster").size()
    if (counts < 2).any() or set(counts.index) != {r["cluster"] for r in regions}:
        raise ValueError(
            "Each defined area must contain at least two eligible tracks; review region definitions"
        )
    if labels.duplicated(["file_name", "id"]).any():
        raise ValueError("Duplicate label keys")
    duplicate_records = _duplicates(dataset.tracks)
    labels.to_csv(data_dir / "dump_labels.csv", index=False)
    all_audit.to_csv(data_dir / "dump_label_audit.csv", index=False)
    dataset.skipped.to_csv(data_dir / "dump_skipped_tracks.csv", index=False)
    duplicate_records.to_csv(data_dir / "dump_duplicates.csv", index=False)
    plot_geographic_labels(dataset.tracks, eligible_labels, regions, data_dir / "dump_labels.png")

    checks = {
        "label_origin": "Agent-created geographic annotations; provisional until human review",
        "ground_truth_note": "Containment and visual consistency do not independently establish scientific ground truth",
        "id_source": "GPX number, falling back to zero-based index only if no number exists",
        "files": len(dataset.files),
        "label_rows": len(labels),
        "eligible_tracks": len(audit),
        "excluded_short_tracks": len(excluded),
        "area_count": len(regions),
        "assigned_tracks": int((eligible_labels >= 0).sum()),
        "unassigned_eligible_tracks": int((eligible_labels == -1).sum()),
        "stationary_eligible_tracks": int(audit.stationary.sum()),
        "exact_geometry_duplicates": len(duplicate_records),
        "checks": {
            "unique_keys": True,
            "complete_inventory_coverage": len(labels) == len(audit) + len(excluded),
            "all_assigned_geometry_inside_its_region": bool(
                (audit[audit.cluster >= 0].status == "contained").all()
            ),
            "all_regions_have_at_least_two_eligible_tracks": True,
        },
        "regions": regions,
        "source_files": [_fingerprint(file) for file in dataset.files],
        "labels": _fingerprint(data_dir / "dump_labels.csv"),
        "region_file": _fingerprint(region_path),
    }
    (data_dir / "dump_label_validation.json").write_text(
        json.dumps(checks, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Geographic labels for dump",
        "",
        "These are **provisional agent-created geographic annotations**, created at the project author's "
        "request to group tracks by area. They are not the author's original hand-labeled ground truth "
        "and do not use DBSCAN or QuickBundlesX predictions.",
        "",
        f"The {len(dataset.files)} GPX files contain {len(labels)} tracks: {len(audit)} eligible tracks and "
        f"{len(excluded)} tracks below the existing five-point minimum. "
        f"{int((eligible_labels >= 0).sum())} eligible tracks occupy {len(regions)} defined areas; "
        f"{int((eligible_labels == -1).sum())} remain unassigned.",
        "",
        "## Annotation rule",
        "",
        "Areas are explicit longitude/latitude rectangles reviewed from the trace inventory. "
        "Every coordinate of a track must lie inside exactly one area. Isolated areas without a "
        "second eligible track, cross-region traces, and ambiguous overlaps are unassigned (-1). "
        "The same -1 is recorded for short tracks, whose status is excluded_short_track and which "
        "are omitted by the evaluator. IDs use the GPX <number> field, not list position.",
        "",
        "Geographic area means proximity within the declared box, not the same route. Area sizes vary "
        "with the observed geographic groups; this is a documented annotation convention rather than "
        "a unique clustering truth. Regions should be reviewed before using these labels for claims "
        "about algorithm quality.",
        "",
        "## Review artifacts",
        "",
        "- dump_labels.csv: compatible file_name,id,cluster label table.",
        "- dump_labels.png: all traces colored by area, with area rectangles and unassigned tracks in dashed gray.",
        "- dump_labels_details.png: a zoomed panel for every assigned area.",
        "- dump_label_regions.json: inspectable region definitions, independent of the tested algorithms.",
        "- dump_label_audit.csv: per-track bounds, assignment/exclusion status, and stationary flags.",
        "- dump_duplicates.csv: exact geometry duplicates retained in the dataset.",
        "- dump_label_validation.json: structural checks and source/label checksums.",
        "",
        f"There are {int(audit.stationary.sum())} stationary eligible traces and "
        f"{len(duplicate_records)} exact duplicate geometries. They are retained and flagged, so duplicated "
        "or point-like records can increase apparent cluster support. Long international/cross-region "
        "traces are not forced into a nearby area.",
        "",
        "## Area inventory",
        "",
        "| Label | Geographic center hint | Eligible tracks | Bounds [lon min, lat min, lon max, lat max] |",
        "| --- | --- | --- | --- |",
    ]
    for region in regions:
        report.append(
            f"| {region['cluster']} | {region['name']} | {counts[region['cluster']]} | {region['bounds']} |"
        )
    report += [
        "",
        "Regenerate deterministically from the repository root:",
        "",
        "```bash",
        "python -m extensions.spatial_analysis.annotate_dump demo_data/dump --regions demo_data/dump/dump_label_regions.json",
        "```",
        "",
        "Structural containment, inventory coverage, and colored plots make the annotations "
        "auditable. They do not replace an independent human assessment of the intended geographic categories.",
    ]
    (data_dir / "LABELS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return labels


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--regions", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        labels = annotate(args.data_dir, args.regions)
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Wrote {len(labels)} track labels, colored plots, and audit files to {args.data_dir}")
    return 0
