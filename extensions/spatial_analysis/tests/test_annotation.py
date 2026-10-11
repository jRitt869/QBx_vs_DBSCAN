import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extensions.spatial_analysis.annotation import annotate, assign_regions, read_regions


def regions() -> list[dict]:
    return [
        {"cluster": 0, "name": "First area", "bounds": [7.9, 49.9, 8.1, 50.1]},
        {"cluster": 1, "name": "Second area", "bounds": [9.9, 51.9, 10.1, 52.1]},
    ]


def test_assignment_requires_every_point_inside_exactly_one_region() -> None:
    assert assign_regions(np.array([[8, 50], [8.05, 50.05]]), regions()) == (0, "contained")
    # A centroid/start-point-only check would incorrectly assign this crossing trace.
    assert assign_regions(np.array([[8, 50], [10, 52]]), regions()) == (-1, "outside_defined_areas")
    overlapping = [*regions(), {"cluster": 2, "bounds": [7, 49, 9, 51]}]
    assert assign_regions(np.array([[8, 50], [8.05, 50.05]]), overlapping) == (
        -1,
        "ambiguous_regions",
    )


@pytest.mark.parametrize(
    "bad",
    [
        [],
        [{"cluster": -1, "bounds": [0, 0, 1, 1]}],
        [{"cluster": 0, "bounds": [1, 0, 0, 1]}],
        [regions()[0], regions()[0]],
    ],
)
def test_invalid_regions_are_rejected(tmp_path: Path, bad: list[dict]) -> None:
    path = tmp_path / "regions.json"
    path.write_text(json.dumps({"regions": bad}), encoding="utf-8")
    with pytest.raises(ValueError):
        read_regions(path)


def test_annotation_covers_inventory_and_writes_review_artifacts(spatial_directory: Path) -> None:
    path = spatial_directory / "dump_label_regions.json"
    path.write_text(json.dumps({"regions": regions()}), encoding="utf-8")
    labels = annotate(spatial_directory, path)
    assert labels.id.tolist() == [2, 4, 7, 11, 13]
    assert labels.cluster.tolist() == [0, 0, 1, 1, -1]
    audit = pd.read_csv(spatial_directory / "dump_label_audit.csv")
    assert audit[audit.id == 13].status.item() == "excluded_short_track"
    checks = json.loads((spatial_directory / "dump_label_validation.json").read_text())
    assert all(checks["checks"].values())
    assert checks["exact_geometry_duplicates"] == 1
    assert checks["assigned_tracks"] == 4
    assert "provisional" in checks["label_origin"]
    for name in ("dump_labels.png", "dump_labels_details.png"):
        assert (spatial_directory / name).read_bytes().startswith(b"\x89PNG")
