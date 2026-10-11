import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from extensions.spatial_analysis.evaluation import (
    align_labels,
    evaluate,
    read_labels,
    score_assignments,
)
from extensions.spatial_analysis.spatial import cluster_spatial, load_spatial_directory
from track_clusters.clustering import get_dbscan_clusters, get_qb_clusters


def labels_for_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {"file_name": ["000001.gpx"] * 5, "id": [2, 4, 7, 11, 13], "cluster": [0, 0, 1, 1, -1]}
    )


def test_spatial_loading_uses_gpx_numbers_and_ignores_elevation(spatial_directory: Path) -> None:
    dataset = load_spatial_directory(spatial_directory)
    assert dataset.tracks.id.tolist() == [2, 4, 7, 11]
    assert dataset.tracks.track_index.tolist() == [0, 1, 2, 3]
    assert dataset.skipped.id.tolist() == [13]
    assert "times" not in dataset.tracks
    for sampled, geometry in zip(dataset.tracks["13_pts"], dataset.tracks.geometry, strict=True):
        assert sampled.shape == (13, 3)
        assert (sampled[:, 2] == 0).all()
        np.testing.assert_allclose(sampled[0, :2], geometry.coords[0])
        np.testing.assert_allclose(sampled[-1, :2], geometry.coords[-1])
    assert load_spatial_directory(spatial_directory, id_source="index").tracks.id.tolist() == [
        0,
        1,
        2,
        3,
    ]


@pytest.mark.parametrize("algorithm,layer", [("dbscan", 3), ("qbx", 1), ("qbx", 2), ("qbx", 3)])
def test_spatial_algorithms_match_application(
    feature_tracks: gpd.GeoDataFrame, algorithm: str, layer: int
) -> None:
    assignments = cluster_spatial(feature_tracks, algorithm, 0.1, seed=42, layer=layer)
    original = (
        get_dbscan_clusters(feature_tracks, 0.1, 0.01, 5)["clusters0"]
        if algorithm == "dbscan"
        else get_qb_clusters(feature_tracks, 0.1, 0.01, 2, 42)[f"clusters{layer - 1}"]
    )
    assert score_assignments(original, assignments)["adjusted_rand"] == 1
    np.testing.assert_array_equal(original == -1, assignments == -1)


def test_label_alignment_reports_short_tracks_and_rejects_mismatches(
    spatial_directory: Path,
) -> None:
    dataset = load_spatial_directory(spatial_directory)
    labels = labels_for_fixture()
    aligned, excluded = align_labels(dataset, labels.iloc[::-1])
    assert aligned.cluster.tolist() == [0, 0, 1, 1]
    assert excluded.id.tolist() == [13]
    with pytest.raises(ValueError, match="Missing labels"):
        align_labels(dataset, labels.iloc[1:])
    with pytest.raises(ValueError, match="unknown tracks"):
        align_labels(
            dataset,
            pd.concat(
                [labels, pd.DataFrame({"file_name": ["000001.gpx"], "id": [99], "cluster": [0]})]
            ),
        )


@pytest.mark.parametrize(
    "bad_column,bad_value", [("id", 1.5), ("id", -1), ("cluster", -2), ("cluster", np.nan)]
)
def test_invalid_labels_are_rejected(tmp_path: Path, bad_column: str, bad_value: float) -> None:
    labels = labels_for_fixture().astype({bad_column: float})
    labels.loc[0, bad_column] = bad_value
    path = tmp_path / "labels.csv"
    labels.to_csv(path, index=False)
    with pytest.raises(ValueError):
        read_labels(path)


def test_duplicate_label_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    pd.concat([labels_for_fixture(), labels_for_fixture().iloc[:1]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        read_labels(path)


def test_pair_metrics_and_singleton_noise() -> None:
    # Six pairs: TP (0,1), FP (0,2)/(1,2), and three TN; no positive noise pair.
    scores = score_assignments([0, 0, -1, -1], [8, 8, 8, -1])
    assert [scores[f"pair_{key}"] for key in ("tp", "fp", "fn", "tn")] == [1, 2, 0, 3]
    assert scores["pair_precision"] == pytest.approx(1 / 3)
    assert scores["pair_recall"] == 1
    assert scores["pair_f1"] == pytest.approx(0.5)
    assert scores["pair_false_positive_rate"] == pytest.approx(2 / 5)
    assert scores["outlier_detection_recall"] == 0.5
    assert score_assignments([0, 0, -1, -1], [9, 9, -1, -1])["adjusted_rand"] == 1
    all_noise = score_assignments([-1, -1, -1], [-1, -1, -1])
    assert all_noise["pair_tn"] == 3
    assert all_noise["pair_false_negative_rate"] is None
    assert all_noise["adjusted_rand"] == 1
    with pytest.raises(ValueError, match="equal one-dimensional"):
        score_assignments([0, 1], [0])


def test_evaluation_writes_reproducible_audit_and_plots(spatial_directory: Path) -> None:
    labels_path = spatial_directory / "labels.csv"
    labels_for_fixture().to_csv(labels_path, index=False)
    output = spatial_directory / "results"
    runs = evaluate(
        spatial_directory,
        labels_path,
        output,
        qbx_eps=[0.1],
        dbscan_eps=[0.1],
        seeds=[0, 7],
        min_samples=2,
    )
    assert runs.algorithm.tolist() == ["dbscan", "qbx", "qbx"]
    assert runs.tracks.tolist() == [4, 4, 4]
    assert pd.read_csv(output / "assignments.csv").id.tolist() == [2, 4, 7, 11]
    assert pd.read_csv(output / "excluded_labels.csv").id.tolist() == [13]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["eligible_tracks"] == 4
    assert manifest["exact_geometry_duplicates"] == 1
    assert manifest["excluded_label_rows"] == 1
    assert len(manifest["input_files"][0]["sha256"]) == 64
    assert "seed0" in manifest["best_parameters"]["qbx"]
    for name in ("comparison.png", "parameter_sweep.png"):
        assert (output / name).read_bytes().startswith(b"\x89PNG")


def test_duplicate_sweep_values_are_rejected(spatial_directory: Path) -> None:
    with pytest.raises(ValueError, match="duplicates"):
        evaluate(
            spatial_directory,
            spatial_directory / "labels.csv",
            spatial_directory / "out",
            qbx_eps=[0.1, 0.1],
            dbscan_eps=[],
            seeds=[0],
        )
