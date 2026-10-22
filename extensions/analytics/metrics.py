"""Clustering agreement metrics shared by database evaluation and its tests."""

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, pair_confusion_matrix


def _separate_noise(labels: Sequence[int]) -> np.ndarray:
    """Replace each noise assignment with a distinct singleton cluster ID."""
    result = np.asarray(labels, dtype=np.int64).copy()
    noise = result == -1
    start = int(result.max(initial=-1)) + 1
    result[noise] = np.arange(start, start + int(noise.sum()))
    return result


def score_assignments(
    truth: Sequence[int], predictions: Sequence[int]
) -> dict[str, int | float | None]:
    """Compute partition, pairwise, and noise-detection agreement scores.

    Args:
        truth: One-dimensional reference labels for at least two tracks; -1 denotes noise.
        predictions: Predicted labels in the same track order and shape as truth.

    Returns:
        Cluster/outlier counts, ARI, AMI, unordered distinct-track pair counts and
        scores, and outlier-detection scores. Undefined pair error rates are None.
        Partition/pair scores treat every -1 as a separate singleton in both inputs.
    """
    truth = np.asarray(truth, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if truth.shape != predictions.shape or truth.ndim != 1 or len(truth) < 2:
        raise ValueError("Scoring requires equal one-dimensional arrays of at least two tracks")
    normalized_truth, normalized_predictions = _separate_noise(truth), _separate_noise(predictions)
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
