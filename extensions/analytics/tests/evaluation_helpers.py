"""Prepare independent single-algorithm evaluations for shared integration fixtures."""

from extensions.analytics.evaluation import evaluate
from track_clusters.validation import ClusteringSettings


def evaluate_sweeps(
    database, *, dbscan_eps=(), qbx_eps=(), algorithms=("dbscan", "qbx"), **settings
):
    evaluation = {
        key: settings.pop(key) for key in ("reference", "seeds", "layer") if key in settings
    }
    for algorithm, thresholds in (("dbscan", dbscan_eps), ("qbx", qbx_eps)):
        if algorithm in algorithms and thresholds:
            configuration = ClusteringSettings(algorithm=algorithm, eps=thresholds[0], **settings)
            evaluate(database, configuration, eps=thresholds, **evaluation)
