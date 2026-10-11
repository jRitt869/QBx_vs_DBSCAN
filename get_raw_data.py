"""Preprocess GPX files; retained as the original command-line entry point."""

from track_clusters.cli import main
from track_clusters.clustering import get_dbscan_clusters as get_DBSCAN_clusters
from track_clusters.clustering import get_qb_clusters as get_QB_clusters
from track_clusters.data import get_geo_data
from track_clusters.pipeline import gen_clusters

__all__ = ["gen_clusters", "get_DBSCAN_clusters", "get_QB_clusters", "get_geo_data", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
