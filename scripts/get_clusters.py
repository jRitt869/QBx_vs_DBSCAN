"""Cluster ingested source tracks and persist assignments in their SQLite database."""

import sqlite3

from extensions.analytics.evaluation import run_clustering
from extensions.analytics.storage import ClusterDatabase
from scripts.config.parser import build_parser
from scripts.config.validate import ClusteringInput, validate_cli
from track_clusters.validation import ClusteringSettings


def main(argv=None):
    parser = build_parser("get_clusters")
    args = validate_cli(parser, ClusteringInput, parser.parse_args(argv))
    try:
        settings = ClusteringSettings(
            algorithm="qbx" if args.quickbundles else "dbscan",
            **args.model_dump(exclude={"database", "quickbundles"}),
        )
        with ClusterDatabase(args.database, writable=True) as database:
            run_id = run_clustering(database, settings)
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Stored clustering run {run_id} in {args.database}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
