"""Store the original spatial/time clustering in an ingested source database.

Run from the repository root: python -m historical.get_raw_data outputs/dump.sqlite 0.5 0.1
"""

import sqlite3

from .clustering import get_dbscan_clusters as get_DBSCAN_clusters
from .clustering import get_qb_clusters as get_QB_clusters
from .config.parser import build_parser
from .config.validate import ClusteringInput, validate_cli
from .data import get_geo_data
from .pipeline import run_clustering
from .storage import ClusterDatabase

__all__ = ["get_DBSCAN_clusters", "get_QB_clusters", "get_geo_data", "main"]


def main(argv=None):
    parser = build_parser("get_raw_data")
    args = validate_cli(parser, ClusteringInput, parser.parse_args(argv))
    try:
        with ClusterDatabase(args.database, writable=True) as database:
            run_id = run_clustering(
                database,
                "qbx" if args.quickbundles else "dbscan",
                **args.model_dump(exclude={"database", "quickbundles"}),
            )
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Stored historical clustering run {run_id} in {args.database}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
