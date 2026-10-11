"""Command-line preprocessing interface; the original arguments still work."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .pipeline import gen_clusters


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cluster timestamped GPX tracks for the Bokeh map."
    )
    parser.add_argument("-q", action="store_true", help="use QuickBundlesX (default: DBSCAN)")
    parser.add_argument(
        "--seed", type=int, default=0, help="QuickBundlesX shuffle seed (default: 0)"
    )
    parser.add_argument(
        "--min_s",
        "--min-samples",
        type=int,
        default=5,
        help="DBSCAN spatial min_samples (default: 5)",
    )
    parser.add_argument(
        "--factor", type=float, default=2, help="QuickBundlesX threshold multiplier (default: 2)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("."), help="directory for generated Feather tables"
    )
    parser.add_argument(
        "--timezone",
        default="local",
        help="IANA timezone, e.g. UTC or Europe/Berlin (default: local)",
    )
    parser.add_argument("path", type=Path, help="directory containing .gpx files")
    parser.add_argument("eps", type=float, help="spatial clustering threshold")
    parser.add_argument("eps_time", type=float, help="temporal clustering threshold")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        time_zone = None if args.timezone == "local" else ZoneInfo(args.timezone)
        tracks, jumps = gen_clusters(
            args.path,
            args.q,
            args.eps,
            args.eps_time,
            args.min_s,
            args.factor,
            args.seed,
            output_dir=args.output_dir,
            time_zone=time_zone,
        )
    except (OSError, ValueError, ZoneInfoNotFoundError) as error:
        parser.exit(1, f"error: {error}\n")
    logging.info(
        "Prepared %d tracks and %d timeskips in %s", len(tracks), len(jumps), args.output_dir
    )
    return 0
