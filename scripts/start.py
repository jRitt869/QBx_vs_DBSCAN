"""Prepare configured source databases sequentially, then serve the Bokeh dashboard."""

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from extensions.analytics.storage import PROJECT, ClusterDatabase
from extensions.analytics.warehouse import read_config
from scripts.config.parser import build_parser
from scripts.config.validate import StartupInput, validate_cli

DEFAULT_REFERENCE_CONFIG = PROJECT / "scripts/config/reference_qbx.toml"
SPATIAL_REFERENCE_CONFIG = PROJECT / "scripts/config/reference_qbx_spatial.toml"


def _run(module, arguments):
    command = [sys.executable, "-m", f"scripts.{module}", *map(str, arguments)]
    print(f"Preparing data: {' '.join(command[2:])}", flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)


def _supports_time(path):
    with ClusterDatabase(path) as database:
        return database.supports_time()


def prepare_sources(config):
    """Import sources and evaluate each mode against labels or the default reference run."""
    sources = read_config(Path(config))
    _run("load_data", ["batch", "--config", config])
    for source in sources.datasets:
        database = source.database
        with ClusterDatabase(database) as stored:
            references = stored.eligible_references()
            labeled = len(references) >= 2 and references.reference_cluster.notna().all()
        feature_modes = ["spatial"]
        if _supports_time(database):
            feature_modes.append("spatiotemporal")
            reference_config = DEFAULT_REFERENCE_CONFIG
        else:
            reference_config = SPATIAL_REFERENCE_CONFIG
        for feature_mode in feature_modes:
            for algorithm in ("dbscan", "qbx"):
                arguments = [database]
                if algorithm == "qbx":
                    arguments.append("--quickbundles")
                if not labeled:
                    arguments.extend(["--reference-config", reference_config])
                if feature_mode == "spatiotemporal":
                    arguments.extend(["--spatiotemporal", "--seeds", "0"])
                _run("evaluate", arguments)


def main(argv=None):
    parser = build_parser("start")
    parsed, server_arguments = parser.parse_known_args(argv)
    arguments = validate_cli(parser, StartupInput, parsed)
    if server_arguments and server_arguments[0] == "--":
        server_arguments = server_arguments[1:]
    try:
        prepare_sources(arguments.config)
        if arguments.prepare_only:
            return 0
        command = [
            sys.executable,
            "-m",
            "bokeh",
            "serve",
            str(PROJECT / "scripts/run_server.py"),
            "--port",
            "5006",
            *server_arguments,
        ]
        print("Source databases are ready. Starting the dashboard on port 5006.", flush=True)
        # Replace the setup process so Bokeh receives container shutdown signals.
        os.chdir(PROJECT)
        os.execv(command[0], command)
    except subprocess.CalledProcessError as error:
        print(
            f"Data preparation failed (exit {error.returncode}); dashboard startup stopped.",
            file=sys.stderr,
        )
        return error.returncode
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.exit(1, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
