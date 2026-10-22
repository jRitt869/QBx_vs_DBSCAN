"""The historical and maintained applications do not import each other's code."""

import ast
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from track_clusters.validation import ClusteringSettings

ROOT = Path(__file__).resolve().parents[2]


def _imports(path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0]


def test_application_dependency_boundaries():
    for path in (ROOT / "historical").rglob("*.py"):
        if "tests" not in path.parts:
            assert not set(_imports(path)) & {"extensions", "scripts", "track_clusters"}, path
    for folder in ("extensions/analytics", "scripts", "track_clusters"):
        for path in (ROOT / folder).rglob("*.py"):
            if "tests" not in path.parts:
                assert "historical" not in set(_imports(path)), path


def test_historical_and_maintained_results_have_separate_viewers(track_directory, ingestion_input):
    from bokeh.document import Document
    from bokeh.models import Select

    from extensions.analytics.dashboard import build_document as dashboard
    from extensions.analytics.evaluation import run_clustering as current_clustering
    from extensions.analytics.storage import ClusterDatabase as CurrentDatabase
    from extensions.analytics.warehouse import ingest_database
    from historical.pipeline import run_clustering
    from historical.storage import ClusterDatabase
    from historical.viewer import build_document

    path = track_directory / "separate.sqlite"
    ingest_database(ingestion_input(track_directory, path))
    with ClusterDatabase(path, writable=True) as database:
        previous = run_clustering(database, "dbscan", 0.5)
    assert dashboard(Document(), path).kinds == ["overview", "analysis"]
    with CurrentDatabase(path, writable=True) as database:
        current = current_clustering(
            database, ClusteringSettings(algorithm="dbscan", eps=0.5, feature_mode="spatiotemporal")
        )
    document = Document()
    app = dashboard(document, path)
    assert app.kinds == ["overview", "dbscan", "analysis"]
    selector = next(
        item for item in document.select({"type": Select}) if item.title == "DBSCAN run"
    )
    assert [key for key, label in selector.options] == [str(current)]
    historical_document = Document()
    assert build_document(historical_document, path) is not None
    assert f"Run {previous}" in historical_document.select_one({"name": "historical_source"}).text


@pytest.mark.parametrize("entry", ["get_raw_data", "run_server"])
def test_historical_cli_runs_from_an_isolated_copy(tmp_path, entry):
    shutil.copytree(
        ROOT / "historical",
        tmp_path / "historical",
        ignore=shutil.ignore_patterns("tests", "__pycache__"),
    )
    # Use only the copied package and installed dependencies, without this checkout.
    command = (
        "import runpy,sys; "
        "sys.path.insert(0,sys.argv.pop(1)); "
        f"sys.argv=['historical.{entry}','--help']; "
        f"runpy.run_module('historical.{entry}',run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", command, str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
