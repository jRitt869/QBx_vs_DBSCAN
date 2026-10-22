# Original internship visualization

The map, layer controls, hierarchical colors, outlier toggles, time-gap markers, and hover details preserve the internship visualization. The later storage adapter reads the same source databases as the dashboard; Feather files are no longer used.

These entry points retain weekday/hour sphere encoding and the original spatial/time metrics. All runtime code is contained in this folder: the historical application does not import `extensions`, `track_clusters`, or `scripts`. The maintained dashboard displays its calibrated runs separately from historical results.

| Files | Responsibility |
| --- | --- |
| `data.py`, `clustering.py`, `colors.py`, `validation.py` | Original feature encoding, numerical methods, coloring, and parameter constraints |
| `pipeline.py`, `storage.py` | Historical run orchestration and access to compatible source databases |
| `map_view.py`, `viewer.py` | Preserved map controls/rendering and source/run selection |
| `config/` | Independent JSON CLI layouts, parser, and input models |
| `tests/` | Original numerical regressions and dependency-isolation checks |

Dependencies are listed in `historical/requirements.txt`. Source ingestion is a separate preparation step; the historical application reads the resulting SQLite format without importing the ingestion code.

## Run the standalone historical application

Use **Python 3.12**. The included [`demo_data/dump`](../demo_data/README.md) sample contains 74 GPX files and 151 recorded tracks. The application loads 125 tracks with at least five points, skips 26 shorter tracks, and detects 31 gaps longer than 30 minutes. All supplied points have timestamps, so both spatial and temporal layers are available. The GPX files are used unchanged.

Create an environment and install dependencies from the repository root:

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on Linux/macOS or `.\.venv\Scripts\Activate.ps1` in PowerShell, then run:

```bash
python -m pip install -r requirements.txt
python -m scripts.load_data single demo_data/dump --database-key dump
python -m historical.get_raw_data outputs/dump.sqlite 0.5 0.1 --timezone UTC
bokeh serve historical/run_server.py --show --args outputs/dump.sqlite
```

Open [http://localhost:5006/run_server](http://localhost:5006/run_server). Historical preprocessing defaults to DBSCAN with timestamp-inclusive resampling. The database is supplied explicitly after Bokeh's `--args`; the source name and stored run ID appear above the map. Both server entry points use that route by default; stop one before starting the other on the same port.

At these settings, DBSCAN produces 9 spatial clusters and 47 outliers on the dump sample.

To try QuickBundlesX, store its run in the same database and open that algorithm:

```bash
python -m historical.get_raw_data -q outputs/dump.sqlite 0.5 0.1 --timezone UTC
bokeh serve historical/run_server.py --show --args outputs/dump.sqlite --quickbundles
```

These are exploratory display settings. Each algorithm uses its own distance convention, and matching epsilon values do not make their thresholds equivalent. The demo colors come from clustering predictions.

## Historical preprocessing options

```text
python -m historical.get_raw_data [-q] [--seed N] [--factor F] [--min_s N]
                                 [--timezone ZONE] [--spatial-only]
                                 database eps [eps_time]
```

Ingest the GPX folder and references first using `scripts/load_data.py single`. The historical clustering script reads only that database's stored tracks. Tracks need at least five points with nonnegative, chronological timestamps for this timestamp-inclusive workflow. Missing timestamps produce a file/track error. Segments within each GPX track are joined, as in the original implementation.

| Argument | Meaning | Default |
| --- | --- | --- |
| `database` | Source SQLite database containing ingested tracks | Required |
| `-q` | Use QuickBundlesX | DBSCAN |
| `--spatial-only` | Skip temporal features and subdivisions | Timestamp-inclusive features |
| `eps` | Spatial clustering threshold | Required |
| `eps_time` | Temporal clustering threshold | 0.1 |
| `--factor` | QBx thresholds: `[factor² × eps, factor × eps, eps]`; at least 1 | 2 |
| `--seed` | Integer seed for QBx input ordering | 0 |
| `--min_s` / `--min-samples` | DBSCAN spatial `min_samples` | 5 |
| `--timezone` | IANA timezone, e.g. `UTC` or `Europe/Berlin`; `local` uses host-local time | UTC |

QBx spatial clusters smaller than five tracks are displayed as outliers. Both temporal layers use a minimum size of three. DBSCAN's spatial minimum can be changed independently. Historical preprocessing uses timestamp-inclusive features by default; `--spatial-only` selects its alternative. The maintained `get_clusters` and `evaluate` commands default to spatial-only features and use `--spatiotemporal` to include timestamps.

For a new settings key, preprocessing reads stored points and saves all layers and colors in `runs` / `assignments`. An existing key immediately returns its stored run ID without recalculating or changing metadata, assignments, or scores. General ingestion/reference/evaluation details are in the [database workflow](../extensions/analytics/README.md).

Map coordinates are reconstructed from stored full-resolution points; time gaps come from the SQL view `v_time_gaps`. Parameters and clustering times are stored in `runs`; `assignments` holds labels and colors for every layer. Evaluation adds scores in `evaluation_results` linked to the same run and layer, without copying predictions. No script creates `raw_data`, `raw_data_jumps`, or `run_metadata.json`. The original GPX-based numerical pipeline remains callable in memory for regression checks.

## Historical viewer

The database argument after `--args` is required. The viewer displays its source name and run ID above the map. It selects the latest timestamp-inclusive DBSCAN run by default, or a QuickBundlesX run with `-q` / `--quickbundles`. Supply `--run-id ID` to select a particular stored run matching that algorithm choice. Sources with no matching run display an explanation without creating data or substituting another source. Add `--help` after `--args` for the historical viewer's argument descriptions. `CLUSTER_DATA_DIR` is no longer used.

For Python callers:

```python
from historical.viewer import build_document

view = build_document(document, "outputs/dump.sqlite", algorithm="qbx")
```

`build_document(document, data_source, *, run_id=None, algorithm=None)` opens the selected database read-only. It reconstructs projected map coordinates from `points`, time gaps from `v_time_gaps`, and layer assignments/colors from `assignments`, then passes the tables to `historical.map_view.ClusterView`. No raw-file access or clustering occurs in the viewer.

The Python API can select the latest timestamp-inclusive run across algorithms when `algorithm=None`.

The basemap uses an external tile service and needs an internet connection. Track data and clustering run locally.

Run `python -m pytest -q historical/tests` for comparisons of direct GPX loading with stored features, labels, colors, projected coordinates, and time-gap markers. Isolation tests launch a copied historical folder with no access to this checkout's application packages.

The verified former root Feather files and their metadata are retained unchanged in the ignored `local_archive/feather-before-sqlite-20261016/`. All values needed to display those results are represented in `outputs/dump.sqlite`; no separate binary copy is required at runtime.
