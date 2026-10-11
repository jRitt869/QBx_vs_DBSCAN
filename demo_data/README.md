# Demo and evaluation data

| Sample | Files | Labels | Purpose |
| --- | --- | --- | --- |
| `003/` | 63 GPX files | `003_labels.csv`, hand-labeled by the project author | Spatial evaluation; these tracks have no timestamps |
| `dump/` | 74 timestamped GPX files | `dump/dump_labels.csv`, provisional geographic annotations created during maintenance | Application demo with spatial/time layers; geographic grouping review and exploratory spatial evaluation |

The project author confirmed authorship of the original code and labels and permission to publish these samples. The original GPX files and `003_labels.csv` are preserved unchanged. New dump annotations are described separately in [dump/LABELS.md](dump/LABELS.md), with a colored overview, regional detail plot, explicit region boundaries, and per-track audit.

See the [later spatial analysis extension](../extensions/spatial_analysis/README.md) for metrics, label IDs, exclusions, and commands. The application's temporal workflow still requires timestamped tracks; the spatial evaluator ignores timestamps even when they exist.

## Application demo

The `dump/` sample is the application's default documented demo. Its 151 tracks include 125 with at least five points and 26 shorter tracks, which the application skips. All points have timestamps; preprocessing detects 31 gaps longer than 30 minutes. The provisional geographic labels are independent of the application's cluster colors and are not required to run the viewer.

From the repository root, after installing the runtime dependencies:

```bash
python get_raw_data.py -q --timezone UTC demo_data/dump 0.5 0.1
bokeh serve run_server.py --show
```

Omit `-q` to use DBSCAN. See the [root README](../README.md#run-the-included-demo) for installation, algorithm conventions, and Docker commands. The selected data subdirectories and their labels/review artifacts are included in the repository; generated tables and local evaluation outputs remain ignored.
