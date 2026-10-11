# Refactoring record

This maintenance pass preserves the original internship project's contribution: GPX preprocessing, comparing QuickBundlesX with DBSCAN, spatial/time layers, the custom hierarchical color scheme, interactive map controls, and Docker startup.

## Preserved behavior

- Existing `python get_raw_data.py ...` and `bokeh serve run_server.py` entry points.
- Existing public imports: `get_geo_data`, `get_QB_clusters`, `get_DBSCAN_clusters`, and `gen_clusters`.
- Required `path eps eps_time` arguments, `-q`, `--factor`, `--seed`, and `--min_s`.
- Default DBSCAN algorithm and parameter defaults: factor 2, seed 0, spatial minimum 5.
- Nonrecursive GPX directory loading, joining segments per track, ignoring tracks under five points.
- Resampling to 13 points with timestamp coordinates included.
- Original weekday/hour-to-unit-sphere encoding and default host-local timezone.
- QBx thresholds `[factor² × eps, factor × eps, eps]`, ordered pointwise Euclidean spatial distance, three spatial layers, minimum spatial size 5.
- DBSCAN flattened/standardized spatial coordinates and first-sample cosine temporal clustering.
- Minimum temporal cluster size 3, custom HSB/RGB mapping, and red outliers.
- Feather output names and the `x/y`, `clustersN`, and `colorsN` columns used by the map.
- Layer selection, outlier visibility, cluster colors, timeskip markers, and hover details.

The four preserved public functions are available through the original script; private helpers now live in the appropriate package modules. Clustering functions return a copy instead of mutating the supplied frame. `gen_clusters` additionally returns the two exported tables.

## Adjustments, performed using Codex

### Intentional corrections

| Issue in the old implementation | Correction | Effect |
| --- | --- | --- |
| DIPY cosine metric received a 13-by-3 sequence although it requires a single vector | Use a small temporal Metric adapter that keeps QBx's three-dimensional centroid buffers and computes normalized angular distance over all concatenated time samples | Temporal QBx executes without dropping samples; this establishes an explicit distance where the old input was unsupported |
| All-noise DBSCAN indexed a zero-length palette | Initialize outliers as red and color only assigned tracks | All-outlier datasets produce valid output |
| QBx color mapping could index nonexistent parents when all tracks were noise | Ignore unassigned parents when coloring children | Small/all-outlier inputs remain red at every layer |
| Empty jumps frame had no geometry/CRS | Keep an explicit typed geometry schema even with no markers | Datasets without time gaps can be exported |
| Display used EPSG:3395 for Web Mercator tiles | Project display geometry to EPSG:3857 | Lines and markers align with the basemap; clustering inputs remain unchanged |
| Docker used a floating Python version and browser-opening startup | Use Python 3.12, explicit port/address, and a portable shell entry point | Container startup is predictable and works without executable-bit assumptions |

The temporal metric follows DIPY's `arccos(similarity) / π` convention. scikit-learn's DBSCAN uses `1 - similarity`; the temporal epsilon scales remain different. Numerical results, displayed cluster IDs, input order, and map positions can differ from the old implementation where these corrections apply. Historical benchmark tables have not been rerun.

### Structure and presentation changes

GPX ingestion/time features, clustering, color mapping, output orchestration, CLI parsing, and the Bokeh view now have focused modules. UI state belongs to one view instance per document. The map fits the loaded dataset and shows track/cluster/outlier counts.

The public demo uses the supplied timestamped GPX files in `demo_data/dump`. The `003` sample and evaluation artifacts are documented separately. Controlled route/time fixtures live within the regression suite. Runtime dependencies include explicit pins for directly imported packages; Matplotlib is an optional evaluation dependency. Development tooling adds pytest and Ruff, with a Python 3.12 GitHub Actions workflow.

The root README provides current setup and explains algorithm units and limitations. `docs/ARCHIVE_README.md` is an unchanged copy of the original documentation.

### Verification

Verified locally on Python 3.12 with the pinned runtime dependencies:

- **32 application regression tests passed**, including known spatial/time groups, the temporal distance convention, all-outlier cases, and per-session Bokeh controls.
- Ruff lint and formatting checks passed; pip reported no broken dependency requirements.
- Both algorithms processed the controlled 25-track regression fixture and exported one timeskip marker.
- Earlier container verification built the Docker image and exercised its actual preprocessing/server entry point at localhost:5006/run_server.
- Browser verification confirmed four temporal clusters, outlier visibility, and timeskip controls.
- Plot export was verified and saved as `docs/demo.png`. A fixed default export filename avoids Bokeh's browser prompt, which the Codex in-app browser does not support.
- Controlled regression data confirmed identical GPX resampling, time-gap durations, DBSCAN spatial labels, and both DBSCAN color layers compared with the preserved pre-refactor code. Temporal label IDs are intentionally corrected.

Regression tests check known route/time groups rather than assuming that historical evaluation scores still apply.

### Recorded-track demo

The documented demo uses `demo_data/dump`, which contains 74 GPX files with 151 tracks and 147,459 timestamped points; the application loads 125 tracks, skips 26 tracks below the five-point minimum, and exports 31 time-gap markers. Source GPX checksums were recorded before the change and verified unchanged afterward.

Both documented algorithm configurations completed with UTC and `eps = 0.5`, `eps_time = 0.1`. DBSCAN produced 9 spatial clusters and 47 outliers. QuickBundlesX produced 7 clusters and 4 outliers at its coarsest spatial layer, using the preserved factor 2 and seed 0. The new `docs/demo.png` is an export of that QuickBundlesX view. These counts describe the demo settings; the separate provisional geographic labels do not control the application colors.

Browser verification exercised the dump's temporal layer and time-gap visibility. All 32 application tests and Ruff checks passed after moving the known route/time GPX fixture into `tests/conftest.py`. The public generator and its generated GPX file were removed. Docker commands now mount `demo_data/dump`.

### Later additions, separate from the application

The new annotation/evaluation tools, plots, dependencies, saved results, and **20 extension tests** live under `extensions/spatial_analysis/`. The application package contains the maintained internship workflow and does not import the extension. The default pytest command runs only application regression tests; the extension suite has a separate command and CI job. Its additional Matplotlib dependency is excluded from the application runtime and Docker image.

The subsequently supplied hand-labeled `003` sample is evaluated by a new spatial-only script. Its timestamp-independent resampling is explicit and does not change the original application pipeline. Labels use GPX track numbers, including gaps; the source GPX files and author's CSV are untouched. Spatial assignments are tested against the application's algorithms on identical features.

The separate `dump` labels were created during maintenance using geographic rectangles. Full-resolution containment, per-track audits, and reviewed colored plots make their convention inspectable. They are marked provisional and are not represented as the author's original hand-labeled ground truth. See [extension instructions and evaluation notes](../extensions/spatial_analysis/README.md).

### References

- [DIPY clustering and metrics documentation](https://docs.dipy.org/stable/reference/dipy.segment.html): QuickBundlesX, metric feature shapes, and normalized angular distance.
- [Bokeh geographical data documentation](https://docs.bokeh.org/en/latest/docs/user_guide/topics/geo.html): tile maps in Web Mercator.
- [Bokeh server documentation](https://docs.bokeh.org/en/latest/docs/user_guide/server.html): serving interactive Python callbacks.
