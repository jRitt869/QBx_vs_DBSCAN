# Analytics technical reference

For installation, launch commands, the dashboard tour, and architecture goals, see the [main README](../../README.md). This guide covers the interfaces and data contracts in this package and its corresponding `scripts/` entry points.

## Module responsibilities

| Element | Responsibility |
| --- | --- |
| [`warehouse.py`](warehouse.py) | Source discovery, TOML loading, GPX/CSV parsing, incremental imports, and database creation |
| [`storage.py`](storage.py) | Connection lifetime, reflected SQLAlchemy metadata, bound queries, transactions, and data loading |
| [`evaluation.py`](evaluation.py) | Feature preparation, clustering orchestration, settings keys, reference configurations, and score persistence |
| [`metrics.py`](metrics.py) | Partition, pairwise, and outlier-detection agreement scores |
| [`temporal.py`](temporal.py) | Spatial resampling, distance calibration, calendar phases, cyclic metrics, and circular centroid adapters |
| [`dashboard.py`](dashboard.py) | Session state, database reads, data preparation, callbacks, and catalog downloads |
| [`layouts.py`](layouts.py), [`presentation.py`](presentation.py) | Passive visual groups, shared components, plot styling, and display adapters |
| [`sql/schema.sql`](sql/schema.sql) | Tables, constraints, triggers, and report views |
| [`scripts/config/`](../../scripts/config/) | JSON argument definitions, parser construction, and Pydantic input models for the entry points |

Internal helpers use underscore names. Functions and methods accessed outside their module or class have detailed purpose-and-argument docstrings; internal helpers have short summaries.

## Ingestion contracts

Batch TOML files contain `[[datasets]]` entries validated as `IngestionInput` objects. Each requires `source_folder` and `dataset_key`; optional fields are `name`, `labels`, `label_kind`, `label_description`, and `on_conflict`. Input paths resolve relative to the TOML file. Source keys must be unique, contain only ASCII letters, digits, underscores, or hyphens, and produce a valid portable filename. GPX discovery is nonrecursive. Both ingestion subcommands return a JSON list of import summaries.

The `on_conflict` field or `--on-conflict` flag controls changed GPX files with an existing filename:

| Policy | Behavior |
| --- | --- |
| `skip` (default) | Keep the stored file and its tracks. |
| `replace` | Replace that file's stored tracks with the supplied contents. |
| `error` | Reject the import and roll back its changes. |

Identical checksums are always reused, regardless of policy. Files absent from the input directory remain stored. Python callers can override the model's policy with `ingest_database(inputs, config=None, on_conflict="replace")`.

Reference CSVs require `file_name`, `id`, and `cluster` columns. `id` is the zero-based track index within its GPX file; `cluster` is an integer greater than or equal to `-1`, with `-1` denoting an outlier. File/track keys must be unique and match stored tracks. Omitting a CSV preserves stored labels. Supplying one replaces labels for files in the import directory or named in that CSV; other files' labels remain. Each CSV path has its own checksum and provenance record.

Eligible input changes invalidate clustering runs and evaluations; changes affecting only short tracks preserve them. Label-only changes invalidate label-based scores while preserving reference-run evaluations. Import updates and their invalidations share one transaction. See [Storage](#storage) for database creation and transaction ownership.

## Clustering and evaluation

### Evaluate one algorithm

The evaluation CLI defaults to DBSCAN; `-q` / `--quickbundles` selects QuickBundlesX. When `--eps` is omitted, the sweeps are:

| Algorithm | Primary thresholds | Seeds | Default scoring layer |
| --- | --- | --- | --- |
| DBSCAN | `0.1 0.25 0.5 1 2 5` | One run with seed `0` | 1 |
| QuickBundlesX | `0.01 0.05 0.1 0.5 1 2 5` | `0 1 2 3 4` | 3 |

`--eps` and `--seeds` accept distinct values. The seed list may be empty for DBSCAN, which always uses seed `0`; QuickBundlesX requires at least one seed. `--layer` selects a primary scoring layer; DBSCAN accepts only 1 and QuickBundlesX accepts 1–3. Commands and configuration use one-based layer numbers; stored indices are zero-based. Target features default to spatial mode; `--eps-time` and `--timezone` apply to the target's spatiotemporal mode independently of reference settings.

Label-based scoring requires at least two eligible tracks and complete reference coverage, including explicit `-1` labels for outliers. Missing or partial coverage causes an error. A supplied reference configuration takes precedence over stored hand-made labels.

### Cyclic temporal metrics

Spatiotemporal runs accept `--p` (default `2`, finite and at least `1`) and use all day/week/month phases. `calendar_phases()` computes local wall-clock fractions of the day, Monday-based week, and actual calendar month, including leap years and subsecond time. The configured timezone determines these coordinates; repeated wall-clock times at a daylight-saving transition have the same phases.

Both algorithms use `temporal_mode = "separate"` by default: primary layers cluster spatial features, followed by one temporal subdivision using `eps_time`. Adding `--joint` selects `temporal_mode = "joint"`: all primary layers use the combined spatial/torus distance and there is no extra time layer. `eps_time` is inactive in joint mode; `eps` controls the joint threshold. These settings are inactive in spatial-only mode.

```bash
python -m scripts.get_clusters outputs/dump.sqlite 0.5 --spatiotemporal --joint
python -m scripts.get_clusters outputs/dump.sqlite 0.5 -q --spatiotemporal --joint
python -m scripts.evaluate outputs/dump.sqlite --spatiotemporal --joint --reference-config scripts/config/reference_qbx.toml
```

For phases in `[0, 1)`, let `delta(t,s) = min(|t-s|, 1-|t-s|)`. Each corresponding pair of positions has Euclidean spatial distance:

```text
d_S(A_i,B_i) = ||x_i-y_i||_2.
```

The purely temporal pointwise distance on the day/week/month torus is

```text
d_T(t,s) = ((2 delta(t_day,s_day))^p
           + (2 delta(t_week,s_week))^p
           + (2 delta(t_month,s_month))^p)^(1/p).
```

Each torus component is a shortest angular distance divided by pi, hence lies in `[0, 1]`. Both algorithms retain all 13 samples and take the arithmetic mean of their pointwise distances:

```text
D_S(A,B) = mean(i=1..13) d_S(A_i,B_i).
D_T(A,B) = mean(i=1..13) d_T(A_i,B_i).
```

The Euclidean/Lp norms operate within each sample; averaging combines samples into a track distance. In joint mode, space and time are combined at each sample before taking the mean, using the calibration below. Repeating every sample equally leaves the distance unchanged. Changing `p` never changes the spatial distance or its fitted scale.

In separate mode, DBSCAN's primary `13_pts` arrays contain only longitude and latitude. Its adapter reshapes flattened buffers before calling the shared spatial/time distance with temporal weight zero, giving mean pointwise Euclidean distance multiplied by the fitted spatial weight. QuickBundlesX uses padded spatial samples and the same weighted mean pointwise Euclidean distance. Day/week/month phases stay in separate `times` arrays and are clustered within each finest primary cluster using the averaged torus metric. The temporal exponent does not affect the spatial layers. Joint DBSCAN instead uses flattened 13-by-5 arrays (x, y, day, week, month) and the same shared distance with both weights active.

`resample_spatial_track()` places DBSCAN's and separate-mode QuickBundlesX's 13 samples at equal spatial arc-length intervals in longitude/latitude, independently of timestamps. Epochs are linearly interpolated onto those positions before calendar encoding. At pauses, time jumps from arrival to departure: the preceding movement ends at the arrival epoch, and the following movement starts at the departure epoch. An interior sample exactly at the pause uses the departure epoch; recorded start/end epochs are retained. Fully stationary tracks keep their fixed position and interpolate epochs by recorded point index. Joint QuickBundlesX uses timestamp-inclusive resampling.

For QuickBundlesX, sine/cosine rows let arithmetic centroid updates represent circular means. Joint features occupy 52 three-coordinate rows: 13 spatial rows followed by 39 circular rows, ordered by sample and day/week/month. Separate temporal buffers contain only those 39 circular rows. The metric reads their directions and evaluates the distances above. A zero circular resultant uses phase zero as a deterministic tie. These packed buffers are internal features; stored source points remain full-resolution longitude/latitude and epochs.

### Distance calibration

`fit_distance_scales()` uses every unordered track pair when there are at most `scale_pairs` pairs, otherwise a uniform sample without replacement. `--scale-pairs` defaults to `10000`. The run's `seed` controls pair sampling and QBx's track ordering; `get_clusters --seed` defaults to `0` for both algorithms. Calibration uses only eligible tracks and the same sampled pairs for both joint components. It never uses reference labels. For a fixed seed and feature configuration, scales are identical across epsilon values. Changing the seed can change the sampled pairs and fitted scales; QBx's `--seeds` sweep therefore measures variation from both calibration and ordering. DBSCAN evaluation uses a single seed-zero run per threshold.

For both algorithms and modes, `D_S` is the raw mean pointwise Euclidean track distance. The spatial weight `w_S` is the reciprocal median strictly positive sampled `D_S`, independent of `p`. Every coordinate and sample uses this same multiplicative weight. Before clustering, all tracks are centered on the same mean 13-point spatial profile. This preserves corresponding-point distances while avoiding precision loss from large absolute offsets, including in QBx's float32 centroid buffers. Stored geometries remain unchanged longitude/latitude degrees; calibration does not convert them to physical distances.

In joint mode, `D_T` is the mean pointwise day/week/month torus distance, and `w_T` its reciprocal median strictly positive value on the same pairs. The primary distance is

```text
D(A,B) = mean(i=1..13) ((w_S * d_S(A_i,B_i))^p
                      + (w_T * d_T(A_i,B_i))^p)^(1/p).
```

Thus each active component's nonzero sampled median is one, with equal normalized contributions. The weights stay fixed for track and centroid comparisons; circular phases and the temporal subdivision metric are unchanged. An active component with no positive sampled distances uses weight `1`; the same fallback applies when there is only one eligible track. Outside joint mode, `w_T = 0` disables temporal calculations in the primary metric. Pair counts and nonzero counts record these cases explicitly.

`settings_json` stores `scale_pairs` and the shared `seed`; these belong to the settings key. The run's JSON `provenance` stores fitted `distance_scales` (`spatial`, `temporal`, `spatial_center`, `sampled_pairs`, `spatial_nonzero_pairs`, `temporal_nonzero_pairs`). Its spatial/temporal values are multiplicative weights, with temporal weight zero outside joint mode. Calibration metadata is saved in the same transaction as assignments.

Both clustering kernels require an explicit primary metric. `run_clustering()` always supplies DBSCAN's shared spatial/time callable. For QBx, it selects `SpatiotemporalMetric` in joint mode or `AveragePointwiseEuclideanMetric` on spatially weighted coordinates otherwise. The kernels do not choose fallback primary metrics.

### Use a reference clustering run

`ReferenceConfiguration` in `evaluation.py` validates these TOML fields:

| Field | Requirement or default |
| --- | --- |
| `algorithm` | Required: `dbscan` or `qbx` |
| `eps` | Required positive primary threshold |
| `feature_mode` | `spatial` or `spatiotemporal`; default `spatial` |
| `eps_time`, `timezone` | Separate-layer threshold and encoding zone; defaults `0.1` and `UTC`. The threshold is inactive in joint mode. |
| `temporal_mode`, `p` | `separate` (default) or `joint`, and the cyclic Lp exponent (default `2`). Inactive in spatial-only mode. |
| `scale_pairs` | Maximum distinct calibration pairs; default `10000` |
| `seed` | Shared pair-sampling and QBx ordering seed; default `0` |
| `min_samples` | DBSCAN neighborhood minimum; default `5` |
| `factor` | QuickBundlesX hierarchy multiplier; default `2` |
| `layer` | Default finest primary layer: DBSCAN 1 or QuickBundlesX 3 |

A temporal reference layer can be selected explicitly in separate spatiotemporal mode: DBSCAN 2 or QuickBundlesX 4. Joint mode has only the primary layers, DBSCAN 1 or QuickBundlesX 1–3. Unknown fields and unavailable layers are rejected. Manual evaluation uses the supplied configuration exactly, without substituting a spatial configuration when timestamps are unsuitable.

The reference algorithm can differ from the target algorithm. Its selected layer must cover every eligible track. The evaluation manifest records the reference run ID, zero-based layer, normalized configuration, noise policy, and scoring interpretation. Different reference runs or layers have separate evaluation identities. Their scores measure agreement with that baseline rather than ground-truth accuracy.

### Container startup

`scripts.start` uses hand-made labels when at least two eligible tracks have complete coverage. Otherwise it applies the source's default reference configuration. All target algorithms and modes share that source's chosen baseline. Spatial and spatiotemporal evaluations use the same algorithm-specific epsilon sweeps above. Spatiotemporal evaluations use seed `0`.

### Scoring conventions

`metrics.score_assignments()` returns adjusted Rand index (ARI), adjusted mutual information (AMI), unordered distinct-track pair counts and scores, cluster/outlier counts, and outlier-detection precision/recall. For partition and pair scores, each `-1` is a separate singleton in both inputs. Outlier detection compares the `-1` masks directly. Pair precision, recall, and F1 use zero when their denominator is zero; undefined pair error rates are stored as NULL.

The parameter-sweep report aggregates the requested QuickBundlesX seeds with means, sample variances, and observed ranges. Threshold selection maximizes mean ARI on the scored sample. These scores describe exploratory in-sample selection, without held-out validation.

## Input validation

Validation is shared across CLI commands, TOML inputs, clustering operations, and database writes:

| Boundary | Mechanism and checks |
| --- | --- |
| CLI syntax and JSON layouts | `scripts/config/cli.json` supplies argument layouts and required help text. `parser.py` checks model/field names, complete field coverage, duplicate/reserved flags, argument counts, and action/type compatibility before building argparse parsers. Binary flags select the model's non-default value; categorical options and defaults are added to help automatically. |
| CLI and TOML values | Pydantic models in `scripts/config/validate.py` reject unknown fields and invalid paths, source keys, modes, numeric ranges, seeds, and duplicate sweep values. TOML paths resolve relative to their file; both ingestion subcommands use the same models and database-key resolution. Errors identify the offending fields. |
| GPX records | Ingestion checks parsing, finite coordinates, schema longitude/latitude bounds, and finite optional elevation. Missing timestamps remain NULL; naive timestamps are treated as UTC and flagged. Short tracks remain stored but are excluded from clustering. |
| Target and reference settings | Validate finite positive thresholds, factor bounds, integer seed limits, time zones, and available layers before clustering. Unused parameters are normalized to None before settings-key construction. |
| Temporal readiness | `v_track_time_quality` requires complete, finite, nonnegative, nondecreasing epochs on eligible tracks. Startup and clustering use `supports_time()`; schema triggers enforce the same rule on temporal-run inserts and mode changes. |
| Database integrity and concurrent updates | SQL keys, constraints, and triggers guard stored records. Write transactions roll back on failure; integrity/foreign-key checks validate imports. Input signatures reject source changes during clustering or evaluation before results are committed. |

## Storage

SQLAlchemy Core reflects the schema objects; the `sqlite3` connection executes bound statements.

| Tables | Contents |
| --- | --- |
| `datasets`, `source_files`, `tracks`, `points` | Source and file identities, file checksums, per-file track indices, ordered points, and eligibility/timestamp inventory |
| `labels` | Reference assignments keyed by track ID |
| `runs`, `assignments` | Unique configuration keys and JSON settings; assignments keyed by `(run_id, track_id, layer)` |
| `evaluations`, `evaluation_results` | Evaluation scope and reference identity; scores uniquely keyed by `(evaluation_id, run_id, layer)` |
| `build_metadata`, `artifacts` | Import settings, software versions, code/schema hashes, and supplied-file provenance |

`Database` exposes `connection`, reflected tables, generic select/insert/update/delete helpers, and `transaction()`. `select()` builds a statement; `select_exec()` returns its cursor and `select_frame()` its DataFrame. Callers own connection lifetime: the object context closes without committing, while `transaction()` commits or rolls back. `ClusterDatabase` adds label alignment, track loading, time readiness, and map/report adaptation.

`warehouse.ingest_database()` opens an existing database or enters `create_database()` for a new one, then calls `load_gpx_sources()` and `replace_references()` in its write transaction. Creation initializes schema and source metadata in a temporary file, closes the connection, and publishes the file atomically only after success. Publication cannot overwrite an existing or concurrently created destination.

`load_tracks(run_id=None)` returns full-resolution EPSG:4326 geometries and aligned raw epoch arrays. A run ID restricts tracks to that run and adds all layers' predictions and RGB colors. `load_map_jumps(run_id)` and `load_layer_names(run_id)` read markers and layer names separately. `run_clustering()` owns feature preparation; its normalized settings-key lookup returns an existing run ID before loading inputs or making writes. New runs insert their assignments in one transaction.

The Python API uses `run_clustering(database, settings)` and `evaluate(database, settings, eps=(...), seeds=(...), reference=...)`, where `settings` is a validated `ClusteringSettings` model. Evaluation creates validated copies for each threshold and seed without changing the supplied model. QBx uses the requested seeds; DBSCAN uses seed zero. Reference-run settings are independent, and the selected reference layer is excluded from its clustering key.

All maintained run keys include `scale_pairs` and the shared `seed`. Spatiotemporal keys also include `p` and `temporal_mode`; joint runs normalize the inactive `eps_time` to None. Parameter-sweep aggregation keeps joint/separate modes and calibration pair limits in distinct profiles while aggregating across requested seeds. Write operations refresh changed schema view definitions transactionally without replacing tables or stored results.

Stored assignment changes invalidate the complete run and its scores. Updating hand-made labels invalidates only label-based evaluations. A reference run is linked by foreign key; its removal invalidates dependent evaluations, while unrelated runs remain. Evaluation headers disappear when their last score is removed. All these effects participate in the caller's transaction.

`Database.report()` exposes six named views: `dataset_inventory`, `track_catalog`, `time_gaps`, `parameter_sweep`, `selected_models`, and `cluster_composition`. All accept a source filter; the last three also accept an evaluation ID to isolate scope and reference.

## Dashboard

`layouts.py` creates fresh Bokeh models without database queries or callbacks. Its visual groups expose their `layout`, named controls, and data sources. The controller in `dashboard.py` fills these sources and attaches callbacks. `presentation.py` supplies shared headings, statistics, separators, tables, option labels, and plot styling.

### Clustering maps

`map_track_frame()` projects stored geometries to EPSG:3857 for display. `ClusteringLayout` arranges the map view's controls without implementing map callbacks. Each source has its own map models; empty jump tables retain the expected columns. Algorithm tabs and evaluation views use calibrated run records with a stored `scale_pairs` setting and, for spatiotemporal runs, an explicit supported `temporal_mode`.

### Analysis

The full-width evaluation selector sits below the main heading, with parameter settings included in its dropdown entries. A profile filter appears below the epsilon-chart heading when an evaluation contains multiple parameter profiles. Performance-measure controls follow. The epsilon chart uses `parameter_sweep` and fills the available width. Below it, the overlap chart sits beside five per-run score cards showing ARI, AMI, pair F1, pair precision, and pair recall to four decimal places. The run selector fills the score section's width below its explanatory text, is restricted to the active evaluation, and initially uses the representative result from `selected_models`. Selecting another run updates both the scores and the overlap chart, using its stored assignments and this evaluation's reference without rerunning clustering.

### Overview

Supplied hand-made labels take priority in the reference statistics and pie chart. Without them, these elements use the reference run/layer of the evaluation selected in Analysis, initially the first recorded evaluation, and update when that selection changes. Eligible reference groups, outliers, unlabeled tracks, and excluded short tracks have separate slices; excluded tracks contribute only to their excluded slice. Provenance includes supplied label paths/descriptions or saved reference-run settings. Reference-run predictions are not written into `labels`; the catalog describes the imported tracks and supplied labels.

The catalog's status filter sits beside its heading and also determines its CSV download. The catalog is 400 pixels tall, with the gap summary aligned to its top. Duplicate counts describe extra eligible geometry copies. The gap table grows with its entries up to the available panel height, then scrolls; shorter tables are vertically centered.

### Session behavior

Each Bokeh session owns its controls and cached data. Available tabs preload when a source is selected, so tab switches reuse prepared views; the browser lays out each panel when first displayed. Inactive panels are excluded from layout sizing. Source changes replace the tab group and clear its cached data and map references. Database connections are shared during a render/callback and released while the session is idle.

## Validation

```bash
python -m pip install -r extensions/analytics/requirements-dev.txt
python -m pytest -q extensions/analytics/tests tests/test_temporal.py
python -m ruff check extensions/analytics scripts
python -m ruff format --check extensions/analytics scripts
```

Run these commands from the repository root. Tests cover import atomicity and conflict policies, label alignment, settings reuse, reference configuration validation, calendar boundaries, cyclic metrics and centroids, score conventions, dependency invalidation, SQL reports, CLI models, and dashboard source/session behavior.
