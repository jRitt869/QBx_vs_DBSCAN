# Refactoring and Extension record

The internship application is preserved independently in `historical/`; the maintained kernels and renderer are in `track_clusters/`. These tables summarize later maintenance and portfolio additions; the linked analytics README sections describe the current implementation.

## Central source databases

| Area | Specific changes | Details |
| --- | --- | --- |
| Source storage | One SQLite database per source holds file metadata, tracks/points, supplied labels, clustering assignments/colors, and evaluation scores. Feather and loose result exports are no longer runtime inputs. | [Storage](../extensions/analytics/README.md#storage) |
| Incremental ingestion | Single-folder and TOML batch imports share one pipeline. New files append; changed filenames follow skip/replace/error policies; omitted files remain stored. New databases publish atomically after successful ingestion. | [Ingestion contracts](../extensions/analytics/README.md#ingestion-contracts) |
| References and provenance | Keep supplied labels and input checksums. Evaluation uses complete hand-made labels or an explicit reference run/layer; synthetic geographic labels and their fallback machinery were removed. | [Reference runs](../extensions/analytics/README.md#use-a-reference-clustering-run) |
| Reusable results | Normalized settings keys reuse existing runs before feature loading. All layers share one assignments table; scores link to runs. Input/reference changes invalidate dependent results transactionally. | [Storage](../extensions/analytics/README.md#storage) |
| SQL reports | Six active report views supply inventory, catalog, gaps, parameter sweeps, selected models, and overlap. Bound SQLAlchemy Core queries replace scattered SQL; unused views and general query interfaces were removed. | [Storage](../extensions/analytics/README.md#storage) |

## Modular workflow

| Area | Specific changes | Details |
| --- | --- | --- |
| Responsibilities | Separate ingestion/database management, storage/query operations, clustering/evaluation, metrics, and presentation. Internal helpers use underscore names; externally accessed functions have argument documentation. | [Module responsibilities](../extensions/analytics/README.md#module-responsibilities) |
| Entry points | Maintained commands live in `scripts/`; ingestion uses `single`/`batch`. Historical launchers remain separate and load the selected database without creating results in the browser. | [Modules](../extensions/analytics/README.md#module-responsibilities), [ingestion](../extensions/analytics/README.md#ingestion-contracts) |
| Feature preparation | Database loaders return full-resolution tracks and raw timestamps. Clustering owns eligibility filtering and 13-point resampling. Spatiotemporal runs use configurable cyclic metrics and torus subdivision; historical entry points retain the original encoding. | [Storage](../extensions/analytics/README.md#storage), [cyclic metrics](../extensions/analytics/README.md#cyclic-temporal-metrics) |
| Distance calibration | Primary metrics use median nonzero pair distances, one shared spatial scalar, and separate spatial/time scales for joint QBx. Pair sampling is seeded and fitted scales are stored per run. | [Distance calibration](../extensions/analytics/README.md#distance-calibration) |
| Numerical corrections | Correct QBx permutation mapping and temporal metric compatibility, distinct DBSCAN temporal IDs, and all-outlier color handling. Historical benchmark tables remain unverified. | [Comparison methods](../README.md#how-the-comparison-works), [historical parameter tables](../historical/ARCHIVE_README.md) |
| Evaluation | Each evaluation scores one algorithm against a reference baseline. Missing target/reference runs are populated through the shared numerical functions, preserving settings-key reuse. | [Clustering and evaluation](../extensions/analytics/README.md#clustering-and-evaluation) |
| Container startup | `start.py` replaces the shell launcher, imports sources, evaluates supported modes, and starts Bokeh only after successful setup. Untimed sources use the spatial reference configuration. | [Container startup](../extensions/analytics/README.md#container-startup) |

## Input validation

| Area | Specific changes | Details |
| --- | --- | --- |
| CLI and configuration | JSON defines argparse layouts; shared Pydantic models validate CLI/TOML values, paths, source keys, modes, and unique sweep settings. Invalid inputs produce field-specific errors. | [Input validation](../extensions/analytics/README.md#input-validation) |
| GPX and labels | Validate parsing, coordinates, elevations, CSV types, and file/track keys. Keep short tracks and missing timestamps explicit; scoring requires complete eligible reference coverage. | [Input validation](../extensions/analytics/README.md#input-validation) |
| Numerical and temporal inputs | Shared constraints check finite thresholds, factors, seeds, time zones, and derived QBx thresholds. One SQL readiness rule checks eligible timestamps for startup, clustering, and temporal-run triggers. | [Input validation](../extensions/analytics/README.md#input-validation) |
| Database writes | Keys, constraints, triggers, integrity checks, and caller-owned transactions protect stored records. Input signatures prevent committing results after concurrent source changes. | [Input validation](../extensions/analytics/README.md#input-validation), [storage](../extensions/analytics/README.md#storage) |
| Regression checks | Tests cover numerical parity, input rejection, transactional invalidation, source isolation, and map controls. Pytest, Ruff, pinned dependencies, and CI support repeatable maintenance checks. | [Validation commands](../extensions/analytics/README.md#validation) |

## Dashboard and visualization

| Area | Specific changes | Details |
| --- | --- | --- |
| Clustering maps | Source-specific DBSCAN/QBx tabs expose stored runs, layers, cluster/outlier counts, and the original map controls. Display projection uses EPSG:3857; empty gaps and fixed-size markers are supported. | [Clustering maps](../extensions/analytics/README.md#clustering-maps) |
| Overview | Add raw/eligible track, point, and file counts; reference statistics and pie chart; a filterable catalog with CSV download; duplicate counts and timestamp gaps. | [Overview](../extensions/analytics/README.md#overview) |
| Reference display | Show reference provenance and saved settings. Without supplied labels, Overview follows the selected evaluation's reference run/layer for both its statistics and pie chart. | [Overview](../extensions/analytics/README.md#overview) |
| Analysis | Show epsilon performance beside reference/prediction overlap, with metric radio buttons and a run selector for ARI, AMI, F1, pair precision, and pair recall. Unevaluated sources display a disclaimer. | [Analysis](../extensions/analytics/README.md#analysis) |
| Visual structure | Keep query/callback execution in the session controller, passive visual groups in `layouts.py`, and shared components in `presentation.py`. | [Dashboard](../extensions/analytics/README.md#dashboard) |
| Session behavior | Preload available tabs, preserve independent session state, replace stale controls on source changes, and prevent tall tabs from stretching shorter ones. Release database connections while idle. | [Session behavior](../extensions/analytics/README.md#session-behavior) |
