PRAGMA foreign_keys = ON;

CREATE TABLE build_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE datasets (
    dataset_id INTEGER PRIMARY KEY,
    dataset_key TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    label_kind TEXT NOT NULL CHECK (label_kind IN ('author', 'unlabeled')),
    label_description TEXT NOT NULL
);

CREATE TABLE source_files (
    file_id INTEGER PRIMARY KEY,
    dataset_id INTEGER NOT NULL REFERENCES datasets,
    file_name TEXT NOT NULL,
    source_path TEXT NOT NULL,
    sha256 TEXT NOT NULL CHECK (length(sha256) = 64),
    byte_count INTEGER NOT NULL CHECK (byte_count >= 0),
    UNIQUE (dataset_id, file_name)
);

CREATE TABLE tracks (
    track_id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES source_files,
    source_track_id INTEGER NOT NULL CHECK (source_track_id >= 0),
    track_index INTEGER NOT NULL CHECK (track_index >= 0),
    name TEXT,
    segment_count INTEGER NOT NULL CHECK (segment_count >= 0),
    point_count INTEGER NOT NULL CHECK (point_count >= 0),
    timestamped_points INTEGER NOT NULL CHECK (timestamped_points BETWEEN 0 AND point_count),
    eligible INTEGER GENERATED ALWAYS AS (point_count >= 5) STORED,
    geometry_sha256 TEXT NOT NULL CHECK (length(geometry_sha256) = 64),
    stationary INTEGER NOT NULL CHECK (stationary IN (0, 1)),
    min_lon REAL, min_lat REAL, max_lon REAL, max_lat REAL,
    UNIQUE (file_id, source_track_id),
    UNIQUE (file_id, track_index)
);

CREATE TABLE points (
    track_id INTEGER NOT NULL REFERENCES tracks,
    point_index INTEGER NOT NULL CHECK (point_index >= 0),
    segment_index INTEGER NOT NULL CHECK (segment_index >= 0),
    longitude REAL NOT NULL CHECK (longitude BETWEEN -180 AND 180),
    latitude REAL NOT NULL CHECK (latitude BETWEEN -90 AND 90),
    elevation REAL,
    recorded_at_utc TEXT,
    recorded_at_epoch REAL,
    time_was_naive INTEGER NOT NULL CHECK (time_was_naive IN (0, 1)),
    PRIMARY KEY (track_id, point_index)
);

CREATE TABLE labels (
    track_id INTEGER PRIMARY KEY REFERENCES tracks,
    reference_cluster INTEGER CHECK (reference_cluster >= -1)
);

CREATE TABLE artifacts (
    artifact_id INTEGER PRIMARY KEY,
    dataset_id INTEGER NOT NULL REFERENCES datasets,
    role TEXT NOT NULL,
    source_path TEXT NOT NULL,
    sha256 TEXT NOT NULL CHECK (length(sha256) = 64),
    UNIQUE (dataset_id, role)
);

CREATE TABLE evaluations (
    evaluation_id INTEGER PRIMARY KEY,
    dataset_id INTEGER NOT NULL REFERENCES datasets,
    scope TEXT NOT NULL DEFAULT 'spatial' CHECK (scope IN ('spatial', 'spatiotemporal')),
    algorithm TEXT NOT NULL CHECK (algorithm IN ('dbscan', 'qbx')),
    reference_run_id INTEGER REFERENCES runs ON DELETE CASCADE,
    reference_layer INTEGER CHECK (reference_layer BETWEEN 0 AND 3),
    manifest_json TEXT NOT NULL,
    selection_note TEXT NOT NULL,
    noise_policy TEXT NOT NULL,
    CHECK ((reference_run_id IS NULL AND reference_layer IS NULL)
        OR (reference_run_id IS NOT NULL AND reference_layer IS NOT NULL))
);

CREATE TABLE runs (
    run_id INTEGER PRIMARY KEY,
    algorithm TEXT NOT NULL CHECK (algorithm IN ('dbscan', 'qbx')),
    feature_mode TEXT NOT NULL CHECK (feature_mode IN ('spatial', 'spatiotemporal')),
    config_key TEXT NOT NULL UNIQUE,
    settings_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    clustering_seconds REAL NOT NULL CHECK (clustering_seconds >= 0),
    provenance TEXT NOT NULL
);

CREATE TABLE assignments (
    run_id INTEGER NOT NULL REFERENCES runs ON DELETE CASCADE,
    track_id INTEGER NOT NULL REFERENCES tracks,
    layer INTEGER NOT NULL CHECK (layer BETWEEN 0 AND 3),
    layer_name TEXT NOT NULL,
    predicted_cluster INTEGER NOT NULL CHECK (predicted_cluster >= -1),
    color TEXT NOT NULL,
    PRIMARY KEY (run_id, track_id, layer)
);

CREATE TABLE evaluation_results (
    evaluation_result_id INTEGER PRIMARY KEY,
    evaluation_id INTEGER NOT NULL REFERENCES evaluations ON DELETE CASCADE,
    run_id INTEGER NOT NULL REFERENCES runs ON DELETE CASCADE,
    layer INTEGER NOT NULL CHECK (layer BETWEEN 0 AND 3),
    adjusted_rand REAL NOT NULL,
    adjusted_mutual_information REAL NOT NULL,
    pair_tp INTEGER NOT NULL CHECK (pair_tp >= 0),
    pair_fp INTEGER NOT NULL CHECK (pair_fp >= 0),
    pair_fn INTEGER NOT NULL CHECK (pair_fn >= 0),
    pair_tn INTEGER NOT NULL CHECK (pair_tn >= 0),
    pair_precision REAL NOT NULL, pair_recall REAL NOT NULL, pair_f1 REAL NOT NULL,
    pair_false_positive_rate REAL, pair_false_negative_rate REAL,
    outlier_detection_precision REAL, outlier_detection_recall REAL,
    UNIQUE (evaluation_id, run_id, layer)
);

CREATE INDEX tracks_geometry ON tracks (geometry_sha256);
CREATE INDEX assignments_track ON assignments (track_id, run_id);
CREATE INDEX runs_algorithm ON runs (algorithm, feature_mode);
CREATE INDEX evaluation_results_run ON evaluation_results (run_id, layer);
CREATE UNIQUE INDEX evaluations_labels ON evaluations (dataset_id, scope, algorithm)
WHERE reference_run_id IS NULL;
CREATE UNIQUE INDEX evaluations_reference_run ON evaluations (
    dataset_id, scope, algorithm, reference_run_id, reference_layer
) WHERE reference_run_id IS NOT NULL;

CREATE TRIGGER evaluation_algorithm_on_insert
BEFORE INSERT ON evaluation_results
WHEN (SELECT algorithm FROM runs WHERE run_id = NEW.run_id)
  <> (SELECT algorithm FROM evaluations WHERE evaluation_id = NEW.evaluation_id)
BEGIN
    SELECT RAISE(ABORT, 'An evaluation contains exactly one algorithm');
END;

CREATE TRIGGER evaluation_algorithm_on_update
BEFORE UPDATE OF run_id, evaluation_id ON evaluation_results
WHEN (SELECT algorithm FROM runs WHERE run_id = NEW.run_id)
  <> (SELECT algorithm FROM evaluations WHERE evaluation_id = NEW.evaluation_id)
BEGIN
    SELECT RAISE(ABORT, 'An evaluation contains exactly one algorithm');
END;

-- Evaluation headers disappear with their last score. Stored assignments are
-- immutable: deleting or changing one invalidates its complete clustering run.
-- Initial parent/child insertion remains atomic in the processing transaction.
CREATE TRIGGER delete_empty_evaluation
AFTER DELETE ON evaluation_results
WHEN NOT EXISTS (
    SELECT 1 FROM evaluation_results WHERE evaluation_id = OLD.evaluation_id
)
BEGIN
    DELETE FROM evaluations WHERE evaluation_id = OLD.evaluation_id;
END;

CREATE TRIGGER invalidate_run_after_assignment_delete
AFTER DELETE ON assignments
WHEN EXISTS (
    SELECT 1 FROM runs WHERE run_id = OLD.run_id
)
BEGIN
    DELETE FROM runs WHERE run_id = OLD.run_id;
END;

CREATE TRIGGER invalidate_run_after_assignment_update
AFTER UPDATE ON assignments
WHEN OLD.run_id IS NOT NEW.run_id
  OR OLD.track_id IS NOT NEW.track_id
  OR OLD.layer IS NOT NEW.layer
  OR OLD.layer_name IS NOT NEW.layer_name
  OR OLD.predicted_cluster IS NOT NEW.predicted_cluster
  OR OLD.color IS NOT NEW.color
BEGIN
    DELETE FROM runs WHERE run_id IN (OLD.run_id, NEW.run_id);
END;

CREATE VIEW v_run_layers AS
SELECT run_id, layer, layer_name, COUNT(*) AS tracks,
       COUNT(DISTINCT CASE WHEN predicted_cluster >= 0 THEN predicted_cluster END) AS clusters,
       SUM(predicted_cluster = -1) AS outliers,
       AVG(predicted_cluster = -1) AS outlier_fraction
FROM assignments GROUP BY run_id, layer, layer_name;

-- Configuration, prediction statistics, and scores are joined only when reading.
-- Layer indices are zero-based in both assignments and evaluation results.
CREATE VIEW v_evaluation_runs AS
SELECT d.dataset_key AS dataset, e.scope, e.reference_run_id, e.reference_layer, s.*,
       r.algorithm, r.feature_mode, r.clustering_seconds,
       json_extract(r.settings_json, '$.eps') AS eps,
       json_extract(r.settings_json, '$.eps_time') AS eps_time,
       json_extract(r.settings_json, '$.seed') AS seed,
       json_extract(r.settings_json, '$.factor') AS factor,
       json_extract(r.settings_json, '$.min_samples') AS min_samples,
       json_extract(r.settings_json, '$.timezone') AS timezone,
       json_extract(r.settings_json, '$.temporal_mode') AS temporal_mode,
       json_extract(r.settings_json, '$.p') AS p,
       json_extract(r.settings_json, '$.scale_pairs') AS scale_pairs,
       r.algorithm || '_run' || r.run_id || '_layer' || s.layer AS run_key,
       l.tracks, l.clusters, l.outliers, l.outlier_fraction
FROM evaluation_results s JOIN runs r USING (run_id)
JOIN evaluations e USING (evaluation_id) JOIN datasets d USING (dataset_id)
JOIN v_run_layers l ON l.run_id = s.run_id AND l.layer = s.layer
WHERE json_extract(r.settings_json, '$.scale_pairs') IS NOT NULL
  AND (r.feature_mode = 'spatial'
       OR json_extract(r.settings_json, '$.temporal_mode') IN ('separate', 'joint'));

CREATE VIEW v_track_catalog AS
SELECT d.dataset_id, d.dataset_key AS dataset, d.label_kind,
       f.file_name, t.*,
       l.reference_cluster AS provided_reference, l.reference_cluster,
       CASE WHEN l.reference_cluster IS NOT NULL THEN 'provided'
            ELSE 'unlabeled' END AS reference_origin,
       CASE WHEN t.eligible = 0 THEN 'Excluded: fewer than five points'
            WHEN l.reference_cluster IS NULL AND d.label_kind = 'unlabeled' THEN 'Eligible: not labeled'
            WHEN l.reference_cluster IS NULL THEN 'Missing eligible label'
            WHEN l.reference_cluster = -1 THEN 'Eligible: reference outlier'
            ELSE 'Eligible: reference group' END AS status,
       CASE WHEN t.eligible = 1 THEN ROW_NUMBER() OVER (
           PARTITION BY d.dataset_id, t.eligible, t.geometry_sha256
           ORDER BY f.file_name, t.track_index
       ) - 1 ELSE 0 END AS geometry_occurrence
FROM tracks t
JOIN source_files f USING (file_id)
JOIN datasets d USING (dataset_id)
LEFT JOIN labels l USING (track_id);

CREATE VIEW v_evaluation_references AS
SELECT e.evaluation_id, t.track_id,
       CASE WHEN e.reference_run_id IS NULL THEN t.reference_cluster
            ELSE a.predicted_cluster END AS reference_cluster
FROM evaluations e JOIN v_track_catalog t USING (dataset_id)
LEFT JOIN assignments a ON a.run_id = e.reference_run_id
    AND a.layer = e.reference_layer AND a.track_id = t.track_id
WHERE t.eligible = 1;

-- Keep untimed/invalid source points available for spatial work. Temporal
-- readiness depends on ordered points, so it cannot be a row-level CHECK.
CREATE VIEW v_track_time_quality AS
WITH ordered AS (
    SELECT track_id, recorded_at_epoch,
           LAG(recorded_at_epoch) OVER (
               PARTITION BY track_id ORDER BY point_index
           ) AS previous_epoch
    FROM points
), quality AS (
    SELECT track_id, COUNT(*) AS point_count,
           SUM(recorded_at_epoch IS NULL) AS missing_times,
           SUM(recorded_at_epoch IS NOT NULL AND NOT (
               recorded_at_epoch BETWEEN -1.7976931348623157e308 AND 1.7976931348623157e308
           )) AS nonfinite_times,
           SUM(COALESCE(recorded_at_epoch < 0, 0)) AS negative_times,
           SUM(COALESCE(recorded_at_epoch < previous_epoch, 0)) AS decreasing_times
    FROM ordered GROUP BY track_id
)
SELECT t.track_id, f.file_name, t.source_track_id, t.eligible,
       COALESCE(q.point_count, 0) AS point_count,
       COALESCE(q.missing_times, 0) AS missing_times,
       COALESCE(q.nonfinite_times, 0) AS nonfinite_times,
       COALESCE(q.negative_times, 0) AS negative_times,
       COALESCE(q.decreasing_times, 0) AS decreasing_times,
       CASE WHEN q.point_count > 0 AND q.missing_times = 0
                 AND q.nonfinite_times = 0 AND q.negative_times = 0
                 AND q.decreasing_times = 0 THEN 1 ELSE 0 END AS temporal_valid
FROM tracks t JOIN source_files f USING (file_id)
LEFT JOIN quality q USING (track_id);

-- Direct writes must obey the same temporal preconditions as the pipeline.
CREATE TRIGGER require_temporal_inputs_on_run_insert
BEFORE INSERT ON runs
WHEN NEW.feature_mode = 'spatiotemporal' AND COALESCE((
    SELECT MIN(temporal_valid) FROM v_track_time_quality WHERE eligible = 1
), 0) = 0
BEGIN
    SELECT RAISE(ABORT, 'Temporal clustering requires eligible tracks with complete, finite, nonnegative, chronological timestamps');
END;

CREATE TRIGGER require_temporal_inputs_on_run_update
BEFORE UPDATE OF feature_mode ON runs
WHEN NEW.feature_mode = 'spatiotemporal' AND COALESCE((
    SELECT MIN(temporal_valid) FROM v_track_time_quality WHERE eligible = 1
), 0) = 0
BEGIN
    SELECT RAISE(ABORT, 'Temporal clustering requires eligible tracks with complete, finite, nonnegative, chronological timestamps');
END;

CREATE VIEW v_dataset_quality AS
SELECT dataset, label_kind,
       COUNT(DISTINCT file_id) AS files_with_tracks,
       COUNT(*) AS raw_tracks,
       SUM(eligible) AS eligible_tracks,
       SUM(1 - eligible) AS excluded_tracks,
       SUM(reference_cluster IS NOT NULL) AS labeled_tracks,
       SUM(provided_reference IS NOT NULL) AS provided_labeled_tracks,
       SUM(eligible = 1 AND reference_cluster IS NULL AND label_kind <> 'unlabeled') AS missing_eligible_labels,
       SUM(eligible = 1 AND reference_cluster IS NULL) AS unlabeled_eligible_tracks,
       SUM(eligible = 0 AND reference_cluster IS NOT NULL) AS excluded_labeled_tracks,
       COUNT(DISTINCT CASE WHEN eligible = 1 AND reference_cluster >= 0
                          THEN reference_cluster END) AS reference_groups,
       SUM(CASE WHEN eligible = 1 AND reference_cluster = -1 THEN 1 ELSE 0 END) AS reference_outliers,
       SUM(point_count) AS points,
       SUM(timestamped_points) AS timestamped_points,
       SUM(eligible = 1 AND timestamped_points = point_count) AS timed_eligible_tracks,
       SUM(eligible = 1 AND geometry_occurrence > 0) AS duplicate_eligible_tracks,
       SUM(eligible = 1 AND stationary = 1) AS stationary_eligible_tracks
FROM v_track_catalog GROUP BY dataset, label_kind;

-- Variance is the sample variance (ddof=1) across scored seeds.
-- Single deterministic runs have a displayed spread of zero.
CREATE VIEW v_parameter_summary AS
SELECT r.dataset, r.evaluation_id, r.scope, r.algorithm, r.eps,
       r.factor, r.layer, r.min_samples, r.eps_time, r.timezone, r.temporal_mode, r.p,
       r.scale_pairs, COUNT(*) AS runs,
       AVG(r.adjusted_rand) AS adjusted_rand_mean,
       CASE WHEN COUNT(*) > 1 THEN MAX(0.0,
           (SUM(r.adjusted_rand * r.adjusted_rand)
            - SUM(r.adjusted_rand) * AVG(r.adjusted_rand)) / (COUNT(*) - 1)
       ) ELSE 0.0 END AS adjusted_rand_variance,
       MIN(r.adjusted_rand) AS adjusted_rand_min, MAX(r.adjusted_rand) AS adjusted_rand_max,
       AVG(r.adjusted_mutual_information) AS adjusted_mutual_information_mean,
       AVG(r.pair_f1) AS pair_f1_mean,
       AVG(r.pair_precision) AS pair_precision_mean,
       AVG(r.pair_recall) AS pair_recall_mean,
       AVG(r.pair_false_positive_rate) AS pair_false_positive_rate_mean,
       AVG(r.pair_false_negative_rate) AS pair_false_negative_rate_mean,
       AVG(r.outlier_fraction) AS outlier_fraction_mean,
       AVG(r.clusters) AS clusters_mean, AVG(r.clustering_seconds) AS clustering_seconds_mean
FROM v_evaluation_runs r
GROUP BY r.dataset, r.evaluation_id, r.algorithm, r.eps, r.factor, r.layer,
         r.min_samples, r.eps_time, r.timezone, r.temporal_mode, r.p, r.scale_pairs;

CREATE VIEW v_selected_parameters AS
WITH ranked AS (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY evaluation_id, algorithm ORDER BY adjusted_rand_mean DESC, eps
    ) AS parameter_position FROM v_parameter_summary
)
SELECT * FROM ranked WHERE parameter_position = 1;

CREATE VIEW v_selected_runs AS
WITH ranked AS (
    SELECT p.dataset, p.evaluation_id, p.algorithm, r.evaluation_result_id,
           r.run_id, r.layer, r.run_key, r.seed, r.eps,
           ROW_NUMBER() OVER (
               PARTITION BY r.evaluation_id, r.algorithm ORDER BY r.evaluation_result_id
           ) AS representative_position
    FROM v_evaluation_runs r JOIN v_selected_parameters p
      ON r.evaluation_id = p.evaluation_id AND r.algorithm = p.algorithm AND r.eps = p.eps
      AND r.factor IS p.factor AND r.layer = p.layer AND r.min_samples IS p.min_samples
      AND r.eps_time IS p.eps_time AND r.timezone IS p.timezone
      AND r.temporal_mode IS p.temporal_mode AND r.p IS p.p
      AND r.scale_pairs IS p.scale_pairs
)
SELECT * FROM ranked WHERE representative_position = 1;

CREATE VIEW v_time_gaps AS
WITH consecutive AS (
    SELECT *, LAG(recorded_at_utc) OVER (
        PARTITION BY track_id ORDER BY point_index
    ) AS previous_time,
    LAG(recorded_at_epoch) OVER (
        PARTITION BY track_id ORDER BY point_index
    ) AS previous_epoch
    FROM points
)
SELECT t.dataset, t.file_name, t.source_track_id, p.point_index, p.segment_index,
       p.previous_time, p.recorded_at_utc,
       p.recorded_at_epoch - p.previous_epoch AS gap_seconds
FROM consecutive p JOIN v_track_catalog t USING (track_id)
WHERE t.eligible = 1
  AND p.recorded_at_epoch - p.previous_epoch > 1800;


-- Dashboard report views. Apply selection filters and ordering when reading.

CREATE VIEW cluster_composition AS
-- Numeric reference/predicted IDs are arbitrary: this is an overlap table,
-- not a classification confusion matrix. Noise is displayed separately as -1.
SELECT t.dataset, r.evaluation_id, r.evaluation_result_id, r.algorithm,
       r.run_id, r.layer, r.run_key,
       truth.reference_cluster, a.predicted_cluster, COUNT(*) AS tracks
FROM assignments a JOIN v_evaluation_runs r ON r.run_id = a.run_id AND r.layer = a.layer
JOIN v_track_catalog t USING (track_id)
JOIN v_evaluation_references truth ON truth.evaluation_id = r.evaluation_id
    AND truth.track_id = t.track_id
JOIN v_selected_runs s USING (evaluation_result_id)
GROUP BY t.dataset, r.evaluation_result_id, truth.reference_cluster, a.predicted_cluster;

CREATE VIEW dataset_inventory AS
-- Keep empty GPX files in the source inventory as well.
SELECT d.dataset_key AS dataset, d.name, d.label_kind, d.label_description,
       (SELECT COUNT(*) FROM source_files f WHERE f.dataset_id = d.dataset_id) AS files,
       q.raw_tracks, q.eligible_tracks, q.excluded_tracks, q.labeled_tracks,
       q.provided_labeled_tracks,
       q.excluded_labeled_tracks, q.reference_groups, q.reference_outliers,
       q.points, q.timestamped_points, q.timed_eligible_tracks,
       q.duplicate_eligible_tracks, q.stationary_eligible_tracks, q.missing_eligible_labels,
       q.unlabeled_eligible_tracks,
       (SELECT COUNT(*) FROM evaluation_results r JOIN evaluations e USING (evaluation_id)
        WHERE e.dataset_id = d.dataset_id) AS evaluation_runs
FROM datasets d LEFT JOIN v_dataset_quality q ON q.dataset = d.dataset_key;

CREATE VIEW parameter_sweep AS
SELECT * FROM v_parameter_summary;

CREATE VIEW selected_models AS
-- Rank thresholds by mean ARI across all requested seeds; break ties by epsilon.
SELECT p.*, s.run_id AS representative_run_id, s.layer AS representative_layer,
       s.evaluation_result_id AS representative_result_id, s.run_key AS representative_run,
       s.seed AS representative_seed
FROM v_selected_parameters p
JOIN v_selected_runs s ON s.evaluation_id = p.evaluation_id AND s.algorithm = p.algorithm;

CREATE VIEW time_gaps AS
-- Only adjacent recorded points with known timestamps; no interpolation.
-- Segment boundaries remain in points, and gaps follow joined-track semantics.
SELECT * FROM v_time_gaps;

CREATE VIEW track_catalog AS
SELECT dataset, file_name, source_track_id, track_id, track_index, name,
       segment_count, point_count, eligible, status, reference_cluster,
       provided_reference, reference_origin,
       timestamped_points, stationary, geometry_occurrence, geometry_sha256,
       min_lon, min_lat, max_lon, max_lat
FROM v_track_catalog;
