# Spatial evaluation

Scored **267 tracks** against 287 label rows; 20 labeled short tracks were excluded.
Reference labels contain 5 clusters and 86 outliers.
Found 25 exact duplicate geometries; all records are retained.

Timestamps are ignored. Both methods use the same 13-point spatial resampling; QBx still uses coordinate degrees and DBSCAN standardized flattened coordinates.

Every -1 track is treated as its own singleton for ARI, AMI, and pairwise metrics. Pair false-positive rate is FP/(FP+TN); false-negative rate is FN/(TP+FN). Undefined rates are blank. Precision/recall/F1 use 0 when their denominator is zero.

Thresholds are selected on this dataset for exploration. These are not held-out performance estimates, and the two epsilon scales are different.

| Algorithm | Epsilon | Mean ARI | ARI std | Mean pair F1 |
| --- | --- | --- | --- | --- |
| dbscan | 1 | 0.9148 | 0.0000 | 0.9328 |
| qbx | 0.5 | 0.8223 | 0.0598 | 0.8585 |

The comparison plot uses the first requested seed at the selected QBx threshold, rather than the most favorable seed. See metrics.csv for all runs, summary.csv for seed means/std, assignments.csv for track-level results, and manifest.json for checksums/settings.
