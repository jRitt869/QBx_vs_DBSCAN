# Geographic labels for dump

These are **provisional agent-created geographic annotations**, created at the project author's request to group tracks by area. They are not the author's original hand-labeled ground truth and do not use DBSCAN or QuickBundlesX predictions.

The 74 GPX files contain 151 tracks: 125 eligible tracks and 26 tracks below the existing five-point minimum. 111 eligible tracks occupy 21 defined areas; 14 remain unassigned.

## Annotation rule

Areas are explicit longitude/latitude rectangles reviewed from the trace inventory. Every coordinate of a track must lie inside exactly one area. Isolated areas without a second eligible track, cross-region traces, and ambiguous overlaps are unassigned (-1). The same -1 is recorded for short tracks, whose status is excluded_short_track and which are omitted by the evaluator. IDs use the GPX <number> field, not list position.

Geographic area means proximity within the declared box, not the same route. Area sizes vary with the observed geographic groups; this is a documented annotation convention rather than a unique clustering truth. Regions should be reviewed before using these labels for claims about algorithm quality.

## Review artifacts

- dump_labels.csv: compatible file_name,id,cluster label table.
- dump_labels.png: all traces colored by area, with area rectangles and unassigned tracks in dashed gray.
- dump_labels_details.png: a zoomed panel for every assigned area.
- dump_label_regions.json: inspectable region definitions, independent of the tested algorithms.
- dump_label_audit.csv: per-track bounds, assignment/exclusion status, and stationary flags.
- dump_duplicates.csv: exact geometry duplicates retained in the dataset.
- dump_label_validation.json: structural checks and source/label checksums.

There are 2 stationary eligible traces and 8 exact duplicate geometries. They are retained and flagged, so duplicated or point-like records can increase apparent cluster support. Long international/cross-region traces are not forced into a nearby area.

## Area inventory

| Label | Geographic center hint | Eligible tracks | Bounds [lon min, lat min, lon max, lat max] |
| --- | --- | --- | --- |
| 0 | 52.0N / 8.5E | 17 | [8.25, 51.8, 8.8, 52.2] |
| 1 | 48.0N / 7.9E | 6 | [7.75, 47.94, 8.05, 48.1] |
| 2 | 48.0N / 8.7E | 2 | [8.6, 48, 8.8, 48.1] |
| 3 | 53.9N / 14.1E | 5 | [14, 53.88, 14.1, 53.96] |
| 4 | 48.8N / 9.3E | 6 | [9.1, 48.5, 9.5, 49] |
| 5 | 52.5N / 7.4E | 2 | [7.3, 52.4, 7.45, 52.5] |
| 6 | 50.6N / 14.3E | 3 | [14, 50.2, 14.55, 50.95] |
| 7 | 48.1N / 11.6E | 5 | [11.2, 47.9, 12, 48.2] |
| 8 | 47.7N / 9.9E | 4 | [9.7, 47.5, 10.05, 47.82] |
| 9 | 47.7N / 12.4E | 4 | [12.4, 47.68, 12.5, 47.81] |
| 10 | 50.2N / 7.5E | 7 | [7.42, 50.14, 7.57, 50.2] |
| 11 | 50.9N / 11.9E | 2 | [11.83, 50.88, 11.9, 50.94] |
| 12 | 50.7N / 7.2E | 2 | [7.12, 50.62, 7.24, 50.68] |
| 13 | 50.3N / 9.1E | 6 | [9.07, 50.17, 9.2, 50.37] |
| 14 | 49.2N / 9.5E | 2 | [9.48, 49.12, 9.6, 49.25] |
| 15 | 51.3N / 7.2E | 17 | [7.03, 51.2, 7.28, 51.29] |
| 16 | 47.7N / 9.1E | 3 | [9, 47.73, 9.11, 47.77] |
| 17 | 52.7N / 7.9E | 9 | [7.87, 52.71, 7.98, 52.77] |
| 18 | 53.7N / 10.9E | 5 | [10.8, 53.63, 10.96, 53.72] |
| 19 | 53.3N / 10.4E | 2 | [10.4, 53.2, 10.45, 53.31] |
| 20 | 51.9N / 10.2E | 2 | [10.12, 51.83, 10.2, 52.04] |

Regenerate deterministically from the repository root:

```bash
python -m extensions.spatial_analysis.annotate_dump demo_data/dump --regions demo_data/dump/dump_label_regions.json
```

Structural containment, inventory coverage, and colored plots make the annotations auditable. They do not replace an independent human assessment of the intended geographic categories.
