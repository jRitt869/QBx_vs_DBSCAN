# Public track samples

| Source | Tracks | Hand-made labels | Use |
| --- | --- | --- | --- |
| `003/` | 466 raw tracks, 267 eligible | `003_labels.csv` | Spatial evaluation; timestamps are absent |
| `dump/` | 151 raw tracks, 125 eligible | None | Timestamped map demo and explicit reference-run comparisons |

The author confirmed authorship and permission to publish the samples. Original GPX files and the author's label assignments are preserved. CSV IDs are zero-based track positions within each GPX file, counting excluded short tracks too.

The dump contains 74 GPX files. All its points have timestamps; 26 tracks are below the five-point minimum and 31 gaps exceed 30 minutes among eligible tracks. Its former generated geographic labels and review artifacts are preserved locally in `local_archive/` and are excluded from the active pipeline.

```bash
python -m scripts.load_data batch
python -m scripts.evaluate outputs/003.sqlite
python -m scripts.evaluate outputs/dump.sqlite --reference-config scripts/config/reference_qbx.toml
```

Reference-run comparisons measure agreement with configured predictions, rather than accuracy against hand-made ground truth.
