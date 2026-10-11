import json
from datetime import UTC
from pathlib import Path

import gpxpy
import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer

from track_clusters.cli import main
from track_clusters.data import _transform_time, get_geo_data, load_directory
from track_clusters.pipeline import gen_clusters


def test_load_resampling_and_timeskips(track_directory: Path) -> None:
    tracks, jumps = load_directory(track_directory)
    assert len(tracks) == 25
    assert tracks.iloc[0]["13_pts"].shape == (13, 3)
    assert len(jumps) == 1
    assert jumps.iloc[0]["jump_minutes"] == 42
    assert jumps.iloc[0]["jump_name"] == tracks.iloc[0]["name"]
    transformed = _transform_time(tracks, UTC)
    np.testing.assert_allclose(transformed.iloc[0]["13_pts"][:, 2], 0)
    np.testing.assert_allclose(np.linalg.norm(transformed.iloc[0]["times"], axis=1), 1)
    assert "times" not in tracks


@pytest.mark.parametrize("qbx", [False, True])
def test_pipeline_feather_and_projection(track_directory: Path, tmp_path: Path, qbx: bool) -> None:
    output = tmp_path / "output"
    tracks, jumps = gen_clusters(
        track_directory, qbx, 0.01 if qbx else 0.5, 0.1, 5, 2, 0, output_dir=output, time_zone=UTC
    )
    restored = pd.read_feather(output / "raw_data")
    assert len(restored) == 25
    assert restored["name"].tolist() == tracks["name"].tolist()
    assert len(pd.read_feather(output / "raw_data_jumps")) == len(jumps) == 1
    assert restored["clusters0"].nunique() == 3
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert metadata["map_crs"] == "EPSG:3857"
    assert metadata["timezone"] == "UTC"
    x, y = Transformer.from_crs(4326, 3857, always_xy=True).transform(13.35, 52.50)
    np.testing.assert_allclose([restored.iloc[0]["x"][0], restored.iloc[0]["y"][0]], [x, y])


def test_no_timeskips(track_directory: Path, tmp_path: Path) -> None:
    source = track_directory / "tracks.gpx"
    gpx = gpxpy.parse(source.read_text())
    gpx.tracks.pop(0)
    source.write_text(gpx.to_xml(), encoding="utf-8")
    _, jumps = gen_clusters(
        track_directory, False, 0.5, 0.1, 5, 2, 0, output_dir=tmp_path / "output"
    )
    assert jumps.empty
    assert set(jumps.columns) == {"jump_name", "jump_hours", "jump_minutes", "x", "y"}
    assert pd.read_feather(tmp_path / "output" / "raw_data_jumps").empty


def test_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No GPX files"):
        gen_clusters(tmp_path, False, 0.5, 0.1, 5, 2, 0)


def _write_track(
    path: Path, count: int = 5, missing: bool = False, backwards: bool = False
) -> None:
    points = []
    for i in range(count):
        minute = count - i if backwards else i
        timestamp = "" if missing else f"<time>2024-01-08T08:{minute:02d}:00Z</time>"
        points.append(f'<trkpt lat="52.5" lon="13.35">{timestamp}</trkpt>')
    path.write_text(
        '<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">'
        "<trk><trkseg>" + "".join(points) + "</trkseg></trk></gpx>",
        encoding="utf-8",
    )


@pytest.mark.parametrize("error", ["missing", "backwards"])
def test_invalid_timestamps_report_context(tmp_path: Path, error: str) -> None:
    _write_track(
        tmp_path / "invalid.gpx", missing=error == "missing", backwards=error == "backwards"
    )
    with pytest.raises(ValueError, match="invalid.gpx, track 0"):
        get_geo_data(tmp_path, "invalid.gpx")


def test_short_tracks(tmp_path: Path) -> None:
    _write_track(tmp_path / "short.gpx", count=4)
    with pytest.raises(ValueError, match="No usable tracks"):
        load_directory(tmp_path)


def test_sorted_and_uppercase_files(tmp_path: Path) -> None:
    _write_track(tmp_path / "z.gpx")
    _write_track(tmp_path / "a.GPX")
    tracks, _ = load_directory(tmp_path)
    assert tracks["name"].tolist() == ["a_0", "z_0"]


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--seed", "1.5", ".", "1", "1"],
        [".", "nan", "1"],
        [".", "1", "0"],
        ["--factor", "0.5", ".", "1", "1"],
        ["--min_s", "0", ".", "1", "1"],
        ["--seed", "-1", ".", "1", "1"],
        ["--timezone", "invalid/timezone", ".", "1", "1"],
    ],
)
def test_cli_invalid_arguments(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code != 0
