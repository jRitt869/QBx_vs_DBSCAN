from datetime import UTC
from pathlib import Path

import gpxpy
import numpy as np
import pytest

from historical.data import _transform_time, get_geo_data, load_directory


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


def test_no_timeskips(track_directory: Path) -> None:
    source = track_directory / "tracks.gpx"
    gpx = gpxpy.parse(source.read_text())
    gpx.tracks.pop(0)
    source.write_text(gpx.to_xml(), encoding="utf-8")
    _, jumps = load_directory(track_directory)
    assert jumps.empty
    assert set(jumps.columns) == {"jump_name", "jump_hours", "jump_minutes", "geometry"}
    assert jumps.crs.to_epsg() == 4326


def test_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No GPX files"):
        load_directory(tmp_path)


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
