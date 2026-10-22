"""Ingest track sources, validate references, and manage source database files."""

import hashlib
import json
import math
import os
import sqlite3
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ConfigDict, ValidationError, validate_call

from scripts.config.validate import (
    ConflictPolicy,
    IngestionInput,
    SourceConfiguration,
    validation_message,
)

from .storage import (
    OUTPUT_DIRECTORY,
    PACKAGE,
    ClusterDatabase,
    Database,
    sha256,
    source_path,
)


def discover_databases(directory: Path = OUTPUT_DIRECTORY) -> dict[str, Path]:
    """Find source databases and reject duplicate keys or multiple sources per file.

    Args:
        directory: Folder to inspect nonrecursively for SQLite database files.

    Returns:
        Source keys mapped to database paths, or an empty mapping for a missing folder.
    """
    sources = {}
    directory = Path(directory)
    paths = sorted(directory.iterdir()) if directory.is_dir() else []
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in {".sqlite", ".sqlite3", ".db"}:
            continue
        with Database(path) as database:
            rows = database.select_exec("datasets", ("dataset_key", "name")).fetchall()
            if len(rows) != 1:
                raise ValueError(f"{path.name}: expected exactly one track source")
            if rows[0][0] in sources:
                raise ValueError(f"Duplicate source {rows[0][0]!r} in {directory}")
            sources[rows[0][0]] = path
    return sources


@contextmanager
def create_database(inputs: IngestionInput, metadata: Mapping[str, str]):
    """Stage a new source database and publish it after the context succeeds.

    Args:
        inputs: Validated source identity, input folder, and destination database key.
        metadata: Additional string provenance values to store with build information.

    Yields:
        An open, writable ClusterDatabase with its schema and source metadata initialized.

    Raises:
        FileExistsError: The destination exists, including a concurrent publication.

    The connection closes before publication. Failed contexts remove the temporary file
    without publishing it; existing destination files are never overwritten.
    """
    path = inputs.database
    if path.exists():
        raise FileExistsError(f"Database already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.stem + "-", suffix=".building", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        ClusterDatabase.initialize(temporary)
        with ClusterDatabase(temporary, writable=True) as database:
            with database.transaction():
                database.set_metadata(
                    {
                        "built_at_utc": datetime.now(UTC).isoformat(),
                        "import_versions": json.dumps(
                            {
                                package: version(package)
                                for package in ("gpxpy", "numpy", "sqlalchemy")
                            }
                        ),
                        "sqlite_version": sqlite3.sqlite_version,
                        "schema_sha256": sha256(PACKAGE / "sql/schema.sql"),
                        "storage_sha256": sha256(PACKAGE / "storage.py"),
                        "warehouse_sha256": sha256(Path(__file__)),
                        "time_policy": "UTC; missing timestamps NULL; naive timestamps treated as UTC and flagged",
                        **metadata,
                    }
                )
                database.insert(
                    "datasets",
                    {
                        "dataset_key": inputs.dataset_key,
                        "name": inputs.name
                        if inputs.name is not None
                        else inputs.source_folder.resolve().name,
                        "label_kind": "unlabeled",
                        "label_description": "No hand-made reference labels supplied.",
                    },
                )
            yield database
        # Same-directory hard-link publication is atomic and fails if another creator won.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_config(path: Path) -> SourceConfiguration:
    """Validate TOML sources and resolve input paths before any import begins.

    Args:
        path: Source-configuration TOML file; relative source paths use its parent folder.

    Returns:
        A SourceConfiguration containing validated ingestion inputs.
    """
    path = path.resolve()
    with path.open("rb") as source:
        config = tomllib.load(source)
    try:
        return SourceConfiguration.model_validate(config, context={"directory": path.parent})
    except ValidationError as error:
        raise ValueError(f"{path.name}: {validation_message(error)}") from None


def read_labels(path: str | Path) -> pd.DataFrame:
    """Read hand-made CSV labels and validate their track keys and integer assignments.

    Args:
        path: CSV with file_name, zero-based id, and cluster columns; -1 denotes noise.

    Returns:
        Those three columns with unique file/track keys and integer IDs and labels.
    """
    labels = pd.read_csv(path, dtype={"file_name": str})
    required = ["file_name", "id", "cluster"]
    if missing := set(required) - set(labels):
        raise ValueError(f"Label CSV is missing columns: {', '.join(sorted(missing))}")
    labels = labels[required].copy()
    if labels.empty or labels.isna().any().any():
        raise ValueError("Label CSV must contain nonempty file_name, id, and cluster values")
    for column, minimum in (("id", 0), ("cluster", -1)):
        values = pd.to_numeric(labels[column], errors="raise")
        if not np.isfinite(values).all() or (values % 1 != 0).any() or (values < minimum).any():
            raise ValueError(f"{column} must contain integers >= {minimum}")
        labels[column] = values.astype(np.int64)
    if labels.duplicated(["file_name", "id"]).any():
        raise ValueError("Label CSV contains duplicate (file_name, id) keys")
    return labels


def _reference_signature(database: ClusterDatabase):
    """Capture eligible references for detecting changes that invalidate label-based scores."""
    return (
        database.select_exec("datasets", ("label_kind",)).fetchone()[0],
        tuple(database.eligible_references().itertuples(index=False, name=None)),
    )


def replace_references(
    database: ClusterDatabase,
    labels,
    *,
    kind="author",
    description=None,
    file_names: Sequence[str] | None = None,
):
    """Replace supplied labels and invalidate affected label-based evaluations without committing.

    Args:
        database: Open writable source database; the caller owns its transaction.
        labels: Hand-made CSV path, or None to clear labels in the selected scope.
        kind: Declared provenance kind for supplied labels, normally "author".
        description: Label description; None retains or supplies the default description.
        file_names: Files to replace, combined with CSV-named files; None replaces all labels.
    """
    before = _reference_signature(database)
    values = database.validate_labels(labels) if labels is not None else []
    catalog = database.select_frame("v_track_catalog", ("track_id", "file_name"))
    if file_names is None:
        condition = None
    else:
        named = set(file_names) | set(
            catalog[catalog.track_id.isin([row[0] for row in values])].file_name
        )
        ids = catalog.loc[catalog.file_name.isin(named), "track_id"].tolist()
        condition = database.table("labels").c.track_id.in_(ids)
    database.delete("labels", where=condition)
    database.insert_many(
        "labels", ({"track_id": track_id, "reference_cluster": label} for track_id, label in values)
    )
    provided = database.select_exec(
        "labels", (1,), where=database.table("labels").c.reference_cluster.is_not(None), limit=1
    ).fetchone()
    dataset_id, previous_description = database.select_exec(
        "datasets", ("dataset_id", "label_description")
    ).fetchone()
    database.update(
        "datasets",
        {
            "label_kind": kind if provided else "unlabeled",
            "label_description": (
                description
                or (
                    previous_description
                    if before[0] != "unlabeled"
                    else "Supplied hand-made reference labels."
                )
            )
            if provided
            else "No hand-made reference labels supplied.",
        },
        where=None,
    )
    artifacts = database.table("artifacts")
    if labels is not None:
        path = source_path(Path(labels))
        database.delete(
            "artifacts",
            where=(artifacts.c.dataset_id == dataset_id)
            & artifacts.c.role.like("labels%")
            & (artifacts.c.source_path == path),
        )
        database.artifact(dataset_id, f"labels:{path}", Path(labels))
    elif file_names is None:
        database.delete("artifacts", where=artifacts.c.role.like("labels%"))
    if _reference_signature(database) != before:
        database.delete("evaluations", where={"reference_run_id": None})


def _finite_optional(value: float | None) -> float | None:
    """Accept missing elevations and reject nonfinite recorded values."""
    if value is not None and not math.isfinite(value):
        raise ValueError("Nonfinite elevation in source GPX")
    return value


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def load_gpx_sources(
    database: ClusterDatabase,
    directory: Path,
    dataset_id: int,
    *,
    on_conflict: ConflictPolicy = "skip",
) -> dict:
    """Append raw GPX tracks and invalidate runs when eligible source inputs change.

    Args:
        database: Open writable database; the caller owns its transaction.
        directory: GPX source folder to inspect nonrecursively in filename order.
        dataset_id: Stored source ID to which the files and tracks belong.
        on_conflict: Policy for changed filenames: skip, replace, or error. Identical
            checksums are always reused; files missing from the folder remain stored.

    Returns:
        Imported filenames and counts under files_added, files_replaced, and files_skipped.
    """
    import gpxpy

    if not directory.is_dir():
        raise ValueError(f"GPX source folder does not exist: {directory}")
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".gpx")
    existing = {
        name: (file_id, checksum)
        for name, file_id, checksum in database.select_exec(
            "source_files", ("file_name", "file_id", "sha256"), where={"dataset_id": dataset_id}
        )
    }
    result = {
        "files": [p.name for p in files],
        "files_added": 0,
        "files_replaced": 0,
        "files_skipped": 0,
    }
    invalidated = False
    for path in files:
        raw = path.read_bytes()
        checksum = hashlib.sha256(raw).hexdigest()
        previous = existing.get(path.name)
        if previous and (previous[1] == checksum or on_conflict == "skip"):
            result["files_skipped"] += 1
            continue
        if previous and on_conflict == "error":
            raise ValueError(
                f"GPX file already exists with different contents: {path.name}; choose skip or replace"
            )
        try:
            document = gpxpy.parse(raw.decode("utf-8-sig"))
        except (gpxpy.gpx.GPXException, UnicodeError) as error:
            raise ValueError(f"Cannot parse GPX file {path.name}: {error}") from error
        eligible_before = (
            previous
            and database.select_exec(
                "tracks", (1,), where={"file_id": previous[0], "eligible": 1}, limit=1
            ).fetchone()
        )
        eligible_after = any(
            sum(len(segment.points) for segment in track.segments) >= 5 for track in document.tracks
        )
        if not invalidated and (eligible_before or eligible_after):
            database.delete("evaluations", where=None)
            database.delete("runs", where=None)
            invalidated = True
        if previous:
            tracks = database.table("tracks")
            ids = database.select("tracks", ("track_id",), where={"file_id": previous[0]})
            database.delete("points", where=database.table("points").c.track_id.in_(ids))
            database.delete("labels", where=database.table("labels").c.track_id.in_(ids))
            database.delete("tracks", where=tracks.c.file_id == previous[0])
            database.delete("source_files", where={"file_id": previous[0]})
        file_id = database.insert(
            "source_files",
            {
                "dataset_id": dataset_id,
                "file_name": path.name,
                "source_path": source_path(path),
                "sha256": checksum,
                "byte_count": len(raw),
            },
        )
        for track_index, track in enumerate(document.tracks):
            coordinates, point_rows = [], []
            for segment_index, segment in enumerate(track.segments):
                for point in segment.points:
                    lon, lat = point.longitude, point.latitude
                    if not math.isfinite(lon) or not math.isfinite(lat):
                        raise ValueError(
                            f"Nonfinite coordinates in {path.name}, track {track_index}"
                        )
                    naive = point.time is not None and point.time.tzinfo is None
                    timestamp, epoch = None, None
                    if point.time is not None:
                        utc = (point.time.replace(tzinfo=UTC) if naive else point.time).astimezone(
                            UTC
                        )
                        timestamp, epoch = utc.isoformat(), utc.timestamp()
                    coordinates.append((lon, lat))
                    point_rows.append(
                        (
                            len(point_rows),
                            segment_index,
                            lon,
                            lat,
                            _finite_optional(point.elevation),
                            timestamp,
                            epoch,
                            int(naive),
                        )
                    )
            coords = np.asarray(coordinates, dtype="<f8").reshape(-1, 2)
            bounds = (*coords.min(axis=0), *coords.max(axis=0)) if len(coords) else (None,) * 4
            track_id = database.insert(
                "tracks",
                {
                    "file_id": file_id,
                    "source_track_id": track_index,
                    "track_index": track_index,
                    "name": track.name,
                    "segment_count": len(track.segments),
                    "point_count": len(coords),
                    "timestamped_points": sum(row[5] is not None for row in point_rows),
                    "geometry_sha256": hashlib.sha256(coords.tobytes()).hexdigest(),
                    "stationary": int(len(coords) > 0 and np.all(coords == coords[0])),
                    "min_lon": bounds[0],
                    "min_lat": bounds[1],
                    "max_lon": bounds[2],
                    "max_lat": bounds[3],
                },
            )
            database.insert_many(
                "points",
                (
                    dict(
                        zip(
                            (
                                "track_id",
                                "point_index",
                                "segment_index",
                                "longitude",
                                "latitude",
                                "elevation",
                                "recorded_at_utc",
                                "recorded_at_epoch",
                                "time_was_naive",
                            ),
                            (track_id, *row),
                            strict=True,
                        )
                    )
                    for row in point_rows
                ),
            )
        result["files_replaced" if previous else "files_added"] += 1
    file_ids = database.select("source_files", ("file_id",), where={"dataset_id": dataset_id})
    has_tracks = database.select_exec(
        "tracks", (1,), where=database.table("tracks").c.file_id.in_(file_ids), limit=1
    ).fetchone()
    if not has_tracks:
        message = "No GPX files" if not files else "No GPX tracks found"
        raise ValueError(
            f"{message} in {directory}; loading is nonrecursive and waypoints/routes are not tracks"
        )
    return result


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def ingest_database(
    inputs: IngestionInput, config: Path | None = None, *, on_conflict: ConflictPolicy | None = None
) -> dict:
    """Import a source transactionally, creating its database only when absent.

    Args:
        inputs: Validated source folder, database key, metadata, and optional label CSV.
        config: Optional source TOML path to record with its contents and checksum.
        on_conflict: Override the input model's skip/replace/error filename policy, or
            None to use inputs.on_conflict. Omitting a label CSV preserves stored labels.

    Returns:
        Database path, input-reuse flag, file-change counts, and stored-table row counts.
    """
    directory = inputs.source_folder.resolve()
    path = inputs.database
    if not directory.is_dir():
        raise ValueError(f"GPX source folder does not exist: {directory}")
    metadata = {
        "config_sha256": sha256(config) if config else "",
        "config_path": source_path(config) if config else "",
        "config_toml": config.read_text(encoding="utf-8") if config else "",
    }
    context = (
        ClusterDatabase(path, writable=True)
        if path.exists()
        else create_database(
            inputs,
            {
                "source_directory": source_path(directory),
                "dataset_key": inputs.dataset_key,
                "import_scope": "Append GPX sources and supplied hand-made labels",
                **metadata,
            },
        )
    )
    with context as database, database.transaction():
        sources = database.select_exec("datasets", ("dataset_id", "dataset_key")).fetchall()
        if len(sources) != 1 or sources[0][1] != inputs.dataset_key:
            raise ValueError("Database belongs to another source; use the matching database key")
        imported = load_gpx_sources(
            database, directory, sources[0][0], on_conflict=on_conflict or inputs.on_conflict
        )
        if inputs.labels is not None:
            replace_references(
                database,
                inputs.labels,
                kind=inputs.label_kind,
                description=inputs.label_description,
                file_names=imported["files"],
            )
        database.set_metadata(metadata, replace=True)
        database.validate()
        report = database.row_counts()
    return {
        "database": str(path),
        "reused_inputs": not (imported["files_added"] or imported["files_replaced"]),
        **{key: imported[key] for key in ("files_added", "files_replaced", "files_skipped")},
        **report,
    }
