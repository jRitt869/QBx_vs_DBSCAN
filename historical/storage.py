"""Frozen SQLite adapter for the historical application; no analytics orchestration."""

import sqlite3
from collections.abc import Mapping
from contextlib import contextmanager
from itertools import chain
from pathlib import Path

import pandas as pd
from sqlalchemy import (
    MetaData,
    and_,
    bindparam,
    create_engine,
    func,
    insert,
    select,
)
from sqlalchemy.dialects.sqlite import dialect
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.pool import NullPool

APPLICATION = "gpx-portfolio-analytics-v1"


def _conditions(table, where):
    """Normalize a generic database filter into Core WHERE conditions."""
    if isinstance(where, Mapping):
        return tuple(table.c[name] == value for name, value in where.items())
    return () if where is None else (where,)


class Database:
    """Historical source connection and track access."""

    def __init__(self, path: str | Path, *, writable: bool = False):
        """Open an existing source database and reflect its tables and views.

        Args:
            path: Existing SQLite source database with the expected application marker.
            writable: Allow writes when True; the default connection is read-only.
        """
        self.path = Path(path).resolve()
        mode = "rw" if writable else "ro"
        self.connection = sqlite3.connect(
            self.path.as_uri() + f"?mode={mode}", uri=True, timeout=30
        )
        try:
            self.connection.execute("PRAGMA foreign_keys = ON")
            try:
                marker = self.connection.execute(
                    "SELECT value FROM build_metadata WHERE key='application'"
                ).fetchone()
            except sqlite3.DatabaseError as error:
                raise ValueError("Output exists and is not an analytics snapshot") from error
            if marker != (APPLICATION,):
                raise ValueError("Output exists and is not an analytics snapshot")
            # Reflection borrows the existing driver connection. Execution and
            # transactions remain with sqlite3, including its original exceptions.
            self._engine = create_engine(
                "sqlite://", creator=lambda: self.connection, poolclass=NullPool
            )
            self._reflection_connection = self._engine.connect()
            self.metadata = MetaData()
            self.metadata.reflect(bind=self._reflection_connection, views=True)
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        """Return this open database without starting a transaction."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the database on context exit without suppressing exceptions.

        Args:
            exc_type: Exception class from the context body, or None on success.
            exc_value: Raised exception instance, or None on success.
            traceback: Exception traceback, or None on success.
        """
        self.close()

    def close(self):
        """Close the connection, rolling back any uncommitted work."""
        if hasattr(self, "_reflection_connection"):
            self._reflection_connection.close()
        if hasattr(self, "_engine"):
            self._engine.dispose()
        self.connection.close()

    @staticmethod
    def compile(statement, parameters=None):
        """Compile a Core statement for SQLite without embedding bound values in SQL.

        Args:
            statement: Core expression, or SQL text that is returned unchanged.
            parameters: Optional bound values overriding compiled defaults, including
                values needed for expanding list parameters.

        Returns:
            SQL text and its parameter mapping, ready for the sqlite3 connection.
        """
        if isinstance(statement, str):
            return statement, dict(parameters or {})
        compiled = statement.compile(dialect=dialect(paramstyle="named"))
        if compiled.post_compile_params or compiled.literal_execute_params:
            expanded = compiled.construct_expanded_state(parameters)
            return expanded.statement, expanded.parameters
        return str(compiled), {**compiled.params, **(parameters or {})}

    def execute(self, statement, parameters=None):
        """Execute one statement without committing the caller's transaction.

        Args:
            statement: SQL text or a Core statement to compile and execute.
            parameters: Optional bound values overriding compiled defaults.

        Returns:
            The sqlite3 cursor, which can fetch rows or report affected-row counts.
        """
        sql, values = self.compile(statement, parameters)
        return self.connection.execute(sql, values)

    def frame(self, statement, parameters: Mapping | None = None) -> pd.DataFrame:
        """Read a statement's complete result into a DataFrame.

        Args:
            statement: SQL text or a Core selection to execute on this connection.
            parameters: Optional bound values overriding compiled defaults.

        Returns:
            A DataFrame with the statement's result columns and rows.
        """
        sql, values = self.compile(statement, parameters)
        return pd.read_sql_query(sql, self.connection, params=values)

    def table(self, name):
        """Look up a reflected schema object by its exact name.

        Args:
            name: Table or view name present in this database's metadata.

        Returns:
            The SQLAlchemy Table object used to build Core statements.
        """
        return self.metadata.tables[name]

    def select(self, table, columns=None, *, where=None, order_by=(), limit=None, distinct=False):
        """Build a reusable Core SELECT without executing it.

        Args:
            table: Reflected table or view name.
            columns: Sequence of column names or Core expressions; None selects all columns.
            where: Equality mapping, Core predicate, or None for all rows.
            order_by: Sequence of column names or Core ordering expressions.
            limit: Maximum rows to select, or None for no limit.
            distinct: Remove duplicate selected rows when True.

        Returns:
            A SELECT that can be chained further and passed to execute() or frame().
        """
        table = self.table(table)
        columns = (
            (table,)
            if columns is None
            else tuple(table.c[column] if isinstance(column, str) else column for column in columns)
        )
        statement = select(*columns).select_from(table).where(*_conditions(table, where))
        if distinct:
            statement = statement.distinct()
        statement = statement.order_by(
            *(table.c[column] if isinstance(column, str) else column for column in order_by)
        )
        return statement if limit is None else statement.limit(limit)

    def select_exec(self, table, columns=None, *, parameters=None, **options) -> sqlite3.Cursor:
        """Build and execute a selection in the caller's transaction.

        Args:
            table: Reflected table or view name.
            columns: Column names or Core expressions; None selects all columns.
            parameters: Optional bound values overriding compiled defaults.
            **options: where, order_by, limit, and distinct options forwarded to select().

        Returns:
            The sqlite3 result cursor without committing.
        """
        return self.execute(self.select(table, columns, **options), parameters)

    def select_frame(self, table, columns=None, *, parameters=None, **options) -> pd.DataFrame:
        """Build a selection and read its complete result into a DataFrame.

        Args:
            table: Reflected table or view name.
            columns: Column names or Core expressions; None selects all columns.
            parameters: Optional bound values overriding compiled defaults.
            **options: where, order_by, limit, and distinct options forwarded to select().

        Returns:
            A DataFrame containing the selected columns and rows.
        """
        return self.frame(self.select(table, columns, **options), parameters)

    def insert_many(self, name, rows, *, replace=False):
        """Insert a batch of row mappings without committing.

        Args:
            name: Reflected destination table name.
            rows: Iterable of mappings sharing the first row's column keys; empty is a no-op.
            replace: Use SQLite INSERT OR REPLACE instead of rejecting conflicting keys.
        """
        rows = iter(rows)
        first = next(rows, None)
        if first is None:
            return
        statement = sqlite_insert(self.table(name)).values({key: bindparam(key) for key in first})
        if replace:
            statement = statement.prefix_with("OR REPLACE")
        self.execute_many(statement, chain((first,), rows))

    def execute_many(self, statement, rows):
        """Execute a statement repeatedly in the caller's transaction.

        Args:
            statement: SQL text or Core statement shared by every row.
            rows: Iterable of parameter mappings overriding the statement's defaults.
        """
        sql, defaults = self.compile(statement)
        self.connection.executemany(sql, ({**defaults, **row} for row in rows))

    @contextmanager
    def transaction(self):
        """Begin an immediate write transaction, committing on success or rolling back.

        Yields:
            This database for writes within the transaction context.

        Raises:
            ValueError: A transaction is already active; nested writes are not supported.
        """
        if self.connection.in_transaction:
            raise ValueError("A transaction is already active on this database")
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            yield self

    def insert(self, table: str, row: Mapping) -> int:
        """Insert one mapped row without committing.

        Args:
            table: Reflected destination table name.
            row: Mapping of supplied column names to values; omitted columns use defaults.

        Returns:
            SQLite's last inserted row ID.
        """
        return self.execute(insert(self.table(table)).values(**row)).lastrowid

    def input_signature(self):
        """Return source keys and ordered file checksums for detecting concurrent input changes."""
        return (
            tuple(self.select_exec("datasets", ("dataset_key",))),
            tuple(
                self.select_exec("source_files", ("file_name", "sha256"), order_by=("file_name",))
            ),
        )


class ClusterDatabase(Database):
    """Historical source connection and track access."""

    def supports_time(self) -> bool:
        """Require eligible tracks with complete, finite, nonnegative, nondecreasing epochs.

        Return False if there are no eligible tracks. Short tracks do not affect readiness.
        """
        quality = self.table("v_track_time_quality")
        valid = self.select_exec(
            "v_track_time_quality", (func.min(quality.c.temporal_valid),), where={"eligible": 1}
        ).fetchone()[0]
        return valid == 1

    def load_tracks(self, run_id: int | None = None):
        """Load all raw tracks, or a run's tracks with per-layer predictions and RGB colors.

        Args:
            run_id: Stored run whose assigned tracks/layers are loaded, or None for all
                raw tracks, including short ones, without prediction/color columns.

        Returns:
            A source-ordered GeoDataFrame with full-resolution EPSG:4326 geometry,
            aligned raw epoch arrays, and, for a run, clustersN/colorsN columns.
        """
        import geopandas as gpd
        from shapely.geometry import LineString, Point

        from .data import GPX_CRS

        view, point_table = (self.metadata.tables[name] for name in ("v_track_catalog", "points"))
        assignments, track_ids = None, None
        if run_id is not None:
            assignments = self.select_frame(
                "assignments", where={"run_id": run_id}, order_by=("track_id", "layer")
            )
            if assignments.empty:
                raise ValueError("No cluster assignments exist for this run")
            track_ids = self.select("assignments", ("track_id",), where={"run_id": run_id})
        catalog = self.select_frame(
            "v_track_catalog",
            where=None if track_ids is None else view.c.track_id.in_(track_ids),
            order_by=("file_name", "track_index"),
        )
        points = self.select_frame(
            "points",
            where=None if track_ids is None else point_table.c.track_id.in_(track_ids),
            order_by=("track_id", "point_index"),
        )
        timestamps, geometry, names = [], [], []
        for track in catalog.itertuples():
            frame = points[points.track_id == track.track_id]
            coordinates = frame[["longitude", "latitude"]].to_numpy(dtype=float)
            timestamps.append(frame.recorded_at_epoch.to_numpy(dtype=float))
            geometry.append(
                Point(coordinates[0]) if len(coordinates) == 1 else LineString(coordinates)
            )
            names.append(f"{Path(track.file_name).stem}_{track.track_index}")
        tracks = gpd.GeoDataFrame(catalog, geometry=geometry, crs=GPX_CRS)
        tracks["timestamps"], tracks["name"] = timestamps, names
        if assignments is not None:
            for index, layer in enumerate(sorted(assignments.layer.unique())):
                selected = (
                    assignments[assignments.layer == layer]
                    .set_index("track_id")
                    .loc[tracks.track_id]
                )
                tracks[f"clusters{index}"] = selected.predicted_cluster.tolist()
                tracks[f"colors{index}"] = [
                    [int(color[j : j + 2], 16) for j in (1, 3, 5)] for color in selected.color
                ]
        return tracks

    def load_map_jumps(self, run_id):
        """Load markers for gaps exceeding 30 minutes among a run's assigned tracks.

        Args:
            run_id: Stored run restricting which tracks contribute timestamp gaps.

        Returns:
            A DataFrame with jump_name, jump_hours, jump_minutes, and EPSG:3857 x/y
            coordinates; an empty frame has the same columns when no gaps exist.
        """
        from pyproj import Transformer

        from .data import GPX_CRS

        if (
            self.select_exec("assignments", (1,), where={"run_id": run_id}, limit=1).fetchone()
            is None
        ):
            raise ValueError("No cluster assignments exist for this run")
        view, point_table, gap_view = (
            self.metadata.tables[name] for name in ("v_track_catalog", "points", "v_time_gaps")
        )
        track_ids = self.select("assignments", ("track_id",), where={"run_id": run_id})
        gaps = self.frame(
            select(gap_view, view.c.track_index, point_table.c.longitude, point_table.c.latitude)
            .select_from(
                gap_view.join(
                    view,
                    and_(
                        gap_view.c.dataset == view.c.dataset,
                        gap_view.c.file_name == view.c.file_name,
                        gap_view.c.source_track_id == view.c.source_track_id,
                    ),
                ).join(
                    point_table,
                    and_(
                        point_table.c.track_id == view.c.track_id,
                        point_table.c.point_index == gap_view.c.point_index,
                    ),
                )
            )
            .where(view.c.track_id.in_(track_ids))
        )
        if gaps.empty:
            return pd.DataFrame(columns=["jump_name", "jump_hours", "jump_minutes", "x", "y"])
        transformer = Transformer.from_crs(GPX_CRS, "EPSG:3857", always_xy=True)
        x, y = transformer.transform(gaps.longitude.to_numpy(), gaps.latitude.to_numpy())
        return pd.DataFrame(
            {
                "jump_name": [
                    f"{Path(r.file_name).stem}_{r.track_index}" for r in gaps.itertuples()
                ],
                "jump_hours": (gaps.gap_seconds // 3600).astype(int),
                "jump_minutes": (gaps.gap_seconds // 60 % 60).astype(int),
                "x": x,
                "y": y,
            }
        )

    def load_layer_names(self, run_id):
        """Load layer names in the stored-layer order used by map prediction columns.

        Args:
            run_id: Stored run whose assignment layer names are selected.

        Returns:
            Unique layer names ordered by their zero-based stored layer indices.
        """
        layers = self.select_frame(
            "assignments",
            ("layer", "layer_name"),
            where={"run_id": run_id},
            order_by=("layer", "track_id"),
        )
        if layers.empty:
            raise ValueError("No cluster assignments exist for this run")
        return layers.drop_duplicates("layer").layer_name.tolist()
