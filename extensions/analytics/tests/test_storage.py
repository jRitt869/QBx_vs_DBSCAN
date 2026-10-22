"""Connection ownership, transaction recovery, and DataFrame reads on shared handles."""

import sqlite3

import pandas as pd
import pytest
from sqlalchemy import CheckConstraint, bindparam, delete, func, select, update

from extensions.analytics.storage import ClusterDatabase, Database
from extensions.analytics.warehouse import ingest_database, replace_references


@pytest.fixture
def source_database(source_copy, ingestion_input):
    path = source_copy / "source.sqlite"
    ingest_database(ingestion_input(source_copy / "gpx", path))
    with Database(path, writable=True) as database, database.transaction():
        database.insert_many(
            "labels",
            (
                {"track_id": track_id, "reference_cluster": None}
                for (track_id,) in database.select_exec("tracks", ("track_id",))
            ),
        )
    return path


def test_frames_share_the_connection_and_preserve_transaction_ownership(source_database):
    with Database(source_database, writable=True) as database:
        connection = database.connection
        with database.transaction():
            database.insert("build_metadata", {"key": "test", "value": "before"})
            assert (
                database.frame("SELECT value FROM build_metadata WHERE key='test'").iloc[0, 0]
                == "before"
            )
            # Reading a DataFrame leaves the transaction and its connection with the caller.
            assert database.connection is connection and connection.in_transaction
            connection.execute("UPDATE build_metadata SET value='committed' WHERE key='test'")
        assert not connection.in_transaction
        with pytest.raises(RuntimeError, match="abort"):
            with database.transaction():
                connection.execute("UPDATE build_metadata SET value='discarded' WHERE key='test'")
                raise RuntimeError("abort")
        assert (
            database.frame("SELECT value FROM build_metadata WHERE key='test'").iloc[0, 0]
            == "committed"
        )
        assert len(database.report("dataset_inventory")) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
    with Database(source_database) as reopened:
        assert (
            reopened.frame("SELECT value FROM build_metadata WHERE key='test'").iloc[0, 0]
            == "committed"
        )


def test_closing_rolls_back_work_outside_an_explicit_transaction(source_database):
    with Database(source_database, writable=True) as database:
        database.insert("build_metadata", {"key": "uncommitted", "value": "discard"})
    with Database(source_database) as database:
        assert database.frame("SELECT * FROM build_metadata WHERE key='uncommitted'").empty
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            database.connection.execute("DELETE FROM build_metadata")


def test_frames_return_complete_results_and_respect_statement_limits(source_database):
    with Database(source_database) as database:
        actual = database.frame(
            """WITH RECURSIVE numbers(value) AS (
                SELECT 0
                UNION ALL
                SELECT value + 1 FROM numbers WHERE value + 1 < :count
            ) SELECT value FROM numbers ORDER BY value""",
            {"count": 10005},
        )
        pd.testing.assert_frame_equal(actual, pd.DataFrame({"value": range(10005)}))
        tracks = database.metadata.tables["tracks"]
        limited = database.frame(select(tracks.c.track_id).order_by(tracks.c.track_id).limit(3))
        assert len(limited) == 3


def test_generic_select_binds_values_and_supports_core_expressions(source_database):
    quoted = "sample' OR 1=1 --"
    with Database(source_database, writable=True) as database:
        with database.transaction():
            database.insert_many(
                "build_metadata",
                ({"key": key, "value": quoted} for key in ("generic_a", "generic_b")),
            )
        metadata = database.table("build_metadata")
        statement = database.select(
            "build_metadata",
            ("key",),
            where={"value": quoted},
            order_by=(metadata.c.key.desc(),),
            limit=1,
        )
        sql, values = database.compile(statement)
        assert quoted not in sql and quoted in values.values()
        assert database.execute(statement).fetchall() == [("generic_b",)]
        frame = database.select_frame(
            "build_metadata",
            ("value",),
            where=metadata.c.key.in_(["generic_a", "generic_b"]),
            distinct=True,
        )
        assert frame.value.tolist() == [quoted]
        labels = database.select_frame("labels", where={"reference_cluster": None})
        assert len(labels) == 14 and labels.reference_cluster.isna().all()
        with pytest.raises(KeyError):
            database.select("build_metadata; DROP TABLE tracks")
        with pytest.raises(KeyError):
            database.select("build_metadata", ("unknown_column",))


def test_select_helpers_forward_parameters_and_preserve_the_callers_transaction(source_database):
    with Database(source_database, writable=True) as database:
        metadata = database.table("build_metadata")
        options = {
            "columns": ("key", "value"),
            "where": metadata.c.key.in_(bindparam("keys", expanding=True))
            & (metadata.c.value == bindparam("match")),
            "order_by": ("key",),
            "distinct": True,
            "limit": 1,
        }
        quoted = "sample' OR 1=1 --"
        parameters = {"keys": ["select_helpers"], "match": quoted}
        with pytest.raises(RuntimeError, match="abort"):
            with database.transaction():
                database.insert("build_metadata", {"key": "select_helpers", "value": quoted})
                cursor = database.select_exec("build_metadata", parameters=parameters, **options)
                assert isinstance(cursor, sqlite3.Cursor)
                assert cursor.fetchall() == [("select_helpers", quoted)]
                cursor.close()
                actual = database.select_frame("build_metadata", parameters=parameters, **options)
                pd.testing.assert_frame_equal(
                    actual, pd.DataFrame({"key": ["select_helpers"], "value": [quoted]})
                )
                empty_parameters = {"keys": [], "match": quoted}
                assert (
                    database.select_exec(
                        "build_metadata", parameters=empty_parameters, **options
                    ).fetchall()
                    == []
                )
                empty = database.select_frame(
                    "build_metadata", parameters=empty_parameters, **options
                )
                assert empty.empty and empty.columns.tolist() == ["key", "value"]
                assert database.connection.in_transaction
                raise RuntimeError("abort")
        assert not database.connection.in_transaction
        assert (
            database.select_exec("build_metadata", parameters=parameters, **options).fetchall()
            == []
        )


def test_generic_writes_filter_rows_and_leave_rollback_to_the_caller(source_database):
    with Database(source_database, writable=True) as database:
        with database.transaction():
            database.insert_many(
                "build_metadata",
                ({"key": key, "value": "before"} for key in ("generic_a", "generic_b")),
            )
            assert (
                database.update("build_metadata", {"value": "after"}, where={"key": "generic_a"})
                == 1
            )
        before = database.select_frame("build_metadata", order_by=("key",))
        with pytest.raises(RuntimeError, match="abort"):
            with database.transaction():
                assert database.delete("build_metadata", where={"key": "generic_a"}) == 1
                metadata = database.table("build_metadata")
                assert (
                    database.update(
                        "build_metadata", {"value": "discard"}, where=metadata.c.key == "generic_b"
                    )
                    == 1
                )
                raise RuntimeError("abort")
        pd.testing.assert_frame_equal(
            database.select_frame("build_metadata", order_by=("key",)), before
        )
        with pytest.raises(TypeError):
            database.delete("build_metadata")
        with pytest.raises(TypeError):
            database.update("build_metadata", {"value": "missing filter"})
        with database.transaction():
            assert database.delete("labels", where=None) == 14
            assert database.update("datasets", {"label_kind": "unlabeled"}, where=None) == 1
        assert database.select_frame("labels").empty


@pytest.mark.parametrize("operation", ["insert", "update", "delete"])
def test_generic_writes_respect_read_only_connections(source_database, operation):
    before = source_database.read_bytes()
    with Database(source_database) as database:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            if operation == "insert":
                database.insert("build_metadata", {"key": "blocked", "value": "blocked"})
            elif operation == "update":
                database.update(
                    "build_metadata", {"value": "blocked"}, where={"key": "application"}
                )
            else:
                database.delete("build_metadata", where={"key": "application"})
    assert source_database.read_bytes() == before


def test_missing_or_unrelated_databases_are_never_created_or_retained(tmp_path):
    missing = tmp_path / "missing.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        Database(missing, writable=True)
    assert not missing.exists()
    unrelated = tmp_path / "unrelated.sqlite"
    connection = sqlite3.connect(unrelated)
    connection.execute("CREATE TABLE user_records (value TEXT)")
    connection.close()
    before = unrelated.read_bytes()
    with pytest.raises(ValueError, match="not an analytics snapshot"):
        Database(unrelated)
    # On Windows this rename also confirms constructor validation released the handle.
    renamed = unrelated.with_name("closed.sqlite")
    unrelated.rename(renamed)
    assert renamed.read_bytes() == before


def test_failed_ingestion_closes_the_database_and_preserves_output(
    source_database, source_copy, ingestion_input, monkeypatch
):
    before = source_database.read_bytes()
    borrowed = []
    import extensions.analytics.warehouse as warehouse

    load_gpx_sources = warehouse.load_gpx_sources

    def fail(database, directory, dataset_id, **options):
        assert isinstance(database, ClusterDatabase)
        borrowed.append(database)
        load_gpx_sources(database, directory, dataset_id, **options)
        raise RuntimeError("abort snapshot")

    monkeypatch.setattr(warehouse, "load_gpx_sources", fail)
    inputs = ingestion_input(source_copy / "gpx", source_database)
    with pytest.raises(RuntimeError, match="abort snapshot"):
        ingest_database(inputs)
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        borrowed[0].connection.execute("SELECT 1")
    assert source_database.read_bytes() == before
    assert not list(source_database.parent.glob("*.building"))


def test_cluster_reference_methods_use_the_callers_transaction(analytics_database, source_copy):
    before = analytics_database.read_bytes()
    with ClusterDatabase(analytics_database, writable=True) as database:
        connection = database.connection
        original_labels = connection.execute("SELECT * FROM labels ORDER BY track_id").fetchall()
        original_runs = database.frame("SELECT COUNT(*) AS count FROM runs").iloc[0, 0]
        values = database.validate_labels(source_copy / "labels.csv")
        assert len(values) == 13
        assert not connection.in_transaction
        assert (
            connection.execute("SELECT * FROM labels ORDER BY track_id").fetchall()
            == original_labels
        )
        with pytest.raises(RuntimeError, match="abort reference update"):
            with database.transaction():
                replace_references(database, source_copy / "labels.csv")
                assert database.connection is connection and connection.in_transaction
                assert (
                    database.frame("SELECT COUNT(*) AS count FROM runs").iloc[0, 0] == original_runs
                )
                assert database.report("dataset_inventory").iloc[0].label_kind == "author"
                raise RuntimeError("abort reference update")
        assert (
            connection.execute("SELECT * FROM labels ORDER BY track_id").fetchall()
            == original_labels
        )
        assert database.frame("SELECT COUNT(*) AS count FROM runs").iloc[0, 0] == original_runs
        assert database.report("dataset_inventory").iloc[0].label_kind == "author"
    assert analytics_database.read_bytes() == before


def test_reflection_retains_keys_constraints_indexes_and_generated_columns(source_database):
    before = source_database.read_bytes()
    with Database(source_database) as database:
        tables = database.metadata.tables
        assert list(tables["tracks"].primary_key.columns.keys()) == ["track_id"]
        assert list(tables["points"].primary_key.columns.keys()) == ["track_id", "point_index"]
        assert {key.target_fullname for key in tables["tracks"].c.file_id.foreign_keys} == {
            "source_files.file_id"
        }
        assert "tracks_geometry" in {index.name for index in tables["tracks"].indexes}
        assert any(
            isinstance(constraint, CheckConstraint) and "longitude" in str(constraint.sqltext)
            for constraint in tables["points"].constraints
        )
        assert tables["tracks"].c.eligible.computed.persisted is True
        assert "point_count >= 5" in str(tables["tracks"].c.eligible.computed.sqltext)
        assert "reference_origin" in tables["v_track_catalog"].c
        # Views do not declare their own keys; Core reads require no invented keys.
        assert not list(tables["v_track_catalog"].primary_key.columns)
    assert source_database.read_bytes() == before


def test_core_statements_bind_values_expand_lists_and_preserve_transactions(source_database):
    with Database(source_database, writable=True) as database:
        metadata = database.metadata.tables["build_metadata"]
        quoted = "sample' OR 1=1 --"
        with database.transaction():
            database.insert("build_metadata", {"key": "quoted", "value": quoted})
            statement = (
                select(metadata.c.key)
                .select_from(metadata)
                .where(metadata.c.value == bindparam("value"))
                .where(metadata.c.key.in_(bindparam("keys", expanding=True)))
                .order_by(metadata.c.key)
            )
            sql, values = database.compile(statement, {"value": quoted, "keys": ["quoted"]})
            assert quoted not in sql and quoted in values.values()
            assert database.frame(
                statement, {"value": quoted, "keys": ["quoted"]}
            ).key.tolist() == ["quoted"]
            assert database.frame(statement, {"value": quoted, "keys": []}).empty
            database.execute_many(
                update(metadata)
                .where(metadata.c.key == bindparam("target"))
                .values(value="updated"),
                [{"target": "quoted"}],
            )
            assert database.connection.in_transaction
            assert (
                database.frame(select(metadata.c.value).where(metadata.c.key == "quoted")).iloc[
                    0, 0
                ]
                == "updated"
            )
        with pytest.raises(RuntimeError, match="rollback"):
            with database.transaction():
                database.execute(delete(metadata).where(metadata.c.key == "quoted"))
                raise RuntimeError("rollback")
        assert (
            database.frame(
                select(func.count()).select_from(metadata).where(metadata.c.key == "quoted")
            ).iloc[0, 0]
            == 1
        )
        with pytest.raises(sqlite3.IntegrityError):
            with database.transaction():
                database.insert("labels", {"track_id": 999999, "reference_cluster": 0})
