"""Reject malformed inputs before commands open databases or modify sources."""

import json
import tomllib

import pytest
from pydantic import ValidationError

from extensions.analytics.evaluation import run_clustering
from extensions.analytics.storage import Database
from extensions.analytics.warehouse import read_config
from scripts import evaluate, get_clusters, load_data, start
from scripts.config.validate import IngestionInput, SourceConfiguration
from track_clusters.validation import ClusteringSettings


@pytest.mark.parametrize("algorithm", ["dbscan", "qbx"])
@pytest.mark.parametrize("feature_mode", ["spatial", "spatiotemporal"])
def test_clustering_settings_round_trip_with_active_sampling_seed(algorithm, feature_mode):
    settings = ClusteringSettings(
        algorithm=algorithm,
        eps=0.5,
        feature_mode=feature_mode,
        eps_time=0.2,
        min_samples=6,
        factor=3,
        seed=7,
        timezone="Europe/Berlin",
    )
    stored = settings.model_dump()
    assert stored["seed"] == 7
    assert stored["scale_pairs"] == 10000
    assert stored.get("p") == (2 if feature_mode == "spatiotemporal" else None)
    assert stored.get("temporal_mode") == ("separate" if feature_mode == "spatiotemporal" else None)
    assert ClusteringSettings.model_validate(stored).model_dump() == stored


@pytest.mark.parametrize(
    "settings",
    [
        {"algorithm": "unknown"},
        {"feature_mode": "unknown"},
        {"eps": float("nan")},
        {"eps_time": 0},
        {"min_samples": 0},
        {"factor": 0.5},
        {"seed": 1.5},
        {"timezone": "invalid/timezone"},
        {"algorithm": "dbscan", "min_samples": None},
        {"algorithm": "qbx", "factor": None},
        {"algorithm": "qbx", "seed": None},
        {"feature_mode": "spatiotemporal", "eps_time": None},
        {"feature_mode": "spatiotemporal", "timezone": None},
        {"algorithm": "qbx", "eps": 1e308, "factor": 2},
        {"algorithm": "qbx", "factor": 1e200},
        {"scale_pairs": 0},
        {"scale_pairs": 1.5},
        {"seed": -1},
        {"seed": True},
        {"algorithm": "dbscan", "seed": None},
    ],
)
def test_library_settings_are_validated_before_reading_database(settings):
    class UnreadDatabase:
        def input_signature(self):
            pytest.fail("Invalid settings must fail before database reads")

    parameters = {"algorithm": "dbscan", "eps": 0.5, **settings}
    with pytest.raises(ValidationError):
        run_clustering(UnreadDatabase(), ClusteringSettings(**parameters))


@pytest.mark.parametrize(
    "module,arguments,field",
    [
        (get_clusters, ["{database}", "nan"], "eps"),
        (get_clusters, ["{database}", "0.5", "--factor", "0.5"], "factor"),
        (get_clusters, ["{database}", "0.5", "--seed", "4294967296"], "seed"),
        (get_clusters, ["{database}", "0.5", "--timezone", "invalid/timezone"], "timezone"),
        (evaluate, ["{database}", "--eps", "0.5", "0"], "eps.1"),
        (evaluate, ["{database}", "--quickbundles", "--eps", "0.1", "0.1"], "eps"),
        (evaluate, ["{database}", "--quickbundles", "--seeds", "0", "0"], "seeds"),
        (evaluate, ["{database}", "--quickbundles", "--seeds"], "seed"),
        (
            load_data,
            ["single", "{source}", "--database-key", "sample", "--labels", "{missing}"],
            "labels",
        ),
        (load_data, ["batch", "--config", "{missing}"], "config"),
        (start, ["--config", "{missing}"], "config"),
    ],
)
def test_cli_validation_precedes_operations(
    module, arguments, field, tmp_path, monkeypatch, capsys
):
    database = tmp_path / "source.sqlite"
    database.write_bytes(b"Existing database must remain untouched")
    paths = dict(database=database, source=tmp_path, missing=tmp_path / "missing")

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid input reached an operation")

    for name in (
        "ClusterDatabase",
        "Database",
        "ingest_database",
        "prepare_sources",
    ):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, unexpected)
    with pytest.raises(SystemExit) as error:
        module.main([argument.format(**paths) for argument in arguments])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert field in message
    assert "Traceback" not in message and "errors.pydantic.dev" not in message
    assert database.read_bytes() == b"Existing database must remain untouched"
    assert set(tmp_path.iterdir()) == {database}


@pytest.mark.parametrize(
    "old,new,field",
    [
        ('dataset_key = "sample"', "", "dataset_key"),
        ('source_folder = "gpx"', "", "source_folder"),
        ('dataset_key = "sample"', 'key = "sample"', "key"),
        ('source_folder = "gpx"', 'gpx_dir = "gpx"', "gpx_dir"),
        ('name = "Controlled sample"', 'name = "   "', "name"),
        ('dataset_key = "sample"', 'dataset_key = "../outside"', "key"),
        ('dataset_key = "sample"', 'dataset_key = "CON"', "key"),
        ('dataset_key = "sample"', 'dataset_key = "mÃ¼nchen"', "key"),
        ('label_kind = "author"', 'label_kind = "unknown"', "label_kind"),
        ('source_folder = "gpx"', 'source_folder = "labels.csv"', "source_folder"),
        ('labels = "labels.csv"', 'labels = "absent.csv"', "labels"),
        ('source_folder = "gpx"', 'source_folder = ""', "source_folder"),
        ('name = "Controlled sample"', "name = 42", "name"),
        ('name = "Controlled sample"', 'name = "Controlled sample"\nunknown = true', "unknown"),
        (
            'name = "Controlled sample"',
            'name = "Controlled sample"\nid_source = "index"',
            "id_source",
        ),
    ],
)
def test_invalid_configuration_fails_before_any_source_build(
    source_copy, tmp_path, monkeypatch, capsys, old, new, field
):
    config = source_copy / "datasets.toml"
    config.write_text(config.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    monkeypatch.setattr(
        load_data,
        "ingest_database",
        lambda *args: pytest.fail("Invalid configuration reached ingestion"),
    )
    with pytest.raises(SystemExit) as error:
        load_data.main(["batch", "--config", str(config)])
    assert error.value.code == 1
    assert field in capsys.readouterr().err
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("kind", ["empty", "duplicate", "unknown_root", "wrong_type"])
def test_configuration_structure_is_validated(source_copy, kind):
    config = source_copy / "datasets.toml"
    text = config.read_text(encoding="utf-8")
    text = {
        "empty": "datasets = []",
        "duplicate": text + text,
        "unknown_root": "unexpected = true\n" + text,
        "wrong_type": 'datasets = "not a list"',
    }[kind]
    config.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        read_config(config)


def test_configuration_paths_are_relative_to_toml_not_working_directory(
    source_copy, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    configuration = read_config(source_copy / "datasets.toml")
    assert isinstance(configuration, SourceConfiguration)
    source = configuration.datasets[0]
    assert source.source_folder == source_copy / "gpx"
    assert source.labels == source_copy / "labels.csv"


def test_cli_and_toml_use_the_same_ingestion_model(source_copy):
    raw = tomllib.loads((source_copy / "datasets.toml").read_text(encoding="utf-8"))
    configured = SourceConfiguration.model_validate(
        raw, context={"directory": source_copy}
    ).datasets[0]
    cli = IngestionInput(
        source_folder=source_copy / "gpx",
        dataset_key="sample",
        name="Controlled sample",
        labels=source_copy / "labels.csv",
        label_kind="author",
        label_description="Controlled test labels",
    )
    assert type(configured) is IngestionInput
    assert configured.model_dump() == cli.model_dump()
    assert configured == read_config(source_copy / "datasets.toml").datasets[0]


@pytest.mark.parametrize("supplied_labels", [False, True])
def test_batch_optional_metadata_uses_cli_defaults(source_copy, capsys, supplied_labels):
    config = source_copy / "datasets.toml"
    text = '[[datasets]]\ndataset_key = "sample"\nsource_folder = "gpx"\n'
    if supplied_labels:
        text += 'labels = "labels.csv"\n'
    config.write_text(text, encoding="utf-8")
    source = read_config(config).datasets[0]
    cli = IngestionInput(
        source_folder=source_copy / "gpx",
        dataset_key="sample",
        labels=source_copy / "labels.csv" if supplied_labels else None,
    )
    assert source.model_dump() == cli.model_dump()
    assert source.name is None and source.label_description is None
    assert source.label_kind == "author"

    assert load_data.main(["batch", "--config", str(config)]) == 0
    summary = json.loads(capsys.readouterr().out)[0]
    assert summary["tracks"] == 14 and summary["evaluation_results"] == summary["runs"] == 0
    with Database(source.database) as database:
        inventory = database.report("dataset_inventory").iloc[0]
    assert inventory["name"] == "gpx"
    assert inventory.label_kind == ("author" if supplied_labels else "unlabeled")
    assert inventory.provided_labeled_tracks == (13 if supplied_labels else 0)
    assert inventory.labeled_tracks == (13 if supplied_labels else 0)


def test_configuration_accepts_keys_instead_of_database_paths(source_copy):
    config = source_copy / "datasets.toml"
    with config.open("a", encoding="utf-8") as target:
        target.write('database = "elsewhere.sqlite"\n')
    with pytest.raises(ValueError, match="database: Extra inputs"):
        read_config(config)
    assert not (source_copy / "elsewhere.sqlite").exists()


def test_configured_keys_are_validated_without_path_context(source_copy):
    with pytest.raises(ValueError, match="dataset_key"):
        IngestionInput(source_folder=source_copy / "gpx", dataset_key="../outside")


@pytest.mark.parametrize(
    "command,key_flag", [("single", "--database-key"), ("single", "--dataset-key"), ("batch", None)]
)
def test_both_ingestion_commands_build_from_model_objects(
    source_copy, monkeypatch, command, key_flag, output_directory
):
    inputs = []

    def ingest(source, config=None):
        assert isinstance(source, IngestionInput)
        assert source.database.is_absolute()
        inputs.append((source, config))
        return {"tracks": 14}

    monkeypatch.setattr(load_data, "ingest_database", ingest)
    monkeypatch.chdir(source_copy)

    config = source_copy / "datasets.toml"
    arguments = ["single", "gpx", key_flag, "sample"]
    if command == "batch":
        arguments = ["batch", "--config", "datasets.toml"]
    assert load_data.main(arguments) == 0
    assert len(inputs) == 1
    source, provenance = inputs[0]
    assert source.source_folder.resolve() == source_copy / "gpx"
    assert provenance == (config if command == "batch" else None)
    assert source.database == output_directory / "sample.sqlite"


def test_invalid_later_batch_source_prevents_all_imports(
    source_copy, monkeypatch, capsys, output_directory
):
    config = source_copy / "datasets.toml"
    text = config.read_text(encoding="utf-8")
    invalid = text.replace('dataset_key = "sample"', 'dataset_key = "other"').replace(
        'label_kind = "author"', 'label_kind = "unknown"'
    )
    config.write_text(text + invalid, encoding="utf-8")
    monkeypatch.setattr(
        load_data, "ingest_database", lambda *args: pytest.fail("Partial batch reached ingestion")
    )
    with pytest.raises(SystemExit) as error:
        load_data.main(["batch", "--config", str(config)])
    assert error.value.code == 1
    assert "datasets.1.label_kind" in capsys.readouterr().err
    assert not output_directory.exists()
