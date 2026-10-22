"""External CLI metadata preserves parsing and rejects inconsistent definitions."""

import json
from pathlib import Path

import pytest

from extensions.analytics.storage import DEFAULT_CONFIG
from scripts.config.parser import CLI_CONFIG, build_parser, load_definitions
from scripts.config.validate import (
    BatchInput,
    ClusteringInput,
    EvaluationInput,
    validate_cli,
)


@pytest.mark.parametrize(
    "command",
    [
        "get_clusters",
        "evaluate",
        "load_data",
        "start",
    ],
)
def test_every_command_builds_and_exposes_help(command, capsys):
    with pytest.raises(SystemExit) as error:
        build_parser(command).parse_args(["--help"])
    assert error.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_external_aliases_and_help_use_model_types_and_defaults(tmp_path):
    contents = json.loads(CLI_CONFIG.read_text(encoding="utf-8"))
    arguments = contents["commands"]["get_clusters"]["arguments"]
    factor = next(argument for argument in arguments if argument["field"] == "factor")
    factor.update(flags=["--multiplier"], help="Multiplier configured in JSON")
    path = tmp_path / "cli.json"
    path.write_text(json.dumps(contents), encoding="utf-8")
    database = tmp_path / "source.sqlite"
    database.touch()
    parser = build_parser("get_clusters", config_path=path)
    parsed = parser.parse_args([str(database), "0.5", "-q", "--min_s", "7", "--multiplier", "3"])
    assert parsed.database == database and parsed.quickbundles is True
    assert "seed" not in vars(parsed)
    args = validate_cli(parser, ClusteringInput, parsed)
    assert args.factor == 3 and args.min_samples == 7 and args.eps == 0.5
    assert args.seed == 0 and args.eps_time == 0.1 and args.feature_mode == "spatial"
    assert "Multiplier configured in JSON" in parser.format_help()


def test_sweep_lists_and_literal_choices_are_derived_from_models(tmp_path, capsys):
    database = tmp_path / "source.sqlite"
    database.touch()
    parser = build_parser("evaluate")
    args = validate_cli(
        parser,
        EvaluationInput,
        parser.parse_args(
            [
                str(database),
                "--quickbundles",
                "--eps",
                "0.1",
                "0.5",
                "--seeds",
                "3",
                "8",
                "--layer",
                "2",
            ]
        ),
    )
    assert args.eps == [0.1, 0.5] and args.seeds == [3, 8] and args.layer == 2
    defaults = validate_cli(parser, EvaluationInput, parser.parse_args([str(database)]))
    assert defaults.eps == [0.1, 0.25, 0.5, 1, 2, 5]
    with pytest.raises(SystemExit) as error:
        parser.parse_args([str(database), "--quickbundles", "--layer", "4"])
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_dbscan_accepts_an_empty_cli_seed_list(tmp_path):
    database = tmp_path / "source.sqlite"
    database.touch()
    parser = build_parser("evaluate")
    args = validate_cli(parser, EvaluationInput, parser.parse_args([str(database), "--seeds"]))
    assert args.algorithm == "dbscan" and args.seeds == []


@pytest.mark.parametrize("flag", ["--force", "--no-force"])
def test_removed_clustering_flags_are_rejected(tmp_path, flag):
    path = tmp_path / "source.sqlite"
    path.touch()
    parser = build_parser("get_clusters")
    with pytest.raises(SystemExit) as error:
        parser.parse_args([str(path), "0.5", flag])
    assert error.value.code == 2


def test_batch_subcommand_keeps_its_model_defaults():
    parser = build_parser("load_data")
    parsed = parser.parse_args(["batch"])
    assert parsed.command == "batch"
    del parsed.command
    batch = validate_cli(parser, BatchInput, parsed)
    assert batch.config == DEFAULT_CONFIG
    assert "single" in parser.format_help() and "batch" in parser.format_help()
    assert "query" not in parser.format_help()


@pytest.mark.parametrize(
    "arguments",
    [
        ["query", "dataset_inventory", "--database", "{database}"],
        ["query", "--sql", "{sql}", "--database", "{database}", "--output", "{output}"],
        ["batch", "--sql", "{sql}"],
        ["build"],
        ["single", "{source}", "--database-key", "sample", "--database", "{database}"],
        ["single", "{source}"],
        ["batch", "--output-dir", "{output}"],
        ["single", "{source}", "--database-key", "../outside"],
        ["single", "{source}", "--database-key", "CON"],
    ],
)
def test_removed_query_options_are_rejected_before_processing(
    tmp_path, monkeypatch, capsys, arguments
):
    from scripts import load_data

    database = tmp_path / "source.sqlite"
    database.write_bytes(b"Existing source database")
    sql = tmp_path / "custom.sql"
    sql.write_text("DELETE FROM tracks", encoding="utf-8")
    output = tmp_path / "export.csv"
    monkeypatch.setattr(
        load_data,
        "ingest_database",
        lambda *args: pytest.fail("Removed options reached ingestion"),
    )
    with pytest.raises(SystemExit) as error:
        load_data.main(
            [
                argument.format(database=database, sql=sql, output=output, source=tmp_path)
                for argument in arguments
            ]
        )
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err
    assert database.read_bytes() == b"Existing source database"
    assert not output.exists()


@pytest.mark.parametrize(
    "change,message",
    [
        (lambda definition: definition.update(model="UnknownInput"), "Unknown input model"),
        (
            lambda definition: definition["arguments"][0].update(field="unknown"),
            "Unknown model field",
        ),
        (lambda definition: definition["arguments"].pop(), "Missing CLI fields"),
        (
            lambda definition: definition["arguments"].append(definition["arguments"][0]),
            "Repeated model field",
        ),
        (
            lambda definition: definition["arguments"][6].update(flags=["--seed"]),
            "Repeated or reserved",
        ),
        (
            lambda definition: definition["arguments"][6].update(flags=["--help"]),
            "Repeated or reserved",
        ),
        (
            lambda definition: definition["arguments"][0].update(flags=["input"]),
            "Positional argument",
        ),
        (lambda definition: definition["arguments"][1].update(nargs="+"), "Argument count"),
        (lambda definition: definition["arguments"][3].pop("action"), "Boolean flags"),
        (lambda definition: definition["arguments"][0].pop("help"), "help"),
        (lambda definition: definition["arguments"][0].update(help=" "), "help"),
        (lambda definition: definition["arguments"][4].pop("action"), "Binary choices"),
        (lambda definition: definition["arguments"][4].update(const="spatial"), "non-default"),
        (lambda definition: definition["arguments"][4].update(const="invalid"), "non-default"),
        (lambda definition: definition["arguments"][3].update(action="store_false"), "non-default"),
        (lambda definition: definition["arguments"][6].update(default=5), "Extra inputs"),
        (lambda definition: definition["arguments"][6].update(hlep="Typo"), "Extra inputs"),
    ],
)
def test_invalid_metadata_fails_before_parser_construction(tmp_path, change, message):
    contents = json.loads(CLI_CONFIG.read_text(encoding="utf-8"))
    change(contents["commands"]["get_clusters"])
    path = tmp_path / "cli.json"
    path.write_text(json.dumps(contents), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_definitions(path)


@pytest.mark.parametrize("contents", [None, "{bad JSON", '{"commands": {}}'])
def test_unreadable_or_invalid_json_is_a_cli_error(tmp_path, contents, capsys):
    path = tmp_path / "cli.json"
    if contents is not None:
        path.write_text(contents, encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        build_parser("start", config_path=path)
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "CLI configuration:" in message and "cli.json" in message
    assert "Traceback" not in message


def test_default_json_is_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parser = build_parser("start")
    parsed, server_arguments = parser.parse_known_args(["--", "--port", "5010"])
    assert vars(parsed) == {} and server_arguments == ["--", "--port", "5010"]
    assert CLI_CONFIG.is_file() and CLI_CONFIG.parent != Path.cwd()


@pytest.mark.parametrize(
    "command,model,positionals,flag,field,default,alternative",
    [
        ("get_clusters", ClusteringInput, ["0.5"], "-q", "quickbundles", False, True),
        (
            "get_clusters",
            ClusteringInput,
            ["0.5"],
            "--spatiotemporal",
            "feature_mode",
            "spatial",
            "spatiotemporal",
        ),
        ("evaluate", EvaluationInput, [], "--quickbundles", "algorithm", "dbscan", "qbx"),
        ("get_clusters", ClusteringInput, ["0.5"], "--joint", "temporal_mode", "separate", "joint"),
        ("evaluate", EvaluationInput, [], "--joint", "temporal_mode", "separate", "joint"),
        (
            "evaluate",
            EvaluationInput,
            [],
            "--spatiotemporal",
            "feature_mode",
            "spatial",
            "spatiotemporal",
        ),
    ],
)
def test_binary_flags_select_the_non_default_without_a_value(
    tmp_path, command, model, positionals, flag, field, default, alternative
):
    database = tmp_path / "source.sqlite"
    database.touch()
    parser = build_parser(command)
    arguments = [str(database), *positionals]
    original = validate_cli(parser, model, parser.parse_args(arguments))
    switched = validate_cli(parser, model, parser.parse_args([*arguments, flag]))
    assert getattr(original, field) == default
    assert getattr(switched, field) == alternative
    with pytest.raises(SystemExit) as error:
        parser.parse_args([*arguments, flag, str(alternative)])
    assert error.value.code == 2


@pytest.mark.parametrize("subcommand", ["single", "batch"])
def test_ingestion_subcommand_help_lists_descriptions_and_categorical_options(subcommand, capsys):
    parser = build_parser("load_data")
    with pytest.raises(SystemExit) as error:
        parser.parse_args([subcommand, "--help"])
    assert error.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    if subcommand == "single":
        assert "subdirectories are not traversed" in help_text
        assert "{skip,replace,error}" in help_text
        assert "Choices: skip, replace, error." in help_text
        assert "Default: skip." in help_text
        assert "Choices: author." in help_text
    else:
        assert "TOML source configuration" in help_text


def test_categorical_binary_help_lists_choices_and_default():
    help_text = " ".join(build_parser("evaluate").format_help().split())
    assert "Choices: dbscan, qbx. Default: dbscan." in help_text
    assert "Choices: spatial, spatiotemporal. Default: spatial." in help_text
    assert "--algorithm" not in help_text and "--feature-mode" not in help_text


def test_boolean_flag_can_switch_a_true_default_off(tmp_path, monkeypatch):
    from scripts.config import parser as parser_module

    class QuickBundlesDefault(ClusteringInput):
        quickbundles: bool = True

    monkeypatch.setitem(parser_module.MODELS, "ClusteringInput", QuickBundlesDefault)
    contents = json.loads(CLI_CONFIG.read_text(encoding="utf-8"))
    switch = contents["commands"]["get_clusters"]["arguments"][3]
    switch.update(
        flags=["--dbscan"], action="store_false", help="Use DBSCAN instead of QuickBundlesX"
    )
    path = tmp_path / "cli.json"
    path.write_text(json.dumps(contents), encoding="utf-8")
    database = tmp_path / "source.sqlite"
    database.touch()
    parser = build_parser("get_clusters", config_path=path)
    original = validate_cli(parser, QuickBundlesDefault, parser.parse_args([str(database), "0.5"]))
    switched = validate_cli(
        parser, QuickBundlesDefault, parser.parse_args([str(database), "0.5", "--dbscan"])
    )
    assert original.quickbundles is True and switched.quickbundles is False
