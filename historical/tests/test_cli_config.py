"""Frozen CLI layouts and defaults belong to the historical package."""

import pytest

from historical.config.parser import build_parser
from historical.config.validate import ClusteringInput, ViewerInput, validate_cli


@pytest.mark.parametrize(
    "command,model,positionals,flag,field,default,selected",
    [
        ("get_raw_data", ClusteringInput, ["0.5"], "-q", "quickbundles", False, True),
        (
            "get_raw_data",
            ClusteringInput,
            ["0.5"],
            "--spatial-only",
            "feature_mode",
            "spatiotemporal",
            "spatial",
        ),
        ("run_server", ViewerInput, [], "-q", "algorithm", "dbscan", "qbx"),
    ],
)
def test_historical_binary_flags(
    tmp_path, command, model, positionals, flag, field, default, selected
):
    database = tmp_path / "source.sqlite"
    database.touch()
    parser = build_parser(command)
    arguments = [str(database), *positionals]
    assert getattr(validate_cli(parser, model, parser.parse_args(arguments)), field) == default
    assert (
        getattr(validate_cli(parser, model, parser.parse_args([*arguments, flag])), field)
        == selected
    )
    assert "--period" not in parser.format_help()
    assert "--scale-pairs" not in parser.format_help()
