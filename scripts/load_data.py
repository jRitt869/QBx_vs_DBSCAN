"""Import one GPX source or a TOML batch into source-specific SQLite databases."""

import json
import sqlite3

from pydantic import ValidationError

from extensions.analytics.warehouse import ingest_database, read_config
from scripts.config.parser import build_parser
from scripts.config.validate import (
    BatchInput,
    IngestionInput,
    SourceConfiguration,
    validate_cli,
    validation_message,
)


def main(argv=None) -> int:
    parser = build_parser("load_data")
    parsed = parser.parse_args(argv)
    command = parsed.command
    del parsed.command
    try:
        config = None
        if command == "batch":
            config_args = validate_cli(parser, BatchInput, parsed)
            config = config_args.config.resolve()
            sources = read_config(config)
        else:
            source = validate_cli(parser, IngestionInput, parsed)
            sources = SourceConfiguration(datasets=[source])

        results = []
        for source in sources.datasets:
            results.append(ingest_database(source, config))
        print(json.dumps(results, indent=2))
    except ValidationError as error:
        parser.error(validation_message(error))
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.exit(1, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
