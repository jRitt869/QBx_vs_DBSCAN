"""Evaluate clustering results stored in one source database."""

import sqlite3
import tomllib

from pydantic import ValidationError

from extensions.analytics.evaluation import ReferenceConfiguration, evaluate
from extensions.analytics.storage import ClusterDatabase
from scripts.config.parser import build_parser
from scripts.config.validate import EvaluationInput, validate_cli, validation_message
from track_clusters.validation import ClusteringSettings


def main(argv=None):
    parser = build_parser("evaluate")
    args = validate_cli(parser, EvaluationInput, parser.parse_args(argv))
    try:
        reference = None
        if args.reference_config is not None:
            with args.reference_config.open("rb") as config:
                reference = ReferenceConfiguration.model_validate(tomllib.load(config))
    except ValidationError as error:
        parser.exit(1, f"error: reference-config: {validation_message(error)}\n")
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    try:
        settings = ClusteringSettings(
            eps=args.eps[0],
            **args.model_dump(exclude={"database", "eps", "reference_config", "seeds", "layer"}),
        )
        with ClusterDatabase(args.database, writable=True) as database:
            evaluate(
                database,
                settings,
                eps=args.eps,
                reference=reference,
                seeds=args.seeds,
                layer=args.layer,
            )
    except ValidationError as error:
        parser.exit(1, f"error: {validation_message(error)}\n")
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Stored evaluation results in {args.database}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
