"""Input models for the isolated historical CLI."""

from argparse import ArgumentParser, Namespace
from typing import Annotated

from pydantic import BaseModel, ConfigDict, FilePath, StringConstraints, ValidationError

from ..validation import (
    Algorithm,
    Factor,
    FeatureMode,
    PositiveInteger,
    PositiveNumber,
    Seed,
    Timezone,
)

NonBlank = Annotated[str, StringConstraints(strict=True, pattern=r"\S")]


class InputModel(BaseModel):
    """Reject unknown arguments and validate defaults."""

    model_config = ConfigDict(extra="forbid", validate_default=True)


class ClusteringInput(InputModel):
    """Select original clustering settings for an existing source database."""

    database: FilePath
    eps: PositiveNumber
    eps_time: PositiveNumber = 0.1
    quickbundles: bool = False
    feature_mode: FeatureMode = "spatiotemporal"
    min_samples: PositiveInteger = 5
    factor: Factor = 2
    seed: Seed = 0
    timezone: Timezone = "UTC"


class ViewerInput(InputModel):
    """Select an existing source and optional stored run for the original map."""

    database: FilePath
    run_id: PositiveInteger | None = None
    algorithm: Algorithm = "dbscan"


def validation_message(error: ValidationError) -> str:
    """Format field locations and validation failures as a concise user-facing message.

    Args:
        error: Pydantic validation failure from a CLI or configuration model.

    Returns:
        Semicolon-separated field/message pairs without input values or documentation URLs.
    """
    return "; ".join(
        f"{'.'.join(map(str, item['loc'])) or 'arguments'}: {item['msg']}"
        for item in error.errors(include_url=False, include_context=False, include_input=False)
    )


def validate_cli[Model: InputModel](
    parser: ArgumentParser, model: type[Model], arguments: Namespace
) -> Model:
    """Validate parsed CLI arguments or exit through argparse's usage-error handler.

    Args:
        parser: Command parser used to report invalid arguments.
        model: InputModel subclass supplying field constraints and defaults.
        arguments: Namespace from argparse with explicitly provided CLI values.

    Returns:
        An instance of the requested model with validated values and defaults.
    """
    try:
        return model.model_validate(vars(arguments))
    except ValidationError as error:
        parser.error(validation_message(error))
