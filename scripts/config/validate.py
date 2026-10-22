"""Shared validation models for command-line inputs and source configuration."""

from argparse import ArgumentParser, Namespace
from pathlib import Path, PureWindowsPath
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    DirectoryPath,
    Field,
    FilePath,
    StringConstraints,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from extensions.analytics.storage import (
    DEFAULT_CONFIG,
    OUTPUT_DIRECTORY,
)
from track_clusters.validation import (
    Algorithm,
    Factor,
    FeatureMode,
    MetricPower,
    PositiveInteger,
    PositiveNumber,
    Seed,
    Seeds,
    TemporalMode,
    Thresholds,
    Timezone,
)


def _portable_key(value: str) -> str:
    """Reject source keys that produce reserved Windows database filenames."""
    if PureWindowsPath(f"{value}.sqlite").is_reserved():
        raise ValueError("Source key must be a portable filename")
    return value


ConflictPolicy = Literal["skip", "replace", "error"]
LabelKind = Literal["author"]
NonBlank = Annotated[str, StringConstraints(strict=True, pattern=r"\S")]
SourceKey = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[A-Za-z0-9_-]+$"),
    AfterValidator(_portable_key),
]


class InputModel(BaseModel):
    """Reject unknown fields and validate defaults for all input and configuration models."""

    model_config = ConfigDict(extra="forbid", validate_default=True)


class DatabaseInput(InputModel):
    """Shared input for commands that require an existing database file."""

    database: FilePath


class ClusteringInput(DatabaseInput):
    """Settings for one maintained clustering run and its distance calibration."""

    eps: PositiveNumber
    eps_time: PositiveNumber = 0.1
    quickbundles: bool = False
    feature_mode: FeatureMode = "spatial"
    min_samples: PositiveInteger = 5
    factor: Factor = 2
    seed: Seed = 0
    timezone: Timezone = "UTC"
    temporal_mode: TemporalMode = "separate"
    p: MetricPower = 2
    scale_pairs: PositiveInteger = 10000


class EvaluationInput(DatabaseInput):
    """One algorithm's sweep, using hand-made labels or a reference-run TOML file."""

    algorithm: Algorithm = "dbscan"
    eps: Thresholds | None = None
    reference_config: FilePath | None = None
    seeds: Seeds = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    factor: Factor = 2
    min_samples: PositiveInteger = 5
    layer: Literal[1, 2, 3] | None = None
    feature_mode: FeatureMode = "spatial"
    eps_time: PositiveNumber = 0.1
    timezone: Timezone = "UTC"
    temporal_mode: TemporalMode = "separate"
    p: MetricPower = 2
    scale_pairs: PositiveInteger = 10000

    @model_validator(mode="after")
    def _sweep_defaults(self):
        """Apply default thresholds and algorithm-specific sweep constraints."""
        if self.eps is None:
            self.eps = (
                [0.1, 0.25, 0.5, 1, 2, 5]
                if self.algorithm == "dbscan"
                else [0.01, 0.05, 0.1, 0.5, 1, 2, 5]
            )
        if self.algorithm == "dbscan" and self.layer not in (None, 1):
            raise ValueError("DBSCAN has one spatial layer")
        if self.algorithm == "qbx" and not self.seeds:
            raise ValueError("QuickBundlesX requires at least one seed")
        return self


class IngestionInput(InputModel):
    """Shared CLI/TOML ingestion inputs using the same field names in both modes."""

    source_folder: DirectoryPath
    dataset_key: SourceKey
    on_conflict: ConflictPolicy = "skip"
    name: NonBlank | None = None
    labels: FilePath | None = None
    label_kind: LabelKind = "author"
    label_description: NonBlank | None = None

    @property
    def database(self) -> Path:
        """Return this source key's absolute SQLite path in the fixed output directory."""
        return (OUTPUT_DIRECTORY / f"{self.dataset_key}.sqlite").resolve()

    @field_validator("source_folder", "labels", mode="before")
    @classmethod
    def _relative_to_configuration(cls, value, info: ValidationInfo):
        """Resolve configured paths before file-existence validation."""
        if isinstance(value, str) and not value.strip():
            raise ValueError("Paths must be nonempty")
        if isinstance(value, (str, Path)) and info.context and "directory" in info.context:
            return (info.context["directory"] / value).resolve()
        return value


class StartupInput(InputModel):
    """Select the source configuration and whether setup should launch the dashboard."""

    config: FilePath = DEFAULT_CONFIG
    prepare_only: bool = False


class BatchInput(InputModel):
    """Select the TOML file listing the track sources to import."""

    config: FilePath = DEFAULT_CONFIG


class SourceConfiguration(InputModel):
    """Sources for a single or batch import, with unique database keys."""

    datasets: Annotated[list[IngestionInput], Field(min_length=1)]

    @field_validator("datasets")
    @classmethod
    def _unique_source_keys(cls, sources):
        """Reject multiple sources targeting the same database key."""
        keys = [source.dataset_key for source in sources]
        if len(set(keys)) != len(keys):
            raise ValueError("Dataset keys must not contain duplicates")
        return sources


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
