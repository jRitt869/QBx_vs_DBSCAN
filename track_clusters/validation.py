"""Numerical constraints shared by clustering calls and command-line models."""

from typing import Annotated, Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
    validate_call,
)

PositiveNumber = Annotated[float, Field(gt=0, allow_inf_nan=False)]
PositiveInteger = Annotated[int, Field(gt=0, strict=True)]
Factor = Annotated[float, Field(ge=1, allow_inf_nan=False)]
MetricPower = Annotated[float, Field(ge=1, allow_inf_nan=False)]
Seed = Annotated[int, Field(ge=0, lt=2**32, strict=True)]
Algorithm = Literal["dbscan", "qbx"]
FeatureMode = Literal["spatial", "spatiotemporal"]
TemporalMode = Literal["separate", "joint"]


def _unique_values(values: list) -> list:
    """Reject duplicate threshold or seed values."""
    if len(set(values)) != len(values):
        raise ValueError("Values must not contain duplicates")
    return values


Thresholds = Annotated[list[PositiveNumber], Field(min_length=1), AfterValidator(_unique_values)]
Seeds = Annotated[list[Seed], AfterValidator(_unique_values)]


def _valid_timezone(value: str) -> str:
    """Validate the timezone used to encode a track's calendar features."""
    if value != "local":
        try:
            ZoneInfo(value)
        except (ValueError, ZoneInfoNotFoundError):
            raise ValueError("Use an IANA time zone, such as UTC, or 'local'") from None
    return value


Timezone = Annotated[
    str, StringConstraints(strict=True, pattern=r"\S"), AfterValidator(_valid_timezone)
]


@validate_call(validate_return=True)
def qbx_thresholds(eps: PositiveNumber, factor: Factor) -> list[PositiveNumber]:
    """Calculate the original hierarchy thresholds and reject nonfinite results.

    Args:
        eps: Positive finest-layer threshold.
        factor: Finite hierarchy multiplier of at least one.

    Returns:
        The validated thresholds [factor**2 * eps, factor * eps, eps].
    """
    try:
        return [factor**2 * eps, factor * eps, eps]
    except OverflowError:
        raise ValueError("factor and eps must produce finite QuickBundlesX thresholds") from None


class ClusteringSettings(BaseModel):
    """Validate a clustering run and normalize inactive settings for stable stored keys."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    algorithm: Algorithm
    eps: PositiveNumber
    feature_mode: FeatureMode = "spatial"
    eps_time: PositiveNumber | None = 0.1
    min_samples: PositiveInteger | None = 5
    factor: Factor | None = 2
    seed: Seed = 0
    timezone: Timezone | None = "UTC"
    temporal_mode: TemporalMode | None = Field(
        default="separate", exclude_if=lambda value: value is None
    )
    p: MetricPower | None = Field(default=2, exclude_if=lambda value: value is None)
    scale_pairs: PositiveInteger = 10000

    @model_validator(mode="after")
    def _active_parameters(self) -> Self:
        """Validate active settings and normalize unused fields to None for stable run keys."""
        if self.feature_mode == "spatial":
            self.eps_time = None
            self.timezone = None
            self.temporal_mode = None
            self.p = None
        else:
            if any(value is None for value in (self.temporal_mode, self.timezone, self.p)):
                raise ValueError("spatiotemporal runs require temporal_mode, timezone, and p")
            if self.temporal_mode == "joint":
                self.eps_time = None
            elif self.eps_time is None:
                raise ValueError("Separate temporal layers require eps_time")
        if self.algorithm == "dbscan":
            if self.min_samples is None:
                raise ValueError("DBSCAN requires min_samples")
            self.factor = None
        else:
            if self.factor is None:
                raise ValueError("QuickBundlesX requires factor")
            qbx_thresholds(self.eps, self.factor)
            self.min_samples = None
        return self
