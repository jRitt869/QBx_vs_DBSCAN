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
Seed = Annotated[int, Field(ge=0, lt=2**32, strict=True)]
Algorithm = Literal["dbscan", "qbx"]
FeatureMode = Literal["spatial", "spatiotemporal"]


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
    """Validate original spatial/time settings and normalize inactive arguments."""

    model_config = ConfigDict(extra="forbid", validate_default=True)
    algorithm: Algorithm
    eps: PositiveNumber
    feature_mode: FeatureMode = "spatiotemporal"
    eps_time: PositiveNumber | None = 0.1
    min_samples: PositiveInteger | None = 5
    factor: Factor | None = 2
    seed: Seed | None = 0
    timezone: Timezone | None = "UTC"

    @model_validator(mode="after")
    def _active_parameters(self) -> Self:
        """Normalize the parameters used by the original algorithms."""
        if self.feature_mode == "spatial":
            self.eps_time = None
            self.timezone = None
        elif self.eps_time is None or self.timezone is None:
            raise ValueError("spatiotemporal runs require eps_time and timezone")
        if self.algorithm == "dbscan":
            if self.min_samples is None:
                raise ValueError("DBSCAN requires min_samples")
            self.factor = self.seed = None
        else:
            if self.factor is None or self.seed is None:
                raise ValueError("QuickBundlesX requires factor and seed")
            qbx_thresholds(self.eps, self.factor)
            self.min_samples = None
        return self
