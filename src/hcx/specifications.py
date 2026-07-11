from dataclasses import dataclass
from typing import ClassVar, Protocol, TypeVar, runtime_checkable

import numpy as np
import torch
import torch.nn.functional as functional

from hcx.output import Forecast

ParametersT = TypeVar("ParametersT")


@runtime_checkable
class OutputSpecification(Protocol[ParametersT]):
    raw_head_width: int

    def parameterize(self, raw: torch.Tensor) -> ParametersT: ...

    def populate_forecast(
        self,
        parameters: ParametersT,
        *,
        sample_ids: tuple[str, ...],
        input_end_indices: np.ndarray,
        target_fill_mask: np.ndarray,
    ) -> Forecast: ...


def _validate_raw_head(raw: torch.Tensor, expected_width: int) -> None:
    expected = f"[B, T_out, {expected_width}]"
    actual = tuple(raw.shape)
    if raw.ndim != 3 or raw.shape[-1] != expected_width:
        raise ValueError(f"expected shape {expected}; actual shape {actual}")


@dataclass(frozen=True)
class PointParameters:
    prediction: torch.Tensor


@dataclass(frozen=True)
class GaussianParameters:
    prediction: torch.Tensor
    variance: torch.Tensor


@dataclass(frozen=True)
class Point:
    raw_head_width: ClassVar[int] = 1

    def parameterize(self, raw: torch.Tensor) -> PointParameters:
        _validate_raw_head(raw, self.raw_head_width)
        return PointParameters(prediction=raw[..., 0])

    def populate_forecast(
        self,
        parameters: PointParameters,
        *,
        sample_ids: tuple[str, ...],
        input_end_indices: np.ndarray,
        target_fill_mask: np.ndarray,
    ) -> Forecast:
        return Forecast(parameters.prediction, sample_ids, input_end_indices, target_fill_mask)


@dataclass(frozen=True)
class Gaussian:
    raw_head_width: ClassVar[int] = 2

    def parameterize(self, raw: torch.Tensor) -> GaussianParameters:
        _validate_raw_head(raw, self.raw_head_width)
        raw_variance = raw[..., 1]
        return GaussianParameters(
            prediction=raw[..., 0],
            variance=functional.softplus(raw_variance) + torch.finfo(raw_variance.dtype).tiny,
        )

    def populate_forecast(
        self,
        parameters: GaussianParameters,
        *,
        sample_ids: tuple[str, ...],
        input_end_indices: np.ndarray,
        target_fill_mask: np.ndarray,
    ) -> Forecast:
        return Forecast(
            parameters.prediction,
            sample_ids,
            input_end_indices,
            target_fill_mask,
            parameters.variance,
        )
