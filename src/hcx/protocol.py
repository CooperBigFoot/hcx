from typing import Protocol, runtime_checkable

import torch

from hcx.batch import Batch
from hcx.output import Forecast
from hcx.specifications import OutputSpecification

MODEL_ENTRY_POINT_GROUP = "hcx.models"
_EMPTY_GRIDDED_SIZES: dict[str, int] = {}


@runtime_checkable
class ForecastModel(Protocol):
    def forward(self, batch: Batch) -> Forecast: ...


@runtime_checkable
class FeatureExtractor(Protocol):
    def forward(self, batch: Batch) -> Batch: ...


@runtime_checkable
class ModelFactory(Protocol):
    def __call__(
        self,
        model_config: dict[str, object],
        *,
        dynamic_inputs: list[str],
        static_inputs: list[str],
        gridded_dynamic_sizes: dict[str, int] = _EMPTY_GRIDDED_SIZES,
        gridded_static_sizes: dict[str, int] = _EMPTY_GRIDDED_SIZES,
        input_size: int,
        static_size: int,
        output_size: int,
        output_specification: OutputSpecification[object],
    ) -> torch.nn.Module: ...
