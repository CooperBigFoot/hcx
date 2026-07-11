from typing import Protocol, runtime_checkable

import torch

from hcx.batch import Batch
from hcx.output import Forecast
from hcx.specifications import OutputSpecification

MODEL_ENTRY_POINT_GROUP = "hcx.models"


@runtime_checkable
class ForecastModel(Protocol):
    def forward(self, batch: Batch) -> Forecast: ...


@runtime_checkable
class ModelFactory(Protocol):
    def __call__(
        self,
        model_config: dict[str, object],
        *,
        dynamic_inputs: list[str],
        static_inputs: list[str],
        input_size: int,
        static_size: int,
        output_size: int,
        output_specification: OutputSpecification[object],
    ) -> torch.nn.Module: ...
