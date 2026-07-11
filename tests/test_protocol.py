import inspect

import numpy as np
import torch

import hcx
from hcx import Batch, BatchMetadata, Forecast, ForecastModel, ModelFactory, OutputSpecification


class MinimalModel(torch.nn.Module):
    def forward(self, batch: Batch) -> Forecast:
        return Forecast(
            batch.target,
            batch.metadata.sample_ids,
            batch.metadata.input_end_indices,
            batch.metadata.target_fill_mask,
        )


def factory(
    model_config: dict[str, object],
    *,
    dynamic_inputs: list[str],
    static_inputs: list[str],
    input_size: int,
    static_size: int,
    output_size: int,
    output_specification: OutputSpecification[object],
) -> torch.nn.Module:
    del model_config, dynamic_inputs, static_inputs, input_size, static_size, output_size, output_specification
    return MinimalModel()


def test_model_and_factory_protocols_are_implementable() -> None:
    metadata = BatchMetadata(("a",), np.array([2]), np.zeros((1, 1), dtype=np.int8))
    batch = Batch(torch.randn(1, 2, 3), None, {}, {}, torch.randn(1, 1), metadata)
    model = MinimalModel()
    assert isinstance(model, ForecastModel)
    assert isinstance(factory, ModelFactory)
    assert isinstance(model.forward(batch), Forecast)


def test_public_surface_and_factory_signature_are_frozen() -> None:
    assert hcx.MODEL_ENTRY_POINT_GROUP == "hcx.models"
    expected = inspect.signature(factory)
    actual_parameters = list(inspect.signature(ModelFactory.__call__).parameters.values())[1:]
    assert tuple(actual_parameters) == tuple(expected.parameters.values())
    assert inspect.signature(ModelFactory.__call__).return_annotation == expected.return_annotation
    for name in hcx.__all__:
        assert hasattr(hcx, name)
