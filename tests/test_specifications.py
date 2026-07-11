from typing import Any

import numpy as np
import pytest
import torch

from hcx import Gaussian, OutputSpecification, Point


@pytest.mark.parametrize("specification,width", [(Point(), 1), (Gaussian(), 2)])
def test_specifications_parameterize_and_populate(specification: Any, width: int) -> None:
    raw = torch.randn(2, 1, width, dtype=torch.float64)
    parameters = specification.parameterize(raw)
    sample_ids = ("a", "b")
    input_end_indices = np.array([2, 4])
    target_fill_mask = np.zeros((2, 1), dtype=np.int8)
    forecast = specification.populate_forecast(
        parameters,
        sample_ids=sample_ids,
        input_end_indices=input_end_indices,
        target_fill_mask=target_fill_mask,
    )

    torch.testing.assert_close(forecast.prediction, raw[..., 0])
    assert forecast.prediction.shape == (2, 1)
    assert forecast.prediction.dtype == raw.dtype
    assert forecast.prediction.device == raw.device
    assert forecast.sample_ids is sample_ids
    assert forecast.input_end_indices is input_end_indices
    assert forecast.target_fill_mask is target_fill_mask
    if isinstance(specification, Gaussian):
        assert forecast.variance is not None
        assert forecast.variance.shape == (2, 1)
        assert forecast.variance.dtype == raw.dtype
        assert forecast.variance.device == raw.device
        assert torch.all(forecast.variance > 0)
    else:
        assert forecast.variance is None
    assert isinstance(specification, OutputSpecification)


@pytest.mark.parametrize(
    ("specification", "raw", "expected"),
    [
        (Point(), torch.randn(2, 1), "[B, T_out, 1]"),
        (Point(), torch.randn(2, 1, 2), "[B, T_out, 1]"),
        (Gaussian(), torch.randn(2, 1), "[B, T_out, 2]"),
        (Gaussian(), torch.randn(2, 1, 1), "[B, T_out, 2]"),
    ],
)
def test_invalid_raw_shapes(specification: Point | Gaussian, raw: torch.Tensor, expected: str) -> None:
    with pytest.raises(ValueError, match="expected shape") as error:
        specification.parameterize(raw)
    assert expected in str(error.value)
    assert str(tuple(raw.shape)) in str(error.value)
