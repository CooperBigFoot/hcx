from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch

from hcx import Forecast


def test_forecast_is_target_space_neutral_frozen_and_preserves_identity() -> None:
    prediction = torch.tensor([[4.0], [8.0]])
    sample_ids = ("a", "b")
    input_end_indices = np.array([1, 3])
    target_fill_mask = np.array([[0], [1]], dtype=np.int8)
    forecast = Forecast(prediction, sample_ids, input_end_indices, target_fill_mask)

    torch.testing.assert_close(forecast.prediction, prediction)
    assert forecast.prediction.shape == (2, 1)
    assert forecast.variance is None
    assert forecast.sample_ids is sample_ids
    assert forecast.input_end_indices is input_end_indices
    assert forecast.target_fill_mask is target_fill_mask
    np.testing.assert_array_equal(forecast.target_fill_mask, target_fill_mask)
    field_name = "prediction"
    with pytest.raises(FrozenInstanceError):
        setattr(forecast, field_name, torch.empty(0))
