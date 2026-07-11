import importlib.metadata

import torch

import hcx.models.lstm
from hcx.conformance import assert_conforms
from hcx.specifications import Gaussian, Point
from hcx.synthetic import make_synthetic_batch


def test_installed_scalar_lstm_entry_point():
    entry_points = importlib.metadata.entry_points(group="hcx.models")
    matching = [item for item in entry_points if item.name == "scalar_lstm"]
    assert len(matching) == 1
    entry_point = next(item for item in entry_points if item.name == "scalar_lstm")
    assert entry_point.value == "hcx.models.lstm:factory"
    loaded_factory = entry_point.load()
    assert loaded_factory is hcx.models.lstm.factory

    batch = make_synthetic_batch(
        include_gridded_dynamic=False,
        include_gridded_static=False,
        seed=1618,
    )
    assert batch.scalar_dynamic is not None
    assert batch.scalar_static is not None
    for seed, specification in [(11, Point()), (12, Gaussian())]:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            model = loaded_factory(
                {},
                dynamic_inputs=["precipitation", "temperature", "pet"],
                static_inputs=["elevation", "area"],
                input_size=batch.scalar_dynamic.shape[-1],
                static_size=batch.scalar_static.shape[-1],
                output_size=batch.target.shape[-1],
                output_specification=specification,
            )
        assert_conforms(model, batch)
