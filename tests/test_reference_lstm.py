import pytest
import torch

from hcx.conformance import assert_conforms
from hcx.models.lstm import ScalarLSTM, factory
from hcx.specifications import Gaussian, Point
from hcx.synthetic import make_synthetic_batch


def _batch(
    *,
    scalar_dynamic_features: int = 3,
    scalar_static_features: int = 2,
    include_scalar_dynamic: bool = True,
):
    return make_synthetic_batch(
        scalar_dynamic_features=scalar_dynamic_features,
        scalar_static_features=scalar_static_features,
        include_scalar_dynamic=include_scalar_dynamic,
        include_gridded_dynamic=False,
        include_gridded_static=False,
        seed=314,
    )


def _model(batch, specification, config=None, **overrides):
    assert batch.scalar_dynamic is not None
    assert batch.scalar_static is not None
    arguments = {
        "dynamic_inputs": ["precipitation", "temperature", "pet"],
        "static_inputs": ["elevation", "area"],
        "input_size": batch.scalar_dynamic.shape[-1],
        "static_size": batch.scalar_static.shape[-1],
        "output_size": batch.target.shape[-1],
        "output_specification": specification,
    }
    arguments.update(overrides)
    with torch.random.fork_rng():
        torch.manual_seed(2718)
        return factory(config or {}, **arguments)


@pytest.mark.parametrize("specification", [Point(), Gaussian()])
def test_reference_lstm_conforms(specification):
    batch = _batch()
    model = _model(
        batch,
        specification,
        {"hidden_size": 8, "num_layers": 2, "dropout": 0.1, "decoder_hidden_size": 5},
    )
    assert isinstance(model, ScalarLSTM)
    forecast = assert_conforms(model, batch)
    assert forecast.prediction.shape == batch.target.shape
    assert forecast.sample_ids is batch.metadata.sample_ids
    assert forecast.input_end_indices is batch.metadata.input_end_indices
    assert forecast.target_fill_mask is batch.metadata.target_fill_mask
    if isinstance(specification, Point):
        assert forecast.variance is None
    else:
        assert forecast.variance is not None
        assert forecast.variance.shape == forecast.prediction.shape
        assert forecast.variance.dtype == forecast.prediction.dtype
        assert forecast.variance.device == forecast.prediction.device
        assert torch.isfinite(forecast.variance).all()
        assert (forecast.variance > 0).all()


def test_resolved_context_controls_names_and_layer_widths():
    batch = _batch()
    model = _model(batch, Point(), {"hidden_size": 7, "decoder_hidden_size": 4})
    assert isinstance(model, ScalarLSTM)
    assert model.dynamic_inputs == ("precipitation", "temperature", "pet")
    assert model.static_inputs == ("elevation", "area")
    assert model.embedding.in_features == 5
    assert model.embedding.out_features == 7
    assert model.decoder[-1].out_features == batch.target.shape[-1]


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({"mystery": 1}, "mystery"),
        ({"hidden_size": True}, "hidden_size"),
        ({"num_layers": 0}, "num_layers"),
        ({"decoder_hidden_size": -1}, "decoder_hidden_size"),
        ({"dropout": float("nan")}, "dropout"),
        ({"dropout": 1.0}, "dropout"),
    ],
)
def test_invalid_model_config(config, match):
    with pytest.raises(ValueError, match=match):
        _model(_batch(), Point(), config)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"input_size": True}, "input_size"),
        ({"static_size": -1}, "static_size"),
        ({"output_size": 0}, "output_size"),
        ({"dynamic_inputs": ["one"]}, "dynamic_inputs"),
        ({"static_inputs": ["same", "same"]}, "static_inputs"),
        ({"dynamic_inputs": ["one", "two", 3]}, "dynamic_inputs"),
    ],
)
def test_invalid_resolved_context(overrides, match):
    with pytest.raises(ValueError, match=match):
        _model(_batch(), Point(), **overrides)


def test_absent_required_scalar_input_fails():
    construction_batch = _batch()
    model = _model(construction_batch, Point())
    runtime_batch = _batch(include_scalar_dynamic=False)
    with pytest.raises(ValueError, match="scalar dynamic input is required"):
        model(runtime_batch)


@pytest.mark.parametrize(
    ("feature", "model_overrides", "batch_kwargs", "match"),
    [
        ("dynamic", {}, {"scalar_dynamic_features": 4}, "scalar dynamic width"),
        ("static", {}, {"scalar_static_features": 3}, "scalar static width"),
        ("unexpected_static", {"static_inputs": [], "static_size": 0}, {}, "static_size is zero"),
    ],
)
def test_runtime_feature_width_mismatch(feature, model_overrides, batch_kwargs, match):
    del feature
    model = _model(_batch(), Point(), **model_overrides)
    with pytest.raises(ValueError, match=match):
        model(_batch(**batch_kwargs))
