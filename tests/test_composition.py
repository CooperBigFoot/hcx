import torch
from torch import nn

from hcx import (
    Batch,
    ComposedModel,
    FeatureExtractor,
    ForecastModel,
    Point,
    assert_conforms,
    make_synthetic_batch,
)
from hcx.models.lstm import LSTMConfig, ScalarLSTM


class PoolingExtractor(nn.Module):
    consumed_quadrants = ("gridded_dynamic",)

    def forward(self, batch: Batch) -> Batch:
        grid = batch.gridded_dynamic["meteorology"]
        return Batch(
            scalar_dynamic=grid.values.mean(dim=2),
            scalar_static=None,
            gridded_dynamic={},
            gridded_static={},
            target=batch.target,
            metadata=batch.metadata,
        )


class UndeclaredExtractor(nn.Module):
    def forward(self, batch: Batch) -> Batch:
        return batch


def _batch() -> Batch:
    return make_synthetic_batch(
        batch_size=4,
        input_length=6,
        output_length=2,
        grid_cells=5,
        gridded_dynamic_features=3,
        include_scalar_dynamic=False,
        include_scalar_static=False,
        include_gridded_dynamic=True,
        include_gridded_static=False,
        seed=314,
    )


def _forecaster() -> ScalarLSTM:
    return ScalarLSTM(
        LSTMConfig(hidden_size=8, decoder_hidden_size=5),
        input_size=3,
        static_size=0,
        output_size=2,
        dynamic_inputs=("latent-0", "latent-1", "latent-2"),
        static_inputs=(),
        output_specification=Point(),  # ty: ignore[invalid-argument-type]
    )


def test_pooling_extractor_composes_with_scalar_lstm():
    batch = _batch()
    extractor = PoolingExtractor()
    extracted = extractor(batch)

    assert isinstance(extractor, FeatureExtractor)
    assert extracted.scalar_dynamic is not None
    assert extracted.scalar_dynamic.shape == (4, 6, 3)
    torch.testing.assert_close(
        extracted.scalar_dynamic,
        batch.gridded_dynamic["meteorology"].values.mean(dim=2),
    )
    assert extracted.scalar_static is None
    assert extracted.gridded_dynamic == {}
    assert extracted.gridded_static == {}
    assert extracted.target is batch.target
    assert extracted.metadata is batch.metadata

    composed = ComposedModel(extractor, _forecaster())
    assert isinstance(composed, ForecastModel)
    assert composed.consumed_quadrants == ("gridded_dynamic",)

    forecast = assert_conforms(composed, batch)

    assert forecast.prediction.shape == (4, 2)
    assert forecast.sample_ids is batch.metadata.sample_ids
    assert forecast.input_end_indices is batch.metadata.input_end_indices
    assert forecast.target_fill_mask is batch.metadata.target_fill_mask


def test_composed_model_without_declared_quadrants_falls_back_to_none():
    extractor = UndeclaredExtractor()
    assert isinstance(extractor, FeatureExtractor)

    composed = ComposedModel(extractor, _forecaster())

    assert composed.consumed_quadrants is None
