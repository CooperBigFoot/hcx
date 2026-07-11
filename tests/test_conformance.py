import numpy as np
import pytest
import torch
from torch import nn

from hcx.batch import Batch
from hcx.conformance import ConformanceError, assert_conforms, check_conformance
from hcx.output import Forecast
from hcx.synthetic import make_synthetic_batch

_NAMES = [
    "structural_protocol",
    "consumed_quadrants_present",
    "returns_forecast",
    "prediction_tensor_dtype_device",
    "prediction_shape_exact",
    "identity_sample_ids_verbatim",
    "identity_input_end_indices_verbatim",
    "identity_target_fill_mask_verbatim",
    "finiteness",
    "stable_across_two_eval_forwards",
    "trainability_grad_flow",
]


class Valid(nn.Module):
    def __init__(self, batch: Batch) -> None:
        super().__init__()
        assert batch.scalar_dynamic is not None
        self.linear = nn.Linear(batch.scalar_dynamic.shape[-1], batch.target.shape[-1]).to(
            dtype=batch.target.dtype, device=batch.target.device
        )

    def forward(self, batch: Batch) -> Forecast:
        assert batch.scalar_dynamic is not None
        return Forecast(
            self.linear(batch.scalar_dynamic[:, -1, :]),
            batch.metadata.sample_ids,
            batch.metadata.input_end_indices,
            batch.metadata.target_fill_mask,
        )


def _failed(model: nn.Module, batch: Batch) -> set[str]:
    return {check.name for check in check_conformance(model, batch)[0] if not check.passed}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_valid_model_conforms_and_restores_mode(dtype: torch.dtype) -> None:
    batch = make_synthetic_batch(dtype=dtype)
    model = Valid(batch)
    model.eval()
    checks, forecast = check_conformance(model, batch)
    assert [check.name for check in checks] == _NAMES
    assert all(check.passed for check in checks)
    assert isinstance(forecast, Forecast)
    assert not model.training
    asserted = assert_conforms(model, batch)
    assert isinstance(asserted, Forecast)


class MissingForward(nn.Module):
    pass


class Wrap(Valid):
    def output(
        self,
        batch: Batch,
        prediction: torch.Tensor,
        *,
        sample_ids: tuple[str, ...] | None = None,
        input_end_indices: np.ndarray | None = None,
        target_fill_mask: np.ndarray | None = None,
        variance: object = None,
    ) -> Forecast:
        return Forecast(
            prediction,
            batch.metadata.sample_ids if sample_ids is None else sample_ids,
            batch.metadata.input_end_indices if input_end_indices is None else input_end_indices,
            batch.metadata.target_fill_mask if target_fill_mask is None else target_fill_mask,
            variance,  # ty: ignore[invalid-argument-type]
        )


class Bare(Wrap):
    def forward(self, batch: Batch) -> torch.Tensor:  # ty: ignore[invalid-method-override]
        assert batch.scalar_dynamic is not None
        return self.linear(batch.scalar_dynamic[:, -1, :])


class Raises(Wrap):
    def forward(self, batch: Batch) -> Forecast:
        raise ValueError("boom")


class WrongShape(Wrap):
    def forward(self, batch: Batch) -> Forecast:
        return self.output(batch, super().forward(batch).prediction[:, :1])


class WrongDtype(Wrap):
    def forward(self, batch: Batch) -> Forecast:
        return self.output(batch, super().forward(batch).prediction.double())


class Nonfinite(Wrap):
    def forward(self, batch: Batch) -> Forecast:
        prediction = super().forward(batch).prediction.clone()
        prediction[0, 0] = torch.inf
        return self.output(batch, prediction)


class BadVariance(Wrap):
    variance: object = None

    def forward(self, batch: Batch) -> Forecast:
        out = super().forward(batch)
        return self.output(batch, out.prediction, variance=self.variance)


class Rebuild(Wrap):
    def forward(self, batch: Batch) -> Forecast:
        out = super().forward(batch)
        return self.output(
            batch,
            out.prediction,
            sample_ids=(*out.sample_ids,),
            input_end_indices=np.array(out.input_end_indices),
            target_fill_mask=np.array(out.target_fill_mask),
        )


class Corrupt(Wrap):
    def forward(self, batch: Batch) -> Forecast:
        batch.metadata.input_end_indices[:] = -1
        batch.metadata.target_fill_mask[:] = 1
        object.__setattr__(batch.metadata, "sample_ids", tuple("bad" for _ in batch.metadata.sample_ids))
        return super().forward(batch)


class Unstable(Wrap):
    def __init__(self, batch: Batch) -> None:
        super().__init__(batch)
        self.calls = 0

    def forward(self, batch: Batch) -> Forecast:
        out = super().forward(batch)
        if not self.training:
            self.calls += 1
        return self.output(batch, out.prediction + self.calls)


class Detached(Wrap):
    def forward(self, batch: Batch) -> Forecast:
        return self.output(batch, super().forward(batch).prediction.detach())


def test_missing_forward_fails_output_checks() -> None:
    checks, forecast = check_conformance(MissingForward(), make_synthetic_batch())
    assert forecast is None
    assert [c.name for c in checks] == _NAMES
    assert not next(c for c in checks if c.name == "returns_forecast").passed


@pytest.mark.parametrize("model_type", [Bare, Raises])
def test_absent_eval_output_has_stable_failures(model_type: type[Wrap]) -> None:
    batch = make_synthetic_batch()
    model = model_type(batch)
    checks, forecast = check_conformance(model, batch)
    assert forecast is None and [c.name for c in checks] == _NAMES
    assert {c.name for c in checks if not c.passed} >= {
        "returns_forecast",
        "prediction_tensor_dtype_device",
        "prediction_shape_exact",
        "finiteness",
        "stable_across_two_eval_forwards",
    }
    with pytest.raises(ConformanceError, match="returns_forecast"):
        assert_conforms(model, batch)


@pytest.mark.parametrize(
    ("model_type", "failure"),
    [
        (WrongShape, "prediction_shape_exact"),
        (WrongDtype, "prediction_tensor_dtype_device"),
        (Nonfinite, "finiteness"),
        (Unstable, "stable_across_two_eval_forwards"),
        (Detached, "trainability_grad_flow"),
    ],
)
def test_single_clause_breakers(model_type: type[Wrap], failure: str) -> None:
    batch = make_synthetic_batch()
    assert _failed(model_type(batch), batch) == {failure}


@pytest.mark.parametrize(
    "variance", ["not-a-tensor", torch.zeros(4, 2), -torch.ones(4, 2), torch.full((4, 2), torch.nan)]
)
def test_invalid_variance_fails_finiteness(variance: object) -> None:
    batch = make_synthetic_batch()
    model = BadVariance(batch)
    model.variance = variance
    expected = {"finiteness"}
    if not isinstance(variance, torch.Tensor) or not torch.equal(variance, variance):
        expected.add("stable_across_two_eval_forwards")
    assert _failed(model, batch) == expected


def test_rebuilt_and_corrupted_metadata_fail_identity() -> None:
    identity = {
        "identity_sample_ids_verbatim",
        "identity_input_end_indices_verbatim",
        "identity_target_fill_mask_verbatim",
    }
    batch = make_synthetic_batch()
    assert _failed(Rebuild(batch), batch) == identity
    batch = make_synthetic_batch()
    assert _failed(Corrupt(batch), batch) == identity


def test_consumed_quadrants_validation() -> None:
    class ConsumesGrid(Valid):
        consumed_quadrants = ("gridded_dynamic",)

    class ConsumesUnknown(Valid):
        consumed_quadrants = ("unknown",)

    batch = make_synthetic_batch(include_gridded_dynamic=False)
    absent = ConsumesGrid(batch)
    assert _failed(absent, batch) == {"consumed_quadrants_present"}
    invalid = ConsumesUnknown(batch)
    assert _failed(invalid, batch) == {"consumed_quadrants_present"}
    populated = ConsumesGrid(make_synthetic_batch())
    assert not _failed(populated, make_synthetic_batch())
