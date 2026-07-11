from contextlib import suppress
from dataclasses import dataclass

import numpy as np
import torch

from hcx.batch import Batch
from hcx.output import Forecast
from hcx.protocol import ForecastModel

_SKIPPED = "skipped: forward did not return a Forecast"
_QUADRANTS = ("scalar_dynamic", "scalar_static", "gridded_dynamic", "gridded_static")


@dataclass(frozen=True)
class ConformanceCheck:
    name: str
    passed: bool
    detail: str


class ConformanceError(AssertionError):
    pass


def _consumed_quadrants(model: torch.nn.Module, batch: Batch) -> ConformanceCheck:
    declared = getattr(model, "consumed_quadrants", None)
    if declared is None:
        return ConformanceCheck("consumed_quadrants_present", True, "no consumed quadrants declared")
    populated = {
        "scalar_dynamic": batch.scalar_dynamic is not None,
        "scalar_static": batch.scalar_static is not None,
        "gridded_dynamic": bool(batch.gridded_dynamic),
        "gridded_static": bool(batch.gridded_static),
    }
    try:
        names = tuple(declared)
        invalid = [name for name in names if name not in _QUADRANTS]
        absent = [name for name in names if name in populated and not populated[name]]
    except Exception as exc:
        return ConformanceCheck("consumed_quadrants_present", False, f"declaration iteration failed: {exc!r}")
    if invalid:
        return ConformanceCheck("consumed_quadrants_present", False, f"invalid quadrants: {invalid!r}")
    if absent:
        return ConformanceCheck("consumed_quadrants_present", False, f"absent quadrants: {absent!r}")
    return ConformanceCheck("consumed_quadrants_present", True, "all declared quadrants are populated")


def _identity(name: str, got: object, original: object, snapshot: object) -> ConformanceCheck:
    same_object = got is original
    if isinstance(snapshot, tuple):
        same_value = got == snapshot
    elif isinstance(got, np.ndarray) and isinstance(snapshot, np.ndarray):
        same_value = bool(np.array_equal(got, snapshot))
    else:
        same_value = False
    passed = bool(same_object and same_value)
    detail = "carried verbatim (original object, unchanged values)" if passed else "rebuilt or changed"
    return ConformanceCheck(name, passed, detail)


def _finiteness(out: Forecast) -> ConformanceCheck:
    prediction = out.prediction
    valid = isinstance(prediction, torch.Tensor) and bool(torch.isfinite(prediction).all())
    detail = "prediction is finite"
    if out.variance is not None:
        variance = out.variance
        valid_variance = (
            isinstance(variance, torch.Tensor)
            and isinstance(prediction, torch.Tensor)
            and variance.shape == prediction.shape
            and variance.dtype == prediction.dtype
            and variance.device == prediction.device
            and bool(torch.isfinite(variance).all())
            and bool((variance > 0).all())
        )
        valid = valid and valid_variance
        detail = "prediction finite; variance must match prediction and be finite and strictly positive"
    return ConformanceCheck("finiteness", bool(valid), detail)


def _trainability(model: torch.nn.Module, batch: Batch) -> ConformanceCheck:
    try:
        model.train()
        model.zero_grad(set_to_none=True)
        out = model.forward(batch)
        if not isinstance(out, Forecast):
            return ConformanceCheck("trainability_grad_flow", False, f"train forward returned {type(out).__name__}")
        requires_grad = isinstance(out.prediction, torch.Tensor) and bool(out.prediction.requires_grad)
        if requires_grad:
            out.prediction.sum().backward()
        gradient = any(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            and float(parameter.grad.abs().sum()) > 0
            for parameter in model.parameters()
        )
        return ConformanceCheck(
            "trainability_grad_flow",
            requires_grad and gradient,
            f"requires_grad={requires_grad}, finite nonzero gradient={gradient}",
        )
    except Exception as exc:
        return ConformanceCheck("trainability_grad_flow", False, f"train forward/backward raised {exc!r}")
    finally:
        with suppress(Exception):
            model.zero_grad(set_to_none=True)


def check_conformance(model: torch.nn.Module, batch: Batch) -> tuple[list[ConformanceCheck], Forecast | None]:
    was_training = model.training
    metadata = batch.metadata
    sample_ids = metadata.sample_ids
    input_end_indices = metadata.input_end_indices
    target_fill_mask = metadata.target_fill_mask
    input_end_snapshot = input_end_indices.copy()
    target_fill_snapshot = target_fill_mask.copy()
    checks = [
        ConformanceCheck("structural_protocol", isinstance(model, ForecastModel), "isinstance(model, ForecastModel)"),
        _consumed_quadrants(model, batch),
    ]
    output: Forecast | None = None
    try:
        model.eval()
        try:
            with torch.no_grad():
                produced = model.forward(batch)
            if isinstance(produced, Forecast):
                output = produced
                checks.append(ConformanceCheck("returns_forecast", True, "forward returned Forecast"))
            else:
                checks.append(
                    ConformanceCheck("returns_forecast", False, f"forward returned {type(produced).__name__}")
                )
        except Exception as exc:
            checks.append(ConformanceCheck("returns_forecast", False, f"forward raised {exc!r}"))

        if output is None:
            for name in (
                "prediction_tensor_dtype_device",
                "prediction_shape_exact",
                "identity_sample_ids_verbatim",
                "identity_input_end_indices_verbatim",
                "identity_target_fill_mask_verbatim",
                "finiteness",
            ):
                checks.append(ConformanceCheck(name, False, _SKIPPED))
            checks.append(ConformanceCheck("stable_across_two_eval_forwards", False, _SKIPPED))
        else:
            prediction = output.prediction
            tensor_match = (
                isinstance(prediction, torch.Tensor)
                and prediction.dtype == batch.target.dtype
                and prediction.device == batch.target.device
            )
            checks.append(
                ConformanceCheck("prediction_tensor_dtype_device", tensor_match, "must match target dtype/device")
            )
            shape = tuple(prediction.shape) if isinstance(prediction, torch.Tensor) else None
            expected = (batch.target.shape[0], batch.target.shape[-1])
            checks.append(
                ConformanceCheck("prediction_shape_exact", shape == expected, f"expected {expected}, got {shape}")
            )
            checks.append(_identity("identity_sample_ids_verbatim", output.sample_ids, sample_ids, sample_ids))
            checks.append(
                _identity(
                    "identity_input_end_indices_verbatim",
                    output.input_end_indices,
                    input_end_indices,
                    input_end_snapshot,
                )
            )
            checks.append(
                _identity(
                    "identity_target_fill_mask_verbatim",
                    output.target_fill_mask,
                    target_fill_mask,
                    target_fill_snapshot,
                )
            )
            checks.append(_finiteness(output))
            try:
                model.eval()
                with torch.no_grad():
                    again = model.forward(batch)
                stable = (
                    isinstance(again, Forecast)
                    and torch.equal(output.prediction, again.prediction)
                    and (
                        (output.variance is None and again.variance is None)
                        or (
                            isinstance(output.variance, torch.Tensor)
                            and isinstance(again.variance, torch.Tensor)
                            and torch.equal(output.variance, again.variance)
                        )
                    )
                )
                detail = "adjacent eval forwards equal; cross-run/device determinism is documented, not enforced"
                checks.append(ConformanceCheck("stable_across_two_eval_forwards", bool(stable), detail))
            except Exception as exc:
                checks.append(
                    ConformanceCheck("stable_across_two_eval_forwards", False, f"second eval forward raised {exc!r}")
                )
        checks.append(_trainability(model, batch))
        return checks, output
    finally:
        model.train(was_training)


def assert_conforms(model: torch.nn.Module, batch: Batch) -> Forecast:
    checks, forecast = check_conformance(model, batch)
    for check in checks:
        if not check.passed:
            raise ConformanceError(f"{check.name}: {check.detail}")
    if forecast is None:
        raise ConformanceError("harness error: all checks passed without a forecast")
    return forecast


__all__ = ["ConformanceCheck", "ConformanceError", "assert_conforms", "check_conformance"]
