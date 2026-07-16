"""A scalar-input LSTM reference model."""

import math
from dataclasses import dataclass, fields
from numbers import Real

import torch
from torch import nn

from hcx.batch import Batch
from hcx.output import Forecast
from hcx.specifications import OutputSpecification

_EMPTY_GRIDDED_SIZES: dict[str, int] = {}


@dataclass(frozen=True)
class LSTMConfig:
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    decoder_hidden_size: int = 32


def _parse_config(values: dict[str, object]) -> LSTMConfig:
    allowed = {field.name for field in fields(LSTMConfig)}
    for key, value in values.items():
        if key not in allowed:
            raise ValueError(f"unknown model config key {key!r} with value {value!r}")

    defaults = LSTMConfig()
    for key in ("hidden_size", "num_layers", "decoder_hidden_size"):
        value = values.get(key, getattr(defaults, key))
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer; got {value!r}")
    dropout_value = values.get("dropout", defaults.dropout)
    if isinstance(dropout_value, bool) or not isinstance(dropout_value, Real):
        raise ValueError(f"dropout must be a finite real number in [0, 1); got {dropout_value!r}")
    dropout = float(dropout_value)
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError(f"dropout must be a finite real number in [0, 1); got {dropout_value!r}")
    hidden_size = values.get("hidden_size", defaults.hidden_size)
    num_layers = values.get("num_layers", defaults.num_layers)
    decoder_hidden_size = values.get("decoder_hidden_size", defaults.decoder_hidden_size)
    assert isinstance(hidden_size, int) and not isinstance(hidden_size, bool)
    assert isinstance(num_layers, int) and not isinstance(num_layers, bool)
    assert isinstance(decoder_hidden_size, int) and not isinstance(decoder_hidden_size, bool)
    return LSTMConfig(
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        decoder_hidden_size=decoder_hidden_size,
    )


def _validate_size(key: str, value: object, *, allow_zero: bool = False) -> None:
    valid_range = value >= 0 if isinstance(value, int) else False
    if isinstance(value, bool) or not isinstance(value, int) or not valid_range or (not allow_zero and value == 0):
        requirement = "a nonnegative integer" if allow_zero else "a positive integer"
        raise ValueError(f"{key} must be {requirement}; got {value!r}")


def _validate_names(key: str, names: list[str], size: int) -> tuple[str, ...]:
    if len(names) != size:
        raise ValueError(f"{key} count must equal its supplied size {size}; got {len(names)}")
    for name in names:
        if not isinstance(name, str):
            raise ValueError(f"{key} names must be strings; got {name!r}")
    if len(set(names)) != len(names):
        raise ValueError(f"{key} names must be unique; got {names!r}")
    return tuple(names)


class ScalarLSTM(nn.Module):
    def __init__(
        self,
        config: LSTMConfig,
        input_size: int,
        static_size: int,
        output_size: int,
        dynamic_inputs: tuple[str, ...],
        static_inputs: tuple[str, ...],
        output_specification: OutputSpecification[object],
    ) -> None:
        super().__init__()
        self.config = config
        self.input_size = input_size
        self.static_size = static_size
        self.output_size = output_size
        self.dynamic_inputs = dynamic_inputs
        self.static_inputs = static_inputs
        self.output_specification = output_specification
        self.consumed_quadrants = ("scalar_dynamic",) if static_size == 0 else ("scalar_dynamic", "scalar_static")

        self.embedding = nn.Linear(input_size + static_size, config.hidden_size)
        self.lstm = nn.LSTM(
            config.hidden_size,
            config.hidden_size,
            config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True,
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.decoder_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.decoder_hidden_size, output_size * output_specification.raw_head_width),
        )

    def forward(self, batch: Batch) -> Forecast:
        x = batch.scalar_dynamic
        if x is None:
            raise ValueError("scalar dynamic input is required")
        if x.shape[-1] != self.input_size:
            raise ValueError(f"scalar dynamic width must be {self.input_size}; got {x.shape[-1]}")
        batch_size, input_length, _ = x.shape

        if self.static_size > 0:
            static = batch.scalar_static
            if static is None:
                raise ValueError(f"scalar static input of width {self.static_size} is required")
            if static.shape[-1] != self.static_size:
                raise ValueError(f"scalar static width must be {self.static_size}; got {static.shape[-1]}")
            x = torch.cat([x, static.unsqueeze(1).expand(-1, input_length, -1)], dim=-1)
        elif batch.scalar_static is not None:
            raise ValueError("scalar static input was supplied when static_size is zero")

        embedded = self.embedding(x)
        recurrent, _ = self.lstm(embedded)
        decoded = self.decoder(recurrent[:, -1, :])
        raw = decoded.view(batch_size, self.output_size, self.output_specification.raw_head_width)
        parameters = self.output_specification.parameterize(raw)
        return self.output_specification.populate_forecast(
            parameters,
            sample_ids=batch.metadata.sample_ids,
            input_end_indices=batch.metadata.input_end_indices,
            target_fill_mask=batch.metadata.target_fill_mask,
        )


def factory(
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
) -> torch.nn.Module:
    del gridded_dynamic_sizes, gridded_static_sizes
    _validate_size("input_size", input_size)
    _validate_size("static_size", static_size, allow_zero=True)
    _validate_size("output_size", output_size)
    dynamic_names = _validate_names("dynamic_inputs", dynamic_inputs, input_size)
    static_names = _validate_names("static_inputs", static_inputs, static_size)
    return ScalarLSTM(
        _parse_config(model_config),
        input_size,
        static_size,
        output_size,
        dynamic_names,
        static_names,
        output_specification,
    )


__all__ = ["LSTMConfig", "ScalarLSTM", "factory"]
