import numpy as np
import torch

from hcx.batch import Batch, BatchMetadata, GriddedDynamic, GriddedStatic


def make_synthetic_batch(
    *,
    batch_size: int = 4,
    input_length: int = 6,
    output_length: int = 2,
    scalar_dynamic_features: int = 3,
    scalar_static_features: int = 2,
    grid_cells: int = 5,
    gridded_dynamic_features: int = 2,
    gridded_static_features: int = 3,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    seed: int = 1729,
    include_scalar_dynamic: bool = True,
    include_scalar_static: bool = True,
    include_gridded_dynamic: bool = True,
    include_gridded_static: bool = True,
) -> Batch:
    dimensions = {
        "batch_size": batch_size,
        "input_length": input_length,
        "output_length": output_length,
        "grid_cells": grid_cells,
    }
    if invalid := [name for name, value in dimensions.items() if value <= 0]:
        raise ValueError(f"dimensions must be positive: {invalid!r}")
    counts = {
        "scalar_dynamic": scalar_dynamic_features,
        "scalar_static": scalar_static_features,
        "gridded_dynamic": gridded_dynamic_features,
        "gridded_static": gridded_static_features,
    }
    if invalid := [name for name, value in counts.items() if value < 0]:
        raise ValueError(f"feature counts must be nonnegative: {invalid!r}")
    included = {
        "scalar_dynamic": include_scalar_dynamic,
        "scalar_static": include_scalar_static,
        "gridded_dynamic": include_gridded_dynamic,
        "gridded_static": include_gridded_static,
    }
    if invalid := [name for name, enabled in included.items() if enabled and counts[name] == 0]:
        raise ValueError(f"included quadrants must have features: {invalid!r}")

    device = torch.device(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    def randn(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, dtype=dtype, device=device, generator=generator)

    def rand(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.rand(shape, dtype=dtype, device=device, generator=generator)

    scalar_dynamic = randn((batch_size, input_length, scalar_dynamic_features)) if include_scalar_dynamic else None
    scalar_static = randn((batch_size, scalar_static_features)) if include_scalar_static else None

    gridded_dynamic: dict[str, GriddedDynamic] = {}
    coordinates: torch.Tensor | None = None
    padding_mask: torch.Tensor | None = None
    if include_gridded_dynamic:
        values = randn((batch_size, input_length, grid_cells, gridded_dynamic_features))
        coordinates = rand((batch_size, grid_cells, 2))
        padding_mask = torch.zeros((batch_size, grid_cells), dtype=torch.bool, device=device)
        gridded_dynamic["meteorology"] = GriddedDynamic(values, coordinates, padding_mask)

    gridded_static: dict[str, GriddedStatic] = {}
    if include_gridded_static:
        values = randn((batch_size, grid_cells, gridded_static_features))
        if coordinates is None:
            coordinates = rand((batch_size, grid_cells, 2))
            padding_mask = torch.zeros((batch_size, grid_cells), dtype=torch.bool, device=device)
        assert padding_mask is not None
        gridded_static["physiography"] = GriddedStatic(values, coordinates, padding_mask)

    target = randn((batch_size, output_length))
    metadata = BatchMetadata(
        sample_ids=tuple(f"sample-{index:03d}" for index in range(batch_size)),
        input_end_indices=np.full((batch_size,), input_length - 1, dtype=np.int64),
        target_fill_mask=np.zeros((batch_size, output_length), dtype=np.int8),
    )
    return Batch(scalar_dynamic, scalar_static, gridded_dynamic, gridded_static, target, metadata)


__all__ = ["make_synthetic_batch"]
