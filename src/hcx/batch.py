from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class GriddedDynamic:
    values: torch.Tensor
    coordinates: torch.Tensor
    padding_mask: torch.Tensor
    resolution: torch.Tensor


@dataclass(frozen=True)
class GriddedStatic:
    values: torch.Tensor
    coordinates: torch.Tensor
    padding_mask: torch.Tensor
    resolution: torch.Tensor


@dataclass(frozen=True)
class BatchMetadata:
    sample_ids: tuple[str, ...]
    input_end_indices: np.ndarray
    target_fill_mask: np.ndarray


@dataclass(frozen=True)
class Batch:
    scalar_dynamic: torch.Tensor | None
    scalar_static: torch.Tensor | None
    gridded_dynamic: dict[str, GriddedDynamic]
    gridded_static: dict[str, GriddedStatic]
    target: torch.Tensor
    metadata: BatchMetadata
