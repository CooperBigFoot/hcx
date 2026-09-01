from dataclasses import dataclass

import numpy as np
import torch

from hcx.geography import GeographicGridGeometry


def _require_geography(geography: object) -> None:
    if not isinstance(geography, GeographicGridGeometry):
        raise TypeError(f"geography must be GeographicGridGeometry; got {geography!r}")


@dataclass(frozen=True)
class GriddedDynamic:
    values: torch.Tensor
    coordinates: torch.Tensor
    padding_mask: torch.Tensor
    resolution: torch.Tensor
    geography: GeographicGridGeometry

    def __post_init__(self) -> None:
        _require_geography(self.geography)


@dataclass(frozen=True)
class GriddedStatic:
    values: torch.Tensor
    coordinates: torch.Tensor
    padding_mask: torch.Tensor
    resolution: torch.Tensor
    geography: GeographicGridGeometry

    def __post_init__(self) -> None:
        _require_geography(self.geography)


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
