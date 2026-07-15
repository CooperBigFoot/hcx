from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch

from hcx import Batch, BatchMetadata, GriddedDynamic, GriddedStatic


def test_native_batch_carriers_preserve_supplied_objects_and_are_frozen() -> None:
    scalar_dynamic = torch.randn(2, 3, 4)
    scalar_static = torch.randn(2, 5)
    coordinates = torch.randn(2, 6, 2)
    padding_mask = torch.zeros(2, 6, dtype=torch.bool)
    resolution = torch.tensor([0.25, -0.25], dtype=coordinates.dtype, device=coordinates.device)
    dynamic = GriddedDynamic(torch.randn(2, 3, 6, 1), coordinates, padding_mask, resolution)
    static = GriddedStatic(torch.randn(2, 6, 2), dynamic.coordinates, dynamic.padding_mask, resolution)
    dynamic_map = {"rain": dynamic}
    static_map = {"terrain": static}
    sample_ids = ("a", "b")
    input_end_indices = np.array([2, 2])
    target_fill_mask = np.zeros((2, 1), dtype=np.int8)
    metadata = BatchMetadata(sample_ids, input_end_indices, target_fill_mask)
    target = torch.randn(2, 1)
    batch = Batch(scalar_dynamic, scalar_static, dynamic_map, static_map, target, metadata)

    assert batch.scalar_dynamic is scalar_dynamic
    assert batch.scalar_static is scalar_static
    assert batch.gridded_dynamic is dynamic_map
    assert batch.gridded_static is static_map
    assert batch.target is target
    assert batch.metadata is metadata
    assert metadata.sample_ids is sample_ids
    assert metadata.input_end_indices is input_end_indices
    assert metadata.target_fill_mask is target_fill_mask
    assert dynamic.coordinates is static.coordinates
    assert dynamic.padding_mask is static.padding_mask
    assert dynamic.resolution is resolution
    assert static.resolution is resolution
    field_name = "target"
    with pytest.raises(FrozenInstanceError):
        setattr(batch, field_name, torch.empty(0))


@pytest.mark.parametrize(
    ("carrier", "values_shape"),
    [(GriddedDynamic, (2, 3, 6, 1)), (GriddedStatic, (2, 6, 2))],
)
def test_gridded_carriers_require_resolution(carrier, values_shape: tuple[int, ...]) -> None:
    values = torch.randn(values_shape)
    coordinates = torch.randn(2, 6, 2)
    padding_mask = torch.zeros(2, 6, dtype=torch.bool)

    with pytest.raises(TypeError):
        carrier(values, coordinates, padding_mask)


@pytest.mark.parametrize(
    ("carrier", "values_shape"),
    [(GriddedDynamic, (2, 3, 6, 1)), (GriddedStatic, (2, 6, 2))],
)
def test_gridded_carriers_preserve_supplied_resolution(carrier, values_shape: tuple[int, ...]) -> None:
    values = torch.randn(values_shape)
    coordinates = torch.randn(2, 6, 2)
    padding_mask = torch.zeros(2, 6, dtype=torch.bool)
    resolution = torch.tensor([0.25, -0.25], dtype=coordinates.dtype, device=coordinates.device)

    leg = carrier(values, coordinates, padding_mask, resolution)

    assert leg.resolution is resolution


def test_absent_quadrants_and_no_compatibility_aliases() -> None:
    metadata = BatchMetadata((), np.array([], dtype=int), np.empty((0, 1), dtype=np.int8))
    batch = Batch(None, None, {}, {}, torch.empty((0, 1)), metadata)
    assert batch.scalar_dynamic is None
    assert batch.scalar_static is None
    assert batch.gridded_dynamic == {}
    assert batch.gridded_static == {}
    for name in ("x", "static", "future"):
        assert not hasattr(batch, name)
