import numpy as np
import pytest
import torch

from hcx.synthetic import make_synthetic_batch


def _floating_tensors(batch):
    yield batch.scalar_dynamic
    yield batch.scalar_static
    for grid in (*batch.gridded_dynamic.values(), *batch.gridded_static.values()):
        yield grid.values
        yield grid.coordinates
        yield grid.resolution
    yield batch.target


def _assert_resolution_contract(batch) -> None:
    for grid in (*batch.gridded_dynamic.values(), *batch.gridded_static.values()):
        expected = torch.tensor([0.25, -0.25], dtype=grid.coordinates.dtype, device=grid.coordinates.device)
        assert grid.resolution.shape == (2,)
        assert grid.resolution[0] > 0
        assert grid.resolution[1] < 0
        assert grid.resolution.dtype == grid.coordinates.dtype
        assert grid.resolution.device == grid.coordinates.device
        torch.testing.assert_close(grid.resolution, expected, rtol=0, atol=0)


def test_default_batch_is_exact_and_reproducible() -> None:
    first = make_synthetic_batch()
    second = make_synthetic_batch()
    assert list(first.gridded_dynamic) == ["meteorology"]
    assert list(first.gridded_static) == ["physiography"]
    assert first.gridded_dynamic["meteorology"].coordinates is first.gridded_static["physiography"].coordinates
    assert first.gridded_dynamic["meteorology"].padding_mask is first.gridded_static["physiography"].padding_mask
    for left, right in zip(_floating_tensors(first), _floating_tensors(second), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    np.testing.assert_array_equal(first.metadata.input_end_indices, second.metadata.input_end_indices)
    np.testing.assert_array_equal(first.metadata.target_fill_mask, second.metadata.target_fill_mask)
    assert first.metadata.sample_ids == ("sample-000", "sample-001", "sample-002", "sample-003")
    assert first.scalar_dynamic is not None and first.scalar_static is not None
    assert first.scalar_dynamic.shape == (4, 6, 3)
    assert first.scalar_static.shape == (4, 2)
    dynamic = first.gridded_dynamic["meteorology"]
    static = first.gridded_static["physiography"]
    assert dynamic.values.shape == (4, 6, 5, 2)
    assert dynamic.coordinates.shape == (4, 5, 2)
    assert dynamic.padding_mask.shape == (4, 5)
    assert static.values.shape == (4, 5, 3)
    assert first.target.shape == (4, 2)
    _assert_resolution_contract(first)
    assert all(t.dtype == torch.float32 and t.device.type == "cpu" for t in _floating_tensors(first))
    assert dynamic.padding_mask.dtype == torch.bool
    assert first.metadata.input_end_indices.dtype == np.int64
    assert first.metadata.target_fill_mask.dtype == np.int8
    np.testing.assert_array_equal(first.metadata.input_end_indices, np.full(4, 5, dtype=np.int64))
    np.testing.assert_array_equal(first.metadata.target_fill_mask, np.zeros((4, 2), dtype=np.int8))


def test_seed_and_float64_propagation() -> None:
    first = make_synthetic_batch(seed=1)
    second = make_synthetic_batch(seed=2)
    assert first.scalar_dynamic is not None and second.scalar_dynamic is not None
    assert not torch.equal(first.scalar_dynamic, second.scalar_dynamic)
    assert not torch.equal(first.target, second.target)
    double = make_synthetic_batch(dtype=torch.float64, device="cpu")
    _assert_resolution_contract(double)
    assert all(t.dtype == torch.float64 and t.device.type == "cpu" for t in _floating_tensors(double))


def test_quadrant_absence_options() -> None:
    scalar = make_synthetic_batch(include_gridded_dynamic=False, include_gridded_static=False)
    assert scalar.scalar_dynamic is not None and scalar.scalar_static is not None
    assert scalar.gridded_dynamic == {} and scalar.gridded_static == {}
    gridded = make_synthetic_batch(include_scalar_dynamic=False, include_scalar_static=False)
    assert gridded.scalar_dynamic is None and gridded.scalar_static is None
    assert list(gridded.gridded_dynamic) == ["meteorology"]
    assert list(gridded.gridded_static) == ["physiography"]
    dynamic_only = make_synthetic_batch(include_gridded_static=False)
    assert list(dynamic_only.gridded_dynamic) == ["meteorology"]
    assert dynamic_only.gridded_static == {}
    _assert_resolution_contract(dynamic_only)
    static_only = make_synthetic_batch(include_gridded_dynamic=False)
    assert static_only.gridded_dynamic == {}
    assert list(static_only.gridded_static) == ["physiography"]
    _assert_resolution_contract(static_only)


@pytest.mark.parametrize("name", ["batch_size", "input_length", "output_length", "grid_cells"])
def test_nonpositive_dimensions_are_rejected(name: str) -> None:
    with pytest.raises(ValueError):
        make_synthetic_batch(**{name: 0})  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "name",
    ["scalar_dynamic_features", "scalar_static_features", "gridded_dynamic_features", "gridded_static_features"],
)
def test_negative_feature_counts_are_rejected(name: str) -> None:
    with pytest.raises(ValueError):
        make_synthetic_batch(**{name: -1})  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("count", "flag"),
    [
        ("scalar_dynamic_features", "include_scalar_dynamic"),
        ("scalar_static_features", "include_scalar_static"),
        ("gridded_dynamic_features", "include_gridded_dynamic"),
        ("gridded_static_features", "include_gridded_static"),
    ],
)
def test_included_empty_quadrant_is_rejected(count: str, flag: str) -> None:
    with pytest.raises(ValueError):
        make_synthetic_batch(**{count: 0, flag: True})  # ty: ignore[invalid-argument-type]
