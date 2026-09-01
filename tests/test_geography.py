from dataclasses import FrozenInstanceError

import pytest

from hcx import (
    CellReferenceConvention,
    CoordinateOrder,
    CoordinateReferenceSystem,
    CoordinateUnit,
    GeographicGridGeometry,
    SignedResolutionConvention,
)


def test_epsg_4326_cell_center_geometry_is_typed_and_frozen() -> None:
    geography = GeographicGridGeometry(
        coordinate_order=CoordinateOrder.LONGITUDE_LATITUDE,
        crs=CoordinateReferenceSystem.EPSG_4326,
        coordinate_unit=CoordinateUnit.DEGREE,
        cell_reference=CellReferenceConvention.CENTER,
        resolution_convention=SignedResolutionConvention.AXIS_ALIGNED_CELL_EXTENT_WITH_AXIS_DIRECTION,
    )

    assert geography.coordinate_order is CoordinateOrder.LONGITUDE_LATITUDE
    assert geography.crs is CoordinateReferenceSystem.EPSG_4326
    assert geography.coordinate_unit is CoordinateUnit.DEGREE
    assert geography.cell_reference is CellReferenceConvention.CENTER
    assert geography.resolution_convention is SignedResolutionConvention.AXIS_ALIGNED_CELL_EXTENT_WITH_AXIS_DIRECTION
    with pytest.raises(FrozenInstanceError):
        geography.crs = CoordinateReferenceSystem.EPSG_4326  # ty: ignore[invalid-assignment]


def test_grid_geometry_rejects_raw_metadata_values() -> None:
    with pytest.raises(TypeError, match="coordinate_order"):
        GeographicGridGeometry(
            coordinate_order="longitude_latitude",  # ty: ignore[invalid-argument-type]
            crs=CoordinateReferenceSystem.EPSG_4326,
            coordinate_unit=CoordinateUnit.DEGREE,
            cell_reference=CellReferenceConvention.CENTER,
            resolution_convention=SignedResolutionConvention.AXIS_ALIGNED_CELL_EXTENT_WITH_AXIS_DIRECTION,
        )
