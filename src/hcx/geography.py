"""geographic grid semantics = coordinate order × CRS × unit × cell reference × resolution convention."""

from dataclasses import dataclass
from enum import Enum


class CoordinateOrder(Enum):
    """The semantic order of the final coordinate tensor axis."""

    LONGITUDE_LATITUDE = "longitude_latitude"


class CoordinateReferenceSystem(Enum):
    """The coordinate reference system in which coordinates are expressed."""

    EPSG_4326 = "EPSG:4326"


class CoordinateUnit(Enum):
    """The unit shared by coordinate and resolution tensor components."""

    DEGREE = "degree"


class CellReferenceConvention(Enum):
    """The point within each represented cell stored in the coordinate tensor."""

    CENTER = "center"


class SignedResolutionConvention(Enum):
    """The meaning of each signed component of the resolution tensor."""

    AXIS_ALIGNED_CELL_EXTENT_WITH_AXIS_DIRECTION = "axis_aligned_cell_extent_with_axis_direction"


@dataclass(frozen=True)
class GeographicGridGeometry:
    """Typed interpretation of a gridded leg's coordinate and resolution tensors.

    Under ``AXIS_ALIGNED_CELL_EXTENT_WITH_AXIS_DIRECTION``, each resolution
    component corresponds to the coordinate component in the same position. Its
    absolute value is the cell extent in ``coordinate_unit``. Its sign is the
    native raster axis direction.
    """

    coordinate_order: CoordinateOrder
    crs: CoordinateReferenceSystem
    coordinate_unit: CoordinateUnit
    cell_reference: CellReferenceConvention
    resolution_convention: SignedResolutionConvention

    def __post_init__(self) -> None:
        members: tuple[tuple[str, object, type[Enum]], ...] = (
            ("coordinate_order", self.coordinate_order, CoordinateOrder),
            ("crs", self.crs, CoordinateReferenceSystem),
            ("coordinate_unit", self.coordinate_unit, CoordinateUnit),
            ("cell_reference", self.cell_reference, CellReferenceConvention),
            ("resolution_convention", self.resolution_convention, SignedResolutionConvention),
        )
        for name, value, enum_type in members:
            if not isinstance(value, enum_type):
                raise TypeError(f"{name} must be {enum_type.__name__}; got {value!r}")


__all__ = [
    "CellReferenceConvention",
    "CoordinateOrder",
    "CoordinateReferenceSystem",
    "CoordinateUnit",
    "GeographicGridGeometry",
    "SignedResolutionConvention",
]
