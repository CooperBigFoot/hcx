import types

import hcx
from hcx import (
    MODEL_ENTRY_POINT_GROUP,
    Batch,
    BatchMetadata,
    ComposedModel,
    ConformanceCheck,
    ConformanceError,
    FeatureExtractor,
    Forecast,
    ForecastModel,
    Gaussian,
    GaussianParameters,
    GriddedDynamic,
    GriddedStatic,
    ModelFactory,
    OutputSpecification,
    Point,
    PointParameters,
    __version__,
    assert_conforms,
    check_conformance,
    make_synthetic_batch,
)

EXPECTED_PUBLIC_API = [
    "MODEL_ENTRY_POINT_GROUP",
    "Batch",
    "BatchMetadata",
    "ComposedModel",
    "ConformanceCheck",
    "ConformanceError",
    "FeatureExtractor",
    "Forecast",
    "ForecastModel",
    "Gaussian",
    "GaussianParameters",
    "GriddedDynamic",
    "GriddedStatic",
    "ModelFactory",
    "OutputSpecification",
    "Point",
    "PointParameters",
    "assert_conforms",
    "check_conformance",
    "make_synthetic_batch",
    "__version__",
]


def test_supported_names_are_explicitly_importable() -> None:
    imported_names = (
        MODEL_ENTRY_POINT_GROUP,
        Batch,
        BatchMetadata,
        ComposedModel,
        ConformanceCheck,
        ConformanceError,
        FeatureExtractor,
        Forecast,
        ForecastModel,
        Gaussian,
        GaussianParameters,
        GriddedDynamic,
        GriddedStatic,
        ModelFactory,
        OutputSpecification,
        Point,
        PointParameters,
        assert_conforms,
        check_conformance,
        make_synthetic_batch,
        __version__,
    )
    assert len(imported_names) == len(EXPECTED_PUBLIC_API)
    assert hcx.__all__ == EXPECTED_PUBLIC_API
    assert hasattr(hcx, "__version__")
    for name in EXPECTED_PUBLIC_API:
        assert getattr(hcx, name) is not None


def test_public_api_is_complete() -> None:
    excluded_module_bindings = {
        name
        for name, value in vars(hcx).items()
        if isinstance(value, types.ModuleType) and value.__package__.split(".", maxsplit=1)[0] == "hcx"
    }
    derived_public_api = {name for name in vars(hcx) if not name.startswith("_")} - excluded_module_bindings
    assert derived_public_api == set(EXPECTED_PUBLIC_API) - {"__version__"}
