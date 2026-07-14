# hcx core contract

## 1. Status and terminology

This document is the normative specification of the hcx public contract. README
text and docstrings are explanatory and defer to this document. “Target space”
is the space represented by `Batch.target` after upstream recipe transformations,
including identity.

## 2. Batch and native metadata contract

`GriddedDynamic` has `values [B, T, N, C]`, `coordinates [B, N, 2]`, and boolean
`padding_mask [B, N]`, where true means padding. `GriddedStatic` has `values
[B, N, S_g]` and the same coordinate and padding-mask shapes. `BatchMetadata`
contains stable `sample_ids` of length B, integer `input_end_indices [B]`, and
`target_fill_mask [B, T_out]`, nonzero where a missing target was filled.

`Batch` has exactly four input quadrants: `scalar_dynamic [B, T, F]` or `None`,
`scalar_static [B, S]` or `None`, and the ordered Python mappings
`gridded_dynamic` and `gridded_static`. An empty mapping means an absent gridded
quadrant. It also contains `target [B, T_out]` and native `metadata`. Feature
ordering for scalar quadrants is supplied separately to model factories.

## 3. Forecast contract and target-space semantics

`Forecast.prediction [B, T_out]` is prediction in target space. Optional
`Forecast.variance [B, T_out]` is variance in that same target space. hcx neither
knows nor inverts upstream transformations. The forecast also carries
`sample_ids`, `input_end_indices`, and `target_fill_mask`. A singleton output
horizon remains `[B, 1]`.

## 4. Output specification contract

The runtime-checkable generic `OutputSpecification[ParametersT]` protocol has an
integer `raw_head_width`, `parameterize(raw)`, and `populate_forecast(parameters,
*, sample_ids, input_end_indices, target_fill_mask)`. `Point` consumes raw shape
`[B, T_out, 1]` and selects the first channel. `Gaussian` consumes
`[B, T_out, 2]`, selects its first channel as prediction, and maps its second
through softplus plus the dtype's smallest positive normal value for strictly
positive variance. Invalid rank or final width raises `ValueError` including
expected and actual shapes. Their neutral parameter carriers contain tensors
directly.

## 5. Model protocol

The runtime-checkable `ForecastModel` structural protocol consists solely of
`forward(batch: Batch) -> Forecast`. Runtime checks establish attribute
structure by name, not annotations or behavior.

## 6. Model factory and hcx.models entry points

Model factories are discovered through the frozen `hcx.models` entry-point
group. Their callable signature is:

```python
def factory(
    model_config: dict[str, object],
    *,
    dynamic_inputs: list[str],
    static_inputs: list[str],
    input_size: int,
    static_size: int,
    output_size: int,
    output_specification: OutputSpecification[object],
) -> torch.nn.Module: ...
```

`model_config` contains model-specific configuration only. `dynamic_inputs` and
`static_inputs` are ordered names whose order factories must preserve. The
caller resolves sizes from the first actual batch as
`batch.scalar_dynamic.shape[-1]`, `0 if batch.scalar_static is None else
batch.scalar_static.shape[-1]`, and `batch.target.shape[-1]`. A scalar model
consumer must reject `scalar_dynamic is None` before factory invocation because
`input_size` is then invalid. `output_specification` is already resolved. Names
carry semantic order; sizes record actual first-batch widths and permit
checkpoint reconstruction. The returned module must behave as a `ForecastModel`;
the annotation remains `torch.nn.Module` because training requires module
methods. `ModelFactory` expresses this exact boundary without fallback
parameters, extra keyword arguments, config classes, batches, devices, or
transformation objects.

### Third-party implementations

A third-party model package must depend on `hcx` at runtime and implement a
`torch.nn.Module` whose `forward(batch: Batch) -> Forecast` behavior satisfies
`ForecastModel`. The returned forecast must obey the normative shape,
dtype/device, and metadata identity requirements in this specification.

Its factory must satisfy `ModelFactory`: it accepts model-specific
`dict[str, object]` configuration plus keyword-only ordered `dynamic_inputs` and
`static_inputs`, resolved `input_size`, `static_size`, and `output_size`, and a
resolved `OutputSpecification[object]`, and returns the module. Configuration
is model-specific only. Names must not be reordered, and neither names nor
first-batch-resolved sizes may be recomputed by the factory.

Discovery uses the constant value `hcx.models`. The loaded entry-point object
must satisfy the frozen factory signature above, and the returned module must
meet the model and forecast contract. A model package declares its factory as:

```toml
[project.entry-points.'hcx.models']
my_model = "my_model_package.factory:create_model"
```

Entry-point names are consumer-facing identifiers and should remain stable and
unique within an environment. `scalar_lstm` is hcx's packaged reference entry
point, not a required base class.

`make_synthetic_batch` and `assert_conforms` provide the package-independent
proof workflow. A complete smoke test may load the entry point, or import the
factory directly, then pass ordered names and first-batch-resolved sizes:

```python
from importlib.metadata import entry_points

from hcx import Point, assert_conforms, make_synthetic_batch

batch = make_synthetic_batch()
assert batch.scalar_dynamic is not None
assert batch.scalar_static is not None

factory = entry_points(group="hcx.models", name="my_model")[0].load()
dynamic_inputs = [
    f"dynamic_{index}" for index in range(batch.scalar_dynamic.shape[-1])
]
static_inputs = [
    f"static_{index}" for index in range(batch.scalar_static.shape[-1])
]
model = factory(
    {},
    dynamic_inputs=dynamic_inputs,
    static_inputs=static_inputs,
    input_size=batch.scalar_dynamic.shape[-1],
    static_size=batch.scalar_static.shape[-1],
    output_size=batch.target.shape[-1],
    output_specification=Point(),
)
assert_conforms(model, batch)
```

This workflow requires no external dataset or training package.

## 7. Shapes, dtype, and device

All shapes are specified in sections 2 through 4. Floating tensor dtype is
batch-defined, not globally pinned. Outputs made from raw heads retain the raw
tensor's dtype and device. Masks marked boolean are boolean. All tensors in a
batch are expected to be device-compatible. Carriers perform no implicit tensor
conversion, copying, dtype coercion, device movement, shape repair, or metadata
rebuilding.

## 8. Identity and mutation rules

All carriers are frozen dataclasses. Implementations must not mutate metadata in
place. `Batch.metadata` values are propagated verbatim by identity into a
`Forecast`; specifications likewise pass the supplied metadata objects through
by identity.

## 9. Python and package compatibility

hcx supports CPython 3.11, 3.12, and 3.13 under `requires-python >=3.11`. The
public contract follows semantic versioning. Additive implementations behind
existing protocols may be compatible. Removing or renaming fields; changing
field order, types, or meaning; changing the model-factory signature or
entry-point group; or changing shape, identity, or target-space semantics is a
breaking change.

## 10. Native model stacking

The runtime-checkable `FeatureExtractor` structural protocol consists solely of
`forward(batch: Batch) -> Batch`. As with `ForecastModel`, runtime checks
establish attribute structure by name, not annotations, return types, identity,
or behavior. Its behavioral contract is separate: the returned value must be a
valid `Batch`, must carry the input `target` object and `metadata` object by
identity, and must place the intermediate latent in `scalar_dynamic` with shape
`[B, T, D]`.

The load-bearing design choice: the intermediate representation is **a
`Batch`**. The extractor is `Batch → Batch` — it emits its latent `[B, T, D]`
in the `scalar_dynamic` leg — so every existing `ForecastModel` (including the
reference `ScalarLSTM`) is a legal forecaster with no modification, and no new
latent or forecaster type is introduced. `D` is the latent width occupying the
same scalar-dynamic feature axis that §2 names `F`; the different name denotes
the axis's latent role, not a different shape. The latent rides in
`scalar_dynamic` (shape `[B, T, F]`); no new `Batch` field.

The extractor must honor §8 by preserving `target` and `metadata` by object
identity. The forecaster and its output-specification path must continue §8's
rule: the metadata objects contained in that same metadata carrier must
propagate verbatim by identity into the final `Forecast`, and no stage may
rebuild or mutate them. In this sense, `target`/`metadata` propagate **verbatim
by identity** through extractor → forecaster → `Forecast`. The target travels
through the intermediate `Batch` for use at the forecaster and conformance
boundary; `Forecast` carries the metadata components specified in §§3 and 8,
not a target field.

`ComposedModel` holds one `FeatureExtractor` and one `ForecastModel`. It is an
`nn.Module` that itself satisfies `ForecastModel`, and is the dumb pipe
represented by `forecaster(extractor(batch))`: its implementation explicitly
calls `extractor.forward(batch)` and passes that returned `Batch` directly to
`forecaster.forward(...)`, then returns the resulting `Forecast` unchanged. Its
`consumed_quadrants` property delegates to the extractor and exposes `None`
when the extractor has no such declaration.

Minimal behavioral contract — return a valid `Batch` that carries `target` and
`metadata` **by identity** from the input and populates `scalar_dynamic` with the
latent `[B, T, D]`; all other leg cleanup (clearing gridded, passing vs. clearing
`scalar_static`) is left to the implementation, not mandated. The protocol and
`ComposedModel` do not normalize other legs. Clearing or preserving gridded
mappings and passing or clearing `scalar_static` are extractor implementation
choices, so this contract does not settle scalar-static fusion. The extractor
must return a valid, forecaster-compatible `Batch`; the composition wrapper
does not clean legs or validate the intermediate batch.

Native stacking is additive and non-breaking under §9. It does not change
`Batch`'s fields or four quadrants, `Forecast`, `OutputSpecification`, the
`ModelFactory` callable signature in §6, or the frozen `hcx.models` entry-point
group. It registers no composed model through entry points, adds no second
forecaster role, and mandates no particular extractor implementation.
