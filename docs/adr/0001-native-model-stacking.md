# ADR-0001: Native model stacking

## Status

Accepted

## Context

hcx historically typed every model as a self-contained end-to-end
`Batch -> Forecast`, leaving no native way to place a downstream forecaster on
an upstream feature extractor. The immediate Perceiver program and ET1–ET5
require a stable composition contract. This decision supplies neither the
Perceiver itself nor application wiring; it records the package-level boundary
that their later implementation can use.

## Decision

hcx expands from supporting only self-contained models to native stacking via
the minimum two primitives already implemented. The runtime-checkable
`FeatureExtractor` protocol consists of `forward(batch: Batch) -> Batch`, and
`ComposedModel` is the dumb pipe `forecaster(extractor(batch))`, implemented
with explicit calls to each component's `forward` method.

The load-bearing design choice: the intermediate representation is **a
`Batch`**. The extractor is `Batch → Batch` — it emits its latent `[B, T, D]`
in the `scalar_dynamic` leg — so every existing `ForecastModel` (including the
reference `ScalarLSTM`) is a legal forecaster with no modification, and no new
latent or forecaster type is introduced. `D` names the latent width on the
existing scalar-dynamic feature axis. No new `Batch` field, bespoke latent type,
or alternate forecaster role is added.

The extractor must return a valid `Batch` carrying `target` and `metadata` by
identity. The downstream forecaster and output-specification path retain the
existing metadata identity rules. `ComposedModel.consumed_quadrants` delegates
to the extractor and falls back to `None` when the extractor has no such
attribute. Cleanup of gridded legs and preservation or clearing of
`scalar_static` belong to extractor implementations. Factory signatures and
the model discovery entry-point group remain unchanged.

## Consequences

This decision unblocks the Perceiver work and ET1–ET5. Existing forecast models,
including `ScalarLSTM`, can serve as downstream forecasters without
modification, and the intermediate remains compatible with the established
`Batch` and conformance path.

The runtime-checkable protocol provides only a weak structural guarantee:
Python runtime protocol checks do not enforce annotations, return types,
identity, or behavior. This limitation is accepted, with behavioral proof left
to end-to-end conformance. Leg-cleanup and scalar-static fusion policy are
deferred to implementations and later work, which may expose pressure on the
minimal interface. No concrete extractor, new entry-point registration,
factory change, or new conformance kit is part of this decision.
