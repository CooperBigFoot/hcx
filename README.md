# hcx

hcx defines typed, runtime-usable contracts for hydrological model batches,
forecasts, output specifications, models, and model factories.

The [contract specification](docs/spec.md) is normative. Public docstrings and
this README are explanatory summaries of that specification.

The distribution combines three parts: the normative specification, an
inline-typed Python contract marked with `py.typed`, and a standalone
conformance harness. Independent model packages can import the same contract
and prove their behavior without an application, training engine, or external
dataset.

## Authoring a model package

1. Add `hcx` as a runtime dependency of the model package.
2. Implement a `torch.nn.Module` whose `forward(batch: Batch) -> Forecast`
   behavior satisfies `ForecastModel`, including the normative shape,
   dtype/device, and metadata identity rules.
3. Implement a callable matching `ModelFactory`. It accepts a model-specific
   `dict[str, object]` and the keyword-only ordered `dynamic_inputs` and
   `static_inputs`, resolved `input_size`, `static_size`, and `output_size`, and
   a resolved `OutputSpecification[object]`. It returns the module. Preserve
   name order; the sizes are resolved from the first batch.
4. Declare the factory in the model package's `pyproject.toml`:

   ```toml
   [project.entry-points.'hcx.models']
   my_model = "my_model_package.factory:create_model"
   ```

5. Exercise the complete contract with synthetic data:

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

The factory can also be imported and called directly in this smoke test. hcx
packages `scalar_lstm` as a reference entry point, not as a required base class.
Entry-point names are consumer-facing identifiers and should remain stable and
unique within an environment.

See the normative [factory and entry-point clauses](docs/spec.md#6-model-factory-and-hcxmodels-entry-points)
and the [changelog](CHANGELOG.md) for version history.
