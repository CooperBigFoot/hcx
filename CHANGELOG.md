# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.7] - 2026-07-11

### Added

- Documented third-party model factory and `hcx.models` entry-point authoring.
- Added public API, privacy-boundary, and built-wheel contract tests.

### Changed

- Declared the supported root import surface explicitly and finalized distribution packaging documentation.

## [0.1.6] - 2026-07-11

### Added

- Added the `ScalarLSTM` reference model with Point and Gaussian output support.
- Registered the reference `scalar_lstm` factory in the `hcx.models` entry-point group.

## [0.1.5] - 2026-07-11

### Added

- Added the standalone `assert_conforms` and `check_conformance` behavioral harness.
- Added deterministic synthetic batches covering all four input quadrants.

## [0.1.4] - 2026-07-11

### Added

- Established the typed batch, forecast, output-specification, model, and model-factory contracts.
- Added the normative hcx specification, `py.typed` marker, Python 3.11 support, and multi-version CI.

[Unreleased]: https://github.com/CooperBigFoot/hcx/compare/v0.1.7...HEAD
[0.1.7]: https://github.com/CooperBigFoot/hcx/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/CooperBigFoot/hcx/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/CooperBigFoot/hcx/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/CooperBigFoot/hcx/releases/tag/v0.1.4
