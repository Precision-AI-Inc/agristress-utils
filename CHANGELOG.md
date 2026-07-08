# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Renamed package from `pai-agribench-utils` to `pai-agristress-utils` to align with the AgriStress-500 dataset name.
- Renamed `AgriBenchDataset` → `AgriStressDataset` throughout the public API.
- Updated HuggingFace default repo ID to `precisionaiinc/AgriStress-500`.
- Switched to `setuptools-scm` for dynamic versioning from git tags.
- Upgraded CI action versions (`checkout@v7`, `setup-python@v6`).
- Added `release.yml` workflow for PyPI publishing on tag push.
- Added `CODE_OF_CONDUCT.md`, `CHANGELOG.md`, `MANIFEST.in`, Dependabot config, and GitHub issue/PR templates.

## [0.1.0] - 2026-07-08

### Added

- Initial release.
- `download_dataset()` — single-shot HuggingFace Hub download.
- `AgriStressDataset` — in-memory loader for the three JSON manifests.
- `normalize_embeddings()` / `validate_embeddings()` — L2-norm utilities.
- Pydantic record models: `ImageRecord`, `InstanceRecord`, `ImageAttributes`.
- Retrieval payload builders: `build_i2i_payload`, `build_p2i_payload`, `build_p2p_payload`.

[Unreleased]: https://github.com/Precision-AI-Inc/agristress-utils/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Precision-AI-Inc/agristress-utils/releases/tag/v0.1.0
