# Changelog

All notable changes to this project are documented here. This project follows
[Semantic Versioning](https://semver.org/).

## [1.1.0] - 2026-07-17

### Added

- Property graph queries, vector similarity search, ownership-aware edges,
  pluggable persistence, React hooks, and transport-agnostic synchronization.
- Public API reference, release metadata, native Node.js ESM support, and npm
  subpath exports for core, React, and sync entry points.

### Fixed

- Persisted edge cleanup during node deletion.
- Concurrent flush and shutdown data-loss paths.
- Node/vector index consistency during replacement, restoration, and eviction.
- IndexedDB bulk-read and vector top-k performance.

[1.1.0]: https://github.com/imattau/polypack/releases/tag/v1.1.0
