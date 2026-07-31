# Releasing polypack

Release rules from the multi-language roadmap (POLYPACK_RUST_PYTHON_PLAN §9).

## Version coordination

The native stack shares one version and is released together:

| Package | Version (current) |
|---|---|
| `polypack-core` crate (`crates/polypack-core`) | `0.1.0` |
| `@0xx0lostcause0xx0/polypack-native` + per-platform packages | `0.1.0` |
| `polypack` Python wheel | `0.1.0` |

The TypeScript package follows its own semver (currently `2.x`); it is not
bumped for native-stack releases unless its public API changes.

## Rules

- **Native packages are built in CI, never on a maintainer workstation.** Each
  platform addon is produced by the `native` CI lane; wheels by the `python`
  lane. `napi pre-publish` stages the built addons into the per-platform
  packages only after every platform artifact is present.
- **Release candidates run the complete suite:** `npm run check` (TypeScript +
  conformance + browser), `cargo test --release` + `cargo clippy -- -D warnings`
  (Rust, incl. recovery + query conformance), `pytest` (Python), and the
  `package` CI job (clean Rust-free native install + clean-venv wheel install
  with the 100K-vector example).
- **Format changes require an explicit persistence-format version.** The
  snapshot/WAL format is byte-compatible v1; bumping it is a coordinated
  breaking change across TypeScript, Rust, Python, and Node native.
- **Behavioural breaking changes require a major release.**
- **Benchmark regressions generate reports** (`benchmarks/`) but only block
  releases after stable thresholds exist.

## Native npm release flow

1. CI `native` lane builds `polypack-native.<triple>.node` per OS and uploads
   them as artifacts.
2. In the `package` (release) job: `napi create-npm-dirs`, then copy each
   downloaded addon into its platform package, then `npm pack` and publish
   each per-platform package, then the wrapper.
3. Verify with the clean-install smoke (installs the tarballs with no Rust
   toolchain and loads the native addon).

## Python release flow

1. CI `python` lane runs `maturin build --release` per OS/Python (abi3 wheels).
2. Publish wheels with `maturin publish` (CI credentials) — see
   [maturin docs](https://www.maturin.rs/publish).
3. Verify with the clean-venv 100K-vector example.
