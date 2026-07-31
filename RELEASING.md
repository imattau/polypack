# Releasing polypack

Release rules from the multi-language roadmap (POLYPACK_RUST_PYTHON_PLAN §9).

## Version coordination

As of `2.4.2`, every package in the repo — the TypeScript package, the native
stack, and the Python wheel — shares a single version number:

| Package | Version (current) |
|---|---|
| `@0xx0lostcause0xx0/polypack` (TypeScript) | `2.4.2` |
| `polypack-core` crate (`crates/polypack-core`) | `2.4.2` |
| `@0xx0lostcause0xx0/polypack-native` + per-platform packages | `2.4.2` |
| `polypack-db` Python wheel | `2.4.2` |

Bump all of them together for every release, even if only one language's code
changed — registries need a fresh version to re-publish, and drift between
the TypeScript version and the native/Python versions has been a source of
confusion.

## How to release

1. Bump versions (TS package + native stack as needed), update
   `CHANGELOG.md`, commit, and push `master`.
2. Create a GitHub release tagged `v<ts-version>` (e.g. `v2.4.1`). The release
   event triggers `.github/workflows/release.yml`.

The release workflow publishes, in order:

- **TypeScript package** (`ts-publish`, npm provenance).
- **npm per-platform packages** then the **native wrapper** (`native-publish`
  builds every addon on a matching runner; `native-package-publish` stages and
  publishes with npm provenance).
- **`polypack-core`** to crates.io (`crate-publish`, uses the
  `CRATES_IO_TOKEN` secret).
- **Python wheels** to PyPI (`wheel-publish` builds abi3 wheels per OS;
  `wheel-publish-upload` publishes via OIDC trusted publishing for the
  `release` environment).
- **TypeScript package** (`ts-publish`).

## One-time registry setup (required before the first release)

### npm

All npm publishing happens in `release.yml` and uses npm OIDC trusted
publishing (audience `npm`; `id-token: write` is set at the workflow level).
npm configures trusted publishing **per package name** (there is no
scope-level grant), so add one trusted publisher for each of these packages,
all pointing at workflow `release.yml`:

- `@0xx0lostcause0xx0/polypack`
- `@0xx0lostcause0xx0/polypack-native`
- `@0xx0lostcause0xx0/polypack-native-darwin-arm64`
- `@0xx0lostcause0xx0/polypack-native-darwin-x64`
- `@0xx0lostcause0xx0/polypack-native-linux-arm64-gnu`
- `@0xx0lostcause0xx0/polypack-native-linux-x64-gnu`
- `@0xx0lostcause0xx0/polypack-native-win32-x64-msvc`

How to add each one:

1. Sign in to [npmjs.com](https://www.npmjs.com) and open
   **Access Tokens → Generate New Token → Generate a token via OIDC (GitHub
   Actions)**.
2. Fill in:
   - **Workflow file**: `release.yml`
   - **Environment**: leave blank (the publish jobs have no environment)
   - **Package**: the exact package name (one per token/publisher)
3. Generate. Repeat for each of the 7 names above.

The native platform packages (`polypack-native-*`) do not exist yet; the first
OIDC publish creates them. If npm refuses to configure a trusted publisher for
a name that doesn't exist, publish each once manually with a classic token
(`npm publish --access public` from the package directory) to create it, then
add the OIDC publisher.

### crates.io

The `crate-publish` job prefers a classic API token from the `CRATES_IO_TOKEN`
repo secret; it falls back to GitHub OIDC if crates.io trusted publishing is
registered for this repository (not available on every account).

**Recommended: classic token**

1. Sign in to [crates.io](https://crates.io) → **Account settings → API
   tokens → New token** (with `publish-new` scope). Copy the token.
2. In the GitHub repo → **Settings → Secrets and variables → Actions →
   New repository secret**: name `CRATES_IO_TOKEN`, value the token.
3. Ensure the account has a **verified email address**
   (Account settings → Profile), or `cargo publish` is rejected with a 400.

**Alternative: OIDC trusted publishing**

Only if crates.io shows **Account settings → OIDC Providers**:

1. Add a GitHub OIDC provider with:
   - **Repository owner**: `imattau`
   - **Repository name**: `polypack`
   - **Workflow name**: `release.yml`
   - **Environment**: blank
2. The workflow already fetches the OIDC token with audience `crates.io`.

Requirements for either path: `cargo` ≥ 1.74 (CI installs stable), workflow
`id-token: write` (set), `release.yml` on the default branch.

### PyPI (trusted publishing)

No token is stored. The `wheel-publish-upload` job uses
`pypa/gh-action-pypi-publish` with `id-token: write` and the `release`
environment, and PyPI validates the GitHub OIDC token against the trusted
publisher you register.

1. Sign in to [PyPI](https://pypi.org), open **Account settings → Publishing
   → Add trusted publisher**.
2. Fill in:
   - **Workflow name**: `release.yml`
   - **Environment name**: `release`
   - **Project name**: `polypack-db`
   - **Owner**: `imattau`
   - **Repository**: `polypack`
3. Save. The `polypack-db` project name matches the wheel's
   `[project] name`; the first trusted publish creates it.

The `release` GitHub environment is referenced by the workflow; create it in
**Settings → Environments → New environment → `release`** (no deployment
branch protection needed, but do not restrict it in a way that blocks the
workflow).

Alternative: set a **`PYPI_TOKEN`** secret and the pypi-publish action will
use it instead of OIDC.

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
