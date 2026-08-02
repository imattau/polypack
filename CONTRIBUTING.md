# Contributing

Contributions are welcome through GitHub issues and pull requests.

## AI-generated code

Code written or assisted by AI tools (Claude, Copilot, ChatGPT, etc.) is
welcome — you don't need permission to use one. The same standards apply
regardless of how a change was produced: keep it focused, add regression
tests for behavioral changes, and make sure you understand and can explain
what it does before opening the PR. Mentioning the tool you used in the PR
description helps reviewers, but isn't required.

## Development

Requires Node.js 18 or newer.

```sh
npm ci
npm test
npm run build
```

Keep changes focused, add regression tests for behavioral changes, and update
the README, API reference, and changelog when the public contract changes.
Before submitting, ensure `npm run check` passes.

## Publishing

Publishing a GitHub release triggers `.github/workflows/release.yml`, which
runs the full check suite (`npm run check`, `cargo test`/`clippy`, Python
tests against a built wheel) as a gate, then publishes across every
ecosystem: the TypeScript package to npm, per-platform native addons plus
the `@0xx0lostcause0xx0/polypack-native` wrapper to npm, `polypack-core` and
`polypack-graph` to crates.io, and Python wheels to PyPI as `polypack-db`.
Every package in the repo shares one version number — see `RELEASING.md`
for the full release procedure, version-coordination rules, and one-time
registry (npm/crates.io/PyPI) trusted-publisher setup.

By contributing, you agree that your contribution is licensed under the MIT License.
