# Contributing

Contributions are welcome through GitHub issues and pull requests.

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

Stable GitHub releases are published to npm by
`.github/workflows/publish-npm.yml`. Create a release whose tag exactly matches
the package version with a `v` prefix (for example, `v2.1.0`). The workflow
rejects mismatched tags, runs the complete check and package dry run, then
publishes the public scoped package.

Publishing uses npm trusted publishing rather than a long-lived token. In the
package settings on npmjs.com, configure a GitHub Actions trusted publisher with:

- GitHub owner: `imattau`
- Repository: `polypack`
- Workflow filename: `publish-npm.yml`
- Allowed action: `npm publish`

The workflow's OIDC identity provides authentication and npm automatically adds
provenance for the public package. Prerelease GitHub releases are intentionally
not published by this workflow.

By contributing, you agree that your contribution is licensed under the MIT License.
