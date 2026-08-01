# `@0xx0lostcause0xx0/polypack-native`

NAPI-RS bindings exposing the [`polypack-core`](https://crates.io/crates/polypack-core)
Rust engine to Node.js: vector indexes, the query-plan executor, and the
directory-backed persistence store, as drop-in accelerators for
[`@0xx0lostcause0xx0/polypack`](https://www.npmjs.com/package/@0xx0lostcause0xx0/polypack).

This package ships prebuilt binaries via optional dependencies for
`darwin-x64`, `darwin-arm64`, `linux-x64-gnu`, `linux-arm64-gnu`, and
`win32-x64-msvc`. If no native binary is available for the current platform,
`isNativeAvailable()` reports `false` and callers should fall back to the
pure-TypeScript engine — this package does not do that fallback for you.

## Install

```sh
npm install @0xx0lostcause0xx0/polypack-native
```

## Usage

```ts
import { isNativeAvailable, NativeVectorIndex, NativeHnswIndex, NativeStore } from '@0xx0lostcause0xx0/polypack-native'

if (isNativeAvailable()) {
  const index = new NativeVectorIndex(undefined, 'cosine')
  index.add('doc_1', [0.95, 0.20, 0.10])
  index.query([0.90, 0.30, 0.10], 5, 0.5)
}
```

`NativeVectorIndex` and `NativeHnswIndex` are drop-in replacements for the
TypeScript `VectorIndex`/`HNSWIndex` classes and are the recommended way to
plug native acceleration into `PolyGraph` via its `createVectorIndex` hook:

```ts
import { PolyGraph } from '@0xx0lostcause0xx0/polypack'
import { createNativeVectorIndex } from '@0xx0lostcause0xx0/polypack-native'

const graph = new PolyGraph(undefined, 50_000, undefined, undefined, createNativeVectorIndex())
```

`installNativeQueryExecutor()` routes in-memory `GraphQuery` execution
through the Rust query planner when available (queries with join predicates
still fall back to TypeScript). `NativeStore` is a directory-backed store
using the same snapshot/WAL byte format as the TypeScript
`BinaryStoreAdapter`, so files are interchangeable between the two.

None of this wiring happens automatically — importing this package does not
change `@0xx0lostcause0xx0/polypack`'s behavior on its own; call the hooks
above explicitly.

## API

See [`src/index.d.ts`](src/index.d.ts) for the full exported surface:
`NativeVectorIndex`, `NativeHnswIndex`, `NativeStore`, `executeQueryPlan`,
`aggregateQueryPlan`, `installNativeQueryExecutor`, `engineInfo`,
`isNativeAvailable`, and `detectEngine`.

## License

MIT
