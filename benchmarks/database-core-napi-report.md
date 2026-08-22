# napi round-trip benchmark — and the fix

Every "Rust" number in `database-core-report.md` and `go-no-go.md` comes
from a standalone Rust process (`cargo run --example database_core_batches`)
or a Python-FFI call — never from the actual `packages/node-native`
`NativeStore` napi module that real Node.js consumers load. This was flagged
as an untried benchmark; `benchmarks/database-core-batches-napi.ts` closes
that gap by driving `NativeStore.apply()` directly from TypeScript, same
workload/schema as `database-core-batches.ts`.

Run with:

```sh
npm run bench:database-core:batches:napi
```

## Root cause

`NativeStore.apply()` in `crates/polypack-node/src/lib.rs` took
`serde_json::Value` and re-parsed it into `CoreChangeBatch`:

```rust
pub fn apply(&self, changes: serde_json::Value) -> Result<()> {
    let batch: CoreChangeBatch = serde_json::from_value(changes)...
```

Every call deserialized the JS object **twice**: once via napi's own
JS-value -> `serde_json::Value` walk, and again via `serde_json::from_value`
into the typed `CoreChangeBatch`.

## The fix

Typed `#[napi(object)]` mirror structs (`NapiNode`, `NapiEdge`,
`NapiVectorEntry`, `NapiChangeBatch`) that convert into the `polypack-core`
types via `TryFrom`. `#[napi(object)]` gives required fields (`id`, `type`,
`insertedAt`, `updatedAt`) direct per-field conversion off the JS value —
no JSON tree built for the batch as a whole. `apply()` now takes
`NapiChangeBatch` directly; the JS-facing shape is unchanged, so
`packages/node-native/src/index.ts` and every existing caller needed zero
changes.

**A real correctness trap along the way**: `#[napi(object)]`'s generated
getter for an `Option<T>` field only treats a genuinely *missing* JS
property as `None`. If the property is present but explicitly `null` (this
codebase's own convention for "no vector" / "no edge data" — confirmed by
grepping every `SerializedNode`/`SerializedEdge` call site, which
consistently write `vector: null` / `data: null` rather than omitting the
key), the derived getter calls `T::from_napi_value` directly on the `null`
value, bypassing `Option<T>`'s own null-handling. For a `T` that doesn't
tolerate `null` itself (`Vec<f64>`, `String`, `f64`, another `#[napi(object)]`
struct), this throws (`Given napi value is not an array on NapiNode.vector`)
instead of yielding `None`. First implementation attempt broke 7 of 12
`tests/native/storage.test.ts` cases this way.

Fix: every optional `Node`/`Edge` field is typed `Option<serde_json::Value>`
on the napi struct (which *does* tolerate `null`, becoming `Value::Null`)
and converted through one helper that treats `None` and `Some(Value::Null)`
identically:

```rust
fn opt_json<T: serde::de::DeserializeOwned>(v: Option<serde_json::Value>, field: &str) -> Result<Option<T>> {
    match v {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(value) => serde_json::from_value(value).map(Some)
            .map_err(|e| Error::from_reason(format!("{field}: {e}"))),
    }
}
```

This still avoids building one JSON tree for the *entire* batch/node, at
the cost of a small per-field JSON parse for each optional value (cheap for
scalars; `vector` arrays are the largest case, still far smaller than a
whole node). Verified: all 12 `tests/native/storage.test.ts` cases, the
full 558-test TS suite, and all 283 Rust tests (`polypack-core` +
`polypack-graph`) pass.

## Results (5,000 nodes, varying batch size = nodes per `apply()`/`flush()` call)

| batch size | ts | rust process | napi (before) | napi (after) | napi vs ts (after) |
|---|---:|---:|---:|---:|---:|
| 1 | 11,754 | 2,019 | 707 | 1,405 | 0.12× |
| 100 | 117,416 | 119,148 | 49,214 | 81,156 | 0.69× |
| 500 | 127,015 | 246,481 | 111,192 | 153,012 | **1.20×** |
| 5,000 | 128,537 | 223,762 | 138,769 | 158,898 | **1.24×** |

(ops/sec; "napi (before)" is the pre-fix, `serde_json::Value`-parameter
measurement.)

The fix gained 1.14×–2.0× across every batch size, and at realistic batch
sizes (≥500) `NativeStore` now **beats** the TypeScript write path instead
of trailing it (was 0.88×/1.08×, now 1.20×/1.24×). At batch size 1 it's
still well behind TS (0.12×, though 2× better than before) — per-call FFI
overhead is inherent to crossing the JS/Rust boundary at all and can't be
fully eliminated, only amortized by batching.

## Not pursued further

`data` (node/edge free-form payload) and the now-`opt_json`-routed optional
fields still each cost one small `serde_json::Value` parse. A fully
zero-JSON path would need typed napi fields for the *shape* of `data` too,
which isn't possible since `data` is genuinely arbitrary per node — not
worth chasing further.
