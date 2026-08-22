# database-core benchmark report

Generated 2026-08-22T14:04:24.356Z by `benchmarks/database-core-compare.ts`.

Measures durable write throughput, mutation-log replay + cold-store recovery,
and in-process sync-server throughput, for the same workload
(count=20000 nodes, 5000 sync ops) across
the TypeScript, Rust, and Python implementations of database-core.

- **Durable write throughput**: `addNode`/`add_node` in batches of 500,
  flushed to a real on-disk store (`BinaryStoreAdapter` / `FileStorage` /
  native `DirectoryStorage`).
- **Mutation log / recovery**: the store is reopened cold; `mutation log
  replay time` reads back the durable mutation log, `cold-store recovery
  time` is how long re-opening + loading the store into the working set
  takes (`PolyGraph.load()` / `Graph::warm()` / `open_store()`).
- **Sync throughput**: ops submitted directly to each language's
  `SyncServer` message handler (no transport/network layer), in batches of
  100.

## Results

| metric | TypeScript | Rust | Python |
|---|---|---|---|
| durable write throughput (ops/sec) | 66163 | 65638 | 4521 |
| durable write time | 302.3ms | 304.7ms | 4423.5ms |
| mutation log records | 40 | 40 | 41 |
| mutation log replay time | 49.40ms | 18.99ms | 2180.92ms |
| cold-store recovery time | 95.91ms | 80.30ms | 1402.54ms |
| sync throughput (ops/sec) | 1680152 | 625739 | 301787 |
| sync submit time | 3.0ms | 8.0ms | 16.6ms |

## Speedups relative to TypeScript

| lane | write speedup vs TS | sync speedup vs TS |
|------|----------------------|---------------------|
| Rust | 0.99× | 0.37× |
| Python | 0.07× | 0.18× |

## Notes

- Rust batches durable writes every 500 nodes via `add_nodes` +
  `flush()`; TypeScript and Python do the same via per-node `addNode`/
  `add_node` + a `flush()`/`save()` every 500 nodes — so mutation-log
  record counts line up across lanes for the same workload.
- Python's numbers include the pure-Python graph layer + native-extension
  FFI overhead per node; only the vector/index primitives are native.
- These are single-run, single-machine numbers meant to catch regressions
  and give a rough cross-language comparison — not a rigorous statistical
  benchmark (no warmup, no repeated trials, no percentiles).
