# Edge-flush cost benchmark (Rust) — and what it uncovered

`crates/polypack-graph/examples/database_core_edge_flush.rs` isolates
`Graph::flush()`'s cost for resolving dirty edge ids, separate from storage
I/O and node writes (nodes are seeded once, unflushed; only edges are added
and flushed per batch).

Run with:

```sh
npm run bench:database-core:edge-flush:rust -- --nodes 2000 --batch-size 200 --batches 100
```

## What it found: three compounding bugs, not one

Chasing why per-edge flush cost kept growing with total edge count (even
after each fix) surfaced three separate O(n) costs stacked on top of each
other. All three are fixed now; all `polypack-core`/`polypack-graph` tests
still pass.

### 1. `Graph::flush()` scanned every source's edge map per dirty edge

`crates/polypack-graph/src/graph.rs::flush()` resolved each dirty edge id via
`self.edges.iter().find_map(...)` — O(dirty_edges × total_edges) per flush.

**Fix**: a persistent `edge_id_index: HashMap<String, (String, String)>`
(edge id -> (source, key)), maintained incrementally in `add_edge`,
`add_edge_with_id`, and every edge-removal path (`record_removed_edge`,
`remove_edge_if_revision`, `remove_edges`), mirrored in the transaction
snapshot (`GraphCheckpoint`) and rebuilt on load (`rebuild_edge_index`).
This also replaced three other linear id-scans in the same file
(`update_edge_if_revision`, `remove_edge_if_revision`,
`add_edge_with_id`'s duplicate check) with O(1) lookups.

### 2. `InMemoryStorage::append()` cloned the whole file on every append

`crates/polypack-core/src/storage/store.rs` — `self.files.get(name).cloned()`
before extending, on every WAL/mutation-log append. O(total bytes appended
so far) per append, i.e. O(n²) over n flushes. Used by any graph opened over
`InMemoryStorage` (both benchmark examples and any ephemeral in-memory
graph).

**Fix**: `self.files.entry(name.to_string()).or_default().extend_from_slice(data)`
— extends in place instead of clone-then-reinsert.

### 3. `validate_pending_schema` deep-cloned and re-validated the entire store on every apply

The big one. `Store::validate_pending_schema`, called on **every**
`apply`/flush regardless of batch size or workload shape, unconditionally
cloned `self.nodes` and `self.edges` in full, then iterated every node in
the store — even though `validate_node_schema`/`validate_edge_schema` are
guaranteed no-ops when no node/edge type is registered (the common case,
including every benchmark and probably most real usage). This one wasn't
edge-specific at all — it hit every write path, node or edge.

**Fix**: early-return before the clone when
`node_type_definitions.is_empty() && edge_type_definitions.is_empty()`.
Fully behavior-preserving (there is nothing to validate against).

## Results

Edge-flush cost, before vs. after all three fixes (2,000 nodes, batch 200):

| | first batch (ms/edge) | last batch, 19,800 edges (ms/edge) | ratio |
|---|---|---|---|
| before | 0.0052 | 0.0986 | 19.0× |
| after | 0.0038 | 0.0035 | **0.9×** (flat) |

At larger scale (5,000 nodes, batch 200, ~100,000 edges), the ratio also
went from 31× down to **0.8× (flat)**.

Because fix #3 hits every flush, not just edge-heavy ones, it also moved the
numbers in the pre-existing cross-language reports:

**`benchmarks/database-core-report.md`** (20,000-node durable write + sync,
`FileStorage`):

| metric | before | after |
|---|---|---|
| Rust durable write throughput | 51,235 ops/sec (0.67× vs TS) | 66,198 ops/sec (**1.00× vs TS**) |

**`benchmarks/database-core-mixed-report.md`** (2,000 initial nodes, 200
mixed rounds, `InMemoryStorage`):

| metric | before | after |
|---|---|---|
| Rust write p50 | 2.733ms | 1.161ms (2.4× faster) |
| Rust update p50 | 2.292ms | 0.451ms (5.1× faster) |
| Rust total | 2457.6ms | 1568.3ms (1.6× faster) |

Rust write throughput is no longer a regression vs. TypeScript — it's at
parity on durable writes and substantially faster on the mixed in-memory
workload, though still behind TS there (TS write p50 is 0.074ms — that gap
is now dominated by other costs, e.g. `serde_json::Map`-based node data vs.
V8 object shapes, not investigated here).

## Not fixed here

- **Sync throughput** was a regression (0.32× vs TS) — investigated
  separately, see `SyncServer::submit()` in `crates/polypack-core/src/sync.rs`:
  fixed a discarded whole-batch checksum computation and a redundant clone
  pass, bringing it to 0.37×. Confirmed via a scaling check (5k/20k/50k ops)
  that this one was never an O(n) bug — throughput was already flat: it's a
  constant-factor gap (JSON `Value` cloning, `format!`-allocated hash keys)
  vs. V8, not fully closed.
- **Schema-registered graphs** still pay the full `validate_pending_schema`
  clone-and-scan cost on every apply. Only validating `changes.put_nodes`/
  `changes.put_edges` (plus cheap targeted lookups for edge cardinality
  checks) instead of cloning+scanning the whole store would fix this too,
  but wasn't attempted — lower priority since it's an opt-in path.
