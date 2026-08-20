# Cross-language compatibility

Version: 1

Polypack has three maintained implementations: TypeScript, Rust, and Python.
The shared fixtures under `fixtures/conformance` are the behavioural source of
truth for portable graph, edge, ownership, traversal, pagination, aggregation,
and vector-index semantics.

## Compatibility levels

| Level | Contract | TypeScript | Rust | Python |
| --- | --- | --- | --- | --- |
| 1 | Node, edge, ownership, detached reads, and query fixtures | CI | CI | CI |
| 2 | Transactions, revisions, patches, schemas, indexes, capabilities, and durable mutations | CI | CI | CI |
| 3 | Query snapshots, persistence verification, backup/restore, migrations, and resource limits | CI | CI | CI |
| 4 | ANN/HNSW behavior | CI | CI | CI |

Level 1 and Level 4 comparisons use the tolerances defined by the individual
fixtures. ANN results are compared by minimum overlap rather than exact order.
All other portable results are expected to match exactly, apart from map
ordering and implementation-specific diagnostic text.

## CI conformance entry points

- TypeScript runs `tests/conformance/conformance.test.ts`.
- Rust runs `polypack-core/tests/query_conformance.rs` and the graph crate's
  unit and persistence tests.
- Python runs `python/tests/test_conformance.py` against the same fixture
  directory, plus its complete API and persistence suite.

The Python runner reports the hot-cache eviction fixture as an explicit skip
because cache sizing is an orchestration concern in that implementation; it
must not silently report that fixture as passing.

## API parity rules

Implementations may use language-native names and synchronous or asynchronous
calling conventions, but must preserve these observable guarantees:

- successful mutations increment revisions and conditional stale writes raise
  a typed conflict error;
- validation occurs before a mutation becomes visible;
- transaction callbacks see their own writes and failed callbacks leave no
  partial state;
- snapshots are detached and remain stable after later graph mutations;
- reads return detached record, edge, vector, and query-result values;
- adapter capability declarations are honored rather than inferred;
- durable mutation records retain operation, transaction, actor, revision,
  timestamp, and metadata identity where the adapter supports logical logs.

When a capability is unavailable, the implementation must expose the declared
degradation or reject the operation; it must not claim a stronger guarantee.
