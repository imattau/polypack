# Cross-language compatibility

Version: 1

Polypack has three maintained implementations: TypeScript, Rust, and Python.
The shared fixtures under `fixtures/conformance` are the behavioural source of
truth for portable graph, edge, ownership, traversal, pagination, aggregation,
and vector-index semantics.

Persistence recovery scenarios under `fixtures/recovery` are also executed by
the TypeScript, Rust, and Python native stores. The shared
`fixtures/database-core/error-taxonomy.json` fixture defines the stable error
codes expected for representative validation, dimension, conflict, and
resource-limit failures.

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
- TypeScript runs `tests/conformance/recovery.test.ts` for the shared recovery
  corpus and `tests/database-core.test.ts` for database-core fixtures,
  including the error taxonomy.
- Rust runs `polypack-core/tests/query_conformance.rs` and the graph crate's
  unit and persistence tests, including
  `polypack-core/tests/recovery_conformance.rs` and the database-core error
  taxonomy test.
- Python runs `python/tests/test_conformance.py` against the same fixture
  directory, plus its complete API and persistence suite, where
  `test_shared_recovery_fixtures` and `test_error_taxonomy_fixture` cover the
  shared persistence and error fixtures.
- Database-core fixtures under `fixtures/database-core` are also shared across
  the language-specific suites. The secondary-index fixture covers candidate
  intersection, selected-index explain output, and stable query results.
  The migration fixture covers batched application-schema transformation and
  record identity preservation. The resource-limits fixture covers write-side
  vector and payload bounds, bulk mutation limits, and transactional rollback
  when a mutation budget is exceeded.

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
- logical-log consumers can resume from an exclusive sequence cursor and use
  bounded pages where the implementation exposes a paged change-feed API;
  TypeScript uses `mutationLogPage`, while Rust and Python use
  `mutation_log_page`.
- implementations that retain logical mutation records must advertise the
  `changeFeed` capability; adapters without that capability must reject
  change-feed reads rather than returning a misleading empty history.

When a capability is unavailable, the implementation must expose the declared
degradation or reject the operation; it must not claim a stronger guarantee.
