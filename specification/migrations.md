# Application migrations

Application migrations are separate from physical snapshot/WAL format
migrations. They transform valid node and edge records between application
schema versions.

Implementations register contiguous steps (`from` → `to`) and reject missing
or overshooting paths. A migration must preserve record identity: node IDs and
types, and edge IDs, endpoints, and types cannot change in a data migration.
Topology changes belong in an explicit transaction.

Before mutation, every transformed record is validated against the graph's
normal record and schema constraints. A non-dry-run is then committed as one
atomic transaction. Unchanged records are not rewritten, so migrations are
safe to retry. Resume cursors are exclusive and process IDs in deterministic
order.

The TypeScript API additionally supports asynchronous callbacks, abort
signals, and progress callbacks. The Rust API is synchronous and supports
bounded validation batches plus synchronous progress callbacks. Python's
`MigrationRegistry` supports dry-run, batch, resume, and progress options.

Example TypeScript usage:

```ts
graph.migrations.register({
  from: 2,
  to: 3,
  migrateNode(node) {
    return { ...node, data: { ...node.data, displayName: node.data.name } }
  },
})

await graph.migrations.run(graph, 2, 3, { batchSize: 500 })
```
