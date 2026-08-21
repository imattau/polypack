# Database core specification

Version: 1 (draft)

This document defines the first portable database-core contract. Language
implementations may expose idiomatic APIs, but observable behavior must match
these rules.

## Transactions

`transaction(callback)` creates a transaction with a unique `id`. Mutations
made through its context are validated before changing graph state and are
persisted as one atomic commit. A failed callback or persistence failure rolls
back all in-memory changes and emits no mutation events. Events are delivered
after commit, in mutation order, and carry the transaction ID.

Reads through the transaction context see earlier writes in that transaction.
Nested transactions are rejected in v1. Implementations may add savepoints in
a later capability level, but must not silently treat nesting as an unrelated
top-level transaction.

Transactions may carry optional `operationId`, `actor`, `baseRevision`, and
application `metadata`. These values are copied into the durable logical
mutation record; `operationId` is the stable caller-supplied identity used for
retry and audit correlation. They do not affect authorization or replace the
revision checks on individual operations.

An adapter used for transactions must declare both `atomicBatches` and
`transactions`. Graphs must reject a transaction when either guarantee is
absent.

## Revisions and conflicts

Nodes and edges have a non-negative `revision`, defaulting to `0`. A successful
replacement or update increments the revision. A conditional mutation may
provide `expectedRevision`; if it differs from the current revision, the
mutation fails with the implementation's typed conflict error and does not
change graph state.

The check applies before validation side effects, in-memory mutation, and
persistence. Multi-record transactions fail atomically if any conditional
operation conflicts.

## Patches

Node patches support these operations:

- `set`: replace values at named data paths;
- `unset`: remove named data paths;
- `increment`: add numeric deltas;
- `compareAndSet`: replace a value only when it equals the expected value.

Patch application produces a complete valid node internally. Schema and
constraint validation runs against the resulting node before any mutation is
visible. A failed patch leaves the original node and revision unchanged.

## Edge identity

An edge's `id` is independent of `(source, type, target)`. Multiple parallel
edges are valid. Adjacency indexes map a source/type/target tuple to one or
more edge IDs. The historical separator-derived `edgeId()` helper remains
available for compatibility, but its output is not a required canonical ID.

## Schema hooks

Node- and edge-type definitions may declare `requiredFields` and `dataTypes`
in addition to a custom validator. Required fields must be present before a
record is accepted. Declared types are `string`, `number`, `integer`,
`boolean`, `object`, and `array`; a mismatch rejects the mutation before
indexes or persistence are changed. Custom validators run against the
resulting complete record after these generic checks.

## Capability levels

Adapters must expose their guarantees through `AdapterCapabilities`. A graph
must not infer transaction or durability guarantees from the adapter class.
Unsupported guarantees produce typed capability errors or use an explicitly
documented fallback that preserves the advertised semantics.
