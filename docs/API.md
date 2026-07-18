# API reference

Polypack is ESM-only and requires Node.js 18 or newer. The browser adapter also
requires IndexedDB. React is optional and only loaded by `@0xx0lostcause0xx0/polypack/react`.

## `@0xx0lostcause0xx0/polypack`

### `PolyGraph`

```ts
new PolyGraph(adapter?: PersistenceAdapter, hotCacheMax?: number)
```

The main property-graph container. Without an adapter it uses `MemoryAdapter`.
The default hot-node limit is 10,000. Edges remain indexed when nodes are
evicted, and dirty evicted nodes are retained until persistence completes.
Synchronous queries and mutations operate on currently loaded nodes; use
`getNodeSafe(id)` to restore an evicted node before mutating it.

Lifecycle:

- `warm()` / `load()` loads persisted nodes, vectors, and edges. Call it before
  querying an existing database.
- `flush()` immediately persists queued mutations. Flushes are serialized.
- `save()` writes the complete currently loaded graph without clearing dirty state.
- `dispose()` flushes queued mutations, clears memory, then closes the adapter.
- `clear()` only clears in-memory state; it does not delete adapter contents.
- `prune(maxNodes)` removes the oldest loaded nodes and applies ownership rules.

Nodes:

- `addNode(node)` inserts or replaces a node. Replacement updates type/vector indexes.
- `getNode(id)` returns a loaded node synchronously.
- `getNodeSafe(id)` restores an evicted node from persistence when necessary.
- `updateNode(id, data, vector?)` shallow-merges data and optionally replaces its vector.
- `updateNodeSafe(id, data, vector?)` restores an evicted node before updating it.
- `removeNode(id)` removes the node, all connected edges, and owned descendants.
- `removeNodeSafe(id)` restores and removes an evicted node, recursively restoring
  owned descendants as required. Call `warm()` first for an existing database so
  ownership edges are indexed.
- `whereType(type)` returns loaded nodes of one type.
- `size` and `loadedSize` are the number of currently loaded nodes, not total
  persisted nodes. `hasLoadedNode(id)` checks membership in that working set.
- `persistedSize()` asynchronously returns the adapter's current node count.

Edges:

- `addEdge(source, type, target, data?, ownership?)` adds one unique directed edge.
- `getEdges(source, type?)` returns outgoing edge descriptors.
- `getEdgeTargets(source, type)` and `getEdgeSources(target, type)` return IDs.
- `removeEdges(source, type?, target?)` removes matching outgoing edges.

Ownership is stored on the edge:

- `reference` (default): removing the edge never removes its target.
- `shared`: keeps the target, and invokes the protected `onOrphan` hook when it
  loses its final incoming edge.
- `owned`: removes the target when it loses its final owning source; cascading
  is cycle-safe.

Reactivity and batching:

- `changes` is an RxJS `Subject<GraphChangeEvent>`.
- `startBatch()` queues notifications until the matching `endBatch()`.
- `endBatch()` throws if no batch is open.

`query()` creates a mutable `GraphQuery`.

`queryPersisted()` creates a mutable `PersistedGraphQuery`. Its terminal methods
are asynchronous and inspect the complete backing store without loading results
into the hot working set. Results are detached node copies, and mutations must be
flushed before they become visible. The persisted-query API supports node type,
attribute and range filters, edge filters, joins, traversal, collection,
ordering, similarity, offset, and limit.

### `GraphQuery`

Filter methods are chainable:

- `where(field, value)` / `whereAttribute(name, value)` use strict equality.
- `whereAttributeRange(name, { above?, below? })` uses exclusive boundaries.
- `whereNodeType(...types)` restricts node types.
- `whereEdge(type, target?)` requires an outgoing edge.
- `whereEdgeSource(source)` requires an incoming edge from that source.
- `join(edgeType, direction?, predicate?)` filters by connected nodes.
- `traverse(edgeType, depth, direction?)` performs breadth-first traversal and
  includes the seed nodes.
- `similarTo(vector, threshold?, topK?)` ranks vector-bearing nodes by cosine similarity.
- `orderBy(field, direction?)`, `offset(n)`, and `limit(n)` shape results.

Terminal methods:

- `toArray()`, `first()`, `count()`, and `ids()` return matched nodes or IDs.
  `count()` respects similarity, traversal, offset, and limit.
- `pluck(...fields)` projects node data into records that also contain `id` and `type`.
- `aggregate(field, op)` supports `sum`, `avg`, `min`, `max`, and `count`.
- `groupAggregate(field, op, groupByField)` aggregates by a data field.
- `having(groups, predicate)` filters aggregate rows.
- `groupByVector(groups, field, op, threshold?)` assigns nodes to their nearest centroid.
- `uniqueKeys(field)` returns distinct values across all currently loaded nodes.
- `collect(edgeType, direction?, predicate?)` returns unique directly connected nodes.

Filters run before traversal. Similarity ranking runs after `orderBy`, so it becomes
the final ordering before offset and limit.

### `PersistedGraphQuery`

Chainable filters are `where`, `whereAttribute`, `whereAttributeRange`,
`whereNodeType`, `whereEdge`, `whereEdgeSource`, `join`, `traverse`, `orderBy`,
`similarTo`, `offset`, and `limit`. The asynchronous terminal methods are
`toArray()`, `first()`, `count()`, `ids()`, and `collect()`.

Adapters may implement `queryNodes(query)` and `countNodes(query)` for optimized
storage-level execution, plus `getEdgesBySources(ids, type?)` and
`getEdgesByTargets(ids, type?)` for indexed graph operations. When absent,
Polypack falls back to the original node and edge methods, preserving
compatibility with existing custom adapters.

For node-only queries, offset and limit are delegated to the adapter. IndexedDB
uses an early-stopping cursor when ordering and query shape permit it. Queries
with similarity or graph post-processing retain pagination in the query layer so
that filtering, traversal, and ranking occur before the page is selected.

### Vector search

- `new VectorIndex(onChange?, distanceFn?)` creates an exact in-memory index.
- `add`, `addMany`, `remove`, `removeMany`, `clear`, `has`, `get`, `entries`, and
  `size` manage vectors.
- `query(vector, topK, threshold?)` returns `{ id, score }[]`, highest first.
- `cosineSimilarity(a, b)` and `euclideanSimilarity(a, b)` are built in.

All compared vectors must have identical dimensions; otherwise similarity
functions throw `RangeError`. Zero vectors have cosine similarity `0`.

### Persistence

- `MemoryAdapter` stores serialized records in memory.
- `IndexedDBAdapter({ name?, version?, nodeIndexes? })` persists browser data. Defaults to
  database `polypack`, version `2`. Its node-type index accelerates persisted
  type queries; existing default databases are upgraded automatically.
- `nodeIndexes` is an array of node data fields, such as `['score', 'createdAt']`.
  Constrained range queries ordered by one of these fields use the corresponding
  IndexedDB index for ordered cursor paging. Fields must be valid dotted
  IndexedDB key paths. When adding indexes to an existing custom-named database,
  increment its configured schema `version` so IndexedDB runs the upgrade.
- `PersistenceAdapter` is the contract for custom storage. It contains node,
  edge, and vector single/bulk operations plus `clearAll()` and `close()`.

Adapter methods should reject on storage errors. Bulk methods should be atomic
where the backing store permits it.

### Types and utilities

The root exports `PolyNode`, `PolyEdge`, `EdgeOwnership`, `GraphChangeEvent`,
`SerializedNode`, `SerializedEdge`, `VectorQuery`, aggregate types,
`DistanceFunction`, `IndexedDBConfig`, and `PersistenceAdapter`.

`edgeId(source, type, target)` produces the persistence edge key. Source IDs and
edge types must not contain `::`; target IDs may contain it.
`yieldToUI()` yields through a zero-delay timer.

## `@0xx0lostcause0xx0/polypack/react`

```ts
useGraphQuery(graph, queryFn, deps, delay?, nodeTypes?)
useLiveQuery(graph, querier, deps?, defaultResult?)
```

Both hooks execute immediately, subscribe to graph changes, and support sync or
async queries. `useGraphQuery` defaults to a 200 ms debounce. When `nodeTypes`
is supplied, events for other known node types are ignored. Query errors are
logged and set the result to `undefined`.

The dependency list controls query closure refresh in the same way as other
React hooks. React 18 and 19 are supported as an optional peer dependency.

## `@0xx0lostcause0xx0/polypack/sync`

- `OpLog(clientId, existing?)` appends sequenced `SyncOp` values and exposes
  `since(seq)`, `all`, `latestSeq`, and `size`.
- `SyncAdapter(inner, clientId)` wraps persistence and records successful node
  and edge writes in its `oplog`; set `onOp` to observe them.
- `SyncClient({ graph, transport, clientId?, autoFlush?, retryMs? })` captures
  graph events, retains operations until acknowledged, retries unacknowledged
  deltas, detects server-cursor gaps, and applies remote operations with echo
  suppression. `retryMs` defaults
  to 1,000 ms and `0` disables automatic retry. Use `flush()` for manual sends,
  inspect `pendingOps`, call `requestSync(fromStart?)` for catch-up, inspect
  `syncCursor`, call `reconnect(transport)` to replace a transport, resend pending
  work, and request missing server operations, and use `disconnect()` to close.
- `SyncServer` is an in-memory relay. `addClient(handle)` returns that client's
  incoming-message handler, `removeClient(handle)` unregisters it, and `ops`
  exposes the received log. The server deduplicates operations by client and
  sequence, acknowledges both first-time and repeated delivery, and serves full
  operation snapshots or cursor-based deltas for late and reconnecting clients.
- `SyncTransport` requires `send`, `onMessage`, and `close`.
- `MemoryTransport.pair()` creates linked asynchronous in-process transports.

The bundled sync layer is intentionally transport-agnostic and in-memory. It
does not provide authentication, durable server storage, compact state snapshots,
or conflict resolution. Applications must detect transport failure and supply a
replacement transport to `reconnect()`; production deployments must also
supply the remaining durability and security guarantees.
