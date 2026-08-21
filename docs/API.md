# API reference

Polypack is ESM-only and requires Node.js 18 or newer. The browser persistence
adapter additionally requires the File System Access API (OPFS). React is
optional and only loaded by `@0xx0lostcause0xx0/polypack/react`.

## `@0xx0lostcause0xx0/polypack`

### `PolyGraph`

```ts
new PolyGraph(
  adapter?: PersistenceAdapter,
  hotCacheMax?: number,
  embedding?: EmbeddingProvider,
  transform?: DataTransform,
  createVectorIndex?: (onChange: (id: string) => void) => VectorIndexLike,
)
```

The main property-graph container. Without an adapter it uses `MemoryAdapter`.
The default hot-node limit is 50,000. Edges remain indexed when nodes are
evicted, and dirty evicted nodes are retained until persistence completes.
Synchronous queries and mutations operate on currently loaded nodes; use
`getNodeSafe(id)` to restore an evicted node before mutating it.

The optional `transform` parameter provides `serialize`/`deserialize` hooks for
non-cloneable data (Blob, File, etc.) that cannot pass through
`structuredClone`. See the DataTransform section below.

The optional `createVectorIndex` factory swaps the vector engine backing the
public `vectors` property (a `VectorIndex` by default). It accepts anything
implementing `VectorIndexLike` — the structural interface shared by
`VectorIndex`, `HNSWIndex` (see Vector search below), and the native engines
from `@0xx0lostcause0xx0/polypack-native` (`NativeVectorIndex`/
`NativeHnswIndex` via `createNativeVectorIndex()`/`createNativeHnswIndex`) —
so any of them can be swapped in and still type-check:

```ts
import { PolyGraph, HNSWIndex } from '@0xx0lostcause0xx0/polypack'

const graph = new PolyGraph(undefined, undefined, undefined, undefined, (onChange) => new HNSWIndex(onChange))
```

Lifecycle:

- `warm()` / `load()` loads persisted nodes, vectors, and edges. Call it before
  querying an existing database. `warm()` is idempotent — subsequent calls are
  no-ops until the graph is cleared or disposed.
- `flush()` immediately persists queued mutations. Flushes are serialized.
- `save()` writes the complete currently loaded graph without clearing dirty state.
- `dispose()` flushes queued mutations, clears memory, then closes the adapter.
- `clear()` only clears in-memory state; it does not delete adapter contents.
- `prune(maxNodes)` removes the oldest loaded nodes and applies ownership rules.
- `transaction(callback, options?)` provides atomic read-your-own-writes,
  rollback, nested-transaction rejection, and post-commit events. Options may
  include `operationId`, `actor`, `baseRevision`, and `metadata`; these values
  are retained in the durable mutation record when the adapter supports a
  change feed.

Nodes:

- `addNode(node)` inserts or replaces a node. Replacement updates type/vector
  indexes. Node data and vectors are structured-cloned on entry.
- `addNodes(nodes)` inserts a batch of nodes. All inputs are validated before
  any are inserted (an invalid entry inserts nothing), change events are
  coalesced into one flush, and the persistence debounce is scheduled once for
  the whole batch. Prefer this over a loop of `addNode` for large inserts.
- `getNode(id)` returns a detached snapshot of a loaded node synchronously.
- `getNodeSafe(id)` restores an evicted node from persistence when necessary.
- `updateNode(id, data, vector?, activation?)` shallow-merges data and optionally
  replaces its vector or durable activation.
- `updateNodeSafe(id, data, vector?)` restores an evicted node before updating it.
- `removeNodeVector(id)` and `removeNodeVectorSafe(id)` explicitly clear a
  loaded or potentially evicted node's vector while retaining the node.
- `removeNode(id)` removes the node, all connected edges, and owned descendants.
- `removeNodeSafe(id)` restores and removes an evicted node, recursively restoring
  owned descendants as required. Call `warm()` first for an existing database so
  ownership edges are indexed.
- `whereType(type)` returns detached snapshots of loaded nodes of one type.
- `size` and `loadedSize` are the number of currently loaded nodes, not total
  persisted nodes. `hasLoadedNode(id)` checks membership in that working set.
- `persistedSize()` asynchronously returns the adapter's current node count.
- `getNodesByType(type)` loads all persisted nodes of a type (async convenience).
- `getNodesByTypeOrdered(type, field, direction?)` loads all persisted nodes of
  a type ordered by a data field (async convenience).
- `countNodesByType(type)` returns the persisted count for a node type (async).
- `deleteNodesByType(type)` removes all persisted nodes of a type (async).
- `vectors` is the public `VectorIndex` (or substituted engine) backing
  similarity search. Mutating it directly (rather than through `addNode`/
  `updateNode`) does not schedule persistence; call `markVectorDirty(id)`
  afterwards so the change is picked up on the next flush.

Activation — durable primitives (see the `activation` subpath for the engine):

- `reinforceNode(id, amount, reason?)` applies a durable reinforcement delta to
  a loaded node: the prior state is decay-corrected to now, `amount` is added to
  `score`, a fraction is folded into `importance`, `reinforcementCount`
  increments, and `lastMeaningfulActivation` re-anchors to now. Persists and
  emits an `activation_updated` change event (with `delta`/`reason`). Returns
  the updated node, or `undefined` when the node isn't loaded.
- `reinforceNodeSafe(id, amount, reason?)` restores an evicted node first.
- `getActivation(id, halfLifeMs?)` returns the current decay-corrected score
  (0 when the node has none). Decay is a pure function of elapsed time from
  `lastMeaningfulActivation`, so replicas with the same stored state converge.
- `getActivationState(id)` returns the decay-corrected durable record or
  `undefined`.
- `topActivated(limit, minScore?)` returns loaded nodes ranked by current
  activation descending — the working-memory primitive.
- `decay(now?)` materializes decayed values for all loaded nodes and re-anchors
  them. Reads already decay lazily, so this only matters for persisting fresh
  values (e.g. before eviction-driven lifecycle events).

Convenience — graph traversal:

- `walkAncestors(id, edgeType)` walks the parent chain backwards through
  incoming edges of `edgeType`. Returns `PolyNode[]` from root to start node
  (inclusive). Detects cycles.
- `walkDescendants(id, edgeType)` walks the child chain forwards through
  outgoing edges of `edgeType`. Returns `PolyNode[]` from start node to deepest
  child (inclusive). Detects cycles.

Full-text search:

- `searchNodes(text, type, threshold?, topK?)` shorthand for
  `queryPersistedText(text, threshold, topK).then(q => q.whereNodeType(type).toArray())`.
  Returns `Promise<PolyNode[]>`.

Edges:

- `addEdge(source, type, target, data?, ownership?)` adds one unique directed edge.
- `getEdges(source, type?)` returns detached outgoing edge descriptors.
- `getEdgeTargets(source, type)` and `getEdgeSources(target, type)` return IDs.
- `removeEdges(source, type?, target?)` removes matching outgoing edges.

Ownership is stored on the edge:

- `reference` (default): removing the edge never removes its target.
- `shared`: keeps the target, and invokes the protected `onOrphan` hook when it
  loses its final incoming edge.
- `owned`: removes the target when it loses its final owning source; cascading
  is cycle-safe.

Reactivity and batching:

- `changes` is an RxJS `Subject<GraphChangeEvent>` — emits `node_added`,
  `node_updated`, `node_removed`, `edge_added`, `edge_removed`, and
  `activation_updated` (the last carries optional `delta`/`reason`).
- `startBatch()` queues notifications until the matching `endBatch()`.
- `endBatch()` throws if no batch is open.

`query()` creates a mutable `GraphQuery`.

Public graph reads, query results, join/collection predicate values, edge data,
and vector-index reads are detached copies. Mutate graph state through graph or
vector-index methods so persistence tracking and reactive notifications remain
correct. Data values must be compatible with the platform's `structuredClone`.

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
- `whereActivated(above)` keeps only nodes whose current (decay-corrected)
  activation exceeds `above`.
- `orderByActivation(direction?)` orders results by current activation instead of
  a data field (default `desc`).

Limits, offsets, traversal depths, and top-K values must be non-negative
integers. Timestamps must be finite and non-negative; vectors and numeric range
boundaries must contain finite numbers. Node IDs/types and edge endpoints/types
must not be empty.

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
`similarTo`, `whereActivated`, `orderByActivation`, `offset`, and `limit`. The
asynchronous terminal methods are `toArray()`, `first()`, `count()`, `ids()`,
and `collect()`. Activation filters/ordering are applied post-load (like
similarity), so they disable adapter-side pagination.

Adapters may implement `queryNodes(query)` and `countNodes(query)` for optimized
storage-level execution, plus `getEdgesBySources(ids, type?)` and
`getEdgesByTargets(ids, type?)` for indexed graph operations. When absent,
Polypack falls back to the original node and edge methods, preserving
compatibility with existing custom adapters. `countNodes({})` returns the total
persisted node count without materialising ids, and type-only queries use a
secondary type index. `getEdgesBySources`/`getEdgesByTargets` are backed by
source/target edge indexes, so persisted traversal no longer scans every edge.

For node-only queries, offset and limit are delegated to the adapter.
`BinaryStoreAdapter` uses an in-memory snapshot for persisted queries. Queries
with similarity or graph post-processing retain pagination in the query layer so
that filtering, traversal, and ranking occur before the page is selected.

### Vector search

- `new VectorIndex(onChange?, distanceFn?)` creates an exact in-memory index.
- `add`, `addMany`, `remove`, `removeMany`, `clear`, `has`, `get`, `entries`, and
  `size` manage vectors. `add`/`addMany` invoke `onChange` for each id;
  `hydrate(id, vector)` sets a vector without triggering `onChange`, for
  restoring persisted state without marking it dirty again.
- `query(vector, topK, threshold?)` returns `{ id, score }[]`, highest first.
- `cosineSimilarity(a, b)` and `euclideanSimilarity(a, b)` are built in.

All compared vectors must have identical dimensions; otherwise similarity
functions throw `RangeError`. Zero vectors have cosine similarity `0`. Added
vectors are copied, and `get()`/`entries()` return detached arrays.

`HNSWIndex` (`new HNSWIndex(onChange?, distanceFn?, config?, rng?)`) is an
approximate, update-safe alternative implementing the same `VectorIndexLike`
surface (`add`/`addMany`/`hydrate`/`remove`/`removeMany`/`query`/`clear`/
`has`/`get`/`entries`/`size`). `config` accepts `M`, `Mmax0`, `efConstruction`,
and `efSearch` (all optional, cosine distance by default) — each, if given,
must be a positive integer, or the constructor throws `RangeError`. It can be
passed directly to `PolyGraph`'s `createVectorIndex` hook — see the
constructor section above. `update(id, vector)` also exists but is
`@deprecated`: it's identical to `add()` (which already overwrites an
existing id) and isn't part of `VectorIndexLike`, so prefer `add()`.

### Text embeddings

- `EmbeddingProvider` defines `embed(text)` and an optional `dimensions`. The
  method may be synchronous or asynchronous and may return a number array,
  `Float32Array`, or `Float64Array`.
- `FeatureHashEmbedding` is the default dependency-free provider. It produces
  deterministic, normalized 384-dimensional lexical vectors without downloading
  a model. Its dimensions can be configured in the constructor.
- Pass a custom provider as the third `PolyGraph` constructor argument:

  ```ts
  const provider = {
    dimensions: 768,
    async embed(text: string) {
      return model.embed(text)
    },
  }
  const graph = new PolyGraph(adapter, 10_000, provider)
  ```

- `embed(text)` generates and validates a detached `Float64Array`.
- `addNodeWithEmbedding(node, text)` adds a node using generated text features.
- `updateNodeWithEmbedding(id, data, text)` and
  `updateNodeSafeWithEmbedding(id, data, text)` regenerate a node vector.
- `queryText(text, threshold?, topK?)` returns a `Promise<GraphQuery>` already
  configured for similarity search.
- `queryPersistedText(text, threshold?, topK?)` does the same across the complete
  persisted dataset.

The default provider captures shared words rather than learned semantic meaning.
Use a model-backed provider when synonyms and deeper language relationships are
required. Keep one provider and dimensionality for a vector index; similarity
comparisons reject vectors with mismatched dimensions.

### Persistence

- `MemoryAdapter(maxNodes?)` stores serialized records in memory. When
  `maxNodes` is set, the least-recently-put node (plus its edges/vector) is
  evicted once the cap is reached; writing a node again (via `putNode`,
  `bulkPutNodes`, or `applyChanges`) bumps it back to most-recently-put.
- `BinaryStoreAdapter({ storeDir, compactThreshold?, fileIO?, syncWrites? })`
  persists data as a MessagePack snapshot plus an append-only write-ahead log
  (WAL). Nodes, edges, and vectors are committed atomically per batch; the WAL
  is compacted into a snapshot once it passes an adaptive threshold and on
  `close()`. `compactThreshold` is the minimum WAL-entry count at which
  compaction is scheduled (default 10,000); the effective threshold also grows
  with the store (`max(threshold, records / 4)`), so a 1M-node build no longer
  rewrites the snapshot quadratically. Startup replays the WAL, then persists a
  snapshot before deleting it so a crash between those steps loses nothing.
  Recovery also tolerates a truncated WAL tail from a mid-append crash.
  Schema definitions supplied in an atomic `applyChanges` commit are included
  in the WAL and snapshot; the legacy `schemas.json` sidecar remains readable
  for older stores.
- `syncWrites: true` fsyncs WAL appends and snapshot writes (including the
  containing directory) for crash durability at a throughput cost.
- `BinaryStoreAdapter` lives behind platform subpaths so the core entry point
  stays free of `node:` built-ins:
  - `@0xx0lostcause0xx0/polypack/persistence/node` — `NodeFileIO` (filesystem).
  - `@0xx0lostcause0xx0/polypack/persistence/opfs` — `OPFSFileIO` (browser
    File System Access API).
  - `@0xx0lostcause0xx0/polypack/persistence` — `MemoryFileIO` and the `FileIO`
    type for tests and custom storage. When `fileIO` is omitted, a platform
    default is created at first use.
- `FileIO` is the storage contract: `readFile`, `writeFile`, `appendFile`,
  `deleteFile`, `fileExists`. Implement it to plug in any backing store.
- `PersistenceAdapter` is the contract for custom storage. It contains node,
  edge, and vector single/bulk operations plus `clearAll()` and `close()`.
- Adapters that expose `changeFeed` provide the durable logical log through
  `graph.mutationLogSince(sequence)`, `graph.mutationLogPage(sequence, limit)`,
  and `graph.latestMutationSequence()`.
  Sequences are exclusive `bigint` cursors, so callers can resume replication
  or audit scans without rereading acknowledged records.
- Adapters may implement `getSchemaDefinitions()` and
  `setSchemaDefinitions()` to persist structural node/edge schema metadata.
  The canonical file shape is `nodeTypes[{ nodeType, ... }]` and
  `edgeTypes[{ edgeType, ... }]`, shared with the Rust and Python bindings.
  Runtime validator callbacks are intentionally not serialized; applications
  must register those callbacks again after opening a store.
- `PersistenceChanges` describes one logical node/edge/vector commit. Adapters
  may implement `applyChanges(changes)` to commit it atomically; `PolyGraph`
  prefers this hook and restores the complete dirty batch when it rejects.

Adapter methods should reject on storage errors. Bulk methods should be atomic
where the backing store permits it. `MemoryAdapter` applies changes through
copy-on-commit maps, while `BinaryStoreAdapter` appends one WAL batch covering
all three record kinds. Existing custom adapters without `applyChanges` remain
compatible but cannot guarantee cross-record atomicity through the fallback
path.

### Types and utilities

The root exports `PolyNode`, `PolyEdge`, `EdgeOwnership`, `GraphChangeEvent`,
`SerializedNode`, `SerializedEdge`, `VectorQuery`, `EdgeTypes`, `DataTransform`,
`NodeActivation`, aggregate types, `DistanceFunction`, `PersistenceAdapter`,
`ActivationEngine`, and `mergeActivation`. Persistence types and adapters beyond
`MemoryAdapter` live under the `persistence` subpaths.

`edgeId(source, type, target)` produces the persistence edge key. Source IDs and
edge types must not contain `::`; target IDs may contain it.
`yieldToUI()` yields through a zero-delay timer.

#### `DataTransform`

Optional hooks for handling non-cloneable data (Blob, File, etc.) that cannot
pass through `structuredClone`. Pass as the fourth PolyGraph constructor
argument:

```ts
const blobStore = new Map<string, Blob>()

const transform: DataTransform = {
  serialize(data) {
    const copy = { ...data }
    if (copy.blob instanceof Blob) {
      blobStore.set(copy.id, copy.blob)
      copy.blob = null
    }
    return { data: copy }
  },
  deserialize(data) {
    const copy = { ...data }
    if ('blob' in copy && copy.blob === null) {
      copy.blob = blobStore.get(copy.id) ?? null
    }
    return copy
  },
}

const graph = new PolyGraph(adapter, 10_000, undefined, transform)
```

`serialize` returns `{ data: cloneableData, sidecar?: unknown }`. The sidecar
is stored in memory alongside the node and re-supplied to `deserialize` on every
read. The sidecar is **not** persisted across sessions.

#### `defineEdges`

Create a frozen edge-type constant object with full literal-type inference:

```ts
import { defineEdges } from '@0xx0lostcause0xx0/polypack'

export const EDGE = defineEdges({
  KNOWS: 'KNOWS',
  LIKES: 'LIKES',
})
// typeof EDGE.KNOWS === 'KNOWS' (literal string type)
```

#### `buildEmbeddingText`

Build a weighted embedding string by repeating fields according to their weight.
The default weight is 1; fields with higher weights are repeated more often so
the feature-hash bag-of-words embedding treats them as more significant:

```ts
import { buildEmbeddingText } from '@0xx0lostcause0xx0/polypack'

buildEmbeddingText(
  { subject: 'Hello', content: 'World' },
  { subject: 3 }
)
// => "Hello Hello Hello World"
```

## `@0xx0lostcause0xx0/polypack/react`

```ts
useGraphQuery(graph, queryFn, deps, delay?, nodeTypes?, onError?)
useLiveQuery(graph, querier, deps?, defaultResult?)
```

Both hooks execute immediately, subscribe to graph changes, and support sync or
async queries. `useGraphQuery` defaults to a 200 ms debounce. When `nodeTypes`
is supplied, events for other known node types are ignored. Query errors are
logged, passed to the optional `onError(error)` callback, and set the result to
`undefined`.

The dependency list controls query closure refresh in the same way as other
React hooks. Changing `nodeTypes` or `delay` updates the active subscription
without requiring dependency-list changes. In-flight queries are serialized;
stale results are discarded after dependency changes or unmounting, and one
follow-up run is retained when mutations arrive during an asynchronous query.
React 18 and 19 are supported as an optional peer dependency.

`useWorkingMemory(graph, limit?, deps?, delay?, nodeTypes?, engine?)` is a live
view of the current working memory: the `limit` most-activated loaded nodes,
re-queried after any graph change (including `activation_updated`). `deps`
controls re-runs and must include `limit` when it can change. Without `engine`
it ranks by `graph.topActivated` (durable score only); pass an
`ActivationEngine` to rank by `engine.workingMemory` instead (durable score
plus transient attention from `bumpAttention`). Sub-threshold `bumpAttention`
calls never emit a graph change event, so they won't trigger an automatic
re-run on their own — include something that changes after a bump in `deps`
if the view needs to reflect it immediately.

## `@0xx0lostcause0xx0/polypack/activation`

The adaptive-memory layer. It splits activation into two tiers:

- **Durable** — `NodeActivation` (`score`, `importance`, `reinforcementCount`,
  `lastMeaningfulActivation`) rides as an optional field on every node, so it
  persists through the snapshot/WAL and adapters and replicates through sync.
- **Transient** — runtime-only attention held by `ActivationEngine`, never
  serialized or synced.

```ts
import { PolyGraph, ActivationEngine } from '@0xx0lostcause0xx0/polypack'

const graph = new PolyGraph()
const engine = new ActivationEngine(graph)

graph.reinforceNode('article', 1.0, 'user_read')      // durable + synced
engine.bumpAttention('article', 0.2)                  // local only
engine.effective('article')                           // durable + attention

const spread = engine.spread(['article'], { depth: 2, decay: 0.5 })  // neighbours warm up
const scores = await engine.pulse('vector search')    // semantic region scoring
await engine.absorb('vector search')                  // pulse + reinforce above threshold

engine.workingMemory(5)                               // current "mental state"
```

- `new ActivationEngine(graph, config?)` composes the scoring layer. `config`
  options: `scoreHalfLifeMs` (24 h default), `importanceHalfLifeMs` (30 days),
  `importanceGain` (0.05), `spreadDecay` (0.5), `spreadDepth` (2),
  `recencyHalfLifeMs` (7 days), `weights` (all 1), `minReinforceDelta` (0.05),
  `pulseThreshold` (0), `absorbThreshold` (0.3), `absorbGain` (0.05).
- `reinforce(id, amount, reason?)` / `reinforceAll(entries)` call through to
  `PolyGraph.reinforceNode` (durable).
- `bumpAttention(id, amount)`, `attentionOf(id)`, and `effective(id)` manage the
  transient tier. `bumpAttention` accumulates locally and is promoted to durable
  reinforcement once it clears `minReinforceDelta`, so tiny events (scrolls,
  focus) stay local while meaningful ones become persisted and synced.
- `spread(seeds, { depth?, decay?, edgeTypes? })` implements spreading
  activation: each hop attenuates the contribution by `decay`; multiple paths to
  a node sum. Returns `{ nodeId: contribution }`.
- `pulse(text | vector, { topK?, semanticThreshold?, pulseThreshold?, ... })`
  scores the activated region around a query — semantic seeds via vector
  similarity (nodes with zero similarity never seed the region) plus outward
  spreading, folded with recency and usage. Read-only.
- `absorb(input, options?)` runs `pulse` and durably reinforces every node whose
  composite clears `absorbThreshold` by `absorbGain * score`.
- `workingMemory(limit?, minScore?)` returns loaded nodes ranked by `effective`
  activation descending.
- `dispose()` unsubscribes from graph changes and drops transient attention.
- `mergeActivation(existing, incoming, now?)` merges two durable total-state
  records: decay-corrects both to `now`, keeps the stronger component of each,
  and re-anchors to `now`. Used by the sync layer (max-merge, idempotent for
  re-delivered snapshots).

Decay is a pure function of elapsed time anchored at
`lastMeaningfulActivation` (`0.5 ** (elapsed / halfLife)`), so two replicas
with the same stored state compute identical current scores. Synchronization is
**additive for deltas** (the `activationUpdate` op, coalesced and gated by
`activationSyncThreshold`, default 0.05) and **max for total-state node
payloads** — activation is accumulated knowledge, not last-write-wins data.

## `@0xx0lostcause0xx0/polypack/sync`

- `OpLog(clientId, existing?)` appends sequenced `SyncOp` values and exposes
  `since(seq)`, `all`, `latestSeq`, and `size`.
- `MemorySyncClientStateStore` and `FileSyncClientStateStore` persist pending
  operations plus the acknowledged client and server cursors. Use
  `await SyncClient.restore({ graph, transport, stateStore, clientId? })` to
  resume a client after restart. `await client.persist()` waits for the latest
  local state to be durable; the file store includes a checksum and rejects
  corrupted state.
- `SyncAdapter(inner, clientId)` wraps persistence and records successful node
  and edge writes in its `oplog`; set `onOp` to observe them.
- `SyncClient({ graph, transport, clientId?, autoFlush?, retryMs?,
  activationSyncThreshold?, stateStore? })` captures graph events, retains operations until
  acknowledged, retries unacknowledged deltas, detects server-cursor gaps, and
  applies remote operations with echo suppression. `retryMs` defaults
  to 1,000 ms and `0` disables automatic retry. `activationSyncThreshold`
  (default 0.05) drops coalesced activation deltas below that magnitude
  instead of syncing them. Use `flush()` for manual sends,
  inspect `pendingOps`, call `requestSync(fromStart?)` for catch-up, inspect
  `syncCursor`, call `reconnect(transport)` to replace a transport, resend pending
  work, and request missing server operations, and use `disconnect()` to flush
  any pending operations and close.
- `SyncServer` is an in-memory relay by default. Pass a `SyncOperationLog` as
  `operationLog` and await `ready()` before accepting clients to restore and
  durably append the server history. `maxBatchOps` bounds an incoming envelope
  and `maxPendingOps` applies back-pressure to the durable queue. A rejected
  envelope receives an `ack` with `pending_too_large`; retry it after the queue
  drains. `await server.flush()` waits for all durable submissions accepted so
  far to reach the operation log. `addClient(handle)` returns that client's
  incoming-message handler, `removeClient(handle)` unregisters it, and `ops`
  exposes the received log. The server deduplicates operations by client and
  sequence, operation, and transaction identity, acknowledges both first-time
  and repeated delivery, and serves full operation snapshots or cursor-based
  deltas for late and reconnecting clients. Authorization and conflict hooks
  validate transaction groups atomically: if one operation is rejected, none of
  that transaction is committed or broadcast. `await server.logStats()` reports
  cursor retention, retained operations, and operation/transaction identity
  counts for monitoring and compaction tooling.
- `SyncTransport` requires `send`, `onMessage`, and `close`.
- `MemoryTransport.pair()` creates linked asynchronous in-process transports.

The bundled sync layer is intentionally transport-agnostic. It provides optional
durable server and client logs, authentication and conflict hooks, bounded
batches, checksums, cursor recovery, and filtered subscriptions; it does not
provide identity management, application permissions, or a domain-specific
conflict resolver. Applications must detect transport failure and supply a
replacement transport to `reconnect()`.
