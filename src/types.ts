/**
 * Durable activation state for a node (see the `activation` module).
 *
 * All fields are persisted and replicated. The two-tier model separates this
 * learned relevance from transient, runtime-only attention (held by
 * {@link import('./activation.js').ActivationEngine}), which is never
 * serialized or synced.
 *
 * Decay is a pure function of elapsed time anchored at
 * `lastMeaningfulActivation`, so replicas compute identical current scores
 * from the same durable state plus their clocks.
 */
export interface NodeActivation {
  /** Current learned activation, decay-corrected on read. Clamped to [0, 1]. */
  score: number
  /** Long-term relevance that decays far slower than `score` (or never). Clamped to [0, 1]. */
  importance: number
  /** How many times this node has been meaningfully reinforced. */
  reinforcementCount: number
  /** Epoch-ms anchor for decay. Both `score` and `importance` decay from here. */
  lastMeaningfulActivation: number
}

/** A typed property-graph node. Timestamps are Unix milliseconds. */
export interface PolyNode<TData extends Record<string, unknown> = Record<string, unknown>> {
  id: string
  type: string
  data: TData
  vector?: Float64Array
  insertedAt: number
  updatedAt: number
  /** Monotonically increasing revision. Inputs may omit this and default to 0. */
  revision?: number
  /** Durable, syncable activation. Optional — absent until the node is first reinforced. */
  activation?: NodeActivation
}

/** Controls what happens to a target when its incoming edge is removed. */
export type EdgeOwnership = 'owned' | 'shared' | 'reference'

/** Serializable representation of a directed property-graph edge. */
export interface PolyEdge<TData extends Record<string, unknown> = Record<string, unknown>> {
  id: string
  source: string
  target: string
  type: string
  data?: TData
  createdAt: number
  /** Monotonically increasing revision. Inputs may omit this and default to 0. */
  revision?: number
}

/** Mutation notification emitted through {@link PolyGraph.changes}. */
export interface GraphChangeEvent {
  type: 'node_added' | 'node_updated' | 'node_removed' | 'edge_added' | 'edge_updated' | 'edge_removed' | 'activation_updated'
  nodeId?: string
  nodeType?: string
  edgeId?: string
  edgeType?: string
  source?: string
  target?: string
  /** Present on `activation_updated` events: the applied reinforcement delta. */
  delta?: number
  /** Present on `activation_updated` events: an optional reinforcement reason. */
  reason?: string
  /** Transaction that produced this event, when the mutation ran in one. */
  transactionId?: string
}

/** Persistence-safe node representation using plain arrays for vectors. */
export interface SerializedNode {
  id: string
  type: string
  data: Record<string, unknown>
  vector: number[] | null
  insertedAt: number
  updatedAt: number
  revision?: number
  /** Durable activation state. Persisted through the existing snapshot/WAL and adapters. */
  activation?: NodeActivation
}

/** Persistence-safe edge representation. */
export interface SerializedEdge {
  id: string
  source: string
  target: string
  type: string
  data: Record<string, unknown> | null
  createdAt: number
  revision?: number
}

export interface WriteOptions {
  expectedRevision?: number
}

export interface NodePatch {
  set?: Record<string, unknown>
  unset?: string[]
  increment?: Record<string, number>
  compareAndSet?: Record<string, { expected: unknown; value: unknown }>
}

export interface AdapterCapabilities {
  atomicBatches: boolean
  transactions: boolean
  fsync: boolean
  secondaryIndexes: boolean
  snapshots: boolean
  changeFeed: boolean
  concurrentWriters: boolean
  vectorSearch: 'none' | 'exact' | 'ann'
}

export interface IndexDefinition {
  name: string
  nodeType?: string
  fields: string[]
  unique?: boolean
  sparse?: boolean
}

export type EdgeCardinality = 'one-to-one' | 'one-to-many' | 'many-to-one' | 'many-to-many'

export interface NodeTypeDefinition {
  validate?: (node: PolyNode) => void | boolean
  indexes?: string[]
  requiredFields?: string[]
  dataTypes?: Record<string, 'string' | 'number' | 'integer' | 'boolean' | 'object' | 'array'>
}

export interface EdgeTypeDefinition {
  sourceTypes?: string[]
  targetTypes?: string[]
  cardinality?: EdgeCardinality
  validate?: (edge: PolyEdge) => void | boolean
  requiredFields?: string[]
  dataTypes?: Record<string, 'string' | 'number' | 'integer' | 'boolean' | 'object' | 'array'>
}

/** Serializable schema metadata; runtime validator callbacks are omitted. */
export interface PersistedSchemaDefinitions {
  nodeTypes: Array<{ nodeType: string } & Omit<NodeTypeDefinition, 'validate'>>
  edgeTypes: Array<{ edgeType: string } & Omit<EdgeTypeDefinition, 'validate'>>
}

export interface QueryExplain {
  index?: string
  indexes?: string[]
  stages: string[]
  estimatedCost: number
  loadedRecords: number
}

export type GraphOperation = {
  type: 'putNode' | 'deleteNode' | 'putEdge' | 'deleteEdge' | 'putVector' | 'deleteVector' | 'setIndexes' | 'setSchema'
  payload: Record<string, unknown>
}

export interface MutationRecord {
  operationId: string
  sequence: bigint
  transactionId: string
  actor?: string
  timestamp: number
  baseRevision?: number
  operations: GraphOperation[]
  metadata?: Record<string, unknown>
}

export interface VerificationReport {
  ok: boolean
  errors: string[]
  nodeCount: number
  edgeCount: number
  vectorCount: number
  mutationCount: number
}

export interface GraphStats {
  loadedNodeCount: number
  persistedNodeCount: number
  edgeCount: number
  vectorCount: number
  dirtyRecordCount: number
  pendingPersistence: boolean
  indexCount: number
  mutationCount?: number
  walBytes?: number
  memoryEstimateBytes?: number
  queryCount?: number
  queryDurationMs?: number
  queryScannedRecords?: number
  queryIndexUsage?: Record<string, number>
}

export interface QueryMetrics {
  durationMs: number
  scannedRecords: number
  index?: string
  indexes?: string[]
}

export interface QueryResourceLimits {
  signal?: AbortSignal
  maxTraversalDepth?: number
  maxNodesVisited?: number
  maxResults?: number
}

export interface GraphResourceLimits {
  maxVectorDimensions?: number
  maxNodePayloadBytes?: number
  maxBatchSize?: number
}

export interface TransactionOptions {
  actor?: string
  baseRevision?: number
  metadata?: Record<string, unknown>
}

export interface GraphTransaction {
  readonly id: string
  addNode(node: PolyNode, options?: WriteOptions): void
  addEdge(edge: PolyEdge, options?: WriteOptions): void
  updateEdge(id: string, data: Partial<Record<string, unknown>>, options?: WriteOptions): PolyEdge | undefined
  updateNode(id: string, data: Partial<Record<string, unknown>>, options?: WriteOptions): PolyNode | undefined
  patchNode(id: string, patch: NodePatch, options?: WriteOptions): PolyNode | undefined
  removeNode(id: string, options?: WriteOptions): void
  removeEdge(id: string, options?: WriteOptions): boolean
  getNode(id: string): PolyNode | undefined
  query(): unknown
}

/** Parameters for a nearest-neighbour vector search. */
export interface VectorQuery {
  vector: number[]
  threshold?: number
  topK: number
}

/**
 * Optional transform for handling non-cloneable data (Blob, File, etc.) that
 * cannot pass through `structuredClone`. Return `sidecar` from `serialize` —
 * it is stored in memory alongside the node and re-supplied to `deserialize`
 * on every read. The sidecar is NOT persisted across sessions.
 */
export interface DataTransform {
  serialize?(data: Record<string, unknown>): { data: Record<string, unknown>; sidecar?: unknown }
  deserialize?(data: Record<string, unknown>, sidecar?: unknown): Record<string, unknown>
}

/** Infer the edge-type value type from a definition object. */
export type EdgeTypes<T extends Record<string, string>> = { readonly [K in keyof T]: T[K] }

/**
 * Create a frozen edge-type constant object with full type inference.
 *
 * ```ts
 * export const EDGE = defineEdges({
 *   HAS_LABEL: 'HAS_LABEL',
 *   REPLIES_TO: 'REPLIES_TO',
 * })
 * // typeof EDGE.HAS_LABEL === 'HAS_LABEL'  (literal type)
 * ```
 */
export function defineEdges<T extends Record<string, string>>(def: T): EdgeTypes<T> {
  return Object.freeze({ ...def }) as EdgeTypes<T>
}
