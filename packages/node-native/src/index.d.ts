/**
 * Type declarations for the polypack-native wrapper.
 *
 * `NativeVectorIndex` and `NativeHnswIndex` are drop-in replacements for the
 * TypeScript `VectorIndex` / `HNSWIndex` classes backed by the Rust core.
 */

export type DistanceFunction = (a: ArrayLike<number>, b: ArrayLike<number>) => number

export interface ScoredId {
  id: string
  score: number
}

export interface EngineInfo {
  graph: string
  vector: string
  storage: string
  available: boolean
  query: 'rust-native' | 'typescript'
}

/** True when the native addon for this platform loaded successfully. */
export function isNativeAvailable(): boolean

/** Report which engine is active (rust-native, or the TypeScript fallback). */
export function engineInfo(): EngineInfo

/** Detect the engine currently used for a graph, for diagnostics. */
export function detectEngine(): string

/** Drop-in replacement for the TypeScript `VectorIndex`. */
export class NativeVectorIndex {
  constructor(onChange?: (id: string) => void, distance?: 'cosine' | 'euclidean')
  add(id: string, vector: number[] | Float64Array): void
  hydrate(id: string, vector: number[] | Float64Array): void
  addMany(entries: Array<{ id: string; vector: number[] | Float64Array }>): void
  remove(id: string): void
  removeMany(ids: string[]): void
  query(vector: number[], topK: number, threshold?: number): ScoredId[]
  clear(): void
  readonly size: number
  entries(): IterableIterator<[string, Float64Array]>
  has(id: string): boolean
  get(id: string): Float64Array | undefined
}

export interface NativeHnswConfig {
  M?: number
  Mmax0?: number
  efConstruction?: number
  efSearch?: number
}

/** Drop-in replacement for the TypeScript `HNSWIndex` (cosine distance only). */
export class NativeHnswIndex {
  constructor(onChange?: (id: string) => void, distanceFn?: DistanceFunction, config?: NativeHnswConfig)
  add(id: string, vector: number[] | Float64Array): void
  update(id: string, vector: number[] | Float64Array): void
  hydrate(id: string, vector: number[] | Float64Array): void
  addMany(entries: Array<{ id: string; vector: number[] | Float64Array }>): void
  remove(id: string): void
  removeMany(ids: string[]): void
  query(vector: number[], topK: number, threshold?: number): ScoredId[]
  clear(): void
  readonly size: number
  entries(): IterableIterator<[string, Float64Array]>
  has(id: string): boolean
  get(id: string): Float64Array | undefined
}

/** Factory for `PolyGraph`'s `createVectorIndex` hook using the native engine. */
export function createNativeVectorIndex(): (onChange: (id: string) => void) => NativeVectorIndex

/** Factory returning a native HNSW index. */
export function createNativeHnswIndex(onChange?: (id: string) => void): NativeHnswIndex

/**
 * Execute a query plan over serialized nodes/edges with the Rust query
 * executor, returning ordered node ids.
 */
export function executeQueryPlan(
  nodes: Array<Record<string, unknown>>,
  edges: Array<Record<string, unknown>>,
  plan: Record<string, unknown>,
): string[]

/** Aggregate a numeric field over the nodes matched by a query plan. */
export function aggregateQueryPlan(
  nodes: Array<Record<string, unknown>>,
  edges: Array<Record<string, unknown>>,
  plan: Record<string, unknown>,
  field: string,
  op: string,
): { value: number; count: number }

/**
 * Route `GraphQuery` (in-memory) through the Rust query executor when the
 * binary is present. Queries with join predicates fall back to TypeScript.
 */
export function installNativeQueryExecutor(): void

export interface NativeChangeBatch {
  putNodes?: Array<Record<string, unknown>>
  deleteNodeIds?: string[]
  putEdges?: Array<Record<string, unknown>>
  deleteEdgeIds?: string[]
  putVectors?: Array<{ id: string; vector: number[] }>
  deleteVectorIds?: string[]
}

/**
 * Directory-backed durable store over the Rust persistence state machine.
 * Files (`snapshot.msgpack`, `wal.msgpack`) are byte-compatible with the
 * TypeScript `BinaryStoreAdapter`.
 */
export class NativeStore {
  constructor(directory: string, compactThreshold?: number)
  static restore(source: string, destination: string, compactThreshold?: number): NativeStore
  apply(changes: NativeChangeBatch): void
  nodeIds(): string[]
  nodeCount(): number
  queryNodes(query: Record<string, unknown>): Array<Record<string, unknown>>
  countNodes(query: Record<string, unknown>): number
  defineIndex(definition: Record<string, unknown>): void
  dropIndex(name: string): boolean
  indexDefinitions(): Array<Record<string, unknown>>
  registerNodeType(definition: Record<string, unknown>): void
  registerEdgeType(definition: Record<string, unknown>): void
  getEdgesBySources(sources: string[], edgeType?: string): Array<Record<string, unknown>>
  getEdgesByTargets(targets: string[], edgeType?: string): Array<Record<string, unknown>>
  getNode(id: string): Record<string, unknown> | undefined
  allEdges(): Array<Record<string, unknown>>
  allVectors(): Array<[string, number[]]>
  compact(): void
  checkpoint(): void
  verify(): {
    ok: boolean
    errors: string[]
    nodeCount: number
    edgeCount: number
    vectorCount: number
    mutationCount: number
  }
  capabilities(): {
    atomicBatches: boolean
    transactions: boolean
    fsync: boolean
    secondaryIndexes: boolean
    snapshots: boolean
    changeFeed: boolean
    concurrentWriters: boolean
    vectorSearch: 'none' | 'exact' | 'ann'
  }
  stats(): {
    persistedNodeCount: number
    edgeCount: number
    vectorCount: number
    mutationCount: number
    walBytes: number
  }
  mutationLogSince(sequence: bigint, limit?: number): Array<Record<string, unknown>>
  latestMutationSequence(): bigint
  backup(destination: string): void
  close(): void
}
